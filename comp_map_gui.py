#!/usr/bin/env python3

"""
Compressor Map GUI

- Reuses map I/O from scmap.py (read/write .fae)
- Plots flow (CFM/10.323) vs PR with speed lines
- Fits a polynomial surface Efficiency(flow_cfm, PR) and draws iso-efficiency contours
- Lets user pick generic operating points (HD_WG / MD_WG) defined as % of max flow/PR
- Interpolates efficiency at those points and reports weighted average
- Allows adjusting contour min/max/step and saving the output table to CSV

Notes
- .fae flow values are in CFM. Plotting uses CFM/10.323 per brief.
- The regression uses (flow_cfm, PR) space; we convert only for plotting axes.
"""

from __future__ import annotations

import os
import csv
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.spatial import Delaunay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Local import: reuse read_map from scmap
import scmap  # type: ignore


# --------------------------------------------------------------------------------------
# Constants / Logging
# --------------------------------------------------------------------------------------

CFM_TO_PLOT_FLOW = 10.323  # Flow plotting in m.t^0.5/p = CFM / 10.323

logger = logging.getLogger("comp_map_gui")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class Point:
    flow_cfm: float
    pr: float
    eff: float


@dataclass
class SpeedLine:
    rpm: float
    pts: List[Point]


@dataclass
class CompMap:
    title: str
    speed_lines: List[SpeedLine]
    max_flow_cfm: float
    max_pr: float

    # learned models
    poly_features: Optional[PolynomialFeatures] = None
    lin_model: Optional[LinearRegression] = None
    poly_degree: Optional[int] = None
    rmse: Optional[float] = None

    # (optional) surge spline flow(PR) in plotting units if desired later
    surge_spline: Optional[UnivariateSpline] = None


# --------------------------------------------------------------------------------------
# Generic operating point sets (Flow%, PR%, Weight)
# --------------------------------------------------------------------------------------

GENERIC_SETS: Dict[str, List[Tuple[float, float, float]]] = {
    "HD_WG": [
        (0.7, 0.7, 0.3),
        (0.6, 0.6, 0.3),
        (0.5, 0.4, 0.4),
    ],
    "MD_WG": [
        (0.7, 0.7, 0.3),
        (0.5, 0.6, 0.3),
        (0.4, 0.4, 0.4),
    ],
}


# --------------------------------------------------------------------------------------
# Map I/O helpers (wrapping scmap.MapData)
# --------------------------------------------------------------------------------------

def load_map(path: str) -> CompMap:
    m: scmap.MapData = scmap.read_map(path, read_format="auto")

    title = "".join(m.title).strip()
    speed_lines: List[SpeedLine] = []
    max_flow_cfm = -math.inf
    max_pr = -math.inf

    for i in range(m.ns):
        rpm = float(m.sp[i])
        line_pts: List[Point] = []
        for j in range(m.nr):
            pr = float(m.pr[i][j])
            flow_cfm = float(m.mf[i][j])  # stored as CFM per brief
            eff = float(m.ef[i][j])
            line_pts.append(Point(flow_cfm=flow_cfm, pr=pr, eff=eff))
            max_flow_cfm = max(max_flow_cfm, flow_cfm)
            max_pr = max(max_pr, pr)
        speed_lines.append(SpeedLine(rpm=rpm, pts=line_pts))

    return CompMap(
        title=title or os.path.basename(path),
        speed_lines=speed_lines,
        max_flow_cfm=max_flow_cfm,
        max_pr=max_pr,
    )


# --------------------------------------------------------------------------------------
# Regression / interpolation
# --------------------------------------------------------------------------------------

def fit_efficiency_regression(cmap: CompMap, degree_min: int = 2, degree_max: int = 5) -> None:
    flows: List[float] = []
    prs: List[float] = []
    effs: List[float] = []
    for sl in cmap.speed_lines:
        for p in sl.pts:
            flows.append(p.flow_cfm)
            prs.append(p.pr)
            effs.append(p.eff)

    X = np.vstack([flows, prs]).T
    y = np.array(effs)

    best_rmse = np.inf
    best_model = None
    best_poly = None
    best_deg = None

    for deg in range(degree_min, degree_max + 1):
        poly = PolynomialFeatures(deg)
        Xp = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        preds = model.predict(Xp)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        logger.debug("Degree %d RMSE=%.5f", deg, rmse)
        if rmse < best_rmse:
            best_rmse, best_model, best_poly, best_deg = rmse, model, poly, deg

    cmap.poly_features = best_poly
    cmap.lin_model = best_model
    cmap.poly_degree = best_deg
    cmap.rmse = best_rmse
    logger.info("Selected polynomial degree %s (RMSE=%.5f)", best_deg, best_rmse)


def predict_efficiency(cmap: CompMap, flow_cfm: float, pr: float) -> float:
    if cmap.poly_features is None or cmap.lin_model is None:
        raise RuntimeError("Regression model not fitted")
    Xp = cmap.poly_features.transform([[flow_cfm, pr]])
    val = float(cmap.lin_model.predict(Xp)[0])
    return float(np.clip(val, 0.0, 100.0))


def compute_convex_hull_mask(cmap: CompMap, F_cfm: np.ndarray, P: np.ndarray) -> np.ndarray:
    data = np.array([[p.flow_cfm, p.pr] for sl in cmap.speed_lines for p in sl.pts])
    tri = Delaunay(data)
    mask1d = tri.find_simplex(np.c_[F_cfm.ravel(), P.ravel()]) >= 0
    return mask1d.reshape(F_cfm.shape)


def contour_data(cmap: CompMap, n: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid (F_cfm, P), predicted efficiency E and mask inside convex hull."""
    if cmap.poly_features is None or cmap.lin_model is None:
        raise RuntimeError("Regression model not fitted")

    flows = [p.flow_cfm for sl in cmap.speed_lines for p in sl.pts]
    prs = [p.pr for sl in cmap.speed_lines for p in sl.pts]
    fmin, fmax = min(flows), max(flows)
    pmin, pmax = min(prs), max(prs)

    cfms = np.linspace(fmin, fmax, n)
    pr_grid = np.linspace(pmin, pmax, n)
    F_cfm, P = np.meshgrid(cfms, pr_grid)
    X = np.vstack([F_cfm.ravel(), P.ravel()]).T
    E = cmap.lin_model.predict(cmap.poly_features.transform(X)).reshape(F_cfm.shape)
    E = np.clip(E, 0.0, 100.0)
    mask = compute_convex_hull_mask(cmap, F_cfm, P)
    E_masked = np.where(mask, E, np.nan)
    return F_cfm, P, E_masked, mask


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Compressor Map Plotter")
        self.geometry("1200x800")

        # State
        self.cmap: Optional[CompMap] = None
        self.current_set_name: str = "HD_WG"

        # Controls
        self._build_topbar()
        self._build_plot_and_table()

    # ---------------- UI build ----------------
    def _build_topbar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # File selector
        ttk.Label(bar, text="Map file:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(bar, textvariable=self.path_var, width=60)
        self.path_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Browse", command=self.on_browse).pack(side=tk.LEFT)
        ttk.Button(bar, text="Load", command=self.on_load).pack(side=tk.LEFT, padx=4)

        # Generic set picker
        ttk.Label(bar, text="Generic set:").pack(side=tk.LEFT, padx=(16, 4))
        self.set_var = tk.StringVar(value=self.current_set_name)
        set_pick = ttk.Combobox(bar, textvariable=self.set_var, values=list(GENERIC_SETS.keys()), state="readonly")
        set_pick.pack(side=tk.LEFT)
        set_pick.bind("<<ComboboxSelected>>", lambda e: self.on_update_generic())

        # Contour controls
        ttk.Label(bar, text="Contours min/max/step:").pack(side=tk.LEFT, padx=(16, 4))
        self.cmin_var = tk.DoubleVar(value=50.0)
        self.cmax_var = tk.DoubleVar(value=80.0)
        self.cstep_var = tk.DoubleVar(value=2.5)
        ttk.Entry(bar, textvariable=self.cmin_var, width=6).pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.cmax_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Entry(bar, textvariable=self.cstep_var, width=6).pack(side=tk.LEFT)
        ttk.Button(bar, text="Redraw", command=self.redraw).pack(side=tk.LEFT, padx=6)

        # Export
        ttk.Button(bar, text="Export Table…", command=self.on_export).pack(side=tk.RIGHT)

    def _build_plot_and_table(self) -> None:
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # Plot area
        plot_frame = ttk.Frame(main)
        main.add(plot_frame, weight=3)
        self.fig, self.ax = plt.subplots(figsize=(7.5, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Table area
        side = ttk.Frame(main)
        main.add(side, weight=1)

        ttk.Label(side, text="Generic Points (Flow[m.t^0.5/p], PR, Eff[%], Weight)").pack(pady=(8, 4))
        self.tree = ttk.Treeview(side, columns=("flow", "pr", "eff", "w"), show="headings", height=20)
        for c, lbl in zip(["flow", "pr", "eff", "w"], ["Flow", "PR", "Eff", "W"]):
            self.tree.heading(c, text=lbl)
            self.tree.column(c, anchor=tk.CENTER, width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.avg_label = ttk.Label(side, text="Weighted Avg Eff: —")
        self.avg_label.pack(pady=8)

    # ---------------- Actions ----------------
    def on_browse(self) -> None:
        path = filedialog.askopenfilename(title="Select compressor map (.fae)", filetypes=[("Map files", "*.fae"), ("All", "*.*")])
        if path:
            self.path_var.set(path)

    def on_load(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            messagebox.showerror("No file", "Please select a .fae map file.")
            return
        try:
            self.cmap = load_map(path)
            fit_efficiency_regression(self.cmap)
            self.redraw()
        except Exception as e:
            logger.exception("Failed to load map")
            messagebox.showerror("Load error", str(e))

    def on_update_generic(self) -> None:
        self.current_set_name = self.set_var.get()
        self.update_table()

    def on_export(self) -> None:
        if self.cmap is None:
            messagebox.showerror("No data", "Load a map first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        rows = self._compute_generic_rows()
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Flow_plot(m.t^0.5/p)", "PR", "Efficiency(%)", "Weight"])
                for r in rows:
                    w.writerow([f"{r['flow_plot']:.3f}", f"{r['pr']:.3f}", f"{r['eff']:.2f}", f"{r['w']:.3f}"])
                w.writerow([])
                w.writerow(["Weighted Avg Efficiency", f"{self._weighted_avg(rows):.2f}"])
            messagebox.showinfo("Exported", f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ---------------- Plotting & Table ----------------
    def redraw(self) -> None:
        self.ax.clear()
        if self.cmap is None:
            self.ax.set_title("Load a .fae compressor map")
            self.canvas.draw_idle()
            return

        # 1) Plot speed lines (flow plot units vs PR)
        for sl in self.cmap.speed_lines:
            flows_plot = [p.flow_cfm / CFM_TO_PLOT_FLOW for p in sl.pts]
            prs = [p.pr for p in sl.pts]
            self.ax.plot(flows_plot, prs, marker="o", lw=1.2, label=f"{sl.rpm:.0f}")

        # 2) Outline boundary via convex hull
        all_pts = np.array([[p.flow_cfm / CFM_TO_PLOT_FLOW, p.pr] for sl in self.cmap.speed_lines for p in sl.pts])
        try:
            tri_plot = Delaunay(all_pts)
            hull_idx = np.unique(tri_plot.convex_hull.flatten())
            hull = all_pts[hull_idx]
            # order hull points by angle around centroid for a neat boundary
            c = hull.mean(axis=0)
            ang = np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0])
            order = np.argsort(ang)
            hull_ord = hull[order]
            self.ax.plot(np.r_[hull_ord[:, 0], hull_ord[0, 0]], np.r_[hull_ord[:, 1], hull_ord[0, 1]], color="black", lw=1.0, alpha=0.6)
        except Exception:
            # If hull fails (e.g., colinear), skip boundary
            pass

        # 3) Efficiency contours (in flow_cfm, PR space then convert X to plot units)
        try:
            F_cfm, P, E_masked, _ = contour_data(self.cmap)
            F_plot = F_cfm / CFM_TO_PLOT_FLOW
            cmin, cmax, cstep = self.cmin_var.get(), self.cmax_var.get(), self.cstep_var.get()
            levels = np.arange(cmin, cmax + 1e-9, cstep)
            cs = self.ax.contour(F_plot, P, E_masked, levels=levels, colors="grey", linewidths=0.8)
            self.ax.clabel(cs, inline=True, fontsize=8, fmt="%d")
        except Exception as e:
            logger.warning("Contour draw failed: %s", e)

        self.ax.set_xlabel("Flow (m.t^0.5/p)")
        self.ax.set_ylabel("Pressure Ratio")
        self.ax.set_title(self.cmap.title)
        self.ax.grid(True, which="both", alpha=0.2)
        self.ax.legend(title="Speed", fontsize=8)

        self.canvas.draw_idle()
        self.update_table()

    def _compute_generic_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        if self.cmap is None:
            return rows
        set_def = GENERIC_SETS.get(self.current_set_name, [])
        for flow_pct, pr_pct, w in set_def:
            flow_cfm = flow_pct * self.cmap.max_flow_cfm
            pr = pr_pct * self.cmap.max_pr
            try:
                eff = predict_efficiency(self.cmap, flow_cfm, pr)
            except Exception:
                eff = float("nan")
            rows.append({
                "flow_plot": flow_cfm / CFM_TO_PLOT_FLOW,
                "pr": pr,
                "eff": eff,
                "w": w,
            })
        return rows

    def _weighted_avg(self, rows: List[Dict[str, float]]) -> float:
        num = 0.0
        den = 0.0
        for r in rows:
            if not np.isnan(r["eff"]):
                num += r["eff"] * r["w"]
                den += r["w"]
        return num / den if den > 0 else float("nan")

    def update_table(self) -> None:
        for i in self.tree.get_children():
            self.tree.delete(i)
        if self.cmap is None:
            self.avg_label.config(text="Weighted Avg Eff: —")
            return
        rows = self._compute_generic_rows()
        for r in rows:
            self.tree.insert("", tk.END, values=(f"{r['flow_plot']:.3f}", f"{r['pr']:.3f}", f"{r['eff']:.2f}", f"{r['w']:.3f}"))
        avg = self._weighted_avg(rows)
        self.avg_label.config(text=f"Weighted Avg Eff: {avg:.2f} %")


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

