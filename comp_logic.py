#!/usr/bin/env python3

# UI REFACTOR: Business-logic-only module extracted from comp_map_gui.py
# Keeps data loading, regression, interpolation, and contour prep unchanged.

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.spatial import Delaunay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline

import scmap  # map I/O


# --------------------------------------------------------------------------------------
# Constants / Logging
# --------------------------------------------------------------------------------------

CFM_TO_PLOT_FLOW = 10.323  # Flow plotting in m.t^0.5/p = CFM / 10.323

logger = logging.getLogger("comp_map_logic")
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
        title=title or path,
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


def compute_generic_rows(cmap: CompMap, set_def: List[Tuple[float, float, float]]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for flow_pct, pr_pct, w in set_def:
        flow_cfm = flow_pct * cmap.max_flow_cfm
        pr = pr_pct * cmap.max_pr
        try:
            eff = predict_efficiency(cmap, flow_cfm, pr)
        except Exception:
            eff = float("nan")
        rows.append({
            "flow_plot": flow_cfm / CFM_TO_PLOT_FLOW,
            "pr": pr,
            "eff": eff,
            "w": w,
        })
    return rows


def weighted_avg_eff(rows: List[Dict[str, float]]) -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        if not np.isnan(r["eff"]):
            num += r["eff"] * r["w"]
            den += r["w"]
    return num / den if den > 0 else float("nan")

