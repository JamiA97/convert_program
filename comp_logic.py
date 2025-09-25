#!/usr/bin/env python3

# UI REFACTOR: Business-logic-only module extracted from comp_map_gui.py
# Keeps data loading, regression, interpolation, and contour prep unchanged.

from __future__ import annotations

import math
import logging
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.spatial import Delaunay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline

import os
from pathlib import Path

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

    # feature scaling for regression inputs: (x_min, x_max) for [flow_cfm, pr]
    scaling: Optional[Tuple[np.ndarray, np.ndarray]] = None

# --------------------------------------------------------------------------------------
# Generic operating point sets (Flow%, PR%, Weight)
#
# Moved to external JSON with safe fallback to defaults below.
# --------------------------------------------------------------------------------------

# Default built-in sets used when JSON missing/invalid
DEFAULT_GENERIC_SETS: Dict[str, List[Tuple[float, float, float]]] = {
    "OH_HD_SCR": [
        (0.7, 0.66, 0.3),
        (0.52, 0.53, 0.3),
        (0.3, 0.34, 0.4),
    ],
    "OH_HD_EGR": [
        (0.57, 0.66, 0.4),
        (0.43, 0.47, 0.3),
        (0.17, 0.28, 0.3),
    ],
    "OH_MR_SCR": [
        (0.67, 0.60, 0.33),
        (0.49, 0.55, 0.33),
        (0.42, 0.43, 0.33),
    ]
}


def _validate_generic_sets(obj: Any) -> Dict[str, List[Tuple[float, float, float]]]:
    """Validate/normalize JSON-loaded generic sets structure.

    Expected shape: { name: [[flow_pct, pr_pct, weight], ...], ... }
    Returns a dict mapping str -> list[tuple(float, float, float)] or raises ValueError.
    """
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object/dict")
    out: Dict[str, List[Tuple[float, float, float]]] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise ValueError("Set names must be strings")
        if not isinstance(v, list):
            raise ValueError(f"Set '{k}' must be a list")
        triplets: List[Tuple[float, float, float]] = []
        for idx, item in enumerate(v):
            if (not isinstance(item, (list, tuple))) or len(item) != 3:
                raise ValueError(f"Set '{k}' entry {idx} must be a list of 3 numbers")
            try:
                f, p, w = float(item[0]), float(item[1]), float(item[2])
            except Exception as e:
                raise ValueError(f"Set '{k}' entry {idx} must be numeric: {e}")
            triplets.append((f, p, w))
        out[k] = triplets
    return out


def _load_generic_sets_from_file(path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _validate_generic_sets(data)


def load_generic_sets() -> Dict[str, List[Tuple[float, float, float]]]:
    """Load generic sets from JSON with fallbacks.

    Search order:
      1) Env var GENERIC_SETS_FILE (if set)
      2) CWD: ./generic_sets.json
      3) Next to this module: <repo>/generic_sets.json
      4) User config: ~/.config/CompressorMapPlotter/generic_sets.json
    """
    tried: List[str] = []

    def try_path(p: Optional[Path]) -> Optional[Dict[str, List[Tuple[float, float, float]]]]:
        if not p:
            return None
        try:
            if p.is_file():
                gs = _load_generic_sets_from_file(p)
                logger.info("Loaded generic sets from: %s", p)
                return gs
            tried.append(str(p))
        except Exception as e:
            logger.warning("Failed to load generic sets from %s: %s", p, e)
            tried.append(f"{p} (error)")
        return None

    # 1) Environment variable
    env_path = os.environ.get("GENERIC_SETS_FILE")
    if env_path:
        res = try_path(Path(env_path))
        if res is not None:
            return res

    # 2) CWD
    res = try_path(Path.cwd() / "generic_sets.json")
    if res is not None:
        return res

    # 3) Next to this module
    res = try_path(Path(__file__).resolve().parent / "generic_sets.json")
    if res is not None:
        return res

    # 4) User config
    res = try_path(Path.home() / ".config" / "CompressorMapPlotter" / "generic_sets.json")
    if res is not None:
        return res

    logger.info("Using built-in default generic sets (no valid JSON found)")
    return DEFAULT_GENERIC_SETS


# Load at import time so UIs get keys immediately
GENERIC_SETS: Dict[str, List[Tuple[float, float, float]]] = load_generic_sets()


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

    # --- Normalise inputs to [-1, 1] for numerical stability ---
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    rng = np.maximum(x_max - x_min, 1e-12)  # protect against zero range
    X_scaled = 2.0 * (X - x_min) / rng - 1.0


    best_rmse = np.inf
    best_model = None
    best_poly = None
    best_deg = None

    for deg in range(degree_min, degree_max + 1):
        poly = PolynomialFeatures(deg)
        #Xp = poly.fit_transform(X)
        Xp = poly.fit_transform(X_scaled)  # use scaled inputs
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
    cmap.scaling = (x_min, x_max)  # save scaling for inference
    logger.info("Selected polynomial degree %s (RMSE=%.5f)", best_deg, best_rmse)


def predict_efficiency(cmap: CompMap, flow_cfm: float, pr: float) -> float:
    if cmap.poly_features is None or cmap.lin_model is None:
        raise RuntimeError("Regression model not fitted")
    #Xp = cmap.poly_features.transform([[flow_cfm, pr]])
    # Apply stored scaling if available
    X_raw = np.array([[flow_cfm, pr]], dtype=float)
    if getattr(cmap, "scaling", None):
        x_min, x_max = cmap.scaling
        rng = np.maximum(x_max - x_min, 1e-12)
        X_scaled = 2.0 * (X_raw - x_min) / rng - 1.0
    else:
        X_scaled = X_raw
    Xp = cmap.poly_features.transform(X_scaled)

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
    #X = np.vstack([F_cfm.ravel(), P.ravel()]).T
    #E = cmap.lin_model.predict(cmap.poly_features.transform(X)).reshape(F_cfm.shape)
    X = np.vstack([F_cfm.ravel(), P.ravel()]).T
    if getattr(cmap, "scaling", None):
        x_min, x_max = cmap.scaling
        rng = np.maximum(x_max - x_min, 1e-12)
        X_scaled = 2.0 * (X - x_min) / rng - 1.0
    else:
        X_scaled = X
    E = cmap.lin_model.predict(cmap.poly_features.transform(X_scaled)).reshape(F_cfm.shape)

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


# --------------------------------------------------------------------------------------
# Batch evaluation helpers
# --------------------------------------------------------------------------------------

def evaluate_map_weighted_eff(path: str, set_def: List[Tuple[float, float, float]]) -> Dict[str, object]:
    """Load a map, fit efficiency model, compute generic rows and weighted efficiency.

    Returns a dict with: { 'path', 'title', 'rows', 'weighted_eff', 'rmse', 'ns', 'nr' }
    Raises exceptions for fatal errors so callers can handle/skip.
    """
    cmap = load_map(path)
    fit_efficiency_regression(cmap)
    rows = compute_generic_rows(cmap, set_def)
    avg = weighted_avg_eff(rows)

    # Compute max efficiency from raw data
    max_eff = max((p.eff for sl in cmap.speed_lines for p in sl.pts), default=float("nan"))

    return {
        "path": os.path.abspath(path),
        "title": cmap.title,
        "rows": rows,
        "weighted_eff": float(avg),
        "rmse": float(cmap.rmse) if cmap.rmse is not None else float("nan"),
        "ns": len(cmap.speed_lines),
        "nr": len(cmap.speed_lines[0].pts) if cmap.speed_lines else 0,
        "max_eff": float(max_eff),
    }


def iter_map_files(folder: str) -> List[str]:
    """Return candidate map file paths under folder (non-recursive), filtered by extension.

    Accepts typical extensions: .fae, .FAE, .txt (if used). Only files are returned.
    """
    exts = {".fae", ".FAE", ".txt", ".TXT"}
    paths: List[str] = []
    if not folder:
        return paths
    p = Path(folder)
    if not p.is_dir():
        return paths
    for child in p.iterdir():
        if child.is_file() and child.suffix in exts:
            paths.append(str(child))
    return sorted(paths)


def batch_evaluate_folder(folder: str, set_def: List[Tuple[float, float, float]]) -> List[Dict[str, object]]:
    """Evaluate all maps in the folder and return sorted results (best first).

    Each item: { path, title, rows, weighted_eff, rmse, ns, nr }
    Invalid or failed maps are skipped with logging warnings.
    """
    results: List[Dict[str, object]] = []
    files = iter_map_files(folder)
    if not files:
        logger.warning("No candidate map files in folder: %s", folder)
    for fpath in files:
        try:
            res = evaluate_map_weighted_eff(fpath, set_def)
            if np.isnan(res["weighted_eff"]):
                logger.warning("Weighted efficiency is NaN for %s; skipping", fpath)
                continue
            results.append(res)
        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", fpath, e)
            continue
    # Sort best to worst by weighted_eff
    results.sort(key=lambda d: d.get("weighted_eff", float("nan")), reverse=True)
    return results


def write_batch_csv(path: str, results: List[Dict[str, object]]) -> None:
    """Write an ordered CSV report of results best→worst."""
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "WeightedEff(%)", "MaxEff(%)", "Title", "Path", "RMSE", "NS", "NR"])
        for i, r in enumerate(results, start=1):
            w.writerow([
                i,
                f"{float(r.get('weighted_eff', float('nan'))):.2f}",
                f"{float(r.get('max_eff', float('nan'))):.2f}",
                str(r.get("title", "")),
                str(r.get("path", "")),
                f"{float(r.get('rmse', float('nan'))):.5f}",
                int(r.get("ns", 0)),
                int(r.get("nr", 0)),
            ])
