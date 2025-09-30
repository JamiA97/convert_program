#!/usr/bin/env python3

"""
Quick CLI to visualize the fitted efficiency surface in 3D and overlay measured points.

Usage examples:
  python plot3d_demo.py comp_map.fae --save surface.png
  python plot3d_demo.py comp_map.fae --grid 100 --units cfm
  python plot3d_demo.py comp_map.fae --no-show  # useful in headless save-only contexts
"""

from __future__ import annotations

import argparse
import sys

from comp_logic import (
    load_map,
    fit_efficiency_regression,
    plot_efficiency_surface_3d,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D efficiency surface plot for a compressor map (.fae)")
    p.add_argument("input", help="Input .fae file path")
    p.add_argument("--grid", type=int, default=80, help="Grid resolution for surface (default: 80)")
    p.add_argument("--units", choices=["plot", "cfm"], default="plot",
                   help="X-axis flow units: 'plot' uses Flow_plot = CFM/10.323; 'cfm' uses raw CFM")
    p.add_argument("--save", default=None, help="Optional output image path to save the figure (e.g., out.png)")
    p.add_argument("--no-show", action="store_true", help="Do not display the window (useful when saving only)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load and fit
    cmap = load_map(args.input)
    fit_efficiency_regression(cmap)

    # Generate plot
    in_plot_units = args.units == "plot"
    plot_efficiency_surface_3d(
        cmap,
        n=args.grid,
        in_plot_units=in_plot_units,
        show=(not args.no_show),
        save_path=args.save,
        surface_cmap="viridis",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

