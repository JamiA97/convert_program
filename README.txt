Compressor Map Tools
====================

This repo contains:
- scmap.py — Read/scale/write compressor maps in .fae format (FREE/FIXED I/O)
- comp_map_gui.py — GUI to plot maps, draw efficiency contours, and evaluate generic points
 - comp_map_qt.py — New PySide6 desktop UI (dockable panel, toolbar, sortable table)


Requirements
------------
Python 3.9+

Python packages:
- numpy
- scipy
- scikit-learn
- matplotlib
- tkinter (usually included with system Python)
 - PySide6 (for the new Qt UI)

Install (example):
  pip install numpy scipy scikit-learn matplotlib


CLI Usage (Scaling)
-------------------
Scale a map and write a new .fae file.

  python scmap.py input.fae output.fae \
    --scalsp 1.02 --scalpr 1.00 --scalmf 1.10 --scalef 1.00 \
    --read-format auto --write-format fixed

Notes:
- "--scal*" options accept a single scalar or CSV of per-speed-line values.
- Default write format is FIXED to match legacy WTMAP.


GUI Usage (Plotting & Interpolation)
------------------------------------
Launch the GUI:

  python comp_map_gui.py

Workflow:
- Browse → select a .fae compressor map → Load.
- The main plot shows Flow vs PR by speed line.
  - Flow axis uses m.t^0.5/p as required: Flow_plot = CFM / 10.323
  - A boundary outline is drawn (convex hull of data).
- Efficiency contours are fitted via a polynomial surface in (flow_cfm, PR) and drawn as thin grey labeled lines.
  - Adjust contour min/max/step and press Redraw.
- Pick a generic operating set from the dropdown (HD_WG or MD_WG).
  - Generic points are defined as % of max flow and % of max PR from the loaded map.
  - The table lists Flow (plot units), PR, interpolated efficiency, and weight, with a weighted average.
  - Toggle "Show Generic Points" to overlay these points on the plot.
- Export Table… saves the table and weighted average to CSV.


Units
-----
- Map flow values (.fae) are treated as CFM for modeling.
- Plotting flow axis uses Flow_plot = CFM / 10.323 (per brief).


Qt UI (New)
-----------
A modern PySide6 UI with a dockable right panel, compact toolbar, and a sortable zebra-striped table is available:

  python comp_map_qt.py

Shortcuts:
- Ctrl+O: Open file dialog
- R: Redraw
- E: Export table to CSV
- Ctrl+Q: Quit
- F: Fit/autoscale plot
- H: Toggle generic points visibility

Notes:
- The main plot remains central and large; the right dock is resizable (default ~360 px).
- Contour controls now use spinboxes with debounced redraws.
- Status bar shows the current map path and the weighted average efficiency.
- UI state (last file, contours, dock, toggles) persists via QSettings.

Batch Folder Analysis
---------------------
Evaluate a whole folder of maps and rank by weighted efficiency of the selected generic set.

- Click "Batch Folder" on the toolbar, choose a directory containing .fae maps.
- The app loads each map, fits efficiency, computes generic points, and ranks by weighted average.
- The best map is automatically plotted. A results window lists all maps (best → worst).
- Use "Export CSV…" in the results window to save the ordered report.
- Use the toolbar buttons "Top-5 Prev" / "Top-5 Next" to step through the top five maps.
- The results window has a "Show Top 5 only" toggle to quickly focus the list.
- Changing the Generic Set will automatically re-run the last batch evaluation and refresh the ranking.


Generic Sets (JSON)
-------------------
The list of generic operating point sets is configurable via a JSON file. Each set is a list of triples: [flow_pct, pr_pct, weight]. The app tries to load JSON at startup in this order:

- Env var path: `GENERIC_SETS_FILE`
- Current working directory: `./generic_sets.json`
- Next to the code: `generic_sets.json` (same folder as comp_logic.py)
- User config: `~/.config/CompressorMapPlotter/generic_sets.json`

If no valid JSON is found, built-in defaults are used.

Example `generic_sets.json`:

  {
    "HD_WG": [[0.7, 0.7, 0.3], [0.6, 0.6, 0.3], [0.5, 0.4, 0.4]],
    "MD_WG": [[0.7, 0.7, 0.3], [0.5, 0.6, 0.3], [0.4, 0.4, 0.4]]
  }

Notes:
- The UI will populate the Generic Set dropdown from the loaded JSON.
- If a previously saved set name is not present in the JSON, the first available set is selected.
