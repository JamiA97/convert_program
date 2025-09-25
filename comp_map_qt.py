#!/usr/bin/env python3

"""
# UI REFACTOR: PySide6 Qt UI embedding Matplotlib

Keeps business logic in comp_logic.py unchanged. Provides:
- QMainWindow with central plot and right dock for controls + data table
- Compact top toolbar + status bar
- Fusion theme + light QSS (see ui_theme.py)
- Spinboxes for contour controls with debounced redraw
- Sortable zebra-striped QTableView + actions (Copy/Export/Clear)
- Keyboard shortcuts: Ctrl+O, R, E, Ctrl+Q, F, H
- UI state persistence with QSettings
"""

from __future__ import annotations

import os
import csv
import sys
from typing import List, Dict

from PySide6 import QtCore, QtWidgets, QtGui

import matplotlib as mpl
mpl.use("QtAgg")
mpl.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.18,
    "figure.constrained_layout.use": True,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 140, "savefig.dpi": 200,
})

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from ui_theme import apply_fusion_theme
from comp_logic import (
    load_map,
    fit_efficiency_regression,
    contour_data,
    compute_generic_rows,
    weighted_avg_eff,
    GENERIC_SETS,
    CFM_TO_PLOT_FLOW,
    batch_evaluate_folder,
    write_batch_csv,
)


class MapTableModel(QtCore.QAbstractTableModel):
    headers = ["Flow", "PR", "Eff", "W"]

    def __init__(self, rows: List[Dict[str, float]] | None = None) -> None:
        super().__init__()
        self._rows: List[Dict[str, float]] = rows or []

    def setRows(self, rows: List[Dict[str, float]]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else 4

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        r = index.row(); c = index.column()
        row = self._rows[r]
        if role == QtCore.Qt.DisplayRole:
            if c == 0: return f"{row['flow_plot']:.3f}"
            if c == 1: return f"{row['pr']:.3f}"
            if c == 2: return f"{row['eff']:.2f}"
            if c == 3: return f"{row['w']:.3f}"
        if role == QtCore.Qt.TextAlignmentRole and c in (0,1,2,3):
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    def sort(self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.AscendingOrder) -> None:
        key_funcs = [
            lambda r: r["flow_plot"],
            lambda r: r["pr"],
            lambda r: r["eff"],
            lambda r: r["w"],
        ]
        reverse = order == QtCore.Qt.DescendingOrder
        self.layoutAboutToBeChanged.emit()
        self._rows.sort(key=key_funcs[column], reverse=reverse)
        self.layoutChanged.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Compressor Map Plotter (Qt)")
        self.resize(1200, 800)

        # State
        self.cmap = None
        # Prefer HD_WG if available, else first available set, else empty
        try:
            first_key = next(iter(GENERIC_SETS.keys())) if GENERIC_SETS else ""
        except Exception:
            first_key = ""
        self.current_set_name = "HD_WG" if "HD_WG" in GENERIC_SETS else first_key
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.redraw)

        # Batch/top-5 state
        self._batch_results: List[Dict[str, object]] = []
        self._top_paths: List[str] = []
        self._top_index: int = -1
        self._last_batch_folder: str | None = None

        # Status bar
        self.status = self.statusBar()

        # Central plot
        self._build_central_plot()

        # Right dock: controls + table
        self._build_right_dock()

        # Top toolbar
        self._build_toolbar()

        # Shortcuts
        self._build_shortcuts()

        # Settings
        self.settings = QtCore.QSettings("Company", "CompressorMapPlotter")
        self._restore_settings()

    # ---------------- Central plot ----------------
    def _build_central_plot(self) -> None:
        fig, ax = plt.subplots(figsize=(7.8, 6.2))
        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvas(self.fig)
        cw = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(cw)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.setCentralWidget(cw)

        # Hover readout
        self._mpl_cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)

    # ---------------- Right dock ----------------
    def _build_right_dock(self) -> None:
        self.dock = QtWidgets.QDockWidget("Controls & Data", self)
        self.dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)

        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # File open row
        file_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Map file (.fae)")
        btn_open = QtWidgets.QToolButton(text="Open")
        btn_open.clicked.connect(self._action_open)
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(btn_open, 0)
        v.addLayout(file_row)

        # Generic set picker
        set_row = QtWidgets.QHBoxLayout()
        set_row.addWidget(QtWidgets.QLabel("Generic set:"))
        self.set_combo = QtWidgets.QComboBox()
        self.set_combo.addItems(list(GENERIC_SETS.keys()))
        if self.current_set_name:
            self.set_combo.setCurrentText(self.current_set_name)
        self.set_combo.setEnabled(bool(GENERIC_SETS))
        self.set_combo.currentTextChanged.connect(self._on_generic_changed)
        set_row.addWidget(self.set_combo, 1)
        v.addLayout(set_row)

        # Group: Contours
        grp_contour = QtWidgets.QGroupBox("Contours")
        form = QtWidgets.QFormLayout(grp_contour)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.spin_cmin = QtWidgets.QDoubleSpinBox()
        self.spin_cmax = QtWidgets.QDoubleSpinBox()
        self.spin_cstep = QtWidgets.QDoubleSpinBox()
        for sp in (self.spin_cmin, self.spin_cmax, self.spin_cstep):
            sp.setDecimals(2)
            sp.setRange(0.0, 100.0)
            sp.setSingleStep(0.5)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._debounced_redraw)
        self.spin_cmin.setValue(50.0)
        self.spin_cmax.setValue(80.0)
        self.spin_cstep.setValue(2.5)
        form.addRow("Min", self.spin_cmin)
        form.addRow("Max", self.spin_cmax)
        form.addRow("Step", self.spin_cstep)
        btn_reset = QtWidgets.QToolButton(text="Reset")
        btn_reset.clicked.connect(self._reset_contours)
        form.addRow(btn_reset)
        v.addWidget(grp_contour)

        # Group: Display
        grp_disp = QtWidgets.QGroupBox("Display")
        vl = QtWidgets.QVBoxLayout(grp_disp)
        self.chk_points = QtWidgets.QCheckBox("Show Generic Points")
        self.chk_points.toggled.connect(self.redraw)
        vl.addWidget(self.chk_points)
        v.addWidget(grp_disp)

        # Inline toolbar above table
        row_toolbar = QtWidgets.QHBoxLayout()
        self.btn_copy = QtWidgets.QToolButton()
        self.btn_copy.setText("Copy")
        self.btn_copy.setToolTip("Copy selected rows")
        self.btn_copy.clicked.connect(self._copy_selected)
        self.btn_export_csv = QtWidgets.QToolButton()
        self.btn_export_csv.setText("Export CSV")
        self.btn_export_csv.setToolTip("Export table to CSV (E)")
        self.btn_export_csv.clicked.connect(self._action_export)
        self.btn_clear = QtWidgets.QToolButton()
        self.btn_clear.setText("Clear")
        self.btn_clear.setToolTip("Clear table (temporary)")
        self.btn_clear.clicked.connect(self._clear_table)
        row_toolbar.addWidget(self.btn_copy)
        row_toolbar.addWidget(self.btn_export_csv)
        row_toolbar.addWidget(self.btn_clear)
        row_toolbar.addStretch(1)
        v.addLayout(row_toolbar)

        # Table view
        self.table = QtWidgets.QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.model = MapTableModel([])
        self.table.setModel(self.model)
        v.addWidget(self.table, 1)

        # --- NEW: footer row for weighted average ---
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.lbl_avg = QtWidgets.QLabel("Weighted Avg Eff: –")
        self.lbl_avg.setStyleSheet("font-weight: 600; padding: 4px;")
        footer.addWidget(self.lbl_avg, 0)
        v.addLayout(footer)
        # --- END NEW ---


        self.dock.setWidget(host)
        self.dock.setMinimumWidth(320)
        self.dock.setMaximumWidth(800)

    # ---------------- Toolbar & actions ----------------
    def _build_toolbar(self) -> None:
        tb = QtWidgets.QToolBar("Main")
        tb.setIconSize(QtCore.QSize(18, 18))
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        def make_action(text: str, icon: str, slot, shortcut: str | None = None, tip: str | None = None):
            act = QtGui.QAction(QtGui.QIcon.fromTheme(icon), text, self)
            if shortcut:
                act.setShortcut(shortcut)
            if tip:
                act.setToolTip(tip); act.setStatusTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)
            return act

        self.act_open = make_action("Open", "document-open", self._action_open, "Ctrl+O", "Open map file")
        self.act_load = make_action("Load", "view-refresh", self._action_load, None, "Load map into plot")
        self.act_redraw = make_action("Redraw", "media-playlist-repeat", self.redraw, "R", "Redraw plot")
        self.act_export = make_action("Export", "document-save-as", self._action_export, "E", "Export table to CSV")
        self.act_batch = make_action("Batch Folder", "folder", self._action_batch_folder, None, "Evaluate all maps in a folder")
        tb.addSeparator()
        self.act_quit = make_action("Quit", "application-exit", self.close, "Ctrl+Q", "Quit application")

        # Fit action (F)
        self.act_fit = QtGui.QAction("Fit", self)
        self.act_fit.setShortcut("F")
        self.act_fit.triggered.connect(self._action_fit)
        self.addAction(self.act_fit)

        # Toggle generic points (H)
        self.act_toggle_pts = QtGui.QAction("Toggle Points", self)
        self.act_toggle_pts.setShortcut("H")
        self.act_toggle_pts.triggered.connect(lambda: self.chk_points.toggle())
        self.addAction(self.act_toggle_pts)

        # Top-5 navigation (disabled until batch run)
        self.act_top_prev = QtGui.QAction("Top-5 Prev", self)
        self.act_top_next = QtGui.QAction("Top-5 Next", self)
        self.act_top_prev.triggered.connect(lambda: self._navigate_top(delta=-1))
        self.act_top_next.triggered.connect(lambda: self._navigate_top(delta=+1))
        self.act_top_prev.setEnabled(False)
        self.act_top_next.setEnabled(False)
        tb.addAction(self.act_top_prev)
        tb.addAction(self.act_top_next)

    def _build_shortcuts(self) -> None:
        # Redraw (R) and Export (E) already assigned; ensure redirection to slots
        pass

    # ---------------- Slots ----------------
    def _action_open(self) -> None:
        dlg = QtWidgets.QFileDialog(self, "Select compressor map (.fae)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilters(["Map files (*.fae)", "All files (*)"])
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            path = dlg.selectedFiles()[0]
            self.path_edit.setText(path)
            self.status.showMessage(path, 4000)

    def _action_load(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "No file", "Please select a .fae map file.")
            return
        self._load_map_into_plot(path)

    def _action_export(self) -> None:
        if self.cmap is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a map first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", filter="CSV (*.csv);;All files (*)")
        if not path:
            return
        rows = self.model._rows
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Flow_plot(m.\u221At/p)", "PR", "Efficiency(%)", "Weight"])
                for r in rows:
                    w.writerow([f"{r['flow_plot']:.3f}", f"{r['pr']:.3f}", f"{r['eff']:.2f}", f"{r['w']:.3f}"])
                w.writerow([])
                w.writerow(["Weighted Avg Efficiency", f"{weighted_avg_eff(rows):.2f}"])
            self.status.showMessage(f"Exported: {os.path.basename(path)}", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def _action_fit(self) -> None:
        try:
            self.ax.relim(); self.ax.autoscale()
            self.canvas.draw_idle()
        except Exception:
            pass

    def _action_batch_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder of compressor maps")
        if not folder:
            return
        self._last_batch_folder = folder
        self._run_batch_on_folder(folder)

    def _run_batch_on_folder(self, folder: str) -> None:
        try:
            set_def = GENERIC_SETS.get(self.current_set_name, [])
            results = batch_evaluate_folder(folder, set_def)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Batch error", str(e))
            return
        if not results:
            QtWidgets.QMessageBox.information(self, "No results", "No valid maps found or evaluation failed.")
            # Reset top-5 state
            self._batch_results = []
            self._top_paths = []
            self._top_index = -1
            self.act_top_prev.setEnabled(False)
            self.act_top_next.setEnabled(False)
            return
        self._batch_results = results
        # Prepare top-5 navigation
        self._top_paths = [str(r.get("path", "")) for r in results[:5] if r.get("path")]
        self._top_index = 0 if self._top_paths else -1
        self.act_top_prev.setEnabled(len(self._top_paths) > 1)
        self.act_top_next.setEnabled(len(self._top_paths) > 1)

        # Load best map into plot
        best_path = str(results[0].get("path"))
        self.path_edit.setText(best_path)
        self._load_map_into_plot(best_path)

        # Show dialog with table + export
        dlg = BatchResultsDialog(self, results)
        dlg.exec()

    def _copy_selected(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        lines = ["Flow\tPR\tEff\tW"]
        for idx in sel:
            r = idx.row(); row = self.model._rows[r]
            lines.append("\t".join([
                f"{row['flow_plot']:.3f}", f"{row['pr']:.3f}", f"{row['eff']:.2f}", f"{row['w']:.3f}"
            ]))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.status.showMessage(f"Copied {len(sel)} row(s)", 2500)

    def _clear_table(self) -> None:
        self.model.setRows([])
        self.status.showMessage("Table cleared (will repopulate on redraw)", 2500)

    def _on_generic_changed(self, text: str) -> None:
        self.current_set_name = text
        self.redraw()
        # Auto re-evaluate last batch folder (if any)
        if self._last_batch_folder:
            self._run_batch_on_folder(self._last_batch_folder)

    def _reset_contours(self) -> None:
        self.spin_cmin.setValue(50.0)
        self.spin_cmax.setValue(80.0)
        self.spin_cstep.setValue(2.5)
        self.redraw()

    def _debounced_redraw(self) -> None:
        self._debounce_timer.start(250)

    # ---------------- Plotting ----------------
    def redraw(self) -> None:
        self.ax.clear()
        if self.cmap is None:
            self.ax.set_title("Load a .fae compressor map")
            self.canvas.draw_idle()
            return

        # Speed lines
        speedlines = []
        labels = []
        for sl in self.cmap.speed_lines:
            flows_plot = [p.flow_cfm / CFM_TO_PLOT_FLOW for p in sl.pts]
            prs = [p.pr for p in sl.pts]
            ln, = self.ax.plot(flows_plot, prs, marker="", lw=1.1, alpha=0.9)
            speedlines.append(ln); labels.append(f"{sl.rpm:.0f}")

        # Boundary via convex hull (plot units)
        all_pts = np.array([[p.flow_cfm / CFM_TO_PLOT_FLOW, p.pr] for sl in self.cmap.speed_lines for p in sl.pts])
        try:
            from scipy.spatial import Delaunay
            tri_plot = Delaunay(all_pts)
            hull_idx = np.unique(tri_plot.convex_hull.flatten())
            hull = all_pts[hull_idx]
            c = hull.mean(axis=0)
            ang = np.arctan2(hull[:, 1] - c[1], hull[:, 0] - c[0])
            order = np.argsort(ang)
            hull_ord = hull[order]
            self.ax.plot(np.r_[hull_ord[:, 0], hull_ord[0, 0]], np.r_[hull_ord[:, 1], hull_ord[0, 1]], color="black", lw=1.0, alpha=0.6)
        except Exception:
            pass

        # Efficiency contours (viridis) from business logic
        try:
            F_cfm, P, E_masked, _ = contour_data(self.cmap)
            F_plot = F_cfm / CFM_TO_PLOT_FLOW
            cmin = self.spin_cmin.value(); cmax = self.spin_cmax.value(); cstep = self.spin_cstep.value()
            levels = np.arange(cmin, cmax + 1e-9, cstep)
            cs = self.ax.contour(F_plot, P, E_masked, levels=levels, cmap="jet", linewidths=0.7, alpha=0.8)

            # --- NEW: contour labels ---
            levels_to_label = cs.levels[::2] if len(cs.levels) > 6 else cs.levels
            def percent_fmt(v): return f"{v:.0f}%"
            lbls = self.ax.clabel(cs, levels=levels_to_label, inline=True,
                                  inline_spacing=5, fontsize=9, fmt=percent_fmt)
            import matplotlib.patheffects as pe
            for t in lbls:
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
            # --- END NEW ---

            # thinner than speedlines, inline labels
            for ln,label in zip(speedlines, labels):
                xdata, ydata = ln.get_xdata(), ln.get_ydata()
                if len(xdata) == 0:
                    continue
                i = max(0, len(xdata)//2 - 1)
                self.ax.text(xdata[i], ydata[i], label, fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.2", fc=(0,0,0,0), ec="none"))
        except Exception as e:
            # graceful degradation
            pass

        # Optional: generic points
        if self.chk_points.isChecked():
            rows = compute_generic_rows(self.cmap, GENERIC_SETS.get(self.current_set_name, []))
            xs = [r["flow_plot"] for r in rows]
            ys = [r["pr"] for r in rows]
            if xs and ys:
                self.ax.scatter(xs, ys, color="black", marker="s", s=46, alpha=0.85)

            avg = weighted_avg_eff(rows)
           # self.status.showMessage(f"Weighted Avg Eff: {avg:.2f}%")
            self.lbl_avg.setText(f"Weighted Avg Eff: {avg:.2f}%")

        else:
            # Still compute average for status
            rows = compute_generic_rows(self.cmap, GENERIC_SETS.get(self.current_set_name, []))
            avg = weighted_avg_eff(rows)
            self.status.showMessage(f"Weighted Avg Eff: {avg:.2f}%")

        self.ax.set_xlabel("Flow (m.\u221At/p)")
        self.ax.set_ylabel("Pressure Ratio")
        self.ax.set_title(self.cmap.title)
        self.ax.grid(True, which="both", alpha=0.18)
        # Remove legend box (labels inline)

        # Update table
        self.model.setRows(rows)
        self.canvas.draw_idle()

    # ---------------- Helpers ----------------
    def _load_map_into_plot(self, path: str) -> None:
        try:
            self.cmap = load_map(path)
            fit_efficiency_regression(self.cmap)
            self._save_last_file(path)
            self.status.showMessage(f"Loaded: {path}")
            # Ensure points visibility remains as user set
            self.redraw()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def _navigate_top(self, delta: int) -> None:
        if not self._top_paths:
            return
        self._top_index = (self._top_index + delta) % len(self._top_paths)
        path = self._top_paths[self._top_index]
        if os.path.isfile(path):
            self.path_edit.setText(path)
            self._load_map_into_plot(path)
        # Update enable state in case of single entry
        self.act_top_prev.setEnabled(len(self._top_paths) > 1)
        self.act_top_next.setEnabled(len(self._top_paths) > 1)

    # (Batch results dialog moved below MainWindow class)

    # ---------------- Hover readout ----------------
    def _on_mpl_motion(self, event) -> None:
        if not event.inaxes:
            return
        x = event.xdata; y = event.ydata
        msg = f"Flow: {x:.3f}  PR: {y:.3f}"
        try:
            # Optional: nearest/predicted efficiency at cursor
            if self.cmap is not None:
                from comp_logic import predict_efficiency
                eff = predict_efficiency(self.cmap, x * CFM_TO_PLOT_FLOW, y)
                msg += f"  Eff~{eff:.2f}%"
        except Exception:
            pass
        self.status.showMessage(msg)

    # ---------------- Settings ----------------
    def _restore_settings(self) -> None:
        self.restoreGeometry(self.settings.value("win/geom", b""))
        self.restoreState(self.settings.value("win/state", b""))
        last = self.settings.value("last/file", "", type=str)
        if last:
            self.path_edit.setText(last)
        self.spin_cmin.setValue(self.settings.value("contours/min", 50.0, type=float))
        self.spin_cmax.setValue(self.settings.value("contours/max", 80.0, type=float))
        self.spin_cstep.setValue(self.settings.value("contours/step", 2.5, type=float))
        self.chk_points.setChecked(self.settings.value("display/points", False, type=bool))
        set_name = self.settings.value("display/generic_set", self.current_set_name, type=str)
        if set_name in GENERIC_SETS:
            self.current_set_name = set_name
            self.set_combo.setCurrentText(set_name)

    def _save_last_file(self, path: str) -> None:
        self.settings.setValue("last/file", path)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.settings.setValue("win/geom", self.saveGeometry())
        self.settings.setValue("win/state", self.saveState())
        self.settings.setValue("contours/min", self.spin_cmin.value())
        self.settings.setValue("contours/max", self.spin_cmax.value())
        self.settings.setValue("contours/step", self.spin_cstep.value())
        self.settings.setValue("display/points", self.chk_points.isChecked())
        self.settings.setValue("display/generic_set", self.current_set_name)
        super().closeEvent(e)


class BatchResultsDialog(QtWidgets.QDialog):
    """Modal dialog showing ranked batch results with export option and Top-5 toggle."""
    def __init__(self, parent: QtWidgets.QWidget, results: List[Dict[str, object]]):
        super().__init__(parent)
        self.setWindowTitle("Batch Results: Ranked Weighted Efficiency")
        self.resize(900, 520)
        self._results: List[Dict[str, object]] = list(results)

        v = QtWidgets.QVBoxLayout(self)

        # Top bar with toggle
        top_row = QtWidgets.QHBoxLayout()
        self.chk_top5 = QtWidgets.QCheckBox("Show Top 5 only")
        self.chk_top5.setToolTip("Toggle to only display the top five entries.")
        self.chk_top5.toggled.connect(self._refresh_table)
        top_row.addWidget(self.chk_top5)
        top_row.addStretch(1)
        v.addLayout(top_row)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Rank", "WeightedEff(%)", "MaxEff(%)", "Title", "File", "RMSE", "Path"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        v.addWidget(self.table, 1)
        self._refresh_table()

        # Buttons
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_export = QtWidgets.QPushButton("Export CSV…")
        btn_export.setToolTip("Export the full ordered list (best → worst) to CSV.")
        btn_export.clicked.connect(self._export_csv)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        row.addWidget(btn_export)
        row.addWidget(btn_close)
        v.addLayout(row)

    def _current_results_view(self) -> List[Dict[str, object]]:
        return self._results[:5] if self.chk_top5.isChecked() else self._results

    def _refresh_table(self) -> None:
        results = self._current_results_view()
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            we = float(r.get("weighted_eff", float("nan")))
            title = str(r.get("title", ""))
            path = str(r.get("path", ""))
            rmse = float(r.get("rmse", float("nan")))
            fn = os.path.basename(path)
            max_eff = float(r.get("max_eff", float("nan")))
            items = [
                QtWidgets.QTableWidgetItem(str(i+1)),
                QtWidgets.QTableWidgetItem(f"{we:.2f}"),
                QtWidgets.QTableWidgetItem(f"{max_eff:.2f}"),
                QtWidgets.QTableWidgetItem(title),
                QtWidgets.QTableWidgetItem(fn),
                QtWidgets.QTableWidgetItem(f"{rmse:.5f}"),
                QtWidgets.QTableWidgetItem(path),
            ]
            # Align numeric columns
            items[0].setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            items[1].setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            items[2].setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            items[5].setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            for c, it in enumerate(items):
                self.table.setItem(i, c, it)

    def _export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save report CSV", filter="CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            # Export full ordered list regardless of toggle
            write_batch_csv(path, self._results)
            QtWidgets.QMessageBox.information(self, "Saved", f"Report written to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    apply_fusion_theme(app)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
