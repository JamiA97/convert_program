#!/usr/bin/env python3

# UI REFACTOR: Centralized Fusion theme + light QSS

from __future__ import annotations

from PySide6 import QtWidgets, QtGui



def apply_fusion_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = app.palette()

    pal.setColor(QtGui.QPalette.Window,        QtGui.QColor("#f4f4f6"))
    pal.setColor(QtGui.QPalette.WindowText,    QtGui.QColor("#111"))
    pal.setColor(QtGui.QPalette.Base,          QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#ececf0"))
    pal.setColor(QtGui.QPalette.ToolTipBase,   QtGui.QColor("#ffffdc"))
    pal.setColor(QtGui.QPalette.ToolTipText,   QtGui.QColor("#111"))
    pal.setColor(QtGui.QPalette.Text,          QtGui.QColor("#111"))
    pal.setColor(QtGui.QPalette.Button,        QtGui.QColor("#f0f0f3"))
    pal.setColor(QtGui.QPalette.ButtonText,    QtGui.QColor("#111"))
    pal.setColor(QtGui.QPalette.Highlight,     QtGui.QColor("#3b82f6"))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))

    app.setPalette(pal)

    app.setStyleSheet("""
        QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QToolButton, QPushButton { padding: 6px 10px; }
        QTableView { alternate-background-color: #ececf0; gridline-color: #444; }
        QDockWidget::title { padding: 4px; }
        QStatusBar { padding: 2px; }
    """)

