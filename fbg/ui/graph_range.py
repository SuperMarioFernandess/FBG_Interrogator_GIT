"""Общие операторские настройки области временных графиков."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from fbg.ui import texts


class GraphRangeControls(QWidget):
    """X: follow/all/manual; Y: auto/manual. Мышь всегда отдаёт власть человеку."""

    FOLLOW = "follow"
    ALL = "all"
    MANUAL = "manual"
    AUTO = "auto"

    def __init__(self, plot: pg.PlotWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = plot
        self._last_t = np.empty(0, dtype=np.float64)
        self._programmatic = False

        self.x_mode = QComboBox()
        self.x_mode.addItem(texts.GRAPH_X_FOLLOW, self.FOLLOW)
        self.x_mode.addItem(texts.GRAPH_X_ALL, self.ALL)
        self.x_mode.addItem(texts.GRAPH_X_MANUAL, self.MANUAL)
        self.follow_s = self._spin(0.1, 604_800.0, 60.0, 1)
        self.x_from = self._spin(-1.0e9, 1.0e9, 0.0, 3)
        self.x_to = self._spin(-1.0e9, 1.0e9, 60.0, 3)

        self.y_mode = QComboBox()
        self.y_mode.addItem(texts.GRAPH_Y_AUTO, self.AUTO)
        self.y_mode.addItem(texts.GRAPH_Y_MANUAL, self.MANUAL)
        self.y_min = self._spin(-1.0e12, 1.0e12, 0.0, 6)
        self.y_max = self._spin(-1.0e12, 1.0e12, 1.0, 6)
        self.show_all_button = QPushButton(texts.BUTTON_GRAPH_SHOW_ALL)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(0)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_X_MODE), 0, 0)
        layout.addWidget(self.x_mode, 0, 1)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_FOLLOW_SECONDS), 0, 2)
        layout.addWidget(self.follow_s, 0, 3)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_X_FROM), 1, 0)
        layout.addWidget(self.x_from, 1, 1)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_X_TO), 1, 2)
        layout.addWidget(self.x_to, 1, 3)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_Y_MODE), 2, 0)
        layout.addWidget(self.y_mode, 2, 1)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_Y_MIN), 2, 2)
        layout.addWidget(self.y_min, 2, 3)
        layout.addWidget(QLabel(texts.LABEL_GRAPH_Y_MAX), 3, 0)
        layout.addWidget(self.y_max, 3, 1)
        layout.addWidget(self.show_all_button, 3, 2, 1, 2)
        layout.setColumnStretch(4, 1)

        self.x_mode.currentIndexChanged.connect(self._x_mode_changed)
        self.follow_s.valueChanged.connect(lambda _value: self._apply_x_program_mode())
        self.x_from.valueChanged.connect(lambda _value: self._apply_manual_x())
        self.x_to.valueChanged.connect(lambda _value: self._apply_manual_x())
        self.y_mode.currentIndexChanged.connect(self._y_mode_changed)
        self.y_min.valueChanged.connect(lambda _value: self._apply_manual_y())
        self.y_max.valueChanged.connect(lambda _value: self._apply_manual_y())
        self.show_all_button.clicked.connect(self.show_all)
        self._plot.getViewBox().sigRangeChangedManually.connect(self._mouse_range_changed)
        self._update_enabled()
        self._plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    @staticmethod
    def _spin(low: float, high: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setMaximumWidth(130)
        return spin

    def apply_time_axis(self, t_s: np.ndarray) -> None:
        """Применяет только явно выбранный программный X-режим.

        В manual никакой такт графика диапазон не трогает.
        """
        values = np.asarray(t_s, dtype=np.float64)
        self._last_t = values[np.isfinite(values)]
        self._apply_x_program_mode()

    def show_all(self) -> None:
        self._set_combo_data(self.x_mode, self.ALL)
        self._set_combo_data(self.y_mode, self.AUTO)
        self._update_enabled()
        self._plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self._apply_x_program_mode()

    def _x_mode_changed(self, _index: int) -> None:
        self._update_enabled()
        self._apply_x_program_mode()

    def _y_mode_changed(self, _index: int) -> None:
        self._update_enabled()
        if self.y_mode.currentData() == self.AUTO:
            self._plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        else:
            self._plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            self._apply_manual_y()

    def _apply_x_program_mode(self) -> None:
        if self._programmatic or self._last_t.size == 0:
            return
        mode = str(self.x_mode.currentData())
        if mode == self.MANUAL:
            return
        low = float(np.min(self._last_t))
        high = float(np.max(self._last_t))
        if not math.isfinite(low) or not math.isfinite(high):
            return
        if high <= low:
            high = low + 1.0e-6
        if mode == self.FOLLOW:
            low = max(low, high - self.follow_s.value())
        self._programmatic = True
        try:
            self._plot.setXRange(low, high, padding=0.0)
        finally:
            self._programmatic = False

    def _apply_manual_x(self) -> None:
        if self._programmatic or self.x_mode.currentData() != self.MANUAL:
            return
        low, high = self.x_from.value(), self.x_to.value()
        if high <= low:
            return
        self._programmatic = True
        try:
            self._plot.setXRange(low, high, padding=0.0)
        finally:
            self._programmatic = False

    def _apply_manual_y(self) -> None:
        if self._programmatic or self.y_mode.currentData() != self.MANUAL:
            return
        low, high = self.y_min.value(), self.y_max.value()
        if high <= low:
            return
        self._programmatic = True
        try:
            self._plot.setYRange(low, high, padding=0.0)
        finally:
            self._programmatic = False

    def _mouse_range_changed(self, *args: object) -> None:
        del args
        if self._programmatic:
            return
        x_range, y_range = self._plot.viewRange()
        self._programmatic = True
        try:
            self._set_combo_data(self.x_mode, self.MANUAL)
            self._set_combo_data(self.y_mode, self.MANUAL)
            self.x_from.setValue(float(x_range[0]))
            self.x_to.setValue(float(x_range[1]))
            self.y_min.setValue(float(y_range[0]))
            self.y_max.setValue(float(y_range[1]))
            self._plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        finally:
            self._programmatic = False
        self._update_enabled()

    def _update_enabled(self) -> None:
        x_mode = str(self.x_mode.currentData())
        manual_x = x_mode == self.MANUAL
        self.follow_s.setEnabled(x_mode == self.FOLLOW)
        self.x_from.setEnabled(manual_x)
        self.x_to.setEnabled(manual_x)
        manual_y = self.y_mode.currentData() == self.MANUAL
        self.y_min.setEnabled(manual_y)
        self.y_max.setEnabled(manual_y)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0 or index == combo.currentIndex():
            return
        blocker = QSignalBlocker(combo)
        combo.setCurrentIndex(index)
        del blocker
