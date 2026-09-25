"""Контракт пользовательской области графика; Qt/offscreen."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from fbg.ui.graph_range import GraphRangeControls

pytestmark = pytest.mark.ui


def test_такт_не_перезаписывает_ручную_область() -> None:
    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    controls = GraphRangeControls(plot)
    try:
        controls.x_mode.setCurrentIndex(controls.x_mode.findData(controls.MANUAL))
        controls.y_mode.setCurrentIndex(controls.y_mode.findData(controls.MANUAL))
        controls.x_from.setValue(2.0)
        controls.x_to.setValue(4.0)
        controls.y_min.setValue(-3.0)
        controls.y_max.setValue(7.0)
        app.processEvents()
        before = plot.getViewBox().viewRange()

        controls.apply_time_axis(np.asarray([0.0, 10.0, 20.0]))
        app.processEvents()
        after = plot.getViewBox().viewRange()

        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])
    finally:
        plot.close()
        plot.deleteLater()


def test_мышь_переводит_обе_оси_в_ручной_режим() -> None:
    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    controls = GraphRangeControls(plot)
    try:
        plot.setXRange(5.0, 8.0, padding=0.0)
        plot.setYRange(-2.0, 3.0, padding=0.0)
        controls._mouse_range_changed()
        app.processEvents()

        assert controls.x_mode.currentData() == controls.MANUAL
        assert controls.y_mode.currentData() == controls.MANUAL
        assert controls.x_from.value() == pytest.approx(5.0)
        assert controls.x_to.value() == pytest.approx(8.0)
        assert controls.y_min.value() == pytest.approx(-2.0)
        assert controls.y_max.value() == pytest.approx(3.0)
    finally:
        plot.close()
        plot.deleteLater()
