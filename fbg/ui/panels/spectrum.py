"""Панель спектра: одиночный и периодический снимок 30 07."""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fbg.core.session import SessionState
from fbg.ui import texts
from fbg.ui.app import AppController
from fbg.ui.docking import DockTab, scrollable
from fbg.ui.models import AppSnapshot, SpectrumModel


class SpectrumPanel(DockTab):
    """Один канал 30 07 с возможностью периодического повторения."""

    layout_key = "spectrum"

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._shown_version = -1
        self._shown_scale_index = -1
        self._actual_period_s: float | None = None
        self.channel = QComboBox()
        for index in range(controller.config.profile.channels):
            self.channel.addItem(texts.channel_label(index), index)
        self.channel.setMaximumWidth(180)
        self.period = QDoubleSpinBox()
        self.period.setDecimals(2)
        self.period.setRange(0.10, 3600.0)
        self.period.setValue(1.0)
        self.period.setMaximumWidth(140)
        self.frequency_label = QLabel()
        self.threshold = QSpinBox()
        self.threshold.setRange(0, controller.config.profile.adc_max)
        self.threshold.setValue(3000)
        self.threshold.setMaximumWidth(140)
        self.scale = QComboBox()
        self.scale.addItems([texts.SPECTRUM_SCALE_ADC, texts.SPECTRUM_SCALE_DBM])
        self.scale.setMaximumWidth(140)
        self.take_button = QPushButton(texts.BUTTON_TAKE_SPECTRUM)
        self.start_button = QPushButton(texts.BUTTON_START_SPECTRUM)
        self.stop_button = QPushButton(texts.BUTTON_STOP_SPECTRUM)
        self.warning = QLabel(texts.SPECTRUM_WARNING_STREAM)
        self.warning.setWordWrap(True)
        self.max_label = QLabel(texts.UNKNOWN)
        self.saturation_label = QLabel(texts.UNKNOWN)
        self.saturation_label.setWordWrap(True)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(240)
        self.plot.hide()
        # Спектр — 2551 реальных ADC-отсчётов, по которым смотрят форму пика.
        # Прореживание здесь запрещено Р76: это не длинный временной ряд, и
        # downsampling превратил бы измеренную форму в огибающую.
        self.plot.setLabel("bottom", "Длина волны, нм")
        self.plot.setLabel("left", texts.SPECTRUM_SCALE_ADC)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve = self.plot.plot()
        self.empty_graph_label = QLabel(texts.EMPTY_GRAPH_HINT)
        self.empty_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["№", "Вершина, нм", "Центроид, нм", "ADC", "дБм", "Ширина, нм", "FWHM, нм", "Насыщено"]
        )
        self._build_layout()
        self.take_button.clicked.connect(self._take_once)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.period.valueChanged.connect(self._update_frequency_label)
        self.scale.currentIndexChanged.connect(lambda _index: self.refresh(controller.snapshot()))
        self._update_frequency_label(self.period.value())
        self.refresh(controller.snapshot())

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow(texts.LABEL_SPECTRUM_CHANNEL, self.channel)
        form.addRow(texts.LABEL_SPECTRUM_PERIOD, self.period)
        form.addRow(texts.LABEL_SPECTRUM_FREQUENCY, self.frequency_label)
        form.addRow(texts.LABEL_SPECTRUM_THRESHOLD, self.threshold)
        form.addRow(texts.LABEL_SPECTRUM_SCALE, self.scale)
        form.addRow(texts.LABEL_SPECTRUM_MAX, self.max_label)
        form.addRow(texts.LABEL_SPECTRUM_SATURATION, self.saturation_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self.take_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        control = QWidget()
        control_layout = QVBoxLayout()
        control_layout.addLayout(form)
        control_layout.addLayout(buttons)
        control_layout.addWidget(self.warning)
        control.setLayout(control_layout)
        graph = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.plot)
        graph.setLayout(graph_layout)
        regions = QWidget()
        regions_layout = QVBoxLayout()
        regions_layout.addWidget(self.table)
        regions.setLayout(regions_layout)
        graph_layout.insertWidget(0, self.empty_graph_label)
        self.control_dock = self.add_panel_dock(
            texts.GROUP_SPECTRUM_CONTROL,
            scrollable(control),
            "spectrum.control",
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.graph_dock = self.add_panel_dock(
            texts.GROUP_SPECTRUM_GRAPH,
            graph,
            "spectrum.graph",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.regions_dock = self.add_panel_dock(
            texts.GROUP_SPECTRUM_REGIONS,
            regions,
            "spectrum.regions",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.reset_layout()

    def _apply_default_splits(self) -> None:
        self.splitDockWidget(
            self.graph_dock,
            self.regions_dock,
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.control_dock, self.graph_dock],
            [320, 730],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.graph_dock, self.regions_dock],
            [430, 250],
            Qt.Orientation.Vertical,
        )

    def _update_frequency_label(self, period_s: float) -> None:
        requested_hz = 1.0 / period_s
        actual = self._actual_period_s
        if actual is None or not math.isfinite(actual) or actual <= 0.0:
            actual_text = texts.UNKNOWN
        else:
            actual_text = f"{1.0 / actual:.3f} Гц ({actual:.3f} с)"
        self.frequency_label.setText(f"задано {requested_hz:.3f} Гц; факт {actual_text}")

    def _take_once(self) -> None:
        try:
            self._controller.take_spectrum_async(
                int(self.channel.currentData()), self.threshold.value()
            )
        except (RuntimeError, ValueError) as exc:
            self._controller.note(f"спектр: {exc}")
        self.refresh(self._controller.snapshot())

    def _start(self) -> None:
        try:
            self._controller.start_spectrum_continuous(
                int(self.channel.currentData()), self.period.value(), self.threshold.value()
            )
        except (RuntimeError, ValueError) as exc:
            self._controller.note(f"спектр: {exc}")
        self.refresh(self._controller.snapshot())

    def _stop(self) -> None:
        self._controller.stop_spectrum_continuous()
        self.refresh(self._controller.snapshot())

    @staticmethod
    def _number(value: float | None, digits: int = 4) -> str:
        return texts.UNKNOWN if value is None or not math.isfinite(value) else f"{value:.{digits}f}"

    def _show_model(self, model: SpectrumModel) -> None:
        dbm = self.scale.currentText() == texts.SPECTRUM_SCALE_DBM
        y = model.power_dbm if dbm else model.adc
        self.curve.setData(model.wavelength_nm, y, connect="finite")
        has_data = model.wavelength_nm.size > 0 and bool(np.any(np.isfinite(y)))
        self.plot.setVisible(has_data)
        self.empty_graph_label.setVisible(not has_data)
        self.plot.setLabel("left", texts.SPECTRUM_SCALE_DBM if dbm else texts.SPECTRUM_SCALE_ADC)
        self.max_label.setText(str(model.max_adc))
        if model.saturated:
            self.saturation_label.setText(
                f"{model.saturated_points} точек. {texts.SPECTRUM_SATURATION_WARNING}"
            )
        else:
            self.saturation_label.setText("нет")
        # Строки не пересоздаются без необходимости: иначе Qt сбрасывает
        # выделение и прокрутку таблицы на каждом тике даже при том же снимке.
        selected_row = self.table.currentRow()
        scroll = self.table.verticalScrollBar().value()
        if self.table.rowCount() != len(model.regions):
            self.table.setRowCount(len(model.regions))
        for row, region in enumerate(model.regions):
            values = [
                str(row + 1),
                self._number(region.peak_nm),
                self._number(region.centroid_nm),
                str(region.amplitude_adc),
                self._number(region.amplitude_dbm, 2),
                self._number(region.width_nm),
                self._number(region.fwhm_nm),
                str(region.saturated_points),
            ]
            for column, value in enumerate(values):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                if item.text() != value:
                    item.setText(value)
        if 0 <= selected_row < self.table.rowCount():
            self.table.selectRow(selected_row)
        self.table.verticalScrollBar().setValue(scroll)

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Один UI-такт: не опрашивает прибор, только показывает последний снимок."""
        self._actual_period_s = snapshot.spectrum_actual_period_s
        self._update_frequency_label(self.period.value())
        scale_index = self.scale.currentIndex()
        if snapshot.spectrum is not None and (
            snapshot.spectrum_version != self._shown_version
            or scale_index != self._shown_scale_index
        ):
            self._show_model(snapshot.spectrum)
            self._shown_version = snapshot.spectrum_version
            self._shown_scale_index = scale_index
        recording = snapshot.recording
        connected = snapshot.state in (SessionState.IDLE, SessionState.STREAMING)
        available = connected and not recording and not snapshot.spectrum_busy
        self.take_button.setEnabled(available)
        self.start_button.setEnabled(available)
        self.stop_button.setEnabled(snapshot.spectrum_running)
        self.channel.setEnabled(not snapshot.spectrum_busy)
        self.period.setEnabled(not snapshot.spectrum_busy)
        self.threshold.setEnabled(not snapshot.spectrum_busy)
        if recording:
            self.warning.setText(
                texts.SPECTRUM_RECORDING_LOCKED + "\n" + texts.SPECTRUM_WARNING_STREAM
            )
        else:
            self.warning.setText(texts.SPECTRUM_WARNING_STREAM)
