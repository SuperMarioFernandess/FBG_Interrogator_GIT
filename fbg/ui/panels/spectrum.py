"""Панель спектра: одиночный и периодический снимок 30 07."""

import math

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
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
from fbg.ui.models import AppSnapshot, SpectrumModel


class SpectrumPanel(QWidget):
    """Один канал 30 07 с возможностью периодического повторения."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.channel = QComboBox()
        for index in range(controller.config.profile.channels):
            self.channel.addItem(texts.channel_label(index), index)
        self.period = QDoubleSpinBox()
        self.period.setDecimals(2)
        self.period.setRange(0.10, 3600.0)
        self.period.setValue(1.0)
        self.frequency_label = QLabel()
        self.threshold = QSpinBox()
        self.threshold.setRange(0, controller.config.profile.adc_max)
        self.threshold.setValue(3000)
        self.scale = QComboBox()
        self.scale.addItems([texts.SPECTRUM_SCALE_ADC, texts.SPECTRUM_SCALE_DBM])
        self.take_button = QPushButton(texts.BUTTON_TAKE_SPECTRUM)
        self.start_button = QPushButton(texts.BUTTON_START_SPECTRUM)
        self.stop_button = QPushButton(texts.BUTTON_STOP_SPECTRUM)
        self.warning = QLabel(texts.SPECTRUM_WARNING_STREAM)
        self.warning.setWordWrap(True)
        self.max_label = QLabel(texts.UNKNOWN)
        self.saturation_label = QLabel(texts.UNKNOWN)
        self.saturation_label.setWordWrap(True)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Длина волны, нм")
        self.plot.setLabel("left", texts.SPECTRUM_SCALE_ADC)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve = self.plot.plot()
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
        control = QGroupBox(texts.GROUP_SPECTRUM_CONTROL)
        control_layout = QVBoxLayout()
        control_layout.addLayout(form)
        control_layout.addLayout(buttons)
        control_layout.addWidget(self.warning)
        control.setLayout(control_layout)
        graph = QGroupBox(texts.GROUP_SPECTRUM_GRAPH)
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.plot)
        graph.setLayout(graph_layout)
        regions = QGroupBox(texts.GROUP_SPECTRUM_REGIONS)
        regions_layout = QVBoxLayout()
        regions_layout.addWidget(self.table)
        regions.setLayout(regions_layout)
        layout = QVBoxLayout(self)
        layout.addWidget(control)
        layout.addWidget(graph, 1)
        layout.addWidget(regions, 1)

    def _update_frequency_label(self, period_s: float) -> None:
        frequency_hz = 1.0 / period_s
        self.frequency_label.setText(f"{frequency_hz:.3f}")

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
        self.plot.setLabel("left", texts.SPECTRUM_SCALE_DBM if dbm else texts.SPECTRUM_SCALE_ADC)
        self.max_label.setText(str(model.max_adc))
        if model.saturated:
            self.saturation_label.setText(
                f"{model.saturated_points} точек. {texts.SPECTRUM_SATURATION_WARNING}"
            )
        else:
            self.saturation_label.setText("нет")
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
                self.table.setItem(row, column, QTableWidgetItem(value))

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Один UI-такт: не опрашивает прибор, только показывает последний снимок."""
        if snapshot.spectrum is not None:
            self._show_model(snapshot.spectrum)
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
