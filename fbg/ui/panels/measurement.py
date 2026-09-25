"""Панель измерения: пользовательский самописец λ(t)/Δλ(t), таблица и запись CSV."""

import math
import threading
from dataclasses import replace
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fbg.core.session import SessionState
from fbg.io.averaging import AveragingExportResult, average_recording
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.docking import DockTab
from fbg.ui.graph_range import GraphRangeControls
from fbg.ui.history import can_aggregate_exactly, history_memory_estimate_bytes
from fbg.ui.models import AppSnapshot, MeasurementTableModel, SlotRef

DEFAULT_SELECTED_SLOTS = 4
MIN_HISTORY_HOURS = 0.01
MAX_HISTORY_HOURS = 168.0


class _MeasurementQtTableModel(QAbstractTableModel):
    """Qt-обёртка над последним кадром; штатный такт не сбрасывает таблицу."""

    def __init__(self, model: MeasurementTableModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model

    @property
    def model(self) -> MeasurementTableModel:
        return self._model

    def replace(self, model: MeasurementTableModel) -> None:
        same_shape = (
            self._model.channels == model.channels and self._model.positions == model.positions
        )
        if not same_shape:
            self.beginResetModel()
            self._model = model
            self.endResetModel()
            return
        self._model = model
        if self.rowCount() and self.columnCount() > 1:
            self.dataChanged.emit(
                self.index(0, 1),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
                [Qt.ItemDataRole.DisplayRole],
            )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else self._model.positions

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if parent.isValid() else 1 + self._model.channels

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row, column = index.row(), index.column()
        if column == 0:
            return str(row + 1)
        channel = column - 1
        value = float(self._model.wavelength_nm[channel, row])
        return texts.UNKNOWN if not self._model.valid[channel, row] else f"{value:.4f}"

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        if section == 0:
            return texts.TABLE_POSITION
        return f"К{section} {texts.TABLE_WAVELENGTH}"


class MeasurementPanel(DockTab):
    """График с пользовательским λ₀, текущий кадр и неизменённая запись CSV."""

    layout_key = "measurement"

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._selection_loading = False
        self._record_settings_loading = False
        self._record_settings_dirty = False
        self._curves: dict[SlotRef, pg.PlotDataItem] = {}
        self._bands: dict[SlotRef, tuple[pg.PlotDataItem, pg.PlotDataItem, pg.FillBetweenItem]] = {}
        self._range_bands: dict[
            SlotRef, tuple[pg.PlotDataItem, pg.PlotDataItem, pg.FillBetweenItem]
        ] = {}
        self._graph_model: models.MeasurementHistoryGraphModel | None = None
        self._slot_checks: dict[SlotRef, QCheckBox] = {}
        self._slot_current: dict[SlotRef, QTableWidgetItem] = {}
        self._slot_lambda0: dict[SlotRef, QLineEdit] = {}
        self._average_thread: threading.Thread | None = None
        self._average_result: AveragingExportResult | None = None
        self._average_error: str | None = None
        self._average_reported = False

        profile = controller.config.profile
        self.position_table = QTableWidget(profile.channels * profile.fbg_per_channel, 5)
        self.position_table.setHorizontalHeaderLabels(
            ["Показывать", "Позиция", "Текущая λ, нм", "λ₀, нм", ""]
        )
        self.position_table.setAlternatingRowColors(True)
        self.position_table.verticalHeader().setVisible(False)
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.position_table.setMinimumWidth(470)
        self.set_all_lambda0_button = QPushButton(texts.BUTTON_LAMBDA0_ALL_CHECKED)
        self.slot_warning = QLabel(texts.GRAPH_LAMBDA0_SLOT_WARNING)
        self.slot_warning.setWordWrap(True)
        self.slot_warning.setToolTip(texts.GRAPH_LAMBDA0_SLOT_WARNING)

        self.history_spin = QDoubleSpinBox()
        self.history_spin.setDecimals(2)
        self.history_spin.setSingleStep(1.0)
        self.history_spin.setRange(MIN_HISTORY_HOURS, MAX_HISTORY_HOURS)
        self.history_spin.setValue(24.0)
        self.history_spin.setMaximumWidth(140)
        self.history_memory = QLabel()
        self.history_memory.setWordWrap(True)
        self.graph_mode = QComboBox()
        self.graph_mode.addItem(texts.GRAPH_MODE_DELTA, "delta")
        self.graph_mode.addItem(texts.GRAPH_MODE_ABSOLUTE, "absolute")
        self.start_graph_button = QPushButton(texts.BUTTON_GRAPH_START)
        self.stop_graph_button = QPushButton(texts.BUTTON_GRAPH_STOP)
        self.clear_graph_button = QPushButton(texts.BUTTON_GRAPH_CLEAR)

        self.averaging_enabled = QCheckBox()
        self.averaging_enabled.setChecked(False)
        self.averaging_window = QDoubleSpinBox()
        self.averaging_window.setDecimals(1)
        self.averaging_window.setRange(models.MIN_AVERAGING_MS, models.MAX_AVERAGING_MS)
        self.averaging_window.setValue(models.DEFAULT_AVERAGING_MS)
        self.averaging_window.setMaximumWidth(140)
        self.averaging_frames = QLabel()
        self.averaging_sigma = QCheckBox()
        self.averaging_sigma.setChecked(True)
        self.averaging_n = QLabel()
        self.averaging_notice = QLabel()
        self.averaging_notice.setWordWrap(True)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(120)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.plot.setLabel("bottom", texts.GRAPH_AXIS_TIME)
        self.plot.setLabel("left", texts.GRAPH_AXIS_DELTA_NM)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend()
        self.range_controls = GraphRangeControls(self.plot)
        self.graph_hint = QLabel(texts.GRAPH_BASELINE_HINT)
        self.graph_hint.setWordWrap(True)
        self.empty_graph_label = QLabel(texts.EMPTY_GRAPH_HINT)
        self.empty_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_label = QLabel()

        empty_table = models.measurement_table_model(controller.snapshot())
        self.table_model = _MeasurementQtTableModel(empty_table, self)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.temperature_label = QLabel()
        self.temperature_label.setWordWrap(True)

        self.record_directory = QLineEdit()
        self.browse_button = QPushButton(texts.BUTTON_BROWSE_DIRECTORY)
        self.record_decimation = QSpinBox()
        self.record_decimation.setRange(1, 100_000)
        self.record_decimation.setMaximumWidth(140)
        self.record_limit = QSpinBox()
        self.record_limit.setRange(0, profile.fbg_per_channel)
        self.record_limit.setSpecialValueText(texts.RECORD_LIMIT_ALL)
        self.record_limit.setMaximumWidth(140)
        self.record_estimate = QLabel()
        self.record_estimate.setWordWrap(True)
        self.record_state = QLabel()
        self.record_file = QLabel()
        self.record_file.setWordWrap(True)
        self.record_rows = QLabel()
        self.record_size = QLabel()
        self.record_elapsed = QLabel()
        self.record_gaps = QLabel()
        self.record_gaps.setWordWrap(True)
        self.record_error = QLabel()
        self.record_error.setWordWrap(True)
        self.record_error.setStyleSheet("font-weight: bold;")
        self.start_record_button = QPushButton(texts.BUTTON_START_RECORDING)
        self.stop_record_button = QPushButton(texts.BUTTON_STOP_RECORDING)
        self.average_recording_button = QPushButton(texts.BUTTON_AVERAGE_RECORDING)
        self.average_recording_state = QLabel()
        self.average_recording_state.setWordWrap(True)

        self._build_position_table(profile.channels, profile.fbg_per_channel)
        self._build_layout()
        self._connect_signals()
        self._sync_history_request()
        self.refresh(controller.snapshot(include_sensor_data=False))

    def _build_position_table(self, channels: int, positions: int) -> None:
        self._selection_loading = True
        try:
            row = 0
            for channel in range(channels):
                for position in range(positions):
                    slot = SlotRef(channel, position)
                    check = QCheckBox()
                    check.setChecked(channel == 0 and position < DEFAULT_SELECTED_SLOTS)
                    check.toggled.connect(
                        lambda _checked, current=slot: self._on_slot_checked(current)
                    )
                    self.position_table.setCellWidget(row, 0, check)
                    label = QTableWidgetItem(texts.slot_label(channel, position))
                    label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.position_table.setItem(row, 1, label)
                    current = QTableWidgetItem(texts.UNKNOWN)
                    current.setFlags(current.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.position_table.setItem(row, 2, current)
                    edit = QLineEdit()
                    edit.setPlaceholderText("нет λ₀ — Δλ скрыта")
                    edit.setToolTip(texts.GRAPH_LAMBDA0_SLOT_WARNING)
                    edit.editingFinished.connect(lambda current=slot: self._lambda0_edited(current))
                    self.position_table.setCellWidget(row, 3, edit)
                    button = QPushButton(texts.BUTTON_LAMBDA0_CURRENT)
                    button.clicked.connect(
                        lambda _checked=False, current=slot: self._take_lambda0(current)
                    )
                    self.position_table.setCellWidget(row, 4, button)
                    self._slot_checks[slot] = check
                    self._slot_current[slot] = current
                    self._slot_lambda0[slot] = edit
                    row += 1
        finally:
            self._selection_loading = False

    def _build_layout(self) -> None:
        controls = QGridLayout()
        controls.addWidget(QLabel(texts.LABEL_GRAPH_HISTORY), 0, 0)
        controls.addWidget(self.history_spin, 0, 1)
        controls.addWidget(self.history_memory, 0, 2, 1, 4)
        controls.addWidget(QLabel(texts.LABEL_GRAPH_MODE), 1, 0)
        controls.addWidget(self.graph_mode, 1, 1)
        controls.addWidget(self.start_graph_button, 1, 2)
        controls.addWidget(self.stop_graph_button, 1, 3)
        controls.addWidget(self.clear_graph_button, 1, 4)
        controls.addWidget(QLabel(texts.LABEL_AVERAGING_ENABLED), 2, 0)
        controls.addWidget(self.averaging_enabled, 2, 1)
        controls.addWidget(QLabel(texts.LABEL_AVERAGING_WINDOW), 2, 2)
        controls.addWidget(self.averaging_window, 2, 3)
        controls.addWidget(QLabel(texts.LABEL_AVERAGING_FRAMES), 3, 0)
        controls.addWidget(self.averaging_frames, 3, 1)
        controls.addWidget(QLabel(texts.LABEL_AVERAGING_SIGMA), 3, 2)
        controls.addWidget(self.averaging_sigma, 3, 3)
        controls.addWidget(QLabel(texts.LABEL_AVERAGING_N), 4, 0)
        controls.addWidget(self.averaging_n, 4, 1)
        controls.addWidget(self.averaging_notice, 4, 2, 1, 4)
        controls.setColumnStretch(5, 1)
        selection_layout = QVBoxLayout()
        selection_layout.addLayout(controls)
        selection_layout.addWidget(self.set_all_lambda0_button)
        selection_layout.addWidget(self.slot_warning)
        selection_layout.addWidget(self.position_table, 1)
        selection_box = QWidget()
        selection_box.setLayout(selection_layout)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.quality_label)
        graph_layout.addWidget(self.range_controls)
        graph_layout.addWidget(self.empty_graph_label, 1)
        graph_layout.addWidget(self.plot, 1)
        graph_layout.addWidget(self.graph_hint)
        graph_box = QWidget()
        graph_box.setLayout(graph_layout)

        table_layout = QVBoxLayout()
        table_layout.addWidget(self.temperature_label)
        table_layout.addWidget(self.table, 1)
        table_box = QWidget()
        table_box.setLayout(table_layout)

        directory_row = QHBoxLayout()
        directory_row.addWidget(self.record_directory, 1)
        directory_row.addWidget(self.browse_button)
        record_form = QFormLayout()
        record_form.addRow(texts.LABEL_RECORD_DIRECTORY, directory_row)
        record_form.addRow(texts.LABEL_RECORD_DECIMATION, self.record_decimation)
        record_form.addRow(texts.LABEL_RECORD_FBG_LIMIT, self.record_limit)
        record_form.addRow(texts.LABEL_RECORD_ESTIMATE, self.record_estimate)
        record_form.addRow(texts.LABEL_RECORD_STATE, self.record_state)
        record_form.addRow(texts.LABEL_RECORD_FILE, self.record_file)
        record_form.addRow(texts.LABEL_RECORD_ROWS, self.record_rows)
        record_form.addRow(texts.LABEL_RECORD_SIZE, self.record_size)
        record_form.addRow(texts.LABEL_RECORD_ELAPSED, self.record_elapsed)
        record_form.addRow(texts.LABEL_RECORD_GAPS, self.record_gaps)
        record_buttons = QHBoxLayout()
        record_buttons.addWidget(self.start_record_button)
        record_buttons.addWidget(self.stop_record_button)
        record_buttons.addWidget(self.average_recording_button)
        record_buttons.addWidget(self.average_recording_state, 1)
        record_form.addRow("", record_buttons)
        record_form.addRow("", self.record_error)
        record_box = QWidget()
        record_box.setLayout(record_form)

        self.selection_dock = self.add_panel_dock(
            texts.GROUP_TRACE_SELECTION,
            selection_box,
            "measurement.selection",
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.graph_dock = self.add_panel_dock(
            texts.GROUP_MEASUREMENT_GRAPH,
            graph_box,
            "measurement.graph",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.table_dock = self.add_panel_dock(
            texts.GROUP_MEASUREMENT_TABLE,
            table_box,
            "measurement.table",
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.record_dock = self.add_panel_dock(
            texts.GROUP_RECORDING,
            record_box,
            "measurement.record",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.reset_layout()

    def _apply_default_splits(self) -> None:
        self.splitDockWidget(self.selection_dock, self.table_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self.graph_dock, self.record_dock, Qt.Orientation.Vertical)
        self.resizeDocks(
            [self.selection_dock, self.graph_dock], [400, 700], Qt.Orientation.Horizontal
        )
        self.resizeDocks([self.graph_dock, self.record_dock], [430, 250], Qt.Orientation.Vertical)

    def _connect_signals(self) -> None:
        self.history_spin.valueChanged.connect(self._on_history_changed)
        self.graph_mode.currentIndexChanged.connect(self._on_graph_mode_changed)
        self.averaging_enabled.toggled.connect(self._on_averaging_changed)
        self.averaging_window.valueChanged.connect(self._on_averaging_changed)
        self.averaging_sigma.toggled.connect(lambda _checked: self._refresh_from_controller())
        self.start_graph_button.clicked.connect(self._start_graph)
        self.stop_graph_button.clicked.connect(self._stop_graph)
        self.clear_graph_button.clicked.connect(self._clear_graph)
        self.set_all_lambda0_button.clicked.connect(self._take_lambda0_for_checked)
        self.record_directory.textEdited.connect(self._on_record_setting_changed)
        self.record_decimation.valueChanged.connect(self._on_record_setting_changed)
        self.record_limit.valueChanged.connect(self._on_record_setting_changed)
        self.browse_button.clicked.connect(self._browse_directory)
        self.start_record_button.clicked.connect(self._start_recording)
        self.stop_record_button.clicked.connect(self._stop_recording)
        self.average_recording_button.clicked.connect(self._start_average_export)

    def selected_slots(self) -> tuple[SlotRef, ...]:
        return tuple(slot for slot, check in self._slot_checks.items() if check.isChecked())

    def _history_depth_s(self) -> float:
        return self.history_spin.value() * 3600.0

    def _sync_history_request(self) -> None:
        self._controller.set_measurement_history_request(
            [(slot.channel, slot.position) for slot in self.selected_slots()],
            self._history_depth_s(),
        )

    def _on_slot_checked(self, _slot: SlotRef) -> None:
        if self._selection_loading:
            return
        self._graph_model = None
        self._sync_history_request()
        self._refresh_from_controller()

    def _on_history_changed(self, _value: float) -> None:
        if self._selection_loading:
            return
        self._sync_history_request()
        self._graph_model = None
        self._refresh_from_controller()

    def _on_graph_mode_changed(self, _index: int) -> None:
        self._graph_model = None
        self.plot.setLabel(
            "left",
            texts.GRAPH_AXIS_DELTA_NM if self.graph_mode.currentData() == "delta" else "λ, нм",
        )
        self._refresh_from_controller()

    def _averaging_window_s(self) -> float | None:
        if not self.averaging_enabled.isChecked():
            return None
        return self.averaging_window.value() / 1000.0

    def _averaging_available(self) -> bool:
        window = self._averaging_window_s()
        return window is None or can_aggregate_exactly(window)

    def _on_averaging_changed(self, _value: object = None) -> None:
        self._graph_model = None
        self._refresh_from_controller()

    def _refresh_from_controller(self) -> None:
        self.refresh(self._controller.snapshot(include_sensor_data=False))

    def _start_graph(self) -> None:
        if not self._averaging_available():
            self._controller.note(texts.GRAPH_AVERAGING_TOO_SHORT)
            return
        self._controller.start_measurement_history(averaging_window_s=self._averaging_window_s())
        self._refresh_from_controller()

    def _stop_graph(self) -> None:
        self._controller.stop_measurement_history()
        self._refresh_from_controller()

    def _clear_graph(self) -> None:
        self._controller.clear_measurement_history(averaging_window_s=self._averaging_window_s())
        self._graph_model = None
        self._refresh_from_controller()

    def _lambda0_edited(self, slot: SlotRef) -> None:
        edit = self._slot_lambda0[slot]
        text = edit.text().strip().replace(",", ".")
        if not text:
            self._controller.set_measurement_lambda0(slot.channel, slot.position, None)
            self._graph_model = None
            return
        try:
            value = float(text)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError
        except ValueError:
            self._controller.note(
                f"{texts.slot_label(slot.channel, slot.position)}: "
                "λ₀ должна быть положительным числом"
            )
            current = self._controller.measurement_lambda0(slot.channel, slot.position)
            edit.setText("" if current is None else f"{current:.6f}")
            return
        self._controller.set_measurement_lambda0(slot.channel, slot.position, value)
        self._graph_model = None

    def _take_lambda0(self, slot: SlotRef) -> bool:
        if not self._averaging_available():
            self._controller.note(texts.GRAPH_AVERAGING_TOO_SHORT)
            return False
        try:
            value = self._controller.current_measurement_wavelength(
                slot.channel,
                slot.position,
                averaging_window_s=self._averaging_window_s(),
            )
        except ValueError as exc:
            self._controller.note(str(exc))
            return False
        if value is None:
            self._controller.note(
                f"{texts.slot_label(slot.channel, slot.position)}: нет текущего валидного λ"
            )
            return False
        self._controller.set_measurement_lambda0(slot.channel, slot.position, value)
        self._slot_lambda0[slot].setText(f"{value:.6f}")
        self._graph_model = None
        return True

    def _take_lambda0_for_checked(self) -> None:
        for slot in self.selected_slots():
            self._take_lambda0(slot)
        self._refresh_from_controller()

    def _update_lambda0_table(self, snapshot: AppSnapshot) -> None:
        current = models.measurement_table_model(snapshot)
        lambda0 = {
            (channel, position): value
            for channel, position, value in snapshot.measurement_lambda0_nm
        }
        for slot, item in self._slot_current.items():
            value = float(current.wavelength_nm[slot.channel, slot.position])
            item.setText(texts.UNKNOWN if not math.isfinite(value) else f"{value:.6f}")
            edit = self._slot_lambda0[slot]
            reference = lambda0.get((slot.channel, slot.position))
            if not edit.hasFocus():
                edit.setText("" if reference is None else f"{reference:.6f}")
            missing = self.graph_mode.currentData() == "delta" and reference is None
            edit.setToolTip(
                "Нет λ₀: линия Δλ не рисуется. " + texts.GRAPH_LAMBDA0_SLOT_WARNING
                if missing
                else texts.GRAPH_LAMBDA0_SLOT_WARNING
            )

    def _update_averaging_controls(self, snapshot: AppSnapshot) -> None:
        enabled = self.averaging_enabled.isChecked()
        self.averaging_sigma.setEnabled(enabled)
        frames = models.expected_averaging_frames(snapshot, self.averaging_window.value())
        self.averaging_frames.setText("—" if frames <= 0 else f"≈ {frames} кадров")
        self.averaging_notice.setText(
            "" if self._averaging_available() else texts.GRAPH_AVERAGING_TOO_SHORT
        )

    def _update_history_estimate(self, snapshot: AppSnapshot) -> None:
        positions = snapshot.profile.channels * snapshot.profile.fbg_per_channel
        estimate = history_memory_estimate_bytes(self._history_depth_s(), positions)
        self.history_memory.setText(
            f"Оценка максимума памяти для {positions} активных позиций: "
            f"{self._format_bytes(estimate)}; не найденные позиции массивы не выделяют."
        )

    def _new_band(
        self, color: object, *, alpha: int
    ) -> tuple[pg.PlotDataItem, pg.PlotDataItem, pg.FillBetweenItem]:
        transparent_pen = pg.mkPen(0, 0, 0, 0)
        upper = pg.PlotDataItem(pen=transparent_pen)
        lower = pg.PlotDataItem(pen=transparent_pen)
        red, green, blue, _alpha = color.getRgb()
        fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(red, green, blue, alpha))
        fill.setZValue(-10)
        self.plot.addItem(upper)
        self.plot.addItem(lower)
        self.plot.addItem(fill)
        return upper, lower, fill

    def _remove_trace(self, slot: SlotRef) -> None:
        curve = self._curves.pop(slot, None)
        if curve is not None:
            self.plot.removeItem(curve)
        for collection in (self._bands, self._range_bands):
            band = collection.pop(slot, None)
            if band is not None:
                for item in band:
                    self.plot.removeItem(item)

    def _update_graph(self, snapshot: AppSnapshot) -> None:
        selected = self.selected_slots()
        if not self._averaging_available():
            for slot in tuple(self._curves):
                self._remove_trace(slot)
            self.plot.hide()
            self.empty_graph_label.show()
            self.averaging_n.setText(texts.UNKNOWN)
            return
        model = models.measurement_history_graph_model(
            snapshot,
            selected,
            self._graph_model,
            mode=str(self.graph_mode.currentData()),
            averaging_window_s=self._averaging_window_s(),
        )
        unchanged = model is self._graph_model
        self._graph_model = model
        selected_set = set(selected)
        for slot in tuple(self._curves):
            if slot not in selected_set:
                self._remove_trace(slot)

        for index, trace in enumerate(model.traces):
            curve = self._curves.get(trace.slot)
            if curve is None:
                color = pg.intColor(index, hues=max(1, len(selected)))
                curve = self.plot.plot(
                    pen=pg.mkPen(color),
                    name=texts.slot_label(trace.slot.channel, trace.slot.position),
                )
                self._curves[trace.slot] = curve
                self._range_bands[trace.slot] = self._new_band(color, alpha=24)
                self._bands[trace.slot] = self._new_band(color, alpha=45)
            if not unchanged:
                curve.setData(model.t_s, trace.values_nm, connect="finite")
                extrema = self._range_bands[trace.slot]
                extrema[0].setData(model.t_s, trace.max_nm, connect="finite")
                extrema[1].setData(model.t_s, trace.min_nm, connect="finite")
                sigma_band = self._bands[trace.slot]
                sigma_band[0].setData(model.t_s, trace.values_nm + trace.sigma_nm, connect="finite")
                sigma_band[1].setData(model.t_s, trace.values_nm - trace.sigma_nm, connect="finite")
            for item in self._range_bands[trace.slot]:
                item.setVisible(True)
            for item in self._bands[trace.slot]:
                item.setVisible(
                    self.averaging_enabled.isChecked() and self.averaging_sigma.isChecked()
                )

        counts: list[int] = []
        for trace in model.traces:
            valid = trace.n[trace.n > 0]
            if valid.size:
                counts.append(int(valid[-1]))
        if self.averaging_enabled.isChecked() and counts:
            low, high = min(counts), max(counts)
            self.averaging_n.setText(str(low) if low == high else f"{low}…{high}")
        else:
            self.averaging_n.setText(texts.UNKNOWN)

        self.range_controls.apply_time_axis(model.t_s)
        missing = [
            trace.slot
            for trace in model.traces
            if model.mode == "delta" and trace.lambda0_nm is None
        ]
        hint = texts.GRAPH_BASELINE_HINT
        if missing:
            labels = ", ".join(texts.slot_label(slot.channel, slot.position) for slot in missing)
            hint += f" Нет λ₀ — линия Δλ скрыта: {labels}."
        self.graph_hint.setText(hint)
        has_data = (
            bool(selected)
            and model.t_s.size > 0
            and any(
                trace.valid_points > 0
                and (model.mode == "absolute" or trace.lambda0_nm is not None)
                for trace in model.traces
            )
        )
        self.plot.setVisible(has_data)
        self.empty_graph_label.setVisible(not has_data)
        self.start_graph_button.setEnabled(not model.running)
        self.stop_graph_button.setEnabled(model.running)

    def _update_table(self, snapshot: AppSnapshot) -> None:
        model = models.measurement_table_model(snapshot)
        self.table_model.replace(model)
        temperatures = []
        for channel, value in enumerate(model.case_temp_c):
            text = texts.UNKNOWN if not math.isfinite(float(value)) else f"{float(value):.2f} °C"
            temperatures.append(f"{texts.channel_label(channel)}: {text}")
        self.temperature_label.setText(" · ".join(temperatures))

    # --- Запись ----------------------------------------------------------------------

    def _on_record_setting_changed(self, _value: object = None) -> None:
        if not self._record_settings_loading:
            self._record_settings_dirty = True

    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            texts.LABEL_RECORD_DIRECTORY,
            self.record_directory.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.record_directory.setText(selected)
            self._record_settings_dirty = True

    def _load_record_settings(self, model: models.RecordingPanelModel) -> None:
        self._record_settings_loading = True
        try:
            self.record_directory.setText(str(model.directory))
            self.record_decimation.setValue(model.decimation)
            self.record_limit.setMaximum(self._controller.config.profile.fbg_per_channel)
            self.record_limit.setValue(0 if model.fbg_limit is None else model.fbg_limit)
        finally:
            self._record_settings_loading = False

    def _start_recording(self) -> None:
        snapshot = self._controller.snapshot()
        if snapshot.state is not SessionState.STREAMING:
            self._controller.note(texts.RECORD_START_REQUIRES_STREAM)
            return
        directory_text = self.record_directory.text().strip()
        if not directory_text:
            self._controller.note("Папка записи не задана")
            return
        try:
            limit_value = self.record_limit.value()
            self._controller.configure_recording(
                directory=Path(directory_text),
                decimation=self.record_decimation.value(),
                fbg_limit=None if limit_value == 0 else limit_value,
            )
            self._controller.start_recording()
            self._record_settings_dirty = False
        except (OSError, RuntimeError, ValueError) as exc:
            self._controller.note(f"запись не запущена: {type(exc).__name__}: {exc}")
        self.refresh(self._controller.snapshot())

    def _stop_recording(self) -> None:
        self._controller.stop_recording()
        self.refresh(self._controller.snapshot())

    def _start_average_export(self) -> None:
        thread = self._average_thread
        if thread is not None and thread.is_alive():
            return
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            texts.BUTTON_AVERAGE_RECORDING,
            str(Path.cwd()),
            "CSV (*.csv)",
        )
        if not filename:
            return
        window_s = self.averaging_window.value() / 1000.0
        self._average_result = None
        self._average_error = None
        self._average_reported = False
        self.average_recording_state.setText("Усреднение выполняется…")

        def worker() -> None:
            try:
                self._average_result = average_recording(Path(filename), window_s)
            except Exception as exc:  # результат возвращается в UI, поток не теряется молча
                self._average_error = f"{type(exc).__name__}: {exc}"

        self._average_thread = threading.Thread(
            target=worker,
            name="fbg-average-export",
            daemon=False,
        )
        self._average_thread.start()

    def _poll_average_export(self, *, recording: bool) -> None:
        thread = self._average_thread
        running = thread is not None and thread.is_alive()
        self.average_recording_button.setEnabled(not running and not recording)
        if thread is None or running or self._average_reported:
            return
        self._average_reported = True
        if self._average_error is not None:
            self.average_recording_state.setText(f"Ошибка: {self._average_error}")
            return
        result = self._average_result
        if result is None:
            self.average_recording_state.setText("Усреднение завершено без результата")
            return
        self.average_recording_state.setText(
            f"Готово: окон {result.windows}, # GAP {result.gaps}; {result.output.name}"
        )

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} ГБ"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f} МБ"
        if value >= 1_000:
            return f"{value / 1_000:.1f} КБ"
        return f"{value} Б"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _pending_record_config(self, snapshot: AppSnapshot):
        """Настройки из полей панели для прогноза до фактического старта."""
        base = snapshot.recorder_config
        if base is None:
            return None
        limit = self.record_limit.value()
        directory_text = self.record_directory.text().strip()
        return replace(
            base,
            directory=base.directory if not directory_text else Path(directory_text),
            decimation=self.record_decimation.value(),
            fbg_limit=None if limit == 0 else limit,
        )

    def _update_recording(self, snapshot: AppSnapshot) -> None:
        model = models.recording_panel_model(snapshot)
        if not self._record_settings_dirty and not model.active:
            self._load_record_settings(model)

        estimate = model.estimated_bytes_10m
        estimate_max = model.estimated_max_bytes_10m
        if self._record_settings_dirty and not model.active:
            pending = self._pending_record_config(snapshot)
            if pending is not None:
                estimate = models.estimate_recording_bytes(snapshot, pending)
                estimate_max = models.estimate_recording_bytes(snapshot, pending, all_valid=True)

        self.record_state.setText(texts.RECORD_ACTIVE if model.active else texts.RECORD_IDLE)
        self.record_file.setText(texts.UNKNOWN if model.path is None else str(model.path))
        self.record_rows.setText(f"{model.rows:,}".replace(",", " "))
        self.record_size.setText(self._format_bytes(model.bytes_written))
        self.record_elapsed.setText(self._format_elapsed(model.elapsed_s))
        self.record_estimate.setText(
            f"≈ {self._format_bytes(estimate)} / "
            f"{self._format_bytes(estimate_max)} "
            f"({texts.RECORD_ESTIMATE_SUFFIX})"
        )
        if model.has_gaps:
            self.record_gaps.setText(
                f"{texts.RECORD_GAP_WARNING} Маркеров: {model.gaps}, "
                f"потеряно кадров: {model.lost_frames}, ожидают маркера: {model.pending_gap}."
            )
        else:
            self.record_gaps.setText(texts.RECORD_NO_GAPS)
        self.record_error.setText("" if model.error is None else f"Ошибка записи: {model.error}")

        editable = not model.active
        self.record_directory.setEnabled(editable)
        self.browse_button.setEnabled(editable)
        self.record_decimation.setEnabled(editable)
        self.record_limit.setEnabled(editable)
        self.start_record_button.setEnabled(
            not model.active and snapshot.state is SessionState.STREAMING
        )
        self.stop_record_button.setEnabled(model.active)
        self._poll_average_export(recording=model.active)

    def _update_quality(self, snapshot: AppSnapshot) -> None:
        metrics = snapshot.metrics
        if (
            snapshot.state is not SessionState.STREAMING
            or metrics is None
            or metrics.frame_rate_hz <= 0.0
        ):
            self.quality_label.setText("Темп: — · оценка потерь: —")
            return
        loss = (
            "—"
            if metrics.loss_estimate is None
            else f"{metrics.loss_estimate * 100.0:.3f} % (оценка)"
        )
        text = f"Темп: {metrics.frame_rate_hz:.2f} Гц · потери: {loss}"
        self.quality_label.setText(text)

    # --- Общий такт ------------------------------------------------------------------

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Один UI-такт; ручной диапазон графика здесь никогда не перезаписывается."""
        self._update_averaging_controls(snapshot)
        self._update_history_estimate(snapshot)
        self._update_lambda0_table(snapshot)
        self._update_graph(snapshot)
        self._update_table(snapshot)
        self._update_recording(snapshot)
        self._update_quality(snapshot)

    def wait_for_background_work(self) -> None:
        thread = self._average_thread
        if thread is not None and thread.is_alive():
            thread.join()

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt
        self.wait_for_background_work()
        super().closeEvent(event)  # type: ignore[arg-type]
