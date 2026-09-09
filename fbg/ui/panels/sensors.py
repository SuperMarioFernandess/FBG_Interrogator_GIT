"""Панель датчиков: калибровка, карта пиков и постобработка записи.

Датчик адресуется длиной волны и окном поиска (Р30), никогда номером слота.
При обычном показе калибровка приходит готовой в `AppSnapshot` на частоте UI
(Р75). При включённом усреднении Qt-свободная модель использует raw-снимок
выбранных каналов; виджет сам кольцо pipeline не читает.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fbg.core.calibration import (
    CalibrationPoint,
    FitKind,
    Sensor,
    SensorType,
    fit_calibration,
)
from fbg.io.recalibrate import RecalibrationResult, recalibrate_recording
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.docking import MAX_UI_PERIOD_MS, DockTab, scrollable
from fbg.ui.models import AppSnapshot, SensorPanelModel

_SENSOR_ID_ROLE = Qt.ItemDataRole.UserRole
#: Запас к окну усреднения гарантирует, что при максимальном периоде UI
#: предыдущий незавершённый bin ещё целиком есть в raw-кольце. Завершённые
#: окна дальше сохраняются инкрементальной моделью и повторно не копируются.
SENSOR_RAW_UI_MARGIN_S = MAX_UI_PERIOD_MS / 1000.0 + 0.1


class SensorsPanel(DockTab):
    """Редактор датчиков и оперативные графики физических величин."""

    layout_key = "sensors"

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._items: dict[str, QTreeWidgetItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._bands: dict[str, tuple[pg.PlotDataItem, pg.PlotDataItem, pg.FillBetweenItem]] = {}
        self._graph_model: models.SensorGraphModel | None = None
        self._map_regions: list[pg.LinearRegionItem] = []
        self._shown_sensor_version = -1
        self._shown_filter = ""
        self._shown_map_channel = -1
        self._editing_id: str | None = None
        self._editing_compensation = None
        self._last_model: SensorPanelModel | None = None
        self._recalc_thread: threading.Thread | None = None
        self._recalc_result: RecalibrationResult | None = None
        self._recalc_error: str | None = None
        self._recalc_reported = True

        profile = controller.config.profile

        self.filter_edit = QLineEdit()
        self.unit_combo = QComboBox()
        self.sensor_tree = QTreeWidget()
        self.sensor_tree.setColumnCount(7)
        self.sensor_tree.setHeaderLabels(
            ["Имя", "Канал", "Ожидаемая λ", "Текущая λ", "Значение", "Ед.", "Статус"]
        )
        self.sensor_tree.setAlternatingRowColors(True)

        self.new_button = QPushButton(texts.BUTTON_SENSOR_NEW)
        self.save_button = QPushButton(texts.BUTTON_SENSOR_SAVE)
        self.delete_button = QPushButton(texts.BUTTON_SENSOR_DELETE)

        self.id_edit = QLineEdit()
        self.name_edit = QLineEdit()
        self.channel_combo = QComboBox()
        for channel in range(profile.channels):
            self.channel_combo.addItem(texts.channel_label(channel), str(channel))
        self.channel_combo.setMaximumWidth(180)
        self.type_combo = QComboBox()
        for sensor_type in SensorType:
            self.type_combo.addItem(sensor_type.name, str(int(sensor_type)))
        self.type_combo.setMaximumWidth(180)

        self.expected_spin = self._nm_spin()
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setDecimals(4)
        self.window_spin.setRange(0.0001, 100.0)
        self.window_spin.setValue(0.35)
        self.window_spin.setMaximumWidth(130)
        self.value0_spin = self._value_spin()
        self.k1_spin = self._coefficient_spin()
        self.k2_spin = self._coefficient_spin()

        self.down_limit_enabled = QCheckBox()
        self.down_limit_spin = self._value_spin()
        self.up_limit_enabled = QCheckBox()
        self.up_limit_spin = self._value_spin()

        self.current_peak_combo = QComboBox()
        self.take_wavelength_button = QPushButton(texts.BUTTON_SENSOR_TAKE_WAVELENGTH)

        self.points_table = QTableWidget(0, 2)
        self.points_table.setHorizontalHeaderLabels(["λ, нм", "Известное значение"])
        self.known_value_spin = self._value_spin()
        self.add_point_button = QPushButton(texts.BUTTON_SENSOR_ADD_POINT)
        self.remove_point_button = QPushButton(texts.BUTTON_SENSOR_REMOVE_POINT)
        self.fit_kind = QComboBox()
        self.fit_kind.addItem(texts.SENSOR_FIT_LINEAR, FitKind.LINEAR.value)
        self.fit_kind.addItem(texts.SENSOR_FIT_QUADRATIC, FitKind.QUADRATIC.value)
        self.fit_button = QPushButton(texts.BUTTON_SENSOR_FIT)
        self.fit_residual = QLabel(texts.UNKNOWN)

        self.value_plot = pg.PlotWidget()
        self.value_plot.setMinimumHeight(220)
        self.value_plot.hide()
        self.value_plot.setLabel("bottom", "Время, с")
        self.value_plot.showGrid(x=True, y=True, alpha=0.2)
        self.value_plot.addLegend()
        self.value_plot.setDownsampling(auto=True, mode="peak")
        self.value_plot.setClipToView(True)
        self.empty_graph_label = QLabel(texts.SENSOR_EMPTY_GRAPH_HINT)
        self.empty_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        self.map_channel = QComboBox()
        for channel in range(profile.channels):
            self.map_channel.addItem(texts.channel_label(channel), str(channel))
        self.peak_map = pg.PlotWidget()
        self.peak_map.setLabel("bottom", "Длина волны, нм")
        self.peak_map.hideAxis("left")
        self.peak_map.setYRange(0.0, 1.0, padding=0.0)
        self.peak_scatter = pg.ScatterPlotItem(size=9)
        self.peak_map.addItem(self.peak_scatter)
        self.peak_map_hint = QLabel(texts.SENSOR_PEAK_MAP_HINT)
        self.peak_map_hint.setWordWrap(True)

        self.recalibrate_button = QPushButton(texts.BUTTON_RECALIBRATE)
        self.recalibrate_state = QLabel(texts.SENSOR_RECALIBRATION_HINT)
        self.recalibrate_state.setWordWrap(True)

        self._build_layout()
        self._connect_signals()
        self._new_sensor()
        self.refresh(controller.snapshot(include_trace_history=False, include_sensor_data=True))

    @staticmethod
    def _nm_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(1000.0, 2000.0)
        spin.setValue(1550.0)
        spin.setMaximumWidth(130)
        return spin

    @staticmethod
    def _value_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        # Значение/пределы могут быть заметно крупнее коэффициентов, но миллиард
        # единиц всё ещё оставляет широкий запас без поля шириной под 1e12.
        spin.setRange(-1.0e9, 1.0e9)
        spin.setMaximumWidth(150)
        return spin

    @staticmethod
    def _coefficient_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        # Для тензодатчика из физической оценки проекта чувствительность порядка
        # 8.3e5 µε/нм; ±1e8 оставляет более чем стократный запас.
        spin.setRange(-1.0e8, 1.0e8)
        spin.setMaximumWidth(150)
        return spin

    def _build_layout(self) -> None:
        list_form = QFormLayout()
        list_form.addRow(texts.LABEL_SENSOR_FILTER, self.filter_edit)
        list_form.addRow(texts.LABEL_SENSOR_UNIT, self.unit_combo)
        list_buttons = QHBoxLayout()
        list_buttons.addWidget(self.new_button)
        list_buttons.addWidget(self.delete_button)
        list_buttons.addWidget(self.recalibrate_button)
        list_buttons.addStretch(1)
        list_layout = QVBoxLayout()
        list_layout.addLayout(list_form)
        list_layout.addWidget(self.sensor_tree, 1)
        list_layout.addLayout(list_buttons)
        list_layout.addWidget(self.recalibrate_state)
        list_box = QWidget()
        list_box.setLayout(list_layout)

        editor_form = QFormLayout()
        editor_form.addRow(texts.LABEL_SENSOR_ID, self.id_edit)
        editor_form.addRow(texts.LABEL_SENSOR_NAME, self.name_edit)
        editor_form.addRow(texts.LABEL_SENSOR_CHANNEL, self.channel_combo)
        editor_form.addRow(texts.LABEL_SENSOR_TYPE, self.type_combo)
        editor_form.addRow(texts.LABEL_SENSOR_EXPECTED, self.expected_spin)
        editor_form.addRow(texts.LABEL_SENSOR_WINDOW, self.window_spin)
        editor_form.addRow(texts.LABEL_SENSOR_VALUE0, self.value0_spin)
        editor_form.addRow(texts.LABEL_SENSOR_K1, self.k1_spin)
        editor_form.addRow(texts.LABEL_SENSOR_K2, self.k2_spin)

        down = QHBoxLayout()
        down.addWidget(self.down_limit_enabled)
        down.addWidget(self.down_limit_spin)
        editor_form.addRow(texts.LABEL_SENSOR_DOWN_LIMIT, down)
        up = QHBoxLayout()
        up.addWidget(self.up_limit_enabled)
        up.addWidget(self.up_limit_spin)
        editor_form.addRow(texts.LABEL_SENSOR_UP_LIMIT, up)

        peak_row = QHBoxLayout()
        peak_row.addWidget(self.current_peak_combo, 1)
        peak_row.addWidget(self.take_wavelength_button)
        editor_form.addRow(texts.LABEL_SENSOR_CURRENT_PEAK, peak_row)
        editor_form.addRow(self.save_button)
        editor_box = QGroupBox(texts.GROUP_SENSOR_EDITOR)
        editor_box.setLayout(editor_form)

        point_controls = QHBoxLayout()
        point_controls.addWidget(QLabel(texts.LABEL_SENSOR_KNOWN_VALUE))
        point_controls.addWidget(self.known_value_spin)
        point_controls.addWidget(self.add_point_button)
        point_controls.addWidget(self.remove_point_button)
        fit_controls = QFormLayout()
        fit_controls.addRow(texts.LABEL_SENSOR_FIT_KIND, self.fit_kind)
        fit_controls.addRow(texts.LABEL_SENSOR_FIT_RESIDUAL, self.fit_residual)
        fit_controls.addRow(self.fit_button)
        cal_layout = QVBoxLayout()
        cal_layout.addWidget(self.points_table)
        cal_layout.addLayout(point_controls)
        cal_layout.addLayout(fit_controls)
        cal_box = QGroupBox(texts.GROUP_SENSOR_CALIBRATION)
        cal_box.setLayout(cal_layout)

        editor_column = QWidget()
        editor_layout = QVBoxLayout(editor_column)
        editor_layout.addWidget(editor_box)
        editor_layout.addWidget(cal_box, 1)

        graph_controls = QFormLayout()
        graph_controls.addRow(texts.LABEL_AVERAGING_ENABLED, self.averaging_enabled)
        graph_controls.addRow(texts.LABEL_AVERAGING_WINDOW, self.averaging_window)
        graph_controls.addRow(texts.LABEL_AVERAGING_FRAMES, self.averaging_frames)
        graph_controls.addRow(texts.LABEL_AVERAGING_SIGMA, self.averaging_sigma)
        graph_controls.addRow(texts.LABEL_AVERAGING_N, self.averaging_n)
        graph_layout = QVBoxLayout()
        graph_layout.addLayout(graph_controls)
        graph_layout.addWidget(self.empty_graph_label, 1)
        graph_layout.addWidget(self.value_plot)
        graph_box = QWidget()
        graph_box.setLayout(graph_layout)

        map_form = QFormLayout()
        map_form.addRow(texts.LABEL_PEAK_MAP_CHANNEL, self.map_channel)
        map_layout = QVBoxLayout()
        map_layout.addLayout(map_form)
        map_layout.addWidget(self.peak_map)
        map_layout.addWidget(self.peak_map_hint)
        map_box = QWidget()
        map_box.setLayout(map_layout)

        self.list_dock = self.add_panel_dock(
            texts.GROUP_SENSOR_LIST,
            list_box,
            "sensors.list",
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.editor_dock = self.add_panel_dock(
            texts.GROUP_SENSOR_EDITOR,
            scrollable(editor_column),
            "sensors.editor",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.graph_dock = self.add_panel_dock(
            texts.GROUP_SENSOR_GRAPH,
            graph_box,
            "sensors.graph",
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.map_dock = self.add_panel_dock(
            texts.GROUP_PEAK_MAP,
            map_box,
            "sensors.map",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.reset_layout()

    def _apply_default_splits(self) -> None:
        self.splitDockWidget(
            self.list_dock,
            self.graph_dock,
            Qt.Orientation.Vertical,
        )
        self.splitDockWidget(
            self.editor_dock,
            self.map_dock,
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.list_dock, self.editor_dock],
            [500, 550],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.list_dock, self.graph_dock],
            [400, 280],
            Qt.Orientation.Vertical,
        )

    def _connect_signals(self) -> None:
        self.filter_edit.textChanged.connect(lambda _text: self._refresh_from_controller())
        self.unit_combo.currentIndexChanged.connect(lambda _index: self._unit_changed())
        self.sensor_tree.itemSelectionChanged.connect(self._selection_changed)
        self.sensor_tree.itemChanged.connect(
            lambda _item, _column: self._on_graph_selection_changed()
        )
        self.averaging_enabled.toggled.connect(self._on_averaging_changed)
        self.averaging_window.valueChanged.connect(self._on_averaging_changed)
        self.averaging_sigma.toggled.connect(lambda _checked: self._refresh_from_controller())
        self.new_button.clicked.connect(self._new_sensor)
        self.save_button.clicked.connect(self._save_sensor)
        self.delete_button.clicked.connect(self._delete_sensor)
        self.take_wavelength_button.clicked.connect(self._take_current_wavelength)
        self.channel_combo.currentIndexChanged.connect(lambda _index: self._update_peak_combo())
        self.add_point_button.clicked.connect(self._add_point)
        self.remove_point_button.clicked.connect(self._remove_point)
        self.fit_button.clicked.connect(self._fit_points)
        self.map_channel.currentIndexChanged.connect(
            lambda _index: self._update_peak_map(force=True)
        )
        self.recalibrate_button.clicked.connect(self._start_recalibration)

    def _refresh_from_controller(self) -> None:
        self.refresh(
            self._controller.snapshot(include_trace_history=False, include_sensor_data=True)
        )

    def _sync_sensor_trace_request(self) -> None:
        selected = self._checked_sensor_ids() if self.averaging_enabled.isChecked() else ()
        window_s = self.averaging_window.value() / 1000.0
        self._controller.set_sensor_trace_request(
            selected,
            window_s + SENSOR_RAW_UI_MARGIN_S,
        )

    def _on_graph_selection_changed(self) -> None:
        self._graph_model = None
        self._sync_sensor_trace_request()
        self._refresh_from_controller()

    def _on_averaging_changed(self, _value: object = None) -> None:
        self._graph_model = None
        self._sync_sensor_trace_request()
        self._refresh_from_controller()

    def _averaging_window_s(self) -> float | None:
        if not self.averaging_enabled.isChecked():
            return None
        return self.averaging_window.value() / 1000.0

    def _update_averaging_controls(self, snapshot: AppSnapshot) -> None:
        enabled = self.averaging_enabled.isChecked()
        self.averaging_window.setEnabled(True)
        self.averaging_sigma.setEnabled(enabled)
        frames = models.expected_averaging_frames(snapshot, self.averaging_window.value())
        self.averaging_frames.setText("—" if frames <= 0 else f"≈ {frames} кадров")

    @staticmethod
    def _finite(value: float, digits: int = 4) -> str:
        return texts.UNKNOWN if not math.isfinite(value) else f"{value:.{digits}f}"

    def _rebuild_tree(self, model: SensorPanelModel) -> None:
        checked = {
            sensor_id
            for sensor_id, item in self._items.items()
            if item.checkState(0) == Qt.CheckState.Checked
        }
        selected = self._editing_id
        scroll = self.sensor_tree.verticalScrollBar().value()
        self.sensor_tree.blockSignals(True)
        try:
            self.sensor_tree.clear()
            self._items.clear()
            groups: dict[int, QTreeWidgetItem] = {}
            for row in model.rows:
                channel = row.sensor.channel
                group = groups.get(channel)
                if group is None:
                    group = QTreeWidgetItem([texts.channel_label(channel)])
                    self.sensor_tree.addTopLevelItem(group)
                    groups[channel] = group
                item = QTreeWidgetItem()
                item.setData(0, _SENSOR_ID_ROLE, row.sensor.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if row.sensor.id in checked else Qt.CheckState.Unchecked,
                )
                group.addChild(item)
                self._items[row.sensor.id] = item
                if row.sensor.id == selected:
                    item.setSelected(True)
            for group in groups.values():
                group.setExpanded(True)
            self.sensor_tree.verticalScrollBar().setValue(scroll)
        finally:
            self.sensor_tree.blockSignals(False)
        self._update_tree_values(model)

    def _update_tree_values(self, model: SensorPanelModel) -> None:
        rows = {row.sensor.id: row for row in model.rows}
        was_blocked = self.sensor_tree.blockSignals(True)
        try:
            for sensor_id, item in self._items.items():
                row = rows.get(sensor_id)
                if row is None:
                    continue
                value = self._finite(row.value, 5)
                if value != texts.UNKNOWN and row.unit:
                    value = f"{value}"
                item.setText(0, row.sensor.name)
                item.setText(1, str(row.sensor.channel + 1))
                item.setText(2, f"{row.sensor.expected_nm:.4f}")
                item.setText(3, self._finite(row.wavelength_nm))
                item.setText(4, value)
                item.setText(5, row.unit or texts.SENSOR_NO_UNIT)
                item.setText(6, texts.SENSOR_STATUS_LABELS[row.status.value])
        finally:
            self.sensor_tree.blockSignals(was_blocked)

    def _update_units(self, model: SensorPanelModel) -> None:
        current = self.unit_combo.currentData()
        current_text = "" if current is None else str(current)
        available = model.units
        existing = tuple(str(self.unit_combo.itemData(i)) for i in range(self.unit_combo.count()))
        if existing == available:
            return
        self.unit_combo.blockSignals(True)
        try:
            self.unit_combo.clear()
            for unit in available:
                self.unit_combo.addItem(unit or texts.SENSOR_NO_UNIT, unit)
            index = self.unit_combo.findData(current_text)
            self.unit_combo.setCurrentIndex(index if index >= 0 else (0 if available else -1))
        finally:
            self.unit_combo.blockSignals(False)
        self._unit_changed()

    def _selected_unit(self) -> str:
        data = self.unit_combo.currentData()
        return "" if data is None else str(data)

    def _unit_changed(self) -> None:
        unit = self._selected_unit()
        was_blocked = self.sensor_tree.blockSignals(True)
        try:
            for sensor in self._controller.sensors:
                item = self._items.get(sensor.id)
                if item is None:
                    continue
                flags = item.flags()
                if sensor.unit == unit:
                    item.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
                else:
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                    item.setFlags(flags & ~Qt.ItemFlag.ItemIsUserCheckable)
        finally:
            self.sensor_tree.blockSignals(was_blocked)
        self.value_plot.setLabel("left", unit or texts.SENSOR_NO_UNIT)
        self._on_graph_selection_changed()

    def _selection_changed(self) -> None:
        items = self.sensor_tree.selectedItems()
        if not items:
            return
        sensor_id = str(items[0].data(0, _SENSOR_ID_ROLE) or "")
        sensor = next(
            (sensor for sensor in self._controller.sensors if sensor.id == sensor_id), None
        )
        if sensor is not None:
            self._load_sensor(sensor)

    def _load_sensor(self, sensor: Sensor) -> None:
        self._editing_id = sensor.id
        self._editing_compensation = sensor.compensation
        self.id_edit.setText(sensor.id)
        self.name_edit.setText(sensor.name)
        self.channel_combo.setCurrentIndex(sensor.channel)
        self.type_combo.setCurrentIndex(int(sensor.type))
        self.expected_spin.setValue(sensor.expected_nm)
        self.window_spin.setValue(sensor.window_nm)
        self.value0_spin.setValue(sensor.value0)
        self.k1_spin.setValue(sensor.k1)
        self.k2_spin.setValue(sensor.k2)
        self.down_limit_enabled.setChecked(sensor.down_limit is not None)
        self.down_limit_spin.setValue(0.0 if sensor.down_limit is None else sensor.down_limit)
        self.up_limit_enabled.setChecked(sensor.up_limit is not None)
        self.up_limit_spin.setValue(0.0 if sensor.up_limit is None else sensor.up_limit)
        self._set_points(sensor.calibration_points)
        self.fit_residual.setText(texts.UNKNOWN)
        self._update_peak_combo()

    def _new_sensor(self) -> None:
        self._editing_id = None
        self._editing_compensation = None
        self.id_edit.clear()
        self.name_edit.clear()
        self.channel_combo.setCurrentIndex(0)
        self.type_combo.setCurrentIndex(int(SensorType.TEMPERATURE))
        self.expected_spin.setValue(1550.0)
        self.window_spin.setValue(0.35)
        self.value0_spin.setValue(0.0)
        self.k1_spin.setValue(0.0)
        self.k2_spin.setValue(0.0)
        self.down_limit_enabled.setChecked(False)
        self.up_limit_enabled.setChecked(False)
        self._set_points(())
        self.fit_kind.setCurrentIndex(0)
        self.fit_residual.setText(texts.UNKNOWN)
        self._update_peak_combo()

    def _sensor_from_editor(self) -> Sensor:
        raw_id = self.id_edit.text().strip()
        if not raw_id:
            raise ValueError("ID датчика не задан")
        name = self.name_edit.text().strip() or raw_id
        return Sensor(
            id=raw_id,
            name=name,
            channel=int(str(self.channel_combo.currentData())),
            type=SensorType(int(str(self.type_combo.currentData()))),
            expected_nm=self.expected_spin.value(),
            window_nm=self.window_spin.value(),
            value0=self.value0_spin.value(),
            k1=self.k1_spin.value(),
            k2=self.k2_spin.value(),
            calibration_points=self._points(),
            down_limit=(
                self.down_limit_spin.value() if self.down_limit_enabled.isChecked() else None
            ),
            up_limit=self.up_limit_spin.value() if self.up_limit_enabled.isChecked() else None,
            compensation=self._editing_compensation,
        )

    def _save_sensor(self) -> None:
        try:
            sensor = self._sensor_from_editor()
            self._controller.upsert_sensor(sensor, previous_id=self._editing_id)
        except (OSError, ValueError) as exc:
            self._controller.note(f"датчик не сохранён: {exc}")
            return
        self._editing_id = sensor.id
        self._refresh_from_controller()

    def _delete_sensor(self) -> None:
        if self._editing_id is None:
            return
        try:
            self._controller.delete_sensor(self._editing_id)
        except (OSError, ValueError) as exc:
            self._controller.note(f"датчик не удалён: {exc}")
            return
        self._new_sensor()
        self._refresh_from_controller()

    def _current_peaks(self) -> tuple[float, ...]:
        model = self._last_model
        if model is None:
            return ()
        channel = int(str(self.channel_combo.currentData()))
        if not 0 <= channel < len(model.peaks_by_channel):
            return ()
        return tuple(float(value) for value in model.peaks_by_channel[channel])

    def _update_peak_combo(self) -> None:
        current = self.current_peak_combo.currentData()
        self.current_peak_combo.clear()
        for wavelength_nm in self._current_peaks():
            token = f"{wavelength_nm:.9f}"
            self.current_peak_combo.addItem(f"{wavelength_nm:.4f} нм", token)
        if current is not None:
            index = self.current_peak_combo.findData(str(current))
            if index >= 0:
                self.current_peak_combo.setCurrentIndex(index)
        available = self.current_peak_combo.count() > 0
        self.take_wavelength_button.setEnabled(available)
        self.add_point_button.setEnabled(available)

    def _selected_peak(self) -> float:
        data = self.current_peak_combo.currentData()
        if data is None:
            raise ValueError("в текущем кадре выбранного канала нет пиков")
        return float(str(data))

    def _take_current_wavelength(self) -> None:
        try:
            self.expected_spin.setValue(self._selected_peak())
        except ValueError as exc:
            self._controller.note(str(exc))

    def _set_points(self, points: tuple[CalibrationPoint, ...]) -> None:
        self.points_table.setRowCount(len(points))
        for row, point in enumerate(points):
            for column, value in enumerate((point.wavelength_nm, point.value)):
                item = self.points_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.points_table.setItem(row, column, item)
                item.setText(f"{value:.9g}")

    def _points(self) -> tuple[CalibrationPoint, ...]:
        points: list[CalibrationPoint] = []
        for row in range(self.points_table.rowCount()):
            wavelength_item = self.points_table.item(row, 0)
            value_item = self.points_table.item(row, 1)
            if wavelength_item is None or value_item is None:
                continue
            points.append(CalibrationPoint(float(wavelength_item.text()), float(value_item.text())))
        return tuple(points)

    def _add_point(self) -> None:
        try:
            point = CalibrationPoint(self._selected_peak(), self.known_value_spin.value())
            self._set_points((*self._points(), point))
        except ValueError as exc:
            self._controller.note(f"опорная точка не добавлена: {exc}")

    def _remove_point(self) -> None:
        row = self.points_table.currentRow()
        if row >= 0:
            self.points_table.removeRow(row)

    def _fit_points(self) -> None:
        try:
            kind = FitKind(str(self.fit_kind.currentData()))
            fit = fit_calibration(self._points(), self.expected_spin.value(), kind=kind)
        except (ValueError, FloatingPointError) as exc:
            self.fit_residual.setText(str(exc))
            return
        self.value0_spin.setValue(fit.value0)
        self.k1_spin.setValue(fit.k1)
        self.k2_spin.setValue(fit.k2)
        self.fit_residual.setText(
            f"RMS {fit.rms:.6g}; max |r| {fit.max_abs_residual:.6g}; точек {fit.points}"
        )

    def _checked_sensor_ids(self) -> tuple[str, ...]:
        return tuple(
            sensor_id
            for sensor_id, item in self._items.items()
            if item.checkState(0) == Qt.CheckState.Checked
        )

    def _update_graph(self, snapshot: AppSnapshot) -> None:
        selected = self._checked_sensor_ids()
        graph = models.sensor_graph_model(
            snapshot,
            selected,
            self._graph_model,
            averaging_window_s=self._averaging_window_s(),
        )
        unchanged = graph is self._graph_model
        self._graph_model = graph
        selected_set = set(selected)
        for sensor_id in tuple(self._curves):
            if sensor_id not in selected_set:
                self.value_plot.removeItem(self._curves.pop(sensor_id))
                band = self._bands.pop(sensor_id, None)
                if band is not None:
                    for item in band:
                        self.value_plot.removeItem(item)
        if graph.t_s.size == 0:
            self.value_plot.hide()
            self.empty_graph_label.show()
            self.averaging_n.setText("—")
            return
        trace_by_id = {trace.sensor_id: trace for trace in graph.traces}
        latest_counts: list[int] = []
        for index, sensor_id in enumerate(selected):
            trace = trace_by_id.get(sensor_id)
            if trace is None:
                continue
            curve = self._curves.get(sensor_id)
            if curve is None:
                color = pg.intColor(index, hues=max(1, len(selected)))
                curve = self.value_plot.plot(pen=pg.mkPen(color), name=sensor_id)
                self._curves[sensor_id] = curve
                upper = pg.PlotDataItem(pen=None)
                lower = pg.PlotDataItem(pen=None)
                red, green, blue, _alpha = color.getRgb()
                fill = pg.FillBetweenItem(
                    upper,
                    lower,
                    brush=pg.mkBrush(red, green, blue, 45),
                )
                fill.setZValue(-10)
                self.value_plot.addItem(upper)
                self.value_plot.addItem(lower)
                self.value_plot.addItem(fill)
                self._bands[sensor_id] = (upper, lower, fill)
            if not unchanged:
                curve.setData(graph.t_s, trace.values, connect="finite")
                if trace.sigma is not None:
                    band = self._bands[sensor_id]
                    band[0].setData(graph.t_s, trace.values + trace.sigma, connect="finite")
                    band[1].setData(graph.t_s, trace.values - trace.sigma, connect="finite")
            band = self._bands[sensor_id]
            band_visible = (
                self.averaging_enabled.isChecked()
                and self.averaging_sigma.isChecked()
                and trace.sigma is not None
            )
            for item in band:
                item.setVisible(band_visible)
            if trace.n is not None and trace.n.size:
                latest_counts.append(int(trace.n[-1]))
        has_data = bool(selected) and any(
            np.any(np.isfinite(trace.values)) for trace in graph.traces
        )
        if latest_counts:
            low, high = min(latest_counts), max(latest_counts)
            self.averaging_n.setText(str(low) if low == high else f"{low}…{high}")
        else:
            self.averaging_n.setText("—")
        self.value_plot.setVisible(has_data)
        self.empty_graph_label.setVisible(not has_data)

    def _update_peak_map(self, *, force: bool = False) -> None:
        model = self._last_model
        if model is None:
            return
        channel = int(str(self.map_channel.currentData()))
        peaks = model.peaks_by_channel[channel] if channel < len(model.peaks_by_channel) else ()
        self.peak_scatter.setData(x=peaks, y=[0.5] * len(peaks))
        if not force and channel == self._shown_map_channel:
            return
        for region in self._map_regions:
            self.peak_map.removeItem(region)
        self._map_regions.clear()
        for sensor in self._controller.sensors:
            if sensor.channel != channel:
                continue
            region = pg.LinearRegionItem((sensor.low_nm, sensor.high_nm), movable=False)
            self.peak_map.addItem(region)
            self._map_regions.append(region)
        self._shown_map_channel = channel

    def _start_recalibration(self) -> None:
        thread = self._recalc_thread
        if thread is not None and thread.is_alive():
            return
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            texts.BUTTON_RECALIBRATE,
            str(Path.cwd()),
            "CSV (*.csv)",
        )
        if not filename:
            return
        sensors = self._controller.sensors
        self._recalc_result = None
        self._recalc_error = None
        self._recalc_reported = False
        self.recalibrate_state.setText("Пересчёт выполняется…")

        def worker() -> None:
            try:
                self._recalc_result = recalibrate_recording(Path(filename), sensors)
            except Exception as exc:  # результат обязан вернуться в UI, а не убить поток
                self._recalc_error = f"{type(exc).__name__}: {exc}"

        self._recalc_thread = threading.Thread(
            target=worker,
            name="fbg-recalibrate",
            daemon=False,
        )
        self._recalc_thread.start()

    def _poll_recalibration(self) -> None:
        thread = self._recalc_thread
        if thread is None or thread.is_alive() or self._recalc_reported:
            self.recalibrate_button.setEnabled(thread is None or not thread.is_alive())
            return
        self._recalc_reported = True
        self.recalibrate_button.setEnabled(True)
        if self._recalc_error is not None:
            self.recalibrate_state.setText(f"Ошибка: {self._recalc_error}")
            return
        result = self._recalc_result
        if result is None:
            self.recalibrate_state.setText("Пересчёт завершён без результата")
            return
        self.recalibrate_state.setText(
            f"Готово: строк {result.rows}, # GAP {result.gaps}, файлов {len(result.outputs)}"
        )

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Показывает уже рассчитанный на частоте UI снимок датчиков."""
        model = models.sensor_panel_model(snapshot, filter_text=self.filter_edit.text())
        self._last_model = model
        filter_text = self.filter_edit.text()
        sensor_changed = snapshot.sensor_version != self._shown_sensor_version
        if sensor_changed or filter_text != self._shown_filter:
            self._rebuild_tree(model)
            self._update_units(model)
            self._shown_sensor_version = snapshot.sensor_version
            self._shown_filter = filter_text
            self._shown_map_channel = -1
            if sensor_changed:
                # Калибровка/канал могли измениться при том же ID. Старые
                # завершённые окна физических величин переиспользовать нельзя.
                self._graph_model = None
                self._sync_sensor_trace_request()
        else:
            self._update_tree_values(model)
        self._update_peak_combo()
        self._update_averaging_controls(snapshot)
        self._update_graph(snapshot)
        self._update_peak_map()
        self._poll_recalibration()

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt
        thread = self._recalc_thread
        if thread is not None and thread.is_alive():
            thread.join()
        super().closeEvent(event)  # type: ignore[arg-type]
