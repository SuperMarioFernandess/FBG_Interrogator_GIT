"""Панель измерения: λ(t), таблица слотов и управление записью.

UI получает данные только через `AppSnapshot`. Историю графика копирует сам
pipeline по запросу выбранных позиций; `RingHistory` сюда не попадает (Р36).
Один общий таймер главного окна (по умолчанию 10 Гц) обновляет панель — никаких сигналов
на каждый кадр и никакого файлового I/O в колбэках ядра.
"""

import math
import threading
from dataclasses import replace
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QCheckBox,
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
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fbg.core.session import SessionState
from fbg.io.averaging import AveragingExportResult, average_recording
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.docking import DockTab
from fbg.ui.models import AppSnapshot, MeasurementTableModel, SlotRef

#: Дефолт не означает «датчики 1–4». Это четыре первых **слота** канала 1,
#: которые прибор заполняет по мере обнаружения пиков (Р30).
DEFAULT_SELECTED_SLOTS = 4

#: Нижняя граница настройки истории. Меньше одного такта UI практического
#: смысла не имеет, но 50 мс совпадает с периодом публикации pipeline.
MIN_HISTORY_S = 0.05

#: Ось Y сжимается раз в секунду, а расширяется на новом выбросе сразу.
#: Это убирает дрожание масштаба без риска спрятать краткий выброс.
Y_RANGE_RECALC_TICKS = 10


class _MeasurementQtTableModel(QAbstractTableModel):
    """Qt-обёртка над неизменяемой моделью последнего кадра.

    При неизменной геометрии кадр подменяется одним ``dataChanged`` на весь
    диапазон значений. ``modelReset`` оставлен только для реальной смены
    числа каналов/позиций: иначе прокрутка и выделение слетали бы 10 раз/с.
    """

    def __init__(self, model: MeasurementTableModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model

    @property
    def model(self) -> MeasurementTableModel:
        return self._model

    def replace(self, model: MeasurementTableModel) -> None:
        """Подменяет кадр без сброса таблицы при неизменной геометрии.

        ``modelReset`` десять раз в секунду сбрасывает выделение и прокрутку.
        Геометрия 4×30 меняется только при принятии другого профиля; обычный
        кадр поэтому обновляется одним ``dataChanged``.
        """
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
        if parent.isValid():
            return 0
        return self._model.positions

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        if parent.isValid():
            return 0
        return 1 + self._model.channels

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = index.row()
        column = index.column()
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
        channel = section - 1
        return f"К{channel + 1} {texts.TABLE_WAVELENGTH}"


class MeasurementPanel(DockTab):
    """График выбранных слотов, таблица 4×30 и запись CSV."""

    layout_key = "measurement"

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._selection_loading = False
        self._record_settings_loading = False
        self._record_settings_dirty = False
        self._curves: dict[SlotRef, pg.PlotDataItem] = {}
        self._bands: dict[SlotRef, tuple[pg.PlotDataItem, pg.PlotDataItem, pg.FillBetweenItem]] = {}
        self._graph_model: models.MeasurementGraphModel | None = None
        self._graph_ticks = 0
        self._average_thread: threading.Thread | None = None
        self._average_result: AveragingExportResult | None = None
        self._average_error: str | None = None
        self._average_reported = False

        profile = controller.config.profile
        self.trace_tree = QTreeWidget()
        self.trace_tree.setHeaderHidden(True)
        self.trace_tree.setMinimumWidth(210)
        self.history_spin = QDoubleSpinBox()
        self.history_spin.setDecimals(2)
        self.history_spin.setSingleStep(0.5)
        self.history_spin.setRange(MIN_HISTORY_S, 86_400.0)
        self.history_spin.setValue(5.0)
        self.history_spin.setMaximumWidth(140)
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

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        # Длинная история графика измерения прореживается **только при
        # отрисовке**. `peak` сохраняет краткие выбросы; `mean` здесь нельзя —
        # он усреднил бы пропавший на один кадр пик. Исходные точки остаются
        # в копии истории и в файле без изменений (Р76).
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.plot.setLabel("bottom", texts.GRAPH_AXIS_TIME)
        self.plot.setLabel("left", texts.GRAPH_AXIS_DELTA_NM)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend()
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

        self._build_trace_tree(profile.channels, profile.fbg_per_channel)
        self._build_layout()
        self._connect_signals()
        self._sync_trace_request()
        self.refresh(controller.snapshot())

    # --- Компоновка -----------------------------------------------------------------

    def _build_trace_tree(self, channels: int, positions: int) -> None:
        """Строит дерево выбора. Данные Qt — строки, не кортежи (KB_05 №36)."""
        self._selection_loading = True
        try:
            self.trace_tree.clear()
            for channel in range(channels):
                channel_item = QTreeWidgetItem([texts.channel_label(channel)])
                self.trace_tree.addTopLevelItem(channel_item)
                for position in range(positions):
                    slot = SlotRef(channel, position)
                    item = QTreeWidgetItem([f"Позиция {position + 1}"])
                    item.setData(0, Qt.ItemDataRole.UserRole, models.slot_token(slot))
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    checked = channel == 0 and position < DEFAULT_SELECTED_SLOTS
                    item.setCheckState(
                        0,
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                    )
                    channel_item.addChild(item)
                channel_item.setExpanded(channel == 0)
        finally:
            self._selection_loading = False

    def _build_layout(self) -> None:
        selection_controls = QGridLayout()
        selection_controls.addWidget(QLabel(texts.LABEL_GRAPH_HISTORY), 0, 0)
        selection_controls.addWidget(self.history_spin, 0, 1)
        selection_controls.addWidget(QLabel(texts.LABEL_AVERAGING_ENABLED), 0, 2)
        selection_controls.addWidget(self.averaging_enabled, 0, 3)
        selection_controls.addWidget(QLabel(texts.LABEL_AVERAGING_WINDOW), 1, 0)
        selection_controls.addWidget(self.averaging_window, 1, 1)
        selection_controls.addWidget(QLabel(texts.LABEL_AVERAGING_FRAMES), 1, 2)
        selection_controls.addWidget(self.averaging_frames, 1, 3)
        selection_controls.addWidget(QLabel(texts.LABEL_AVERAGING_SIGMA), 2, 0)
        selection_controls.addWidget(self.averaging_sigma, 2, 1)
        selection_controls.addWidget(QLabel(texts.LABEL_AVERAGING_N), 2, 2)
        selection_controls.addWidget(self.averaging_n, 2, 3)
        selection_controls.setColumnStretch(3, 1)
        selection_controls.setVerticalSpacing(0)
        selection_layout = QVBoxLayout()
        selection_layout.addLayout(selection_controls)
        selection_layout.addWidget(self.trace_tree, 1)
        selection_box = QWidget()
        selection_box.setLayout(selection_layout)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.quality_label)
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
        self.splitDockWidget(
            self.selection_dock,
            self.table_dock,
            Qt.Orientation.Vertical,
        )
        self.splitDockWidget(
            self.graph_dock,
            self.record_dock,
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.selection_dock, self.graph_dock],
            [270, 780],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.graph_dock, self.record_dock],
            [430, 250],
            Qt.Orientation.Vertical,
        )

    def _connect_signals(self) -> None:
        self.trace_tree.itemChanged.connect(self._on_trace_changed)
        self.history_spin.valueChanged.connect(self._on_history_changed)
        self.averaging_enabled.toggled.connect(self._on_averaging_changed)
        self.averaging_window.valueChanged.connect(self._on_averaging_changed)
        self.averaging_sigma.toggled.connect(lambda _checked: self._refresh_from_controller())
        self.record_directory.textEdited.connect(self._on_record_setting_changed)
        self.record_decimation.valueChanged.connect(self._on_record_setting_changed)
        self.record_limit.valueChanged.connect(self._on_record_setting_changed)
        self.browse_button.clicked.connect(self._browse_directory)
        self.start_record_button.clicked.connect(self._start_recording)
        self.stop_record_button.clicked.connect(self._stop_recording)
        self.average_recording_button.clicked.connect(self._start_average_export)

    # --- Выбор графика ---------------------------------------------------------------

    def selected_slots(self) -> tuple[SlotRef, ...]:
        """Текущий пользовательский выбор в порядке дерева."""
        selected: list[SlotRef] = []
        for channel_index in range(self.trace_tree.topLevelItemCount()):
            channel_item = self.trace_tree.topLevelItem(channel_index)
            if channel_item is None:
                continue
            for position_index in range(channel_item.childCount()):
                item = channel_item.child(position_index)
                if item.checkState(0) != Qt.CheckState.Checked:
                    continue
                token = str(item.data(0, Qt.ItemDataRole.UserRole))
                selected.append(models.parse_slot_token(token))
        return tuple(selected)

    def _sync_trace_request(self) -> None:
        history_s = self.history_spin.value()
        averaging_s = self._averaging_window_s()
        if averaging_s is not None:
            history_s = max(history_s, averaging_s)
        self._controller.set_measurement_trace_request(
            [(slot.channel, slot.position) for slot in self.selected_slots()],
            history_s,
        )

    def _on_trace_changed(self, _item: QTreeWidgetItem, _column: int) -> None:
        if self._selection_loading:
            return
        self._sync_trace_request()

    def _on_history_changed(self, _value: float) -> None:
        if not self._selection_loading:
            self._sync_trace_request()

    def _on_averaging_changed(self, _value: object = None) -> None:
        self._graph_model = None
        self._sync_trace_request()
        self._refresh_from_controller()

    def _refresh_from_controller(self) -> None:
        self.refresh(self._controller.snapshot(include_sensor_data=False))

    def _averaging_window_s(self) -> float | None:
        if not self.averaging_enabled.isChecked():
            return None
        return self.averaging_window.value() / 1000.0

    def _update_averaging_controls(self, snapshot: AppSnapshot) -> None:
        enabled = self.averaging_enabled.isChecked()
        # Окно остаётся редактируемым и при выключенном live-усреднении:
        # то же поле задаёт окно офлайн-экспорта готовой записи. На график
        # оно не влияет, пока флажок ``averaging_enabled`` снят.
        self.averaging_window.setEnabled(True)
        self.averaging_sigma.setEnabled(enabled)
        frames = models.expected_averaging_frames(snapshot, self.averaging_window.value())
        self.averaging_frames.setText("—" if frames <= 0 else f"≈ {frames} кадров")

    def _update_history_limit(self, snapshot: AppSnapshot) -> None:
        metrics = snapshot.metrics
        if metrics is None:
            return
        rate = metrics.expected_rate_hz or snapshot.profile.sweep_speed_hz
        if rate <= 0:
            return
        maximum = max(MIN_HISTORY_S, metrics.history_frames / rate)
        current = self.history_spin.value()
        if abs(self.history_spin.maximum() - maximum) > 1e-9:
            self._selection_loading = True
            try:
                self.history_spin.setMaximum(maximum)
                if current > maximum:
                    self.history_spin.setValue(maximum)
            finally:
                self._selection_loading = False
            self._sync_trace_request()

    def _update_graph(self, snapshot: AppSnapshot) -> None:
        selected = self.selected_slots()
        self._graph_ticks += 1
        model = models.measurement_graph_model(
            snapshot,
            selected,
            self._graph_model,
            recalculate_y=self._graph_ticks % Y_RANGE_RECALC_TICKS == 0,
            averaging_window_s=self._averaging_window_s(),
        )
        unchanged = model is self._graph_model
        self._graph_model = model
        selected_set = set(selected)
        for slot in tuple(self._curves):
            if slot not in selected_set:
                self.plot.removeItem(self._curves.pop(slot))
                band = self._bands.pop(slot, None)
                if band is not None:
                    for item in band:
                        self.plot.removeItem(item)

        for index, trace in enumerate(model.traces):
            curve = self._curves.get(trace.slot)
            if curve is None:
                color = pg.intColor(index, hues=max(1, len(selected)))
                pen = pg.mkPen(color)
                curve = self.plot.plot(
                    pen=pen, name=texts.slot_label(trace.slot.channel, trace.slot.position)
                )
                self._curves[trace.slot] = curve
                upper = pg.PlotDataItem(pen=None)
                lower = pg.PlotDataItem(pen=None)
                red, green, blue, _alpha = color.getRgb()
                fill = pg.FillBetweenItem(
                    upper,
                    lower,
                    brush=pg.mkBrush(red, green, blue, 45),
                )
                fill.setZValue(-10)
                self.plot.addItem(upper)
                self.plot.addItem(lower)
                self.plot.addItem(fill)
                self._bands[trace.slot] = (upper, lower, fill)
            # `connect="finite"` — принципиальная часть отображения: NaN
            # разрывает линию и никогда не соединяется через пропавший пик.
            if not unchanged:
                curve.setData(model.t_s, trace.delta_nm, connect="finite")
                band = self._bands[trace.slot]
                sigma = trace.sigma_nm
                if sigma is not None:
                    band[0].setData(model.t_s, trace.delta_nm + sigma, connect="finite")
                    band[1].setData(model.t_s, trace.delta_nm - sigma, connect="finite")
            band = self._bands[trace.slot]
            band_visible = (
                self.averaging_enabled.isChecked()
                and self.averaging_sigma.isChecked()
                and trace.sigma_nm is not None
            )
            for item in band:
                item.setVisible(band_visible)

        counts: list[int] = []
        for trace in model.traces:
            if trace.n is not None and trace.n.size:
                counts.append(int(trace.n[-1]))
        if self.averaging_enabled.isChecked() and counts:
            low, high = min(counts), max(counts)
            self.averaging_n.setText(str(low) if low == high else f"{low}…{high}")
        else:
            self.averaging_n.setText(texts.UNKNOWN)

        self.plot.setXRange(-self.history_spin.value(), 0.0, padding=0.0)
        if not unchanged:
            self.plot.setYRange(model.y_min_nm, model.y_max_nm, padding=0.0)
        if not selected:
            self.graph_hint.setText(texts.GRAPH_NO_SELECTION + "\n" + texts.GRAPH_BASELINE_HINT)
        else:
            self.graph_hint.setText(texts.GRAPH_BASELINE_HINT)
        has_data = (
            bool(selected)
            and model.t_s.size > 0
            and any(trace.valid_points > 0 for trace in model.traces)
        )
        self.plot.setVisible(has_data)
        self.empty_graph_label.setVisible(not has_data)

    # --- Таблица ---------------------------------------------------------------------

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
        if metrics is None or metrics.frame_rate_hz <= 0.0:
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
        """Один UI-такт: график, таблица и состояние записи целиком."""
        self._update_history_limit(snapshot)
        self._update_averaging_controls(snapshot)
        self._update_graph(snapshot)
        self._update_table(snapshot)
        self._update_recording(snapshot)
        self._update_quality(snapshot)

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt
        thread = self._average_thread
        if thread is not None and thread.is_alive():
            thread.join()
        super().closeEvent(event)  # type: ignore[arg-type]
