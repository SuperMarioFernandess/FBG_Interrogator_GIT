"""Панель измерения: λ(t), таблица слотов и управление записью.

UI получает данные только через `AppSnapshot`. Историю графика копирует сам
pipeline по запросу выбранных позиций; `RingHistory` сюда не попадает (Р36).
Один таймер главного окна обновляет панель 10 раз в секунду — никаких сигналов
на каждый кадр и никакого файлового I/O в колбэках ядра.
"""

import math
from dataclasses import replace
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fbg.core.session import SessionState
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.models import AppSnapshot, MeasurementTableModel, SlotRef

#: Дефолт не означает «датчики 1–4». Это четыре первых **слота** канала 1,
#: которые прибор заполняет по мере обнаружения пиков (Р30).
DEFAULT_SELECTED_SLOTS = 4

#: Нижняя граница настройки истории. Меньше одного такта UI практического
#: смысла не имеет, но 50 мс совпадает с периодом публикации pipeline.
MIN_HISTORY_S = 0.05


class _MeasurementQtTableModel(QAbstractTableModel):
    """Qt-обёртка над неизменяемой моделью последнего кадра.

    Обновление — один `modelReset` на весь кадр, а не 240 `dataChanged` по
    ячейкам. Это и есть требуемое обновление «целиком» 10 Гц.
    """

    def __init__(self, model: MeasurementTableModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model

    @property
    def model(self) -> MeasurementTableModel:
        return self._model

    def replace(self, model: MeasurementTableModel) -> None:
        """Подменяет кадр одним сигналом модели."""
        self.beginResetModel()
        self._model = model
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        if parent.isValid():
            return 0
        return self._model.positions

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008, N802
        if parent.isValid():
            return 0
        return 1 + 2 * self._model.channels

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = index.row()
        column = index.column()
        if column == 0:
            return str(row + 1)
        channel = (column - 1) // 2
        validity_column = (column - 1) % 2 == 1
        if validity_column:
            return (
                texts.TABLE_VALID_YES
                if bool(self._model.valid[channel, row])
                else texts.TABLE_VALID_NO
            )
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
        channel = (section - 1) // 2
        suffix = texts.TABLE_VALID if (section - 1) % 2 else texts.TABLE_WAVELENGTH
        return f"К{channel + 1} {suffix}"


class MeasurementPanel(QWidget):
    """График выбранных слотов, таблица 4×30 и запись CSV."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._selection_loading = False
        self._record_settings_loading = False
        self._record_settings_dirty = False
        self._curves: dict[SlotRef, pg.PlotDataItem] = {}

        profile = controller.config.profile
        self.trace_tree = QTreeWidget()
        self.trace_tree.setHeaderHidden(True)
        self.trace_tree.setMinimumWidth(210)
        self.history_spin = QDoubleSpinBox()
        self.history_spin.setDecimals(2)
        self.history_spin.setSingleStep(0.5)
        self.history_spin.setRange(MIN_HISTORY_S, 86_400.0)
        self.history_spin.setValue(5.0)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", texts.GRAPH_AXIS_TIME)
        self.plot.setLabel("left", texts.GRAPH_AXIS_DELTA_NM)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend()
        self.graph_hint = QLabel(texts.GRAPH_BASELINE_HINT)
        self.graph_hint.setWordWrap(True)
        self.quality_label = QLabel()

        empty_table = models.measurement_table_model(controller.snapshot())
        self.table_model = _MeasurementQtTableModel(empty_table, self)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.temperature_label = QLabel()
        self.temperature_label.setWordWrap(True)

        self.record_directory = QLineEdit()
        self.browse_button = QPushButton(texts.BUTTON_BROWSE_DIRECTORY)
        self.record_decimation = QSpinBox()
        self.record_decimation.setRange(1, 100_000)
        self.record_limit = QSpinBox()
        self.record_limit.setRange(0, profile.fbg_per_channel)
        self.record_limit.setSpecialValueText(texts.RECORD_LIMIT_ALL)
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
        selection_form = QFormLayout()
        selection_form.addRow(texts.LABEL_GRAPH_HISTORY, self.history_spin)
        selection_layout = QVBoxLayout()
        selection_layout.addLayout(selection_form)
        selection_layout.addWidget(self.trace_tree, 1)
        selection_box = QGroupBox(texts.GROUP_TRACE_SELECTION)
        selection_box.setLayout(selection_layout)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.quality_label)
        graph_layout.addWidget(self.plot, 1)
        graph_layout.addWidget(self.graph_hint)
        graph_box = QGroupBox(texts.GROUP_MEASUREMENT_GRAPH)
        graph_box.setLayout(graph_layout)

        graph_split = QSplitter(Qt.Orientation.Horizontal)
        graph_split.addWidget(selection_box)
        graph_split.addWidget(graph_box)
        graph_split.setStretchFactor(0, 0)
        graph_split.setStretchFactor(1, 1)

        table_layout = QVBoxLayout()
        table_layout.addWidget(self.temperature_label)
        table_layout.addWidget(self.table, 1)
        table_box = QGroupBox(texts.GROUP_MEASUREMENT_TABLE)
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
        record_form.addRow("", record_buttons)
        record_form.addRow("", self.record_error)
        record_box = QGroupBox(texts.GROUP_RECORDING)
        record_box.setLayout(record_form)

        lower_split = QSplitter(Qt.Orientation.Horizontal)
        lower_split.addWidget(table_box)
        lower_split.addWidget(record_box)
        lower_split.setStretchFactor(0, 1)
        lower_split.setStretchFactor(1, 0)

        outer_split = QSplitter(Qt.Orientation.Vertical)
        outer_split.addWidget(graph_split)
        outer_split.addWidget(lower_split)
        outer_split.setStretchFactor(0, 1)
        outer_split.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(outer_split)

    def _connect_signals(self) -> None:
        self.trace_tree.itemChanged.connect(self._on_trace_changed)
        self.history_spin.valueChanged.connect(self._on_history_changed)
        self.record_directory.textEdited.connect(self._on_record_setting_changed)
        self.record_decimation.valueChanged.connect(self._on_record_setting_changed)
        self.record_limit.valueChanged.connect(self._on_record_setting_changed)
        self.browse_button.clicked.connect(self._browse_directory)
        self.start_record_button.clicked.connect(self._start_recording)
        self.stop_record_button.clicked.connect(self._stop_recording)

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
        self._controller.set_measurement_trace_request(
            [(slot.channel, slot.position) for slot in self.selected_slots()],
            self.history_spin.value(),
        )

    def _on_trace_changed(self, _item: QTreeWidgetItem, _column: int) -> None:
        if self._selection_loading:
            return
        self._sync_trace_request()

    def _on_history_changed(self, _value: float) -> None:
        if not self._selection_loading:
            self._sync_trace_request()

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
        model = models.measurement_graph_model(snapshot, selected)
        selected_set = set(selected)
        for slot in tuple(self._curves):
            if slot not in selected_set:
                self.plot.removeItem(self._curves.pop(slot))

        for index, trace in enumerate(model.traces):
            curve = self._curves.get(trace.slot)
            if curve is None:
                pen = pg.mkPen(pg.intColor(index, hues=max(1, len(selected))))
                curve = self.plot.plot(
                    pen=pen, name=texts.slot_label(trace.slot.channel, trace.slot.position)
                )
                self._curves[trace.slot] = curve
            # `connect="finite"` — принципиальная часть отображения: NaN
            # разрывает линию и никогда не соединяется через пропавший пик.
            curve.setData(model.t_s, trace.delta_nm, connect="finite")

        self.plot.setXRange(-self.history_spin.value(), 0.0, padding=0.0)
        self.plot.setYRange(model.y_min_nm, model.y_max_nm, padding=0.0)
        if not selected:
            self.graph_hint.setText(texts.GRAPH_NO_SELECTION + "\n" + texts.GRAPH_BASELINE_HINT)
        else:
            self.graph_hint.setText(texts.GRAPH_BASELINE_HINT)

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
        self.quality_label.setText(f"Темп: {metrics.frame_rate_hz:.2f} Гц · потери: {loss}")

    # --- Общий такт ------------------------------------------------------------------

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Один такт 10 Гц: график, таблица и состояние записи целиком."""
        self._update_history_limit(snapshot)
        self._update_graph(snapshot)
        self._update_table(snapshot)
        self._update_recording(snapshot)
        self._update_quality(snapshot)
