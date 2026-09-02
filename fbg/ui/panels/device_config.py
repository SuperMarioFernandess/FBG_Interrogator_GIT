"""Панель записи настроек прибора.

Логика значений и валидация живут в `fbg.ui.models` без Qt. Виджет только
показывает модель и отправляет уже проверенные значения через `AppController`.

Критичное ограничение — R14: номер канала берётся только из живого ответа
`10 04`. Профиль для этого недостаточен: прибор не проверяет номер и запись
в несуществующий канал реально портила настройки физических каналов.
"""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fbg.core.session import SessionState
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.models import AppSnapshot


class DeviceConfigPanel(QWidget):
    """Порог, усиление, интервал пиков, развёртка и сохранение порогов."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = models.device_config_model(controller.snapshot())
        self._loading = False
        self._channel_dirty = False
        self._gap_dirty = False
        self._sweep_dirty = False

        self.availability_label = QLabel()
        self.availability_label.setWordWrap(True)

        self.channel_box = QComboBox()
        self.threshold_auto = QCheckBox(texts.LABEL_THRESHOLD_AUTO)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, self._model.adc_max)
        self.threshold_status = QLabel()
        self.threshold_hint = QLabel()
        self.threshold_hint.setWordWrap(True)
        self.apply_threshold_button = QPushButton(texts.BUTTON_APPLY_THRESHOLD)

        self.gain_mode = QComboBox()
        self.gain_mode.addItems([texts.LABEL_GAIN_AUTO, texts.LABEL_GAIN_MANUAL])
        self.gain_level = QSpinBox()
        self.gain_level.setRange(0, self._model.gain_max_level)
        self.gain_status = QLabel()
        self.gain_hint = QLabel()
        self.gain_hint.setWordWrap(True)
        self.apply_gain_button = QPushButton(texts.BUTTON_APPLY_GAIN)

        self.peak_gap = QSpinBox()
        self.peak_gap.setRange(1, 0xFF)
        self.peak_gap_status = QLabel()
        self.apply_peak_gap_button = QPushButton(texts.BUTTON_APPLY_PEAK_GAP)

        self.start_param = self._u16_spin()
        self.step_param = self._u16_spin(minimum=1)
        self.stop_param = self._u16_spin()
        self.adc_step_param = self._u16_spin(minimum=1)
        self.sweep_preview = QLabel()
        self.sweep_preview.setWordWrap(True)
        self.sweep_hint = QLabel(texts.SWEEP_RECONNECT_HINT)
        self.sweep_hint.setWordWrap(True)
        self.sweep_status = QLabel()
        self.apply_sweep_button = QPushButton(texts.BUTTON_APPLY_SWEEP)

        self.save_button = QPushButton(texts.BUTTON_SAVE_THRESHOLDS)
        self.save_status = QLabel()
        self.save_hint = QLabel(texts.SAVE_THRESHOLDS_HINT)
        self.save_hint.setWordWrap(True)

        self._build_layout()
        self._connect_signals()
        self.refresh(controller.snapshot())

    @staticmethod
    def _u16_spin(*, minimum: int = 0) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, 0xFFFF)
        return spin

    def _build_layout(self) -> None:
        channel_form = QFormLayout()
        channel_form.addRow(texts.LABEL_CHANNEL, self.channel_box)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addWidget(self.threshold_auto)
        threshold_row.addWidget(self.apply_threshold_button)
        channel_form.addRow(texts.LABEL_THRESHOLD, threshold_row)
        channel_form.addRow("", self.threshold_status)
        channel_form.addRow("", self.threshold_hint)

        channel_form.addRow(texts.LABEL_GAIN_MODE, self.gain_mode)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self.gain_level)
        gain_row.addWidget(self.apply_gain_button)
        channel_form.addRow(texts.LABEL_GAIN_LEVEL, gain_row)
        channel_form.addRow("", self.gain_status)
        channel_form.addRow("", self.gain_hint)
        self.channel_group = QGroupBox(texts.GROUP_CHANNEL_CONFIG)
        self.channel_group.setLayout(channel_form)

        gap_form = QFormLayout()
        gap_row = QHBoxLayout()
        gap_row.addWidget(self.peak_gap)
        gap_row.addWidget(self.apply_peak_gap_button)
        gap_form.addRow(texts.LABEL_PEAK_GAP, gap_row)
        gap_form.addRow("", self.peak_gap_status)
        self.gap_group = QGroupBox(texts.GROUP_PEAK_GAP_CONFIG)
        self.gap_group.setLayout(gap_form)

        sweep_form = QFormLayout()
        sweep_form.addRow(texts.LABEL_START_PARAM, self.start_param)
        sweep_form.addRow(texts.LABEL_STEP_PARAM, self.step_param)
        sweep_form.addRow(texts.LABEL_STOP_PARAM, self.stop_param)
        sweep_form.addRow(texts.LABEL_ADC_STEP_PARAM, self.adc_step_param)
        sweep_form.addRow("", self.sweep_preview)
        sweep_form.addRow("", self.sweep_hint)
        sweep_form.addRow("", self.sweep_status)
        sweep_form.addRow("", self.apply_sweep_button)
        self.sweep_group = QGroupBox(texts.GROUP_SWEEP_CONFIG)
        self.sweep_group.setLayout(sweep_form)

        save_layout = QVBoxLayout()
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.save_status)
        save_layout.addWidget(self.save_hint)
        self.save_group = QGroupBox(texts.GROUP_SAVE_THRESHOLDS)
        self.save_group.setLayout(save_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(self.availability_label)
        layout.addWidget(self.channel_group)
        layout.addWidget(self.gap_group)
        layout.addWidget(self.sweep_group)
        layout.addWidget(self.save_group)
        layout.addStretch(1)

    def _connect_signals(self) -> None:
        self.channel_box.currentIndexChanged.connect(self._on_channel_changed)
        self.threshold_auto.toggled.connect(self._on_channel_edited)
        self.threshold_spin.valueChanged.connect(self._on_channel_edited)
        self.gain_mode.currentIndexChanged.connect(self._on_channel_edited)
        self.gain_level.valueChanged.connect(self._on_channel_edited)
        self.peak_gap.valueChanged.connect(self._on_gap_edited)
        for spin in (self.start_param, self.step_param, self.stop_param, self.adc_step_param):
            spin.valueChanged.connect(self._on_sweep_edited)

        self.apply_threshold_button.clicked.connect(self._apply_threshold)
        self.apply_gain_button.clicked.connect(self._apply_gain)
        self.apply_peak_gap_button.clicked.connect(self._apply_peak_gap)
        self.apply_sweep_button.clicked.connect(self._apply_sweep)
        self.save_button.clicked.connect(self._save_thresholds)

    def _on_channel_edited(self, _value: object = None) -> None:
        if self._loading:
            return
        self._channel_dirty = True
        self.threshold_spin.setEnabled(not self.threshold_auto.isChecked() and self._model.enabled)
        self._update_threshold_hint()

    def _on_gap_edited(self, _value: int) -> None:
        if not self._loading:
            self._gap_dirty = True

    def _on_sweep_edited(self, _value: int) -> None:
        if not self._loading:
            self._sweep_dirty = True
        self._update_sweep_preview()

    def _selected_channel(self) -> int | None:
        data = self.channel_box.currentData()
        return None if data is None else int(data)

    def _on_channel_changed(self, _index: int) -> None:
        if self._loading:
            return
        self._channel_dirty = False
        self._load_selected_channel()
        self._update_channel_status()
        self._update_threshold_hint()

    def _load_selected_channel(self) -> None:
        channel = self._selected_channel()
        if channel is None:
            return
        item = next((item for item in self._model.channels if item.channel == channel), None)
        if item is None:
            return
        self._loading = True
        try:
            auto = item.threshold is None
            self.threshold_auto.setChecked(auto)
            if item.threshold is not None:
                self.threshold_spin.setValue(item.threshold)
            self.threshold_spin.setEnabled(not auto and self._model.enabled)
            self.gain_mode.setCurrentIndex(1 if item.gain.manual else 0)
            self.gain_level.setValue(item.gain.level)
        finally:
            self._loading = False

    def _sync_channels(self) -> None:
        current = self._selected_channel()
        available = tuple(item.channel for item in self._model.channels)
        listed = tuple(int(self.channel_box.itemData(i)) for i in range(self.channel_box.count()))
        if listed == available:
            return
        self._loading = True
        try:
            self.channel_box.clear()
            for channel in available:
                self.channel_box.addItem(str(channel + 1), channel)
            if current in available:
                self.channel_box.setCurrentIndex(available.index(current))
        finally:
            self._loading = False
        self._channel_dirty = False

    def _load_gap(self) -> None:
        if self._model.peak_gap_ghz is None:
            return
        self._loading = True
        try:
            self.peak_gap.setValue(self._model.peak_gap_ghz)
        finally:
            self._loading = False

    def _load_sweep(self) -> None:
        sweep = self._model.sweep
        if sweep is None:
            self.sweep_preview.setText("")
            return
        self._loading = True
        try:
            self.start_param.setValue(sweep.start_param)
            self.step_param.setValue(sweep.step_param)
            self.stop_param.setValue(sweep.stop_param)
            self.adc_step_param.setValue(sweep.adc_step_param)
        finally:
            self._loading = False
        self._update_sweep_preview()

    def _selected_channel_model(self) -> models.ChannelConfigModel | None:
        channel = self._selected_channel()
        if channel is None:
            return None
        return next((item for item in self._model.channels if item.channel == channel), None)

    def _update_channel_status(self) -> None:
        item = self._selected_channel_model()
        if item is None:
            self.threshold_status.setText("")
            self.gain_status.setText("")
            return
        self.threshold_status.setText(
            texts.UNCONFIRMED if item.threshold_unconfirmed else texts.CONFIRMED
        )
        self.gain_status.setText(texts.UNCONFIRMED if item.gain_unconfirmed else texts.CONFIRMED)

    def _update_threshold_hint(self) -> None:
        threshold = None if self.threshold_auto.isChecked() else self.threshold_spin.value()
        self.threshold_hint.setText(
            texts.threshold_spectrum_hint(self._model.last_spectrum_max_adc, threshold)
        )

    def _update_gain_hint(self) -> None:
        self.gain_hint.setText(texts.gain_spectrum_hint(self._model.last_spectrum_saturated_points))

    def _update_verification_status(self) -> None:
        self._update_channel_status()
        if self._model.peak_gap_ghz is None:
            self.peak_gap_status.setText("")
        elif self._model.peak_gap_unconfirmed:
            self.peak_gap_status.setText(texts.UNCONFIRMED)
        else:
            self.peak_gap_status.setText(texts.CONFIRMED)

        sweep = self._model.sweep
        if sweep is None:
            self.sweep_status.setText("")
        elif sweep.unconfirmed:
            self.sweep_status.setText(texts.UNCONFIRMED)
        else:
            self.sweep_status.setText(texts.CONFIRMED)

    def _update_sweep_preview(self) -> None:
        try:
            preview = models.sweep_edit_model(
                self._controller.config.profile,
                self.start_param.value(),
                self.step_param.value(),
                self.stop_param.value(),
                self.adc_step_param.value(),
            )
        except ValueError as exc:
            self.sweep_preview.setText(texts.config_error(str(exc)))
            self.apply_sweep_button.setEnabled(False)
            return
        self.sweep_preview.setText(
            texts.sweep_preview_text(
                preview.adc_points,
                preview.start_ghz,
                preview.stop_ghz,
                preview.start_nm,
                preview.stop_nm,
            )
        )
        self.apply_sweep_button.setEnabled(self._model.sweep_enabled)

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Обновляет поля снимком, не затирая ввод пользователя на каждом тике."""
        self._model = models.device_config_model(snapshot)
        self.threshold_spin.setMaximum(self._model.adc_max)
        self.gain_level.setMaximum(self._model.gain_max_level)
        self._sync_channels()
        if not self._channel_dirty:
            self._load_selected_channel()
        if not self._gap_dirty:
            self._load_gap()
        if not self._sweep_dirty:
            self._load_sweep()

        enabled = self._model.enabled
        for group in (self.channel_group, self.gap_group, self.save_group):
            group.setEnabled(enabled)
        self.sweep_group.setEnabled(self._model.sweep_enabled)
        self.threshold_spin.setEnabled(enabled and not self.threshold_auto.isChecked())
        if snapshot.profile_mismatch:
            availability = texts.CONFIG_PROFILE_MISMATCH
        elif snapshot.recording and enabled:
            availability = (
                texts.CONFIG_STREAMING_ALLOWED + "\n" + texts.CONFIG_RECORDING_SWEEP_LOCKED
            )
        elif snapshot.state is SessionState.STREAMING and enabled:
            availability = texts.CONFIG_STREAMING_ALLOWED
        elif enabled:
            availability = texts.CONFIG_READY
        else:
            availability = texts.CONFIG_UNAVAILABLE
        self.availability_label.setText(availability)
        self._update_verification_status()
        if self._model.saved_thresholds_unconfirmed:
            self.save_status.setText(texts.UNCONFIRMED)
        self._update_threshold_hint()
        self._update_gain_hint()
        self._update_sweep_preview()

    def _report_local_error(self, exc: Exception) -> None:
        self._controller.note(texts.config_error(str(exc)))

    def _apply_threshold(self) -> None:
        channel = self._selected_channel()
        if channel is None:
            return
        try:
            models.validate_channel(channel, self._model.channel_count)
            threshold = models.threshold_value(
                self.threshold_auto.isChecked(), self.threshold_spin.value(), self._model.adc_max
            )
            result = self._controller.set_threshold(channel, threshold)
        except (RuntimeError, ValueError) as exc:
            self._report_local_error(exc)
            return
        if result.ok:
            self._channel_dirty = False
        self.refresh(self._controller.snapshot())

    def _apply_gain(self) -> None:
        channel = self._selected_channel()
        if channel is None:
            return
        try:
            models.validate_channel(channel, self._model.channel_count)
            gain = models.gain_value(
                self.gain_mode.currentIndex() == 1,
                self.gain_level.value(),
                self._model.gain_max_level,
            )
            result = self._controller.set_gain(channel, gain.manual, gain.level)
        except (RuntimeError, ValueError) as exc:
            self._report_local_error(exc)
            return
        if result.ok:
            self._channel_dirty = False
        self.refresh(self._controller.snapshot())

    def _apply_peak_gap(self) -> None:
        try:
            result = self._controller.set_peak_gap(self.peak_gap.value())
        except (RuntimeError, ValueError) as exc:
            self._report_local_error(exc)
            return
        if result.ok:
            self._gap_dirty = False
        self.refresh(self._controller.snapshot())

    def _apply_sweep(self) -> None:
        try:
            preview = models.sweep_edit_model(
                self._controller.config.profile,
                self.start_param.value(),
                self.step_param.value(),
                self.stop_param.value(),
                self.adc_step_param.value(),
            )
            result = self._controller.set_sweep(
                preview.start_param,
                preview.step_param,
                preview.stop_param,
                preview.adc_step_param,
            )
        except (RuntimeError, ValueError) as exc:
            self._report_local_error(exc)
            return
        if result.ok:
            self._sweep_dirty = False
        self.refresh(self._controller.snapshot())

    def _save_thresholds(self) -> None:
        try:
            result = self._controller.save_thresholds()
        except RuntimeError as exc:
            self._report_local_error(exc)
            return
        if result.ok:
            self.save_status.setText(texts.SAVE_READBACK_OK)
        self.refresh(self._controller.snapshot())
