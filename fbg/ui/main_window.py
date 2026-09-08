"""Главное окно: шесть вкладок-доков, строка состояния и один UI-таймер.

Крупные области каждой вкладки — ``QDockWidget``: пользователь может менять
их ширину/высоту, переносить и временно отделять. Раскладка хранится отдельно
от конфигурации прибора, потому что это чисто Qt-состояние и оно не должно
попадать в ``fbg.io``.

Снимок контроллера по-прежнему берётся один на такт. Обычно обновляется только
активная вкладка; исключение — видимый floating-док скрытой вкладки: раз он
остаётся отдельным окном, его график/таблица не должны замирать.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.docking import (
    DEFAULT_UI_PERIOD_MS,
    MAX_UI_PERIOD_MS,
    MIN_UI_PERIOD_MS,
    DockTab,
    UiLayoutState,
    layout_path_for_config,
    load_layout,
    quarantine_layout,
    save_layout,
)
from fbg.ui.panels.connection import ConnectionPanel
from fbg.ui.panels.device_config import DeviceConfigPanel
from fbg.ui.panels.device_info import DeviceInfoPanel
from fbg.ui.panels.measurement import MeasurementPanel
from fbg.ui.panels.packet_log import PacketLogPanel
from fbg.ui.panels.sensors import SensorsPanel
from fbg.ui.panels.spectrum import SpectrumPanel

# Совместимость с тестами/кодом до чата №16: теперь это именно дефолт, а не
# жёстко прошитый период таймера.
UI_PERIOD_MS = DEFAULT_UI_PERIOD_MS


class MainWindow(QMainWindow):
    """Главное окно: шесть вкладок с настраиваемой док-раскладкой."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._layout_path = layout_path_for_config(controller.config_path)
        self._layout_locked = False
        self.setWindowTitle(texts.APP_TITLE)
        self.resize(1100, 750)

        self.connection_panel = ConnectionPanel(controller)
        self.measurement_panel = MeasurementPanel(controller)
        self.sensors_panel = SensorsPanel(controller)
        self.spectrum_panel = SpectrumPanel(controller)
        self.device_config_panel = DeviceConfigPanel(controller)
        self.device_panel = DeviceInfoPanel(controller, self.device_config_panel)
        self.packet_log_panel = PacketLogPanel(controller)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.connection_panel, texts.TAB_CONNECTION)
        self.tabs.addTab(self.measurement_panel, texts.TAB_MEASUREMENT)
        self.tabs.addTab(self.sensors_panel, texts.TAB_SENSORS)
        self.tabs.addTab(self.spectrum_panel, texts.TAB_SPECTRUM)
        self.tabs.addTab(self.device_panel, texts.TAB_DEVICE)
        self.tabs.addTab(self.packet_log_panel, texts.TAB_PACKET_LOG)
        self.setCentralWidget(self.tabs)

        self.status_label = QLabel()
        status = QStatusBar()
        status.addWidget(self.status_label)
        self.setStatusBar(status)

        self._build_layout_toolbar()

        self._timer = QTimer(self)
        self._timer.setInterval(DEFAULT_UI_PERIOD_MS)
        self._timer.timeout.connect(self.tick)
        self.tabs.currentChanged.connect(lambda _index: self.tick())

        self._load_saved_layout()

    @property
    def panels(self) -> tuple[DockTab, ...]:
        """Все шесть вкладок в порядке интерфейса."""

        return (
            self.connection_panel,
            self.measurement_panel,
            self.sensors_panel,
            self.spectrum_panel,
            self.device_panel,
            self.packet_log_panel,
        )

    @property
    def timer(self) -> QTimer:
        """Таймер обновления. Открыт наружу ради тестов и остановки."""

        return self._timer

    @property
    def layout_path(self) -> Path | None:
        """Файл состояния Qt или ``None`` у безфайлового тестового окна."""

        return self._layout_path

    def _build_layout_toolbar(self) -> None:
        toolbar = QToolBar("Раскладка", self)
        toolbar.setObjectName("layout.toolbar")
        toolbar.setMovable(False)

        self.save_layout_button = QPushButton(texts.LAYOUT_SAVE)
        self.reset_layout_button = QPushButton(texts.LAYOUT_RESET)
        self.lock_layout_button = QPushButton(texts.LAYOUT_LOCK)
        self.lock_layout_button.setCheckable(True)
        self.ui_period_spin = QSpinBox()
        self.ui_period_spin.setRange(MIN_UI_PERIOD_MS, MAX_UI_PERIOD_MS)
        self.ui_period_spin.setSingleStep(50)
        self.ui_period_spin.setValue(DEFAULT_UI_PERIOD_MS)
        self.ui_period_spin.setMaximumWidth(100)

        toolbar.addWidget(self.save_layout_button)
        toolbar.addWidget(self.reset_layout_button)
        toolbar.addWidget(self.lock_layout_button)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(texts.LAYOUT_PERIOD))
        toolbar.addWidget(self.ui_period_spin)
        self.addToolBar(toolbar)
        self.layout_toolbar = toolbar

        self.save_layout_button.setEnabled(self._layout_path is not None)
        self.save_layout_button.clicked.connect(self.save_layout_now)
        self.reset_layout_button.clicked.connect(self.reset_layout)
        self.lock_layout_button.toggled.connect(self.set_layout_locked)
        self.ui_period_spin.valueChanged.connect(self._set_ui_period)

    def _set_ui_period(self, period_ms: int) -> None:
        self._timer.setInterval(period_ms)

    def set_layout_locked(self, locked: bool) -> None:
        """Одновременно закрепляет доки всех вкладок."""

        self._layout_locked = bool(locked)
        self.lock_layout_button.blockSignals(True)
        try:
            self.lock_layout_button.setChecked(self._layout_locked)
        finally:
            self.lock_layout_button.blockSignals(False)
        for panel in self.panels:
            panel.set_layout_locked(self._layout_locked)

    def _layout_state(self) -> UiLayoutState:
        return UiLayoutState(
            docks={panel.layout_key: panel.saveState() for panel in self.panels},
            locked=self._layout_locked,
            period_ms=self.ui_period_spin.value(),
        )

    def save_layout_now(self) -> None:
        """Сохраняет текущее положение доков атомарно."""

        if self._layout_path is None:
            return
        try:
            save_layout(self._layout_path, self._layout_state())
        except OSError as exc:
            self._controller.note(f"раскладка интерфейса не сохранена: {exc}")
            return
        self._controller.note(texts.LAYOUT_SAVED)

    def _restore_layout(self, state: UiLayoutState) -> None:
        expected = {panel.layout_key for panel in self.panels}
        if set(state.docks) != expected:
            raise ValueError("набор вкладок в файле раскладки не совпадает с приложением")
        for panel in self.panels:
            if not panel.restoreState(state.docks[panel.layout_key]):
                raise ValueError(f"Qt не восстановил раскладку {panel.layout_key}")
        self.ui_period_spin.setValue(state.period_ms)
        self.set_layout_locked(state.locked)

    def _load_saved_layout(self) -> None:
        path = self._layout_path
        if path is None or not path.exists():
            return
        try:
            self._restore_layout(load_layout(path))
        except (OSError, ValueError) as exc:
            for panel in self.panels:
                panel.reset_layout()
            self.ui_period_spin.setValue(DEFAULT_UI_PERIOD_MS)
            self.set_layout_locked(False)
            with contextlib.suppress(OSError):
                quarantine_layout(path)
            self._controller.note(f"{texts.LAYOUT_LOAD_FAILED} {exc}")

    def reset_layout(self) -> None:
        """Возвращает доки и общий период к штатному виду."""

        self.set_layout_locked(False)
        for panel in self.panels:
            panel.reset_layout()
        self.ui_period_spin.setValue(DEFAULT_UI_PERIOD_MS)
        self._controller.note(texts.LAYOUT_RESET_DONE)

    def start_updates(self) -> None:
        """Запускает таймер и сразу делает первый такт."""

        self.tick()
        self._timer.start()

    def stop_updates(self) -> None:
        """Останавливает таймер. Повторный вызов безвреден."""

        self._timer.stop()

    def _refresh_targets(self) -> tuple[DockTab, ...]:
        active = self.tabs.currentWidget()
        return tuple(panel for panel in self.panels if panel.needs_refresh(active=panel is active))

    def tick(self) -> None:
        """Один такт: один снимок на активную вкладку и floating-доки."""

        targets = self._refresh_targets()
        snapshot = self._controller.snapshot(
            include_trace_history=self.measurement_panel in targets,
            include_sensor_data=self.sensors_panel in targets,
        )
        for panel in targets:
            panel.refresh(snapshot)
        self.status_label.setText(models.status_line(snapshot))

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt
        """Сохраняет раскладку и гасит таймер перед закрытием окна."""

        if self._layout_path is not None:
            try:
                save_layout(self._layout_path, self._layout_state())
            except OSError as exc:
                self._controller.note(f"раскладка интерфейса не сохранена: {exc}")
        self.stop_updates()
        super().closeEvent(event)  # type: ignore[arg-type]
