"""Каркас окна: вкладки, строка состояния и единственный таймер обновления.

Таймер один на всё окно. Событийного обновления нет намеренно: журнал видит
каждую датаграмму, а при 2000 Гц это две тысячи событий в секунду, каждое
из которых дошло бы до отрисовки. Панели вместо этого читают **снимок**
десять раз в секунду и ничего из ядра не держат (Р36).

Снимок берётся один на такт и раздаётся всем панелям. Так они гарантированно
показывают один и тот же момент: состояние в строке состояния не может
разойтись с состоянием на панели подключения.
"""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QMainWindow, QStatusBar, QTabWidget, QWidget

from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.panels.connection import ConnectionPanel
from fbg.ui.panels.device_config import DeviceConfigPanel
from fbg.ui.panels.device_info import DeviceInfoPanel
from fbg.ui.panels.measurement import MeasurementPanel
from fbg.ui.panels.packet_log import PacketLogPanel
from fbg.ui.panels.spectrum import SpectrumPanel

#: Период обновления панелей, мс. 10 Гц — тот же порядок, что у децимации
#: снимков pipeline (Р39): человек больше не различает, а UI не захлёбывается.
UI_PERIOD_MS = 100


class MainWindow(QMainWindow):
    """Главное окно: шесть вкладок и строка состояния."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle(texts.APP_TITLE)
        self.resize(1100, 750)

        self.connection_panel = ConnectionPanel(controller)
        self.measurement_panel = MeasurementPanel(controller)
        self.spectrum_panel = SpectrumPanel(controller)
        self.device_panel = DeviceInfoPanel(controller)
        self.device_config_panel = DeviceConfigPanel(controller)
        self.packet_log_panel = PacketLogPanel(controller)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.connection_panel, texts.TAB_CONNECTION)
        self.tabs.addTab(self.measurement_panel, texts.TAB_MEASUREMENT)
        self.tabs.addTab(self.spectrum_panel, texts.TAB_SPECTRUM)
        self.tabs.addTab(self.device_panel, texts.TAB_DEVICE)
        self.tabs.addTab(self.device_config_panel, texts.TAB_DEVICE_CONFIG)
        self.tabs.addTab(self.packet_log_panel, texts.TAB_PACKET_LOG)
        self.setCentralWidget(self.tabs)

        self.status_label = QLabel()
        status = QStatusBar()
        status.addWidget(self.status_label)
        self.setStatusBar(status)

        self._timer = QTimer(self)
        self._timer.setInterval(UI_PERIOD_MS)
        self._timer.timeout.connect(self.tick)

    @property
    def panels(self) -> tuple[QWidget, ...]:
        """Все панели окна в порядке вкладок."""
        return (
            self.connection_panel,
            self.measurement_panel,
            self.spectrum_panel,
            self.device_panel,
            self.device_config_panel,
            self.packet_log_panel,
        )

    @property
    def timer(self) -> QTimer:
        """Таймер обновления. Открыт наружу ради тестов и остановки."""
        return self._timer

    def start_updates(self) -> None:
        """Запускает таймер и сразу делает первый такт."""
        self.tick()
        self._timer.start()

    def stop_updates(self) -> None:
        """Останавливает таймер. Повторный вызов безвреден."""
        self._timer.stop()

    def tick(self) -> None:
        """Один такт: снимок берётся один раз и раздаётся всем панелям."""
        snapshot = self._controller.snapshot()
        self.connection_panel.refresh(snapshot)
        self.measurement_panel.refresh(snapshot)
        self.spectrum_panel.refresh(snapshot)
        self.device_panel.refresh(snapshot)
        self.device_config_panel.refresh(snapshot)
        self.packet_log_panel.refresh(snapshot)
        self.status_label.setText(models.status_line(snapshot))

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt
        """Гасит таймер при закрытии окна.

        Само приложение останавливает не окно, а `main.py`: порядок остановки
        должен соблюдаться и тогда, когда окна не было вовсе.
        """
        self.stop_updates()
        super().closeEvent(event)  # type: ignore[arg-type]
