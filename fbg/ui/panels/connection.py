"""Панель подключения: адреса, кнопки, индикатор состояния и диагностика.

Половина ценности панели — текст диагностики. «Ошибка подключения» не говорит
человеку за стендом ничего; лестница из KB_01 говорит, что проверять и в каком
порядке, и половину пунктов закрывает сама счётчиками транспорта. Составляет
её `fbg.ui.diagnostics` — здесь только расстановка по виджетам.

Подключение уходит в фоновый поток (`AppController.connect_async`): `Probing`
на молчащем приборе — это Stop плюс пять чтений с повторами, то есть секунды
замороженного окна, за которые человек за стендом успеет решить, что
приложение зависло. Результат забирает `refresh`, то есть тот же таймер 10 Гц,
которым панель и так читает снимки. Пока поток жив, обе кнопки связи выключены:
повторное нажатие дало бы `WRONG_STATE`, а отключение посреди опроса — гонку
с командой, которая уже в полёте.
"""

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fbg.core.endpoint import MAX_PORT, Endpoint
from fbg.core.session import SessionState
from fbg.ui import diagnostics, models, texts
from fbg.ui.app import AppController
from fbg.ui.models import AppSnapshot


class ConnectionPanel(QWidget):
    """Подключение, поток и диагностика отказа."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._local_addresses = diagnostics.local_ipv4_addresses()
        self._diagnostics_text = ""
        self._notices_text = ""

        self.device_ip = QLineEdit()
        self.device_port = QSpinBox()
        self.device_port.setRange(1, MAX_PORT)
        self.local_ip = QLineEdit()
        self.local_ip.setReadOnly(True)
        self.local_ip.setToolTip(texts.LOCAL_IP_LOCKED)
        self.local_port = QSpinBox()
        self.local_port.setRange(0, MAX_PORT)

        self.connect_button = QPushButton(texts.BUTTON_CONNECT)
        self.disconnect_button = QPushButton(texts.BUTTON_DISCONNECT)
        self.start_button = QPushButton(texts.BUTTON_START_STREAM)
        self.stop_button = QPushButton(texts.BUTTON_STOP_STREAM)
        self.apply_profile_button = QPushButton(texts.BUTTON_APPLY_DEVICE_PROFILE)

        self.state_label = QLabel()
        self.mismatch_label = QLabel()
        self.mismatch_label.setWordWrap(True)
        self.diagnostics_view = QPlainTextEdit()
        self.diagnostics_view.setReadOnly(True)
        self.notices_view = QPlainTextEdit()
        self.notices_view.setReadOnly(True)
        self.notices_view.setMaximumHeight(110)

        self._build_layout()
        self.connect_button.clicked.connect(self._on_connect)
        self.disconnect_button.clicked.connect(self._on_disconnect)
        self.start_button.clicked.connect(self._on_start_stream)
        self.stop_button.clicked.connect(self._on_stop_stream)
        self.apply_profile_button.clicked.connect(self._on_apply_profile)

        self._load_endpoint(controller.config.endpoint)
        self.refresh(controller.snapshot())

    def _build_layout(self) -> None:
        """Расстановка виджетов. Ничего, кроме компоновки."""
        addresses = QFormLayout()
        addresses.addRow(texts.LABEL_DEVICE_IP, self.device_ip)
        addresses.addRow(texts.LABEL_DEVICE_PORT, self.device_port)
        addresses.addRow(texts.LABEL_LOCAL_IP, self.local_ip)
        addresses.addRow(texts.LABEL_LOCAL_PORT, self.local_port)
        address_box = QGroupBox(texts.GROUP_ENDPOINT)
        address_box.setLayout(addresses)

        buttons = QHBoxLayout()
        buttons.addWidget(self.connect_button)
        buttons.addWidget(self.disconnect_button)
        buttons.addStretch(1)

        stream = QHBoxLayout()
        stream.addWidget(self.start_button)
        stream.addWidget(self.stop_button)
        stream.addStretch(1)
        stream_box = QGroupBox(texts.GROUP_STREAM)
        stream_box.setLayout(stream)

        diagnostics_layout = QVBoxLayout()
        diagnostics_layout.addWidget(self.diagnostics_view)
        diagnostics_box = QGroupBox(texts.GROUP_DIAGNOSTICS)
        diagnostics_box.setLayout(diagnostics_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(address_box)
        layout.addLayout(buttons)
        layout.addWidget(self.state_label)
        layout.addWidget(self.mismatch_label)
        layout.addWidget(self.apply_profile_button)
        layout.addWidget(stream_box)
        layout.addWidget(diagnostics_box, 1)
        layout.addWidget(self.notices_view)

    # --- Данные ------------------------------------------------------------------------

    def _load_endpoint(self, endpoint: Endpoint) -> None:
        """Показывает текущие адреса в полях."""
        self.device_ip.setText(endpoint.device_ip)
        self.device_port.setValue(endpoint.device_port)
        self.local_ip.setText(endpoint.local_ip)
        self.local_port.setValue(endpoint.local_port)

    def _endpoint_from_fields(self, current: Endpoint) -> Endpoint | None:
        """Собирает `Endpoint` из полей. None — значения не приняты проверкой.

        `local_ip` не читается из поля: оно только для показа. Требование
        Р29 — приём на `0.0.0.0`, и делать его редактируемым значило бы
        предлагать оператору самый дорогой способ сломать связь.
        """
        try:
            return Endpoint(
                device_ip=self.device_ip.text().strip(),
                device_port=self.device_port.value(),
                local_ip=current.local_ip,
                local_port=self.local_port.value(),
                rcvbuf_bytes=current.rcvbuf_bytes,
                read_timeout_s=current.read_timeout_s,
                write_timeout_s=current.write_timeout_s,
                retries=current.retries,
                rx_poll_timeout_s=current.rx_poll_timeout_s,
                rx_queue_capacity=current.rx_queue_capacity,
                strict_source_port=current.strict_source_port,
            )
        except ValueError as exc:
            self._controller.note(str(exc))
            return None

    # --- Действия ----------------------------------------------------------------------

    def _on_connect(self) -> None:
        """Применяет адреса и запускает подключение в фоне.

        Синхронный `connect()` замораживал бы окно на всё время опроса —
        на молчащем приборе это секунды, — поэтому работа уходит в обычный
        поток, а результат забирает таймер окна в `refresh`.
        """
        current = self._controller.config.endpoint
        endpoint = self._endpoint_from_fields(current)
        if endpoint is None:
            self.refresh(self._controller.snapshot())
            return
        if endpoint != current:
            self._controller.set_endpoint(endpoint)
        self._local_addresses = diagnostics.local_ipv4_addresses()
        self._controller.connect_async()
        self.refresh(self._controller.snapshot())

    def _on_disconnect(self) -> None:
        self._controller.disconnect()
        self.refresh(self._controller.snapshot())

    def _on_start_stream(self) -> None:
        self._controller.start_stream()
        self.refresh(self._controller.snapshot())

    def _on_stop_stream(self) -> None:
        self._controller.stop_stream()
        self.refresh(self._controller.snapshot())

    def _on_apply_profile(self) -> None:
        """Принимает геометрию прибора и пересобирает тракт."""
        try:
            self._controller.apply_device_profile()
        except RuntimeError as exc:
            self._controller.note(str(exc))
        self._load_endpoint(self._controller.config.endpoint)
        self.refresh(self._controller.snapshot())

    # --- Обновление --------------------------------------------------------------------

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Читает снимок и обновляет виджеты. Зовётся таймером окна, 10 Гц."""
        # Результат фонового подключения забирается здесь и ровно один раз:
        # иначе один и тот же отказ повторялся бы в сообщениях десять раз
        # в секунду.
        result = self._controller.take_connect_result()
        if result is not None and result.error is not None:
            self._controller.note(f"подключиться не удалось: {result.error}")

        view = models.state_view(snapshot)
        self.state_label.setText(f"● {view.text}")
        self.state_label.setStyleSheet(f"color: {texts.TONE_COLORS[view.tone]}; font-weight: bold;")

        disconnected = snapshot.state is SessionState.DISCONNECTED
        # Пока поток подключения жив, состояние сессии ещё может быть
        # Disconnected — команда только собирается уйти. Кнопка при этом
        # обязана быть выключена, иначе повторное нажатие даст WRONG_STATE.
        idle_ui = disconnected and not snapshot.connecting
        self.connect_button.setEnabled(idle_ui)
        # Отключение во время `Probing` — гонка: команда уже в полёте.
        # Пока поток жив, кнопка выключена, а состояние показывает `Probing`.
        self.disconnect_button.setEnabled(not idle_ui and not snapshot.connecting)
        self.device_ip.setEnabled(idle_ui)
        self.device_port.setEnabled(idle_ui)
        self.local_port.setEnabled(idle_ui)
        self.start_button.setEnabled(snapshot.state is SessionState.IDLE)
        self.stop_button.setEnabled(snapshot.state is SessionState.STREAMING)

        lines = models.profile_mismatch_lines(snapshot)
        self.mismatch_label.setText("\n".join(lines))
        self.mismatch_label.setVisible(bool(lines))
        self.apply_profile_button.setVisible(bool(lines))
        self.apply_profile_button.setEnabled(idle_ui and not snapshot.recording)

        diagnosis = diagnostics.diagnose(
            snapshot.endpoint,
            stats=snapshot.transport,
            error=snapshot.last_error,
            local_addresses=self._local_addresses,
        )
        # Текст переставляется только при изменении: иначе десять раз в секунду
        # сбрасывалась бы позиция прокрутки и выделение.
        text = diagnostics.format_diagnosis(diagnosis)
        if text != self._diagnostics_text:
            self._diagnostics_text = text
            self.diagnostics_view.setPlainText(text)

        notices = "\n".join(snapshot.notices)
        if notices != self._notices_text:
            self._notices_text = notices
            self.notices_view.setPlainText(notices)
