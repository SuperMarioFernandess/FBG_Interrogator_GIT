"""Тесты окна и панелей. Маркер `ui`: требуется Qt.

Гоняются с `QT_QPA_PLATFORM=offscreen`. Окно при этом никто не видит, поэтому
проверяется **устройство**, а не вид: вкладки на месте, таймер стартует и
гаснет, кнопки включаются по состоянию сессии, таблица журнала не держит
кольцо. Как это выглядит — вопрос визуальной приёмки, и тестом он не
подменяется.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")

# Платформа выставляется до первого создания QApplication: в CI её задаёт
# окружение job'а, локально удобнее не забывать про неё вовсе.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import Direction, PacketLogConfig
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.main_window import UI_PERIOD_MS, MainWindow
from fbg.ui.panels.device_info import sections_shape

pytestmark = pytest.mark.ui


@pytest.fixture(scope="session")
def application() -> QApplication:
    """Одно `QApplication` на прогон: второе Qt создать не даст."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


@pytest.fixture
def controller() -> Iterator[AppController]:
    """Контроллер без сети и без файлов: журнал только в памяти."""
    app_controller = AppController(AppConfig(packet_log=PacketLogConfig(directory=None)))
    app_controller.start()
    try:
        yield app_controller
    finally:
        app_controller.shutdown()


@pytest.fixture
def window(application: QApplication, controller: AppController) -> Iterator[MainWindow]:
    """Окно поверх контроллера."""
    main = MainWindow(controller)
    try:
        yield main
    finally:
        main.stop_updates()
        main.close()
        main.deleteLater()


def push(controller: AppController, count: int, prefix: bytes = b"\x10\x04\x04\x00") -> None:
    """Кладёт записи в журнал и переносит их в кольцо без запуска потока."""
    for _ in range(count):
        controller.packet_log.log_tx(prefix)
    controller.packet_log.pump()


# --------------------------------------------------------------------------------------
# Каркас окна
# --------------------------------------------------------------------------------------


def test_окно_создаётся_с_семью_вкладками(window: MainWindow) -> None:
    """Вкладки на месте и подписаны по-русски."""
    titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert titles == [
        texts.TAB_CONNECTION,
        texts.TAB_MEASUREMENT,
        texts.TAB_SENSORS,
        texts.TAB_SPECTRUM,
        texts.TAB_DEVICE,
        texts.TAB_DEVICE_CONFIG,
        texts.TAB_PACKET_LOG,
    ]
    assert window.windowTitle() == texts.APP_TITLE
    assert len(window.panels) == 7


def test_таймер_запускается_и_гаснет(window: MainWindow) -> None:
    """Событийного обновления нет: при 2 кГц оно утопило бы UI."""
    assert not window.timer.isActive()
    window.start_updates()
    assert window.timer.isActive()
    assert window.timer.interval() == UI_PERIOD_MS
    window.stop_updates()
    assert not window.timer.isActive()


def test_закрытие_окна_гасит_таймер(window: MainWindow) -> None:
    """Иначе таймер продолжал бы дёргать панели закрытого окна."""
    window.start_updates()
    window.close()
    assert not window.timer.isActive()


def test_такт_обновляет_строку_состояния(window: MainWindow) -> None:
    """Строка состояния — то, что видно, не открывая вкладок."""
    window.tick()
    assert texts.STATE_LABELS[SessionState.DISCONNECTED][0] in window.status_label.text()


def test_такт_берёт_один_снимок_на_все_панели(
    window: MainWindow, controller: AppController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Иначе состояние в строке могло бы разойтись с состоянием на панели."""
    calls = 0
    original = controller.snapshot

    def counted(**kwargs: object) -> models.AppSnapshot:
        nonlocal calls
        calls += 1
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(controller, "snapshot", counted)
    window.tick()
    assert calls == 1


def test_такт_обновляет_только_видимую_панель(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Р76: скрытые графики и таблицы не получают бесполезные refresh 10 Гц."""
    calls = {panel: 0 for panel in window.panels}
    for panel in window.panels:
        original = panel.refresh

        def counted(snapshot: models.AppSnapshot, *, _panel=panel, _original=original) -> None:
            calls[_panel] += 1
            _original(snapshot)

        monkeypatch.setattr(panel, "refresh", counted)

    window.tabs.setCurrentWidget(window.packet_log_panel)
    calls = dict.fromkeys(window.panels, 0)
    window.tick()

    assert calls[window.packet_log_panel] == 1
    assert sum(calls.values()) == 1


def test_история_копируется_только_для_видимого_графика(
    window: MainWindow, controller: AppController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Скрытая вкладка измерения не копирует 20 000×120 чисел из кольца."""
    calls: list[tuple[bool, bool]] = []
    original = controller.snapshot

    def counted(
        *, include_trace_history: bool = True, include_sensor_data: bool = True
    ) -> models.AppSnapshot:
        calls.append((include_trace_history, include_sensor_data))
        return original(
            include_trace_history=include_trace_history, include_sensor_data=include_sensor_data
        )

    monkeypatch.setattr(controller, "snapshot", counted)
    window.tabs.setCurrentWidget(window.packet_log_panel)
    calls.clear()
    window.tick()
    assert calls == [(False, False)]

    window.tabs.setCurrentWidget(window.measurement_panel)
    calls.clear()
    window.tick()
    assert calls == [(True, False)]


# --------------------------------------------------------------------------------------
# Панель подключения
# --------------------------------------------------------------------------------------


def test_кнопки_включаются_по_состоянию_сессии(window: MainWindow) -> None:
    """Нажать то, что сессия отвергнет решением Р25, быть не должно."""
    panel = window.connection_panel
    panel.refresh(
        models.AppSnapshot(
            endpoint=window._controller.config.endpoint,
            profile=DeviceProfile(),
            state=SessionState.DISCONNECTED,
        )
    )
    assert panel.connect_button.isEnabled()
    assert not panel.disconnect_button.isEnabled()
    assert not panel.start_button.isEnabled()
    assert not panel.stop_button.isEnabled()


def test_поток_запускается_только_из_idle(window: MainWindow) -> None:
    """`start_stream` допустим из Idle, `stop_stream` осмыслен в Streaming."""
    panel = window.connection_panel
    endpoint = window._controller.config.endpoint

    def show(state: SessionState) -> None:
        panel.refresh(models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=state))

    show(SessionState.IDLE)
    assert panel.start_button.isEnabled() and not panel.stop_button.isEnabled()
    show(SessionState.STREAMING)
    assert not panel.start_button.isEnabled() and panel.stop_button.isEnabled()
    show(SessionState.DEGRADED)
    assert not panel.start_button.isEnabled() and not panel.stop_button.isEnabled()


def test_адрес_приёма_не_редактируется(window: MainWindow) -> None:
    """Р29: приём обязан оставаться на `0.0.0.0`, и поле это не предлагает менять."""
    panel = window.connection_panel
    assert panel.local_ip.isReadOnly()
    assert panel.local_ip.text() == "0.0.0.0"
    assert texts.LOCAL_IP_LOCKED in panel.local_ip.toolTip()


def test_индикатор_состояния_меняет_цвет(window: MainWindow) -> None:
    """Degraded не должен выглядеть как Disconnected."""
    panel = window.connection_panel
    endpoint = window._controller.config.endpoint

    def style(state: SessionState) -> str:
        panel.refresh(models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=state))
        return panel.state_label.styleSheet()

    assert style(SessionState.DEGRADED) != style(SessionState.DISCONNECTED)
    assert texts.TONE_COLORS[texts.Tone.WARN] in style(SessionState.RECONNECTING)


def test_восстановление_можно_явно_отменить(window: MainWindow) -> None:
    """Во время Degraded/Reconnecting та же кнопка честно называется отменой."""
    panel = window.connection_panel
    endpoint = window._controller.config.endpoint
    for state in (SessionState.DEGRADED, SessionState.RECONNECTING):
        panel.refresh(models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=state))
        assert panel.disconnect_button.isEnabled()
        assert panel.disconnect_button.text() == texts.BUTTON_CANCEL_RECOVERY

    panel.refresh(
        models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=SessionState.IDLE)
    )
    assert panel.disconnect_button.text() == texts.BUTTON_DISCONNECT


def test_расхождение_профиля_показывается_и_прячется(window: MainWindow) -> None:
    """Пока расхождения нет, места оно не занимает; появилось — видно сразу."""
    panel = window.connection_panel
    endpoint = window._controller.config.endpoint
    panel.refresh(
        models.AppSnapshot(endpoint=endpoint, profile=DeviceProfile(), state=SessionState.IDLE)
    )
    assert not panel.mismatch_label.isVisibleTo(panel)
    panel.refresh(
        models.AppSnapshot(
            endpoint=endpoint,
            profile=DeviceProfile(),
            state=SessionState.DISCONNECTED,
            profile_mismatch=(models.ProfileDifference("fbg_per_channel", 25, 30),),
        )
    )
    assert panel.mismatch_label.isVisibleTo(panel)
    assert "fbg_per_channel" in panel.mismatch_label.text()
    assert panel.apply_profile_button.isEnabled()


def test_диагностика_наполняется_текстом(window: MainWindow) -> None:
    """«Ошибка подключения» человеку за стендом не говорит ничего."""
    window.tick()
    text = window.connection_panel.diagnostics_view.toPlainText()
    assert "Wireshark" in text
    assert texts.EXPECTED_LOCAL_IP in text or "определить не удалось" in text


# --------------------------------------------------------------------------------------
# Панель прибора
# --------------------------------------------------------------------------------------


def test_дерево_прибора_строится_по_модели(window: MainWindow) -> None:
    """Группы и строки те же, что отдала модель."""
    window.tick()
    tree = window.device_panel.tree
    sections = models.device_sections(window._controller.snapshot())
    assert tree.topLevelItemCount() == len(sections)
    first = tree.topLevelItem(0)
    assert first is not None and first.text(0) == texts.SECTION_DEVICE
    assert first.childCount() == len(sections[0].rows)


def test_дерево_не_пересобирается_на_каждом_такте(window: MainWindow) -> None:
    """Дерево, пересобираемое десять раз в секунду, теряет прокрутку и раскрытие.

    То есть смотреть на него невозможно ровно тогда, когда это нужно.
    """
    window.tick()
    tree = window.device_panel.tree
    item = tree.topLevelItem(0)
    window.tick()
    assert tree.topLevelItem(0) is item


def test_форма_модели_меняется_когда_прибор_опрошен() -> None:
    """Строки каналов появляются после опроса — тогда дерево и пересобирается."""
    from tests.test_ui_models import DEVICE, snapshot

    without = sections_shape(models.device_sections(snapshot()))
    with_device = sections_shape(models.device_sections(snapshot(device=DEVICE)))
    assert without != with_device


# --------------------------------------------------------------------------------------
# Панель журнала
# --------------------------------------------------------------------------------------


def test_таблица_журнала_наполняется(window: MainWindow, controller: AppController) -> None:
    """Записи попадают в таблицу через снимок кольца."""
    push(controller, 3)
    window.tabs.setCurrentWidget(window.packet_log_panel)
    window.tick()
    model = window.packet_log_panel.model
    assert model.rowCount() == 3
    assert model.columnCount() == len(texts.LOG_COLUMNS)
    index = model.index(0, 5)
    assert model.data(index) == "10 04"


def test_панель_журнала_не_держит_кольцо(window: MainWindow, controller: AppController) -> None:
    """UI читает снимки и ничего из ядра не держит (Р36).

    После такта в журнал добавляются записи; таблица обязана остаться такой,
    какой была, — иначе у неё в руках само кольцо, а не его копия, и оно
    менялось бы под ней прямо во время отрисовки.
    """
    push(controller, 2)
    window.tabs.setCurrentWidget(window.packet_log_panel)
    window.tick()
    model = window.packet_log_panel.model
    assert model.rowCount() == 2
    held = model.records
    push(controller, 5)
    assert model.rowCount() == 2
    assert model.records is held
    assert model.records is not controller.packet_log._ring
    window.tick()
    assert model.rowCount() == 7


def test_фильтр_по_направлению(window: MainWindow, controller: AppController) -> None:
    """Фильтр применяется к снимку, а не к кольцу."""
    push(controller, 2)
    controller.packet_log.log_rx(b"\x10\x04\x00\x0c\x00\xca\x00\x04\x00\x1e\x00\x1e", 1.0)
    controller.packet_log.pump()
    panel = window.packet_log_panel
    window.tabs.setCurrentWidget(panel)
    window.tick()
    assert panel.model.rowCount() == 3
    panel.direction_box.setCurrentIndex(panel.direction_box.findData(Direction.RX.value))
    window.tick()
    assert panel.model.rowCount() == 1
    assert panel.model.data(panel.model.index(0, 1)) == "RX"


def test_фильтр_по_паре_не_сбрасывается_на_такте(
    window: MainWindow, controller: AppController
) -> None:
    """Выбор пользователя не должен слетать десять раз в секунду."""
    push(controller, 2)
    push(controller, 1, prefix=b"\x30\x01\x06\x00\x00\x00")
    panel = window.packet_log_panel
    window.tabs.setCurrentWidget(panel)
    window.tick()
    index = panel.pair_box.findData(models.format_id_fc_pair((0x30, 0x01)))
    assert index > 0
    panel.pair_box.setCurrentIndex(index)
    window.tick()
    assert panel.selected_pair() == (0x30, 0x01)
    assert panel.model.rowCount() == 1
    push(controller, 1, prefix=b"\x10\x05\x04\x00")
    window.tick()
    assert panel.selected_pair() == (0x30, 0x01)


def test_пауза_останавливает_обновление(window: MainWindow, controller: AppController) -> None:
    """Журнал при этом продолжает писаться — стоит только таблица."""
    push(controller, 2)
    window.tabs.setCurrentWidget(window.packet_log_panel)
    window.tick()
    panel = window.packet_log_panel
    panel.pause_box.setChecked(True)
    push(controller, 3)
    window.tick()
    assert panel.model.rowCount() == 2
    panel.pause_box.setChecked(False)
    window.tick()
    assert panel.model.rowCount() == 5


def test_экспорт_пишет_файл_с_фильтром(
    window: MainWindow, controller: AppController, tmp_path: Path
) -> None:
    """Выгружается то, что видно в панели, вместе с применённым фильтром."""
    push(controller, 2)
    controller.packet_log.log_rx(b"\x10\x04\x00\x0c\x00\xca\x00\x04\x00\x1e\x00\x1e", 1.0)
    controller.packet_log.pump()
    panel = window.packet_log_panel
    panel.direction_box.setCurrentIndex(panel.direction_box.findData(Direction.TX.value))
    target = tmp_path / "export.log"
    assert panel.export_to(target) == 2
    text = target.read_bytes().decode("ascii")
    assert text.startswith("seq;dir;")
    assert "dir=TX" in text
