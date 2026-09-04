"""Тесты сборки приложения: связывание, сверка профиля, порядок остановки.

Qt здесь нет и быть не должно. `fbg/ui/app.py` — это порядок действий,
а не окно, и весь он проверяется без экрана. Окно в песочнице открыть нельзя,
поэтому всё, что можно вынести из виджета, вынесено именно сюда.

Прибор подменяется симулятором `fbg.sim.device_sim`, как и в тестах сессии:
⚠️ тесты фиксируют поведение **приложения и симулятора**, а не факты о приборе.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from fbg.core.calibration import Sensor, SensorType
from fbg.core.endpoint import Endpoint
from fbg.core.pipeline import PipelineConfig, RingHistory
from fbg.core.profile import DeviceProfile
from fbg.core.session import Result, SessionConfig, SessionState
from fbg.io.config import AppConfig, load, save_sensors
from fbg.io.packet_log import Direction, PacketLogConfig
from fbg.io.recorder import RecorderConfig
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene
from fbg.ui.app import AppController

#: Короткие сроки: прогон не должен ждать реальные полсекунды на команду.
TEST_ENDPOINT_KWARGS: dict[str, object] = {
    "local_ip": "127.0.0.1",
    "local_port": 0,
    "read_timeout_s": 0.15,
    "write_timeout_s": 0.2,
    "retries": 1,
    "rx_poll_timeout_s": 0.02,
}

#: Keepalive практически выключен: служебный трафик мешал бы счётчикам.
QUIET = SessionConfig(
    keepalive_period_s=30.0,
    keepalive_failures_to_degrade=2,
    stream_stall_floor_s=0.3,
    stream_resume_wait_s=0.5,
    backoff_schedule=(0.05, 0.1, 0.2),
    retry_pause_s=0.02,
    reassembly_timeout_s=0.3,
    watchdog_tick_s=0.02,
    settle_before_readback_s=0.01,
    auto_reconnect=False,
)


class Rig:
    """Симулятор плюс контроллер, связанные через эфемерный порт.

    Порт открывается **до** подключения и сообщается симулятору: прибор
    отвечает на прописанный адрес назначения, а не на source-порт запроса
    (KB_01), поэтому адрес приёма нужен ему уже для ответа на `Stop`.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        device_profile: DeviceProfile | None = None,
        config_profile: DeviceProfile | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.device_profile = device_profile or DeviceProfile()
        self.sim = DeviceSimulator(
            profile=self.device_profile,
            scene=Scene(self.device_profile, [Grating(0, 0, 1545.0), Grating(1, 0, 1560.0)]),
            reply_to=("127.0.0.1", 1),
            frame_rate_hz=200.0,
        )
        self.sim.start()
        host, port = self.sim.address
        endpoint = Endpoint(device_ip=host, device_port=port, **TEST_ENDPOINT_KWARGS)  # type: ignore[arg-type]
        self.config_path = config_path
        config = AppConfig(
            endpoint=endpoint,
            profile=config_profile or self.device_profile,
            session=QUIET,
            pipeline=PipelineConfig(history_frames=512),
            recorder=RecorderConfig(directory=tmp_path / "data"),
            packet_log=PacketLogConfig(directory=None),
            calibration_path=tmp_path / "sensors.json",
        )
        self.controller = AppController(config, config_path=config_path)
        self.controller.start()
        self.controller.session._transport.open()
        self.sim.reply_to = self.controller.session.local_address

    def close(self) -> None:
        """Сначала замолкает прибор, потом гаснет приложение."""
        self.sim.stop()
        self.controller.shutdown()


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    """Стенд с совпадающей геометрией."""
    stand = Rig(tmp_path, config_path=tmp_path / "fbg_config.json")
    try:
        yield stand
    finally:
        stand.close()


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Ждёт условия. False — не дождались."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# --------------------------------------------------------------------------------------
# Сборка
# --------------------------------------------------------------------------------------


def test_контроллер_связывает_все_компоненты(tmp_path: Path) -> None:
    """Телеметрия идёт в pipeline, а байты обмена — в журнал (Р54)."""
    controller = AppController(AppConfig(packet_log=PacketLogConfig(directory=None)))
    session = controller.session
    assert session._on_telemetry == controller.pipeline.on_telemetry
    assert session._on_stream_gap == controller._on_stream_gap
    assert session._log_rx == controller.packet_log.log_rx
    assert session._log_tx == controller.packet_log.log_tx


def test_журнал_и_тракт_поднимаются_до_подключения(tmp_path: Path) -> None:
    """Журнал обязан записать сам обмен подключения, значит стартует раньше него."""
    controller = AppController(AppConfig(packet_log=PacketLogConfig(directory=None)))
    controller.start()
    try:
        assert controller.packet_log.is_running
        assert controller.pipeline.is_running
        assert controller.session.state is SessionState.DISCONNECTED
    finally:
        controller.shutdown()


def test_компоненты_строятся_по_профилю_настроек(tmp_path: Path) -> None:
    """Кольцо и буферы берут размеры из профиля, а не из умолчаний классов."""
    profile = DeviceProfile(channels=2, fbg_per_channel=8)
    controller = AppController(
        AppConfig(profile=profile, packet_log=PacketLogConfig(directory=None))
    )
    assert controller.pipeline.profile is profile
    assert controller.pipeline.history.freq_ghz.shape[1:] == (2, 8)


def test_снимок_не_содержит_живых_объектов_ядра(rig: Rig) -> None:
    """UI читает снимки и не держит кольцо (Р36)."""
    snapshot = rig.controller.snapshot()
    values = vars(snapshot).values()
    assert not any(isinstance(value, RingHistory) for value in values)
    assert not any(hasattr(value, "on_telemetry") for value in values)


def _temperature_sensor(sensor_id: str = "T1", *, expected_nm: float = 1545.0) -> Sensor:
    return Sensor(
        id=sensor_id,
        name="Температура",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=expected_nm,
        window_nm=0.20,
        value0=25.0,
        k1=100.0,
    )


def test_контроллер_загружает_датчики_при_старте(tmp_path: Path) -> None:
    """Слой калибровки действительно подключён, а не остаётся мёртвым модулем."""
    calibration_path = tmp_path / "sensors.json"
    saved = (_temperature_sensor(),)
    save_sensors(saved, calibration_path)
    controller = AppController(
        AppConfig(
            calibration_path=calibration_path,
            packet_log=PacketLogConfig(directory=None),
        )
    )
    try:
        assert controller.sensors == saved
        assert controller.snapshot(include_sensor_data=False).sensors == saved
    finally:
        controller.shutdown()


def test_датчики_считаются_только_когда_их_просит_ui(tmp_path: Path) -> None:
    """Р75: 2 кГц тракт не калибрует; расчёт делает snapshot панели датчиков."""
    calibration_path = tmp_path / "sensors.json"
    save_sensors((_temperature_sensor(),), calibration_path)
    stand = Rig(tmp_path)
    # Rig создаёт свой путь; явно заменяем набор тем же публичным методом.
    stand.controller.replace_sensors((_temperature_sensor(),))
    try:
        assert stand.controller.connect().ok
        assert stand.controller.start_stream().ok
        assert wait_until(lambda: stand.controller.pipeline.sequence >= 5)
        before = stand.controller._sensor_last_ui_seq

        hidden = stand.controller.snapshot(include_sensor_data=False)
        assert hidden.sensor_readings == () and hidden.sensor_history is None
        assert stand.controller._sensor_last_ui_seq == before

        visible = stand.controller.snapshot(include_sensor_data=True)
        assert len(visible.sensor_readings) == 1
        assert visible.sensor_readings[0].value == pytest.approx(25.0, abs=0.5)
        assert visible.sensor_history is not None and visible.sensor_history.frames == 1
        same = stand.controller.snapshot(include_sensor_data=True)
        assert same.sensor_history is not None and same.sensor_history.frames == 1
    finally:
        stand.close()


def test_сохранение_датчика_проверяет_пересечение_окон(rig: Rig) -> None:
    """Невалидный набор не попадает ни в память, ни в sensors.json."""
    first = _temperature_sensor("A", expected_nm=1545.0)
    rig.controller.replace_sensors((first,))
    path = rig.controller.config.calibration_path
    before = path.read_text(encoding="utf-8")
    overlapping = _temperature_sensor("B", expected_nm=1545.1)

    with pytest.raises(ValueError, match="пересекаются"):
        rig.controller.upsert_sensor(overlapping)

    assert rig.controller.sensors == (first,)
    assert path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------------------
# Сверка профиля с прибором — первое, что делает приложение
# --------------------------------------------------------------------------------------


def test_совпавшая_геометрия_не_даёт_расхождения(rig: Rig) -> None:
    """Профиль настроек совпал с прибором — сообщать не о чем."""
    assert rig.controller.connect().ok
    assert rig.controller.profile_mismatch == ()


def test_после_подключения_запоминается_идентификация_прибора(rig: Rig) -> None:
    """Серийный номер и прошивка едут в настройки: со следующего запуска
    шапка журнала верна с первого байта, а не с `unknown`."""
    device = rig.controller.connect().unwrap()
    config = rig.controller.config
    assert config.serial == device.serial
    assert config.firmware == device.version
    # Подстановка идентификации в шапки файлов — методами AppConfig, а не
    # копией поля в каждой секции (иначе serial разъехался бы по разделам).
    assert config.packet_log_config().serial == device.serial
    assert config.recorder_config().firmware == device.version


def test_настройки_сохраняются_на_диск_после_подключения(rig: Rig) -> None:
    """Идентификация прибора доживает до следующего запуска."""
    device = rig.controller.connect().unwrap()
    assert rig.config_path is not None and rig.config_path.exists()
    saved = json.loads(rig.config_path.read_text(encoding="utf-8"))
    assert saved["serial"] == device.serial
    assert saved["firmware"] == device.version


def test_ожидаемый_темп_задаётся_из_ответа_прибора(rig: Rig) -> None:
    """`10 04` сообщает скорость развёртки — до подключения её знать неоткуда."""
    assert rig.controller.pipeline.metrics().expected_rate_hz is None
    rig.controller.connect().unwrap()
    assert rig.controller.pipeline.metrics().expected_rate_hz == 2000


def test_расхождение_геометрии_сообщается_поимённо(tmp_path: Path) -> None:
    """Прибор сказал одно, настройки помнят другое — это видно построчно."""
    stand = Rig(
        tmp_path,
        config_profile=DeviceProfile(fbg_per_channel=25, peak_gap_ghz=40),
        config_path=tmp_path / "fbg_config.json",
    )
    try:
        assert stand.controller.connect().ok
        fields = {difference.field for difference in stand.controller.profile_mismatch}
        assert fields == {"fbg_per_channel", "peak_gap_ghz"}
        difference = next(
            item for item in stand.controller.profile_mismatch if item.field == "fbg_per_channel"
        )
        assert (difference.configured, difference.device) == (25, 30)
    finally:
        stand.close()


def test_расхождение_геометрии_не_перезаписывает_настройки_молча(tmp_path: Path) -> None:
    """Другая геометрия означает другой прибор либо испорченный файл.

    Ни профиль, ни идентификация в файл не уезжают: решение принимает человек.
    """
    config_path = tmp_path / "fbg_config.json"
    stand = Rig(
        tmp_path,
        config_profile=DeviceProfile(fbg_per_channel=25),
        config_path=config_path,
    )
    try:
        stand.controller.connect().unwrap()
        assert stand.controller.profile_mismatch
        assert stand.controller.config.profile.fbg_per_channel == 25
        assert stand.controller.config.serial is None
        assert not config_path.exists()
    finally:
        stand.close()


def test_расхождение_попадает_в_сообщения(tmp_path: Path) -> None:
    """Молча расхождение не проходит — оно и в сообщениях панели тоже."""
    stand = Rig(tmp_path, config_profile=DeviceProfile(fbg_per_channel=25))
    try:
        stand.controller.connect().unwrap()
        assert any("fbg_per_channel" in notice for notice in stand.controller.notices)
    finally:
        stand.close()


def test_принять_геометрию_прибора_можно_явно(tmp_path: Path) -> None:
    """Явное согласие переписывает профиль и пересобирает тракт."""
    config_path = tmp_path / "fbg_config.json"
    stand = Rig(
        tmp_path,
        config_profile=DeviceProfile(fbg_per_channel=25),
        config_path=config_path,
    )
    try:
        device = stand.controller.connect().unwrap()
        old_pipeline = stand.controller.pipeline
        stand.controller.disconnect()
        stand.controller.apply_device_profile()

        assert stand.controller.config.profile.fbg_per_channel == 30
        assert stand.controller.config.serial == device.serial
        assert stand.controller.profile_mismatch == ()
        # Кольцо и буфер кадра построены заново: подменить профиль
        # на живых объектах нельзя, у них уже выделены массивы.
        assert stand.controller.pipeline is not old_pipeline
        assert stand.controller.pipeline.history.freq_ghz.shape[1:] == (4, 30)
        assert stand.controller.pipeline.is_running
        # В файл пишутся отклонения от умолчаний, а 30 — умолчание, поэтому
        # проверяется круговорот, а не буквальное содержимое секции.
        assert load(config_path).config.profile.fbg_per_channel == 30
    finally:
        stand.close()


def test_принять_геометрию_нельзя_при_открытой_связи(tmp_path: Path) -> None:
    """Компоненты пересобираются, а сессия в это время работать не может."""
    stand = Rig(tmp_path, config_profile=DeviceProfile(fbg_per_channel=25))
    try:
        stand.controller.connect().unwrap()
        with pytest.raises(RuntimeError, match="закрытой связи"):
            stand.controller.apply_device_profile()
    finally:
        stand.close()


def test_принять_геометрию_нельзя_во_время_записи(tmp_path: Path) -> None:
    """Курсор писателя смотрит в старое кольцо — сначала запись, потом профиль."""
    stand = Rig(tmp_path, config_profile=DeviceProfile(fbg_per_channel=25))
    try:
        stand.controller.connect().unwrap()
        stand.controller.start_recording()
        stand.controller.disconnect()
        with pytest.raises(RuntimeError, match="записи"):
            stand.controller.apply_device_profile()
    finally:
        stand.controller.stop_recording()
        stand.close()


def test_смена_адресов_пересобирает_сессию(tmp_path: Path) -> None:
    """Новый `Endpoint` — новый сокет, значит и новая сессия."""
    stand = Rig(tmp_path, config_path=tmp_path / "fbg_config.json")
    try:
        old_session = stand.controller.session
        endpoint = Endpoint(
            device_ip="192.168.0.19",
            device_port=4567,
            **TEST_ENDPOINT_KWARGS,  # type: ignore[arg-type]
        )
        stand.controller.set_endpoint(endpoint)
        assert stand.controller.session is not old_session
        assert stand.controller.config.endpoint.device_ip == "192.168.0.19"
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Порядок остановки
# --------------------------------------------------------------------------------------


def _recording_controller(order: list[str]) -> AppController:
    """Контроллер, у которого каждый шаг остановки только отмечается."""
    controller = AppController(AppConfig(packet_log=PacketLogConfig(directory=None)))
    controller._stop_device = lambda: order.append("stop")  # type: ignore[method-assign]
    controller._session.disconnect = lambda: order.append("session")  # type: ignore[method-assign]
    controller.stop_recording = lambda: order.append("recorder")  # type: ignore[method-assign]
    controller._packet_log.stop = lambda: order.append("log")  # type: ignore[method-assign]
    controller._pipeline.stop = lambda: order.append("pipeline")  # type: ignore[method-assign]
    return controller


def test_порядок_остановки_соблюдается() -> None:
    """Stop прибору → сессия → recorder → журнал → pipeline.

    Источник замолкает первым, писатели дописывают хвосты после закрытия
    связи, и последним гаснет кольцо, из которого они тянут.
    """
    order: list[str] = []
    assert _recording_controller(order).shutdown() == ()
    assert order == ["stop", "session", "recorder", "log", "pipeline"]


@pytest.mark.parametrize("broken", ["stop", "session", "recorder", "log", "pipeline"])
def test_отказ_одного_шага_не_мешает_остальным(broken: str) -> None:
    """Каждый шаг идёт в своём `try`: отказ одного не оставляет прибор в потоке."""
    order: list[str] = []
    controller = _recording_controller(order)

    def explode() -> None:
        order.append(broken)
        raise OSError(f"отказ шага {broken}")

    owner, attribute = {
        "stop": (controller, "_stop_device"),
        "session": (controller._session, "disconnect"),
        "recorder": (controller, "stop_recording"),
        "log": (controller._packet_log, "stop"),
        "pipeline": (controller._pipeline, "stop"),
    }[broken]
    setattr(owner, attribute, explode)

    failures = controller.shutdown()
    # Главное утверждение теста: сломанный шаг не срезал остальные.
    assert order == ["stop", "session", "recorder", "log", "pipeline"]
    assert len(failures) == 1
    assert failures[0].error.endswith(f"отказ шага {broken}")
    assert any("отказ шага" in notice for notice in controller.notices)


def test_отказ_остановки_не_проходит_молча() -> None:
    """Отказ обязан быть виден: и в возвращаемом списке, и в сообщениях."""
    order: list[str] = []
    controller = _recording_controller(order)

    def explode() -> None:
        raise RuntimeError("диск отвалился")

    controller._packet_log.stop = explode  # type: ignore[method-assign]
    failures = controller.shutdown()
    assert [failure.step for failure in failures] == ["журнал пакетов"]
    assert "диск отвалился" in failures[0].error
    assert any("диск отвалился" in notice for notice in controller.notices)


def test_остановка_гасит_поток_прибора(rig: Rig) -> None:
    """Правило KB_05 №6: `Stop` уходит и при штатном завершении."""
    rig.controller.connect().unwrap()
    assert rig.controller.start_stream().ok
    assert wait_until(lambda: rig.controller.pipeline.metrics().frames > 0)
    assert rig.controller.shutdown() == ()
    assert rig.controller.session.state is SessionState.DISCONNECTED
    assert not rig.controller.pipeline.is_running
    assert not rig.controller.packet_log.is_running


# --------------------------------------------------------------------------------------
# Фоновое подключение
# --------------------------------------------------------------------------------------


def test_фоновое_подключение_не_блокирует_вызывающего(rig: Rig) -> None:
    """Окно не должно замирать на время `Probing`.

    Обычный `threading.Thread`, результат которого забирает существующий
    таймер, — тот же приём, каким UI читает снимки pipeline.
    """
    assert rig.controller.connect_async()
    assert wait_until(lambda: not rig.controller.is_connecting)
    result = rig.controller.take_connect_result()
    assert result is not None and result.ok
    assert rig.controller.session.state is SessionState.IDLE


def test_результат_подключения_отдаётся_один_раз(rig: Rig) -> None:
    """Иначе один и тот же отказ повторялся бы в сообщениях каждый такт."""
    rig.controller.connect_async()
    rig.controller.join_connect()
    assert rig.controller.take_connect_result() is not None
    assert rig.controller.take_connect_result() is None


def test_повторное_нажатие_во_время_подключения_игнорируется(rig: Rig) -> None:
    """Вторая попытка получила бы `WRONG_STATE` за нажатие, сделанное один раз."""
    rig.sim.stop()
    assert rig.controller.connect_async()
    assert not rig.controller.connect_async()
    rig.controller.join_connect()


def test_пока_подключение_идёт_снимок_это_показывает(rig: Rig) -> None:
    """Панель гасит кнопки по `connecting`, а не только по состоянию сессии.

    Между нажатием и переходом в `Probing` состояние ещё `Disconnected`,
    и без этого признака кнопка успела бы стать нажимаемой снова.
    """
    rig.sim.stop()
    rig.controller.connect_async()
    assert rig.controller.snapshot().connecting
    rig.controller.join_connect()
    assert not rig.controller.snapshot().connecting


def test_остановка_дожидается_фонового_подключения(rig: Rig) -> None:
    """`Stop` не должен уйти одновременно с чтением конфигурации."""
    rig.controller.connect_async()
    assert rig.controller.shutdown() == ()
    assert not rig.controller.is_connecting


# --------------------------------------------------------------------------------------
# Запись и журнал
# --------------------------------------------------------------------------------------


def test_запись_поднимается_после_опроса(rig: Rig) -> None:
    """Шапка файла обязана нести серийный номер, а он известен после `Probing`."""
    device = rig.controller.connect().unwrap()
    recorder = rig.controller.start_recording()
    try:
        assert recorder.config.serial == device.serial
        assert rig.controller.start_recording() is recorder
        assert rig.controller.is_recording
    finally:
        rig.controller.stop_recording()
    assert not rig.controller.is_recording


def test_запись_переживает_обрыв_и_получает_один_gap(rig: Rig) -> None:
    """Recorder не останавливается при N11, а границы паузы попадают в файл."""
    rig.controller.connect().unwrap()
    assert rig.controller.start_stream().ok
    assert wait_until(lambda: rig.controller.session.stats().telemetry_frames > 10)
    recorder = rig.controller.start_recording()
    assert wait_until(lambda: recorder.stats.rows > 3)

    rows_before = recorder.stats.rows
    rig.sim.go_silent(0.6)
    assert wait_until(lambda: rig.controller.session.state is SessionState.DEGRADED)
    assert rig.controller.is_recording
    assert wait_until(lambda: rig.controller.session.state is SessionState.STREAMING)
    assert rig.controller.is_recording
    assert wait_until(lambda: recorder.stats.rows > rows_before + 3)

    assert rig.controller.stop_stream().ok
    rig.controller.stop_recording()
    paths = sorted(recorder.config.directory.glob("data_*.csv"))
    assert len(paths) == 1
    lines = paths[0].read_text(encoding="ascii").splitlines()
    gap_indexes = [index for index, line in enumerate(lines) if line.startswith("# GAP")]
    assert len(gap_indexes) == 1
    gap_index = gap_indexes[0]
    gap = lines[gap_index]
    assert "frames=unknown" in gap

    before = next(line for line in reversed(lines[:gap_index]) if line[:1].isdigit())
    after = next(line for line in lines[gap_index + 1 :] if line[:1].isdigit())
    before_t = float(before.split(";")[1])
    after_t = float(after.split(";")[1])
    assert f"t_mono_from={before_t:.6f}" in gap
    assert f"t_mono_to={after_t:.6f}" in gap
    assert recorder.stats.gaps == 1
    assert recorder.stats.lost_frames == 0


def test_журнал_видит_обмен_подключения(rig: Rig) -> None:
    """Первой командой в журнале стоит `Stop`, отправленный при подключении.

    ⚠️ Проверяется первая запись **направления TX**, а не первая запись вообще.
    `log_tx` зовётся **после** успешной отправки (журнал ведёт запись провода),
    а `log_rx` — из потока-диспетчера. На loopback ответ возвращается за
    микросекунды и успевает попасть в очередь журнала раньше, чем отправитель
    доберётся до `log_tx`, поэтому номера `seq` не являются порядком провода.
    На реальной сети такого не будет — там ответ идёт миллисекунды, — но
    закреплять несуществующую гарантию тестом нельзя.
    """
    rig.controller.connect().unwrap()
    assert wait_until(lambda: len(rig.controller.packet_records()) > 2)
    outgoing = rig.controller.packet_records(direction=Direction.TX)
    assert outgoing
    assert outgoing[0].data[:2] == bytes([0x30, 0x01])


def test_фильтр_записей_журнала(rig: Rig) -> None:
    """Панель просит отфильтрованный снимок, а не фильтрует кольцо сама."""
    rig.controller.connect().unwrap()
    assert wait_until(lambda: len(rig.controller.packet_records()) > 2)
    everything = rig.controller.packet_records()
    outgoing = rig.controller.packet_records(id_fc=(0x10, 0x01))
    assert 0 < len(outgoing) < len(everything)
    assert all(record.id_fc == (0x10, 0x01) for record in outgoing)


def test_экспорт_журнала_пишет_файл(rig: Rig, tmp_path: Path) -> None:
    """Экспорт выгружает кольцо с применённым фильтром."""
    rig.controller.connect().unwrap()
    assert wait_until(lambda: len(rig.controller.packet_records()) > 2)
    target = tmp_path / "export.log"
    written = rig.controller.export_packets(target)
    assert written > 0
    assert target.read_bytes().decode("ascii").startswith("seq;dir;")


# --------------------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------------------


def test_замечания_настроек_попадают_в_сообщения(tmp_path: Path) -> None:
    """Битый файл настроек не роняет старт, но и молчать о себе не должен."""
    path = tmp_path / "fbg_config.json"
    path.write_text("{ это не json", encoding="utf-8")
    loaded = load(path)
    controller = AppController(loaded.config, config_path=path, issues=loaded.issues)
    try:
        assert controller.notices
        assert any("JSON" in notice for notice in controller.notices)
    finally:
        controller.shutdown()


def test_отказ_сохранения_настроек_не_роняет_подключение(tmp_path: Path) -> None:
    """Не записавшиеся настройки — сообщение, а не потеря установленной связи.

    `config_module.save` намеренно не глотает `OSError`: пользователь обязан
    узнать об отказе. Но узнать он должен сообщением — связь к этому моменту
    уже работает, и рвать её из-за файла настроек было бы хуже.
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("это файл, а не каталог", encoding="utf-8")
    stand = Rig(tmp_path, config_path=blocked / "fbg_config.json")
    try:
        assert stand.controller.connect().ok
        assert stand.controller.session.state is SessionState.IDLE
        assert any("не сохранены" in notice for notice in stand.controller.notices)
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Спектр: режим 30 07 вытесняет поток и приложение обязано вернуть его
# --------------------------------------------------------------------------------------


def test_спектр_во_время_потока_полностью_перезапускает_поток(rig: Rig) -> None:
    rig.controller.connect().unwrap()
    assert rig.controller.start_stream().ok
    assert wait_until(lambda: rig.controller.session.stats().telemetry_frames > 3)
    before = rig.controller.session.stats().telemetry_frames
    before_version = rig.controller.snapshot().spectrum_version
    spectrum = rig.controller.take_spectrum(0, 3000).unwrap()
    assert spectrum.adc.size == rig.controller.config.profile.adc_points
    assert rig.controller.snapshot().spectrum_version == before_version + 1
    assert rig.controller.session.state is SessionState.STREAMING
    assert wait_until(lambda: rig.controller.session.stats().telemetry_frames > before)
    snap = rig.controller.snapshot()
    assert snap.last_spectrum_max_adc == spectrum.max_adc
    assert snap.last_spectrum_saturated_points == spectrum.saturated_points


def test_спектр_из_idle_после_30_07_обязательно_посылает_stop(rig: Rig) -> None:
    rig.controller.connect().unwrap()
    before = len(rig.controller.packet_records(direction=Direction.TX))
    rig.controller.take_spectrum(0, 3000).unwrap()
    outgoing = rig.controller.packet_records(direction=Direction.TX)[before:]
    assert [record.id_fc for record in outgoing] == [(0x30, 0x07), (0x30, 0x01)]
    assert rig.controller.session.state is SessionState.IDLE


def test_одиночный_спектр_не_блокирует_вызывающий_поток(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig.controller.connect().unwrap()
    entered = threading.Event()
    release = threading.Event()

    def slow_take(_channel: int, _threshold_adc: int = 3000) -> Result[None]:
        entered.set()
        assert release.wait(2.0)
        return Result(value=None)

    monkeypatch.setattr(rig.controller, "take_spectrum", slow_take)
    assert rig.controller.take_spectrum_async(0, 3000)
    assert entered.wait(1.0)
    assert rig.controller.spectrum_busy
    assert not rig.controller.take_spectrum_async(0, 3000)
    release.set()
    assert wait_until(lambda: not rig.controller.spectrum_busy)


def test_debug_из_idle_после_30_03_обязательно_посылает_stop(rig: Rig) -> None:
    rig.controller.connect().unwrap()
    before = len(rig.controller.packet_records(direction=Direction.TX))
    spectra = rig.controller.take_debug_spectra(3000).unwrap()
    assert len(spectra) == rig.controller.config.profile.channels
    outgoing = rig.controller.packet_records(direction=Direction.TX)[before:]
    assert [record.id_fc for record in outgoing] == [(0x30, 0x03), (0x30, 0x01)]
    assert rig.controller.session.state is SessionState.IDLE


def test_спектр_во_время_записи_запрещён(rig: Rig) -> None:
    rig.controller.connect().unwrap()
    assert rig.controller.start_stream().ok
    assert wait_until(lambda: rig.controller.session.stats().telemetry_frames > 3)
    rig.controller.start_recording()
    try:
        with pytest.raises(RuntimeError, match="во время записи"):
            rig.controller.take_spectrum(0, 3000)
    finally:
        rig.controller.stop_recording()


def test_фактический_период_continuous_измеряется_по_завершённым_циклам(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KB_05 №38: UI получает факт, а не обратную величину заданного периода."""
    rig.controller.connect().unwrap()
    completed = 0

    def slow_take(_channel: int, _threshold_adc: int = 3000) -> Result[None]:
        nonlocal completed
        time.sleep(0.05)
        completed += 1
        return Result(value=None)

    monkeypatch.setattr(rig.controller, "take_spectrum", slow_take)
    assert rig.controller.start_spectrum_continuous(0, 0.01, 3000)
    assert wait_until(lambda: completed >= 3, timeout=2.0)
    assert wait_until(
        lambda: rig.controller.snapshot().spectrum_actual_period_s is not None, timeout=1.0
    )
    actual = rig.controller.snapshot().spectrum_actual_period_s
    rig.controller.stop_spectrum_continuous()

    assert actual is not None
    assert actual >= 0.045
    assert actual > rig.controller.snapshot().spectrum_period_s * 3


def test_непрерывный_спектр_делает_несколько_снимков_без_наложения(rig: Rig) -> None:
    rig.controller.connect().unwrap()
    assert rig.controller.start_stream().ok
    assert rig.controller.start_spectrum_continuous(0, 0.15, 3000)
    assert wait_until(lambda: rig.controller.snapshot().last_spectrum_max_adc is not None)
    first_commands = rig.controller.session.stats().commands
    assert wait_until(
        lambda: rig.controller.session.stats().commands > first_commands + 5, timeout=3.0
    )
    rig.controller.stop_spectrum_continuous()
    assert not rig.controller.spectrum_running
    assert rig.controller.session.state is SessionState.STREAMING
