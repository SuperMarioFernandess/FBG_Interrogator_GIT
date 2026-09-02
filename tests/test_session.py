"""Тесты сессии: автомат, корреляция, повторы, watchdog, верификация записи.

Всё гоняется против симулятора `fbg.sim.device_sim`: живого прибора нет.

⚠️ Тесты фиксируют поведение **сессии и симулятора**, а факты о приборе
берутся из захватов. D4 и N9 закрыты скринингом 01.09.2026: `20 06` молчит,
а во время потока группы 0x10/0x20 проходят, тогда как 0x30 вытесняет поток.
Тесты ниже лишь закрепляют перенос этих измеренных фактов в код.

Отдельно про D5: датаграммы «в несколько кусков» здесь производит
тестовый `FragmentingDevice`, а не наблюдение за прибором. Тест проверяет,
что сборка по объявленному LEN работает, а не что прибор режет ответ именно так.
"""

import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.endpoint import Endpoint
from fbg.core.frames import GainSetting, SweepConfig
from fbg.core.profile import DeviceProfile
from fbg.core.session import (
    DeviceConfig,
    Result,
    Session,
    SessionConfig,
    SessionError,
    SessionErrorKind,
    SessionState,
)
from fbg.sim import encode as sim_encode
from fbg.sim.device_sim import (
    FACTORY_SERIAL,
    FACTORY_SPEED_CODE,
    FACTORY_VERSION_RAW,
    DeviceSimulator,
)
from fbg.sim.scene import Grating, Scene

#: Щедрый предел ожидания в тестах: они не должны мигать на загруженной машине.
WAIT_TIMEOUT_S = 5.0

#: Темп телеметрии в функциональных тестах.
TEST_FRAME_RATE_HZ = 200.0

#: Сетевые параметры тестов: короткие сроки, чтобы прогон был быстрым.
#: `retries=1` означает одну **повторную** отправку, то есть две всего.
TEST_ENDPOINT_KWARGS = {
    "local_ip": "127.0.0.1",
    "local_port": 0,
    "read_timeout_s": 0.15,
    "write_timeout_s": 0.2,
    "retries": 1,
    "rx_poll_timeout_s": 0.02,
}

#: Сессия по умолчанию: keepalive практически отключён, чтобы служебный трафик
#: не мешал функциональным тестам. Watchdog проверяется отдельными тестами,
#: которые задают короткий период явно.
QUIET = SessionConfig(
    keepalive_period_s=30.0,
    keepalive_failures_to_degrade=2,
    stream_stall_floor_s=0.3,
    backoff_schedule=(0.05, 0.1, 0.2),
    retry_pause_s=0.02,
    reassembly_timeout_s=0.3,
    watchdog_tick_s=0.02,
    settle_before_readback_s=0.01,
)

#: Тот же профиль с включённым keepalive — для тестов watchdog.
WATCHFUL = SessionConfig(
    keepalive_period_s=0.05,
    keepalive_failures_to_degrade=2,
    stream_stall_floor_s=0.3,
    backoff_schedule=(0.05, 0.1, 0.2),
    retry_pause_s=0.02,
    reassembly_timeout_s=0.3,
    watchdog_tick_s=0.02,
    settle_before_readback_s=0.01,
)


def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_TIMEOUT_S) -> bool:
    """Ждёт выполнения условия. False — не дождались."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# --------------------------------------------------------------------------------------
# Симуляторы с особенностями
# --------------------------------------------------------------------------------------


class RecordingSimulator(DeviceSimulator):
    """Симулятор, запоминающий пары (ID, FC) всех принятых команд.

    Нужен там, где проверяется **порядок** команд: Stop первым при подключении
    и Stop последним при отключении.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.requests: list[tuple[int, int]] = []
        self._requests_lock = threading.Lock()
        self.drop_remaining = 0
        """Сколько ближайших команд оставить без ответа — стимул для retry."""

    def _handle(self, request: bytes) -> list[bytes]:
        if len(request) >= 2:
            with self._requests_lock:
                self.requests.append((request[0], request[1]))
        if self.drop_remaining > 0:
            self.drop_remaining -= 1
            return []
        return super()._handle(request)

    def seen(self) -> list[tuple[int, int]]:
        """Копия журнала принятых команд."""
        with self._requests_lock:
            return list(self.requests)


class LyingSimulator(RecordingSimulator):
    """Подтверждает запись `00 01`, но настройку не применяет.

    Ровно та ситуация, ради которой read-back и введён: «успех» без проверки
    чтением означал бы только то, что прибор ответил, а не то, что значение
    применилось.
    """

    def _apply_threshold(self, request: bytes) -> bool:
        return True

    def _apply_peak_gap(self, request: bytes) -> bool:
        return True


class FragmentingDevice:
    """Мини-прибор, режущий длинный ответ `30 07` на несколько датаграмм.

    ⚠️ Стимул, а не наблюдение. Сколькими датаграммами приходит ответ
    на самом деле — открытый вопрос D5; здесь проверяется только то, что
    сборка по объявленному LEN работает.

    Отвечает ещё на пять команд чтения и на Stop, чтобы через него проходило
    подключение. Ответы собираются функциями `fbg.sim.encode` — тем же
    независимым от кодека кодом, что и у полного симулятора (KB_05 №11).
    """

    def __init__(
        self, profile: DeviceProfile, chunk_bytes: int = 1400, telemetry_at: int | None = None
    ) -> None:
        self.profile = profile
        self.chunk_bytes = chunk_bytes
        self.telemetry_at = telemetry_at
        """Номер куска длинного ответа, перед которым вклинить кадр `30 02`."""
        self.reply_to: tuple[str, int] = ("127.0.0.1", 1)
        self.datagrams_sent = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.02)
        self._sock.bind(("127.0.0.1", 0))
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="frag-dev", daemon=True)
        self._thread.start()

    @property
    def address(self) -> tuple[str, int]:
        """Адрес, на который слать команды."""
        host, port = self._sock.getsockname()[:2]
        return str(host), int(port)

    def stop(self) -> None:
        """Останавливает поток и закрывает сокет."""
        self._shutdown.set()
        self._thread.join(timeout=2.0)
        self._sock.close()

    def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                request, _ = self._sock.recvfrom(4096)
            except (TimeoutError, OSError):
                continue
            for payload in self._respond(request):
                self._send(payload)

    def _respond(self, request: bytes) -> list[bytes]:
        if len(request) < 2:
            return []
        ident, fc = request[0], request[1]
        width = self.profile.mode_len_width
        if ident == sim_encode.SIM_ID_READ:
            if fc == 0x01:
                return [sim_encode.encode_version(FACTORY_VERSION_RAW)]
            if fc == 0x03:
                return [sim_encode.encode_serial(FACTORY_SERIAL)]
            if fc == 0x04:
                return [
                    sim_encode.encode_module_params(
                        FACTORY_SPEED_CODE,
                        self.profile.channels,
                        self.profile.fbg_per_channel,
                        self.profile.peak_gap_ghz,
                    )
                ]
            if fc == 0x05:
                return [
                    sim_encode.encode_sweep(
                        self.profile.start_param,
                        self.profile.step_param,
                        self.profile.stop_param,
                        self.profile.adc_step_param,
                    )
                ]
            if fc == 0x06:
                return [
                    sim_encode.encode_channel_setup([(0xFFFF, 0x00, 5)] * self.profile.channels)
                ]
        if ident == sim_encode.SIM_ID_MODE:
            if fc == 0x01:
                return [sim_encode.encode_stop_ack(True, width)]
            if fc == 0x03:
                adc = np.arange(self.profile.adc_points, dtype=np.uint16) % 4096
                body = b"".join(
                    sim_encode.encode_adc_block(channel, 0x00, 5, adc)
                    for channel in range(self.profile.channels)
                )
                return [sim_encode.encode_debug(body, width)]
            if fc == 0x07:
                adc = np.arange(self.profile.adc_points, dtype=np.uint16) % 4096
                return [sim_encode.encode_raw_adc(request[5], 0x00, 5, adc, width)]
        return []

    def _telemetry_frame(self) -> bytes:
        """Кадр `30 02` для проверки того, что он не попадает в тело длинного ответа."""
        freq = np.zeros((self.profile.channels, self.profile.fbg_per_channel), dtype=np.uint32)
        freq[0, 0] = 0x1D9CC0
        temp = np.full(self.profile.channels, 1685, dtype=np.int32)
        return sim_encode.encode_measurement(self.profile, freq, temp)

    def _send(self, payload: bytes) -> None:
        # Длинный ответ уходит кусками, короткий — одной датаграммой.
        for index, offset in enumerate(range(0, len(payload), self.chunk_bytes)):
            if self.telemetry_at is not None and index == self.telemetry_at:
                self._sock.sendto(self._telemetry_frame(), self.reply_to)
                self.datagrams_sent += 1
            self._sock.sendto(payload[offset : offset + self.chunk_bytes], self.reply_to)
            self.datagrams_sent += 1


# --------------------------------------------------------------------------------------
# Стенд
# --------------------------------------------------------------------------------------


class Rig:
    """Симулятор плюс сессия, связанные так же, как прибор и приложение.

    Порядок как в реальности: прибор поднимается первым и сообщает адрес,
    сессия открывает приёмный порт, после чего симулятору проставляется
    `reply_to` — прибор отвечает на прописанный адрес назначения, а не
    на source-порт запроса (KB_01).
    """

    def __init__(
        self,
        *,
        simulator: type[DeviceSimulator] = RecordingSimulator,
        profile: DeviceProfile | None = None,
        session_config: SessionConfig = QUIET,
        rate_hz: float = TEST_FRAME_RATE_HZ,
        **endpoint_kwargs: object,
    ) -> None:
        self.profile = profile or DeviceProfile()
        self.sim = simulator(
            profile=self.profile,
            scene=Scene(self.profile, [Grating(0, 0, 1545.0), Grating(1, 0, 1560.0)]),
            reply_to=("127.0.0.1", 1),
            frame_rate_hz=rate_hz,
        )
        self.sim.start()
        host, port = self.sim.address
        self.endpoint = Endpoint(
            device_ip=host,
            device_port=port,
            **{**TEST_ENDPOINT_KWARGS, **endpoint_kwargs},  # type: ignore[arg-type]
        )
        self.telemetry: list[tuple[bytes, float]] = []
        self.states: list[tuple[SessionState, SessionState]] = []
        self.mismatches: list[tuple[str, ...]] = []
        self.session = Session(
            self.endpoint,
            self.profile,
            session_config,
            on_telemetry=lambda data, t: self.telemetry.append((data, t)),
            on_state=lambda old, new: self.states.append((old, new)),
            on_config_mismatch=self.mismatches.append,
        )

    def connect(self) -> Result[DeviceConfig]:
        """Подключается. Порт уже открыт в `make_rig`, `connect` его не меняет."""
        return self.session.connect()

    def close(self) -> None:
        """Сначала замолкает прибор, потом закрывается сессия."""
        self.sim.stop()
        self.session.disconnect()


def open_port_and_announce(session: Session, announce: Callable[[tuple[str, int]], None]) -> None:
    """Открывает приёмный порт заранее и сообщает его прибору.

    Прибор отвечает на прописанный адрес назначения, а не на source-порт
    запроса (KB_01), поэтому адрес приёма нужен ему **до** первой команды —
    иначе уже Stop при подключении остался бы без ответа. В приложении этот
    адрес задан настройками, в тестах он эфемерный, поэтому порт открывается
    заранее и остаётся открытым: `UdpTransport.open` идемпотентен, и `connect`
    просто ничего не делает с уже открытым сокетом.
    """
    session._transport.open()
    announce(session.local_address)


def make_rig(**kwargs: object) -> Rig:
    """Собирает стенд, открывает приёмный порт и сообщает его симулятору."""
    rig = Rig(**kwargs)  # type: ignore[arg-type]

    def announce(address: tuple[str, int]) -> None:
        rig.sim.reply_to = address

    open_port_and_announce(rig.session, announce)
    return rig


@pytest.fixture
def rig() -> Iterator[Rig]:
    """Подключённый стенд по умолчанию."""
    stand = make_rig()
    try:
        assert stand.connect().ok
        yield stand
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Result, ошибки и конфигурация
# --------------------------------------------------------------------------------------


def test_result_ok_and_error() -> None:
    """Result хранит либо значение, либо ошибку, и распаковывается."""
    good: Result[int] = Result(value=7)
    assert good.ok and good.unwrap() == 7
    bad: Result[int] = Result(error=SessionError(SessionErrorKind.TIMEOUT, "нет ответа"))
    assert not bad.ok
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        bad.unwrap()


def test_session_config_rejects_nonsense() -> None:
    """Некорректные параметры сессии — баг вызывающего, значит ValueError."""
    with pytest.raises(ValueError, match="keepalive_period_s"):
        SessionConfig(keepalive_period_s=0.0)
    with pytest.raises(ValueError, match="backoff_schedule"):
        SessionConfig(backoff_schedule=())
    with pytest.raises(ValueError, match="keepalive_failures_to_degrade"):
        SessionConfig(keepalive_failures_to_degrade=0)


def test_device_config_differences_lists_changes(rig: Rig) -> None:
    """Сравнение конфигураций перечисляет то, что изменилось."""
    was = rig.session.device_config
    assert was is not None
    assert was.differences(was) == ()
    other = DeviceConfig(
        version_raw=was.version_raw,
        serial=was.serial + 1,
        module=was.module,
        sweep=was.sweep,
        channels=was.channels,
    )
    diffs = was.differences(other)
    assert len(diffs) == 1
    assert "серийный номер" in diffs[0]


# --------------------------------------------------------------------------------------
# Подключение и опрос
# --------------------------------------------------------------------------------------


def test_connect_reads_full_config(rig: Rig) -> None:
    """Полный цикл connect → Probing → Idle: конфигурация прочитана целиком."""
    assert rig.session.state is SessionState.IDLE
    config = rig.session.device_config
    assert config is not None
    assert config.version_raw == FACTORY_VERSION_RAW
    assert config.version == "4.10"
    assert config.serial == FACTORY_SERIAL
    assert config.module.speed_hz == 2000
    assert config.module.channels == rig.profile.channels
    assert config.sweep.start_param == rig.profile.start_param
    assert len(config.channels) == rig.profile.channels
    assert all(setup.threshold is None for setup in config.channels)


def test_connect_visits_probing(rig: Rig) -> None:
    """Автомат проходит через Probing, а не прыгает сразу в Idle."""
    transitions = [new for _, new in rig.states]
    assert transitions[:2] == [SessionState.PROBING, SessionState.IDLE]


def test_stop_goes_first_on_connect(rig: Rig) -> None:
    """Stop уходит первой командой: прибор мог остаться в потоке (KB_05 №6)."""
    seen = rig.sim.seen()
    assert seen[0] == (codec.ID_MODE, codec.FC_STOP)
    # N7: порядок опроса выбран нами, у вендора он не подсмотрен.
    assert seen[1:6] == [
        (codec.ID_READ, codec.FC_VERSION),
        (codec.ID_READ, codec.FC_SERIAL),
        (codec.ID_READ, codec.FC_MODULE_PARAMS),
        (codec.ID_READ, codec.FC_SWEEP),
        (codec.ID_READ, codec.FC_CHANNEL_SETUP),
    ]


def test_stop_goes_last_on_disconnect() -> None:
    """Stop уходит в finally при отключении, даже если поток шёл."""
    stand = make_rig()
    try:
        assert stand.connect().ok
        assert stand.session.start_stream().ok
        assert wait_until(lambda: stand.session.stats().telemetry_frames > 0)
        stand.session.disconnect()
        assert stand.sim.seen()[-1] == (codec.ID_MODE, codec.FC_STOP)
        assert not stand.sim.streaming
    finally:
        stand.sim.stop()
        stand.session.disconnect()


def test_connect_twice_is_wrong_state(rig: Rig) -> None:
    """Повторный connect из Idle — ошибка состояния, а не тихий повтор опроса."""
    result = rig.session.connect()
    assert result.error is not None
    assert result.error.kind is SessionErrorKind.WRONG_STATE


def test_commands_before_connect_are_not_connected() -> None:
    """Команда до подключения возвращает NotConnected, а не бросает."""
    session = Session(Endpoint(device_ip="127.0.0.1", device_port=1, **TEST_ENDPOINT_KWARGS))
    result = session.read_version()
    assert result.error is not None
    assert result.error.kind is SessionErrorKind.NOT_CONNECTED


def test_connect_failure_goes_to_reconnecting() -> None:
    """Неудачный опрос переводит в Reconnecting, а ошибку возвращает вызывающему."""
    stand = make_rig()
    try:
        stand.sim.go_silent(30.0)
        result = stand.connect()
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.TIMEOUT
        assert wait_until(lambda: stand.session.state is SessionState.RECONNECTING)
    finally:
        stand.close()


def test_connect_failure_without_autoreconnect_returns_to_disconnected() -> None:
    """С auto_reconnect=False неудачный опрос закрывает транспорт."""
    import dataclasses

    stand = make_rig(session_config=dataclasses.replace(QUIET, auto_reconnect=False))
    try:
        stand.sim.go_silent(30.0)
        result = stand.connect()
        assert result.error is not None
        assert stand.session.state is SessionState.DISCONNECTED
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Команды чтения
# --------------------------------------------------------------------------------------


def test_read_version(rig: Rig) -> None:
    """10 01 — версия прошивки."""
    assert rig.session.read_version().unwrap() == FACTORY_VERSION_RAW


def test_read_serial(rig: Rig) -> None:
    """10 03 — серийный номер."""
    assert rig.session.read_serial().unwrap() == FACTORY_SERIAL


def test_read_module_params(rig: Rig) -> None:
    """10 04 — параметры модуля."""
    module = rig.session.read_module_params().unwrap()
    assert module.speed_code == FACTORY_SPEED_CODE
    assert module.peak_gap_ghz == rig.profile.peak_gap_ghz


def test_read_sweep(rig: Rig) -> None:
    """10 05 — параметры развёртки с пересчётом в ГГц."""
    sweep = rig.session.read_sweep().unwrap()
    assert sweep.stop_param == rig.profile.stop_param
    assert sweep.start_ghz == rig.profile.start_ghz


def test_read_channel_setup(rig: Rig) -> None:
    """10 06 — пороги и усиления всех каналов."""
    setups = rig.session.read_channel_setup().unwrap()
    assert len(setups) == rig.profile.channels
    assert setups[0].gain == GainSetting(manual=False, level=5)


def test_refresh_config_updates_stored(rig: Rig) -> None:
    """refresh_config перечитывает конфигурацию и запоминает её."""
    assert rig.session.set_peak_gap(40).ok
    fresh = rig.session.refresh_config().unwrap()
    assert fresh.module.peak_gap_ghz == 40
    assert rig.session.device_config == fresh


# --------------------------------------------------------------------------------------
# Команды записи и верификация чтением
# --------------------------------------------------------------------------------------


def test_set_sweep_verified(rig: Rig) -> None:
    """20 01 применяется и подтверждается чтением 10 05."""
    wanted = SweepConfig.from_params(2, 2, 5000, 2, rig.profile)
    applied = rig.session.set_sweep(wanted).unwrap()
    assert (applied.start_param, applied.stop_param) == (2, 5000)
    assert rig.session.unconfirmed == frozenset()
    assert rig.sim.state.stop_param == 5000


def test_set_threshold_verified(rig: Rig) -> None:
    """20 02 применяется и подтверждается чтением 10 06."""
    setup = rig.session.set_threshold(2, 1200).unwrap()
    assert setup.threshold == 1200
    assert rig.sim.state.thresholds[2] == 1200


def test_set_threshold_auto_verified(rig: Rig) -> None:
    """Порог None означает авторасчёт и читается обратно как None."""
    assert rig.session.set_threshold(1, 900).ok
    setup = rig.session.set_threshold(1, None).unwrap()
    assert setup.threshold is None


def test_set_gain_verified(rig: Rig) -> None:
    """20 03 применяется и подтверждается чтением 10 06."""
    setup = rig.session.set_gain(3, GainSetting(manual=True, level=2)).unwrap()
    assert setup.gain == GainSetting(manual=True, level=2)


def test_set_peak_gap_verified(rig: Rig) -> None:
    """20 04 применяется и подтверждается чтением 10 04."""
    assert rig.session.set_peak_gap(40).unwrap() == 40
    assert rig.sim.state.peak_gap_ghz == 40


def test_verification_mismatch_reported() -> None:
    """Прибор подтвердил запись, но не применил её → отдельный код ошибки."""
    stand = make_rig(simulator=LyingSimulator)
    try:
        assert stand.connect().ok
        result = stand.session.set_threshold(0, 1200)
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.VERIFICATION_MISMATCH
        assert "threshold:0" in stand.session.unconfirmed
        assert stand.session.stats().verification_mismatches == 1
    finally:
        stand.close()


def test_verification_success_clears_unconfirmed() -> None:
    """Успешная верификация снимает пометку «не подтверждено»."""
    stand = make_rig(simulator=LyingSimulator)
    try:
        assert stand.connect().ok
        assert not stand.session.set_peak_gap(40).ok
        assert "peak_gap" in stand.session.unconfirmed
        # Тот же симулятор, но теперь запись действительно применяется.
        stand.sim.state.peak_gap_ghz = 41
        assert stand.session.set_peak_gap(41).ok
        assert "peak_gap" not in stand.session.unconfirmed
    finally:
        stand.close()


def test_device_rejection_is_separate_error(rig: Rig) -> None:
    """Отказ прибора `00 00` — не таймаут и не расхождение, а свой код."""
    # Симулятор отвергает развёртку с нарушенным инвариантом; кодек такую
    # не соберёт, поэтому отказ вызывается некорректным номером канала
    # в 20 02 — прибор отвечает 00 00.
    request = bytes([codec.ID_WRITE, codec.FC_SET_THRESHOLD, 0x06, 0xFF]) + struct.pack(">H", 100)
    ack = rig.session._write(request, codec.FC_SET_THRESHOLD)
    assert ack.unwrap() is False
    assert rig.sim.stats.rejected_requests == 1


def test_save_thresholds_does_not_wait_for_response(rig: Rig) -> None:
    """20 06 — fire-and-forget: ответа не ждём, проверяем чтением 10 06 (D4)."""
    assert rig.session.set_threshold(0, 800).ok
    started = time.perf_counter()
    result = rig.session.save_thresholds()
    elapsed = time.perf_counter() - started
    assert result.ok
    # Ждать было бы нечего: read_timeout один только занял бы больше.
    assert elapsed < rig.endpoint.read_timeout_s * 2
    assert rig.sim.state.saved_thresholds == rig.sim.state.thresholds
    assert (codec.ID_WRITE, codec.FC_SAVE_THRESHOLDS) in rig.sim.seen()


def test_save_thresholds_readback_mismatch() -> None:
    """Если после сохранения пороги в приборе другие, это расхождение."""
    stand = make_rig()
    try:
        assert stand.connect().ok
        # Прибор «сам по себе» сменил порог между чтением и сохранением.
        stand.sim.state.thresholds[0] = 777
        result = stand.session.save_thresholds()
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.VERIFICATION_MISMATCH
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Одна команда в полёте
# --------------------------------------------------------------------------------------


def test_second_command_while_first_in_flight_is_busy(rig: Rig) -> None:
    """Второй одновременный запрос — ошибка вызывающего, а не очередь."""
    rig.sim.faults.response_delay_s = 0.3
    first: list[Result[int]] = []
    worker = threading.Thread(target=lambda: first.append(rig.session.read_version()))
    worker.start()
    try:
        assert wait_until(lambda: rig.session._lock_owner == "user", timeout=1.0)
        second = rig.session.read_serial()
        assert second.error is not None
        assert second.error.kind is SessionErrorKind.BUSY
    finally:
        worker.join(timeout=WAIT_TIMEOUT_S)
        rig.sim.faults.response_delay_s = 0.0
    assert first and first[0].ok


# --------------------------------------------------------------------------------------
# Таймауты, повторы, поздние ответы
# --------------------------------------------------------------------------------------


def test_timeout_then_retry_succeeds(rig: Rig) -> None:
    """Первая попытка потерялась, вторая прошла: retry делает своё дело."""
    rig.sim.drop_remaining = 1
    before = rig.session.stats()
    assert rig.session.read_version().unwrap() == FACTORY_VERSION_RAW
    after = rig.session.stats()
    assert after.retries == before.retries + 1
    assert after.timeouts == before.timeouts + 1


def test_timeout_exhausts_retries(rig: Rig) -> None:
    """Молчание дольше всех попыток → Timeout, а не исключение."""
    rig.sim.drop_remaining = 10
    result = rig.session.read_version()
    assert result.error is not None
    assert result.error.kind is SessionErrorKind.TIMEOUT
    # retries=1 означает две отправки всего.
    assert rig.session.stats().timeouts == 2


def test_late_response_is_dropped_and_counted() -> None:
    """Ответ, пришедший после таймаута, отбрасывается со счётчиком orphan."""
    stand = make_rig(retries=0)
    try:
        assert stand.connect().ok
        before = stand.session.stats().orphan_responses
        stand.sim.faults.response_delay_s = 0.4
        result = stand.session.read_version()
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.TIMEOUT
        assert wait_until(lambda: stand.session.stats().orphan_responses > before)
        stand.sim.faults.response_delay_s = 0.0
    finally:
        stand.close()


def test_garbage_response_does_not_break_session(rig: Rig) -> None:
    """Мусор вместо ответа: команда падает по таймауту, сессия жива."""
    rig.sim.faults.garbage = True
    result = rig.session.read_version()
    assert result.error is not None
    assert result.error.kind is SessionErrorKind.TIMEOUT
    assert rig.session.stats().orphan_responses > 0
    rig.sim.faults.garbage = False
    assert rig.session.read_version().ok


def test_bad_len_response_is_bad_response(rig: Rig) -> None:
    """Испорченный LEN: ответ пришёл, но не разобрался — отдельный код."""
    rig.sim.faults.bad_len = True
    result = rig.session.read_version()
    assert result.error is not None
    assert result.error.kind is SessionErrorKind.BAD_RESPONSE
    assert "LEN" in result.error.message
    rig.sim.faults.bad_len = False
    assert rig.session.read_version().ok


def test_delayed_but_inside_timeout_succeeds(rig: Rig) -> None:
    """Задержка меньше таймаута — обычный успех, без повторов."""
    rig.sim.faults.response_delay_s = 0.05
    before = rig.session.stats().retries
    assert rig.session.read_version().ok
    assert rig.session.stats().retries == before
    rig.sim.faults.response_delay_s = 0.0


# --------------------------------------------------------------------------------------
# Поток телеметрии и блокировка настроек
# --------------------------------------------------------------------------------------


def test_start_and_stop_stream(rig: Rig) -> None:
    """Старт потока переводит в Streaming, Stop возвращает в Idle."""
    assert rig.session.start_stream().ok
    assert rig.session.state is SessionState.STREAMING
    assert wait_until(lambda: rig.session.stats().telemetry_frames > 5)
    assert rig.session.stop_stream().unwrap() is True
    assert rig.session.state is SessionState.IDLE


def test_telemetry_goes_to_callback_raw(rig: Rig) -> None:
    """Кадры уходят потребителю сырыми байтами: разбор — не дело сессии."""
    assert rig.session.start_stream().ok
    assert wait_until(lambda: len(rig.telemetry) > 3)
    assert rig.session.stop_stream().ok
    data, t_mono = rig.telemetry[0]
    assert data[:2] == bytes([codec.ID_MODE, codec.FC_STREAM])
    assert len(data) == rig.profile.frame_size
    assert t_mono > 0.0


def test_reads_and_writes_allowed_during_streaming(rig: Rig) -> None:
    """Р62: 0x10 и 0x20 проходят в Streaming и поток продолжает идти."""
    assert rig.session.start_stream().ok
    try:
        assert wait_until(lambda: rig.session.stats().telemetry_frames > 5)
        before = rig.session.stats().telemetry_frames
        assert rig.session.read_module_params().ok
        setup = rig.session.set_threshold(0, 1000).unwrap()
        assert setup.threshold == 1000
        assert rig.sim.state.thresholds[0] == 1000
        assert rig.session.state is SessionState.STREAMING
        assert rig.sim.streaming
        assert wait_until(lambda: rig.session.stats().telemetry_frames > before + 5)
    finally:
        assert rig.session.stop_stream().ok


def test_mode_group_rejected_during_streaming(rig: Rig) -> None:
    """Р62/R13: 30 03 и 30 07 не выпускаются, потому что вытесняют поток."""
    assert rig.session.start_stream().ok
    assert wait_until(lambda: rig.sim.streaming)
    try:
        for result in (rig.session.read_raw_adc(0), rig.session.debug_once()):
            assert result.error is not None
            assert result.error.kind is SessionErrorKind.WRONG_STATE
        assert rig.session.state is SessionState.STREAMING
        assert rig.sim.streaming
    finally:
        assert rig.session.stop_stream().ok


def test_stop_allowed_during_streaming(rig: Rig) -> None:
    """Из режимной группы 0x30 в потоке разрешён Stop."""
    assert rig.session.start_stream().ok
    assert rig.session.stop_stream().ok
    assert rig.session.state is SessionState.IDLE


def test_start_stream_from_streaming_is_wrong_state(rig: Rig) -> None:
    """Повторный старт потока — ошибка состояния."""
    assert rig.session.start_stream().ok
    try:
        result = rig.session.start_stream()
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.WRONG_STATE
    finally:
        assert rig.session.stop_stream().ok


def test_lost_telemetry_is_not_a_session_error(rig: Rig) -> None:
    """Потерянные кадры не считаются отказом: у потока нет подтверждений."""
    rig.sim.faults.frame_drop_probability = 0.5
    assert rig.session.start_stream().ok
    assert wait_until(lambda: rig.session.stats().telemetry_frames > 5)
    stats = rig.session.stats()
    assert stats.timeouts == 0
    assert stats.degraded_events == 0
    assert rig.sim.stats.frames_dropped > 0
    rig.sim.faults.frame_drop_probability = 0.0
    assert rig.session.stop_stream().ok


# --------------------------------------------------------------------------------------
# Отладка и длинные ответы
# --------------------------------------------------------------------------------------


def test_debug_once_parses_channel_blocks(rig: Rig) -> None:
    """30 03: ✅ тело разбирается на блоки каналов (N14 закрыт скринингом)."""
    response = rig.session.debug_once().unwrap()

    assert response.channels == rig.profile.channels
    for index, block in enumerate(response.blocks):
        assert block.channel == index
        assert block.points == rig.profile.adc_points
    # Сырые байты сохраняются рядом с разбором — правило KB_05 №3.
    assert len(response.payload) == 20430 - 6
    assert rig.session.state is SessionState.IDLE


def test_debug_once_survives_unsolicited_telemetry(rig: Rig) -> None:
    """⚠️ Команда 30 03 порождает ДВА ответа с разными парами (ID, FC).

    Прибор шлёт кадр телеметрии `30 02` отдельной датаграммой непосредственно
    перед ответом `30 03` (✅ скрининг, N14). Корреляция ведётся по паре
    (ID, FC) с единственным ожидающим, и незапрошенный кадр обязан её пережить:
    телеметрия отбирается в `_on_datagram` до всякой корреляции и уходит
    в колбэк, а не в счётчик потерянных ответов.

    Это главный риск правки: если бы кадр попал в ветку корреляции, он бы
    либо увеличил `orphan_responses`, либо — при уже начатой сборке длинного
    ответа — дописался в его тело и испортил бы разбор.
    """
    rig.telemetry.clear()
    orphans_before = rig.session.stats().orphan_responses

    response = rig.session.debug_once().unwrap()

    assert response.channels == rig.profile.channels
    assert len(rig.telemetry) == 1, "кадр 30 02 обязан дойти до потребителя телеметрии"
    data, _t_mono = rig.telemetry[0]
    assert codec.classify(data) == (codec.ID_MODE, codec.FC_STREAM)
    assert len(data) == rig.profile.frame_size
    assert rig.session.stats().orphan_responses == orphans_before
    assert rig.session.stats().telemetry_frames >= 1


def test_debug_once_repeats_cleanly(rig: Rig) -> None:
    """Две отладки подряд: лишний кадр 30 02 не смещает корреляцию следующей команды."""
    assert rig.session.debug_once().ok
    assert rig.session.debug_once().ok
    assert rig.session.read_version().unwrap() == 410
    assert rig.session.stats().orphan_responses == 0


def test_read_raw_adc_single_datagram(rig: Rig) -> None:
    """30 07 одной датаграммой: 5112 байт разбираются целиком."""
    block = rig.session.read_raw_adc(1).unwrap()
    assert block.channel == 1
    assert block.points == rig.profile.adc_points


def test_read_raw_adc_reassembled_from_fragments() -> None:
    """30 07, разрезанный на куски, собирается по объявленному LEN (D5).

    ⚠️ Куски производит тестовый прибор, а не наблюдение: гипотеза D5
    этим тестом не подтверждается.
    """
    profile = DeviceProfile()
    device = FragmentingDevice(profile, chunk_bytes=1400)
    endpoint = Endpoint(
        device_ip=device.address[0], device_port=device.address[1], **TEST_ENDPOINT_KWARGS
    )
    session = Session(endpoint, profile, QUIET)
    try:
        open_port_and_announce(session, lambda address: setattr(device, "reply_to", address))
        assert session.connect().ok
        block = session.read_raw_adc(2).unwrap()
        assert block.channel == 2
        assert block.points == profile.adc_points
        # 5112 байт кусками по 1400 — четыре датаграммы на один ответ.
        assert device.datagrams_sent > 6
    finally:
        session.disconnect()
        device.stop()


def test_incomplete_long_response_is_not_a_codec_error() -> None:
    """Недобор длинного ответа — своя ошибка сборки, а не ошибка разбора."""
    profile = DeviceProfile()
    device = FragmentingDevice(profile, chunk_bytes=1400)
    endpoint = Endpoint(
        device_ip=device.address[0], device_port=device.address[1], **TEST_ENDPOINT_KWARGS
    )
    session = Session(endpoint, profile, QUIET)
    try:
        open_port_and_announce(session, lambda address: setattr(device, "reply_to", address))
        assert session.connect().ok
        # Прибор замолкает после первой датаграммы длинного ответа.
        original = device._send

        def truncated(payload: bytes) -> None:
            if payload[:2] == bytes([codec.ID_MODE, codec.FC_RAW_ADC]):
                device._sock.sendto(payload[: device.chunk_bytes], device.reply_to)
                return
            original(payload)

        device._send = truncated  # type: ignore[method-assign]
        result = session.read_raw_adc(0)
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.INCOMPLETE_RESPONSE
        assert session.stats().incomplete_responses > 0
    finally:
        session.disconnect()
        device.stop()


# --------------------------------------------------------------------------------------
# Watchdog
# --------------------------------------------------------------------------------------


def test_watchdog_degrades_on_silence_in_idle() -> None:
    """Idle: keepalive не отвечает N раз подряд → Degraded, потом возврат."""
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        stand.sim.go_silent(0.6)
        assert wait_until(lambda: stand.session.state is SessionState.DEGRADED)
        assert stand.session.stats().degraded_events >= 1
        assert stand.session.stats().keepalive_failures >= 2
        # Прибор ожил: пробник проходит и сессия возвращается в Idle.
        assert wait_until(lambda: stand.session.state is SessionState.IDLE, timeout=WAIT_TIMEOUT_S)
        assert stand.session.read_version().ok
    finally:
        stand.close()


def test_watchdog_degrades_on_stalled_stream() -> None:
    """Streaming: телеметрия пропала → Degraded, keepalive при этом не шлётся."""
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        assert stand.session.start_stream().ok
        assert wait_until(lambda: stand.session.stats().telemetry_frames > 3)
        commands_before = len(stand.sim.seen())
        stand.sim.faults.frame_drop_probability = 1.0
        assert wait_until(lambda: stand.session.state is SessionState.DEGRADED)
        assert wait_until(lambda: len(stand.sim.seen()) > commands_before)
        # За время потока сессия не отправила ни одной команды: keepalive
        # в Streaming не шлётся, и первая же команда после — Stop пробника.
        assert stand.sim.seen()[commands_before] == (codec.ID_MODE, codec.FC_STOP)
        assert wait_until(lambda: stand.session.state is SessionState.IDLE)
        assert stand.session.stream_interrupted
    finally:
        stand.sim.faults.frame_drop_probability = 0.0
        stand.close()


def test_recovery_returns_to_idle_not_streaming() -> None:
    """После восстановления состояние Idle: пробник сам остановил поток.

    Отступление от диаграммы KB_03 («возврат в прежнее состояние») сделано
    осознанно: пробник содержит Stop, поэтому объявить Streaming значило бы,
    что автомат врёт про прибор. Факт обрыва виден в `stream_interrupted`.
    """
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        assert stand.session.start_stream().ok
        assert wait_until(lambda: stand.session.stats().telemetry_frames > 3)
        stand.sim.faults.frame_drop_probability = 1.0
        assert wait_until(lambda: stand.session.state is SessionState.DEGRADED)
        stand.sim.faults.frame_drop_probability = 0.0
        assert wait_until(lambda: stand.session.state is SessionState.IDLE)
        assert stand.session.stream_interrupted is True
    finally:
        stand.close()


def test_reconnect_uses_backoff_and_cancels() -> None:
    """Пробник не прошёл → Reconnecting с паузами; отмена во время паузы."""
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        stand.sim.go_silent(30.0)
        assert wait_until(lambda: stand.session.state is SessionState.RECONNECTING)
        assert wait_until(lambda: stand.session.stats().reconnect_attempts >= 2)
        # Отмена посреди backoff обязана отработать быстро, а не ждать паузу.
        started = time.perf_counter()
        stand.session.disconnect()
        assert time.perf_counter() - started < 2.0
        assert stand.session.state is SessionState.DISCONNECTED
    finally:
        stand.sim.stop()
        stand.session.disconnect()


def test_user_command_in_degraded_is_wrong_state() -> None:
    """Пока сессия восстанавливается, пользовательские команды не пускаются."""
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        stand.sim.go_silent(30.0)
        assert wait_until(
            lambda: stand.session.state in (SessionState.DEGRADED, SessionState.RECONNECTING)
        )
        result = stand.session.read_serial()
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.WRONG_STATE
    finally:
        stand.close()


def test_reboot_mid_session_is_noticed() -> None:
    """Прибор перезагрузился: расхождение конфигурации замечено, не перезаписано."""
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        assert stand.session.set_peak_gap(40).ok
        assert stand.session.device_config is not None
        assert stand.session.device_config.module.peak_gap_ghz == 40

        stand.sim.go_silent(0.5)
        stand.sim.reboot()
        assert wait_until(lambda: stand.session.state is SessionState.DEGRADED)
        assert wait_until(lambda: stand.session.state is SessionState.IDLE)

        assert stand.session.config_mismatch, "расхождение конфигурации не замечено"
        assert any("параметры модуля" in diff for diff in stand.session.config_mismatch)
        assert stand.mismatches, "колбэк о расхождении не вызван"
        # Молча возвращать 40 сессия не имеет права: решает оператор.
        assert stand.sim.state.peak_gap_ghz == stand.profile.peak_gap_ghz
    finally:
        stand.close()


def test_keepalive_is_skipped_while_user_command_runs() -> None:
    """Успешная пользовательская команда откладывает keepalive."""
    stand = make_rig(session_config=WATCHFUL)
    try:
        assert stand.connect().ok
        for _ in range(5):
            assert stand.session.read_version().ok
            time.sleep(0.02)
        assert stand.session.stats().keepalive_failures == 0
        assert stand.session.state is SessionState.IDLE
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Закрытие
# --------------------------------------------------------------------------------------


def test_close_joins_threads_and_frees_port() -> None:
    """close не зависает, потоки присоединены, порт освобождён."""
    stand = make_rig()
    assert stand.connect().ok
    address = stand.session.local_address
    stand.sim.stop()

    started = time.perf_counter()
    stand.session.disconnect()
    assert time.perf_counter() - started < 5.0

    names = {thread.name for thread in threading.enumerate()}
    assert "fbg-watchdog" not in names
    assert "fbg-rx" not in names
    assert "fbg-tap" not in names

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(address)
    finally:
        probe.close()


def test_disconnect_is_idempotent() -> None:
    """Повторное отключение безвредно."""
    stand = make_rig()
    assert stand.connect().ok
    stand.sim.stop()
    stand.session.disconnect()
    stand.session.disconnect()
    assert stand.session.state is SessionState.DISCONNECTED


def test_context_manager_stops_device() -> None:
    """Выход из `with` останавливает прибор и закрывает порт."""
    stand = make_rig()
    try:
        with stand.session as session:
            assert session.connect().ok
            assert session.start_stream().ok
            assert wait_until(lambda: session.stats().telemetry_frames > 0)
        assert stand.session.state is SessionState.DISCONNECTED
        assert not stand.sim.streaming
    finally:
        stand.sim.stop()
        stand.session.disconnect()


def test_debug_response_reassembled_from_fragments() -> None:
    """30 03 из 20430 байт собирается по объявленному LEN и разбирается на блоки.

    ⚠️ Куски режет тестовый прибор. Реальная фрагментация со стенда — 20430
    байт при полезной нагрузке 1472, то есть 13 × 1472 + 1294 = 14 датаграмм.
    """
    profile = DeviceProfile()
    device = FragmentingDevice(profile, chunk_bytes=1472)
    endpoint = Endpoint(
        device_ip=device.address[0], device_port=device.address[1], **TEST_ENDPOINT_KWARGS
    )
    session = Session(endpoint, profile, QUIET)
    try:
        open_port_and_announce(session, lambda address: setattr(device, "reply_to", address))
        assert session.connect().ok
        response = session.debug_once().unwrap()
        assert response.channels == profile.channels
        assert response.blocks[2].channel == 2
        assert response.blocks[2].points == profile.adc_points
        assert len(response.payload) == 20430 - 6
    finally:
        session.disconnect()
        device.stop()


def test_telemetry_between_fragments_does_not_corrupt_long_response() -> None:
    """Кадр 30 02, пришедший ПОСРЕДИ сборки длинного ответа, не попадает в тело.

    Худший случай правки этого чата. Прибор шлёт телеметрию не спрашивая,
    а сборка длинного ответа дописывает **любую** датаграмму как продолжение —
    заголовка у продолжений нет (D5). Единственное, что разделяет эти два
    случая, — отбор телеметрии по паре (ID, FC) до всякой корреляции.

    Если бы отбор стоял позже, 494 байта кадра встали бы внутрь массива АЦП:
    LEN добрался бы раньше, тело оказалось бы не кратно блоку канала, и
    разбор упал бы с LEN_MISMATCH — либо, что хуже, сошёлся бы со сдвигом.
    """
    profile = DeviceProfile()
    device = FragmentingDevice(profile, chunk_bytes=1472, telemetry_at=5)
    endpoint = Endpoint(
        device_ip=device.address[0], device_port=device.address[1], **TEST_ENDPOINT_KWARGS
    )
    telemetry: list[bytes] = []
    session = Session(
        endpoint, profile, QUIET, on_telemetry=lambda data, _t: telemetry.append(data)
    )
    try:
        open_port_and_announce(session, lambda address: setattr(device, "reply_to", address))
        assert session.connect().ok
        response = session.debug_once().unwrap()

        assert response.channels == profile.channels
        assert len(response.payload) == 20430 - 6
        for index, block in enumerate(response.blocks):
            assert block.channel == index
        # Кадр не потерялся: он ушёл потребителю телеметрии, а не в тело ответа.
        assert any(len(frame) == profile.frame_size for frame in telemetry)
        assert session.stats().orphan_responses == 0
    finally:
        session.disconnect()
        device.stop()
