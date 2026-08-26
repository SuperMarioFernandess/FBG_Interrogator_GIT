"""Симулятор интеррогатора: UDP-сервер, отвечающий на все 14 команд протокола.

Симулятор — не ядро: ему разрешены сокеты, потоки и часы. `fbg.core` он
не импортирует ничего, кроме `DeviceProfile`, и ничего в нём не меняет.

Ответы собираются модулем `fbg.sim.encode` **независимо от `fbg.core.codec`**.
Если бы симулятор вызывал функции кодека наоборот, интеграционные тесты
проверяли бы согласованность кодека с самим собой.

Поведение, воспроизведённое намеренно:
  * прибор отвечает на **жёстко прописанный** адрес назначения, а не на
    source-порт запроса (KB_01, раздел «Сеть») — отсюда обязательный `reply_to`;
  * на `20 06` ответа нет (гипотеза D4);
  * `30 02` не подтверждается ответом: сразу начинается поток.

Гипотезы, которые симулятор **не выдумывает, а параметризует**: единицы
частоты (D1) — `Scene.divisor`; код «пик не найден» (N3) — `Scene.missing_raw`;
раскладка кадра (N4) — та же, что в кодеке; знаковость и масштаб температуры
(N2, N2b) — из профиля; ширина поля LEN в ответах 0x30 — `profile.mode_len_width`.
"""

import contextlib
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from fbg.core.profile import DeviceProfile
from fbg.sim.encode import (
    SIM_ID_MODE,
    SIM_ID_READ,
    SIM_ID_WRITE,
    SIM_SPEED_KEEP_CURRENT,
    MeasurementEncoder,
    encode_channel_setup,
    encode_debug,
    encode_module_params,
    encode_raw_adc,
    encode_serial,
    encode_stop_ack,
    encode_sweep,
    encode_version,
    encode_write_ack,
    sim_decode_speed_code,
)
from fbg.sim.scene import Scene

#: Заводские значения, ✅ прочитанные с прибора командами 10 01, 10 03, 10 04…10 06.
FACTORY_VERSION_RAW = 410
FACTORY_SERIAL = 94_401_220
FACTORY_SPEED_CODE = 0x00CA
FACTORY_THRESHOLD_AUTO = 0xFFFF
FACTORY_GAIN_MODE = 0x00
FACTORY_GAIN_LEVEL = 5

#: Байт ручного режима усиления.
SIM_GAIN_MANUAL_FLAG = 0x80

#: Допустимые длины кадра команды 20 01 на проводе (вопрос D3 открыт: прибор
#: принимает какую-то одну, симулятор принимает обе, чтобы не предрешать ответ).
SET_SWEEP_ACCEPTED_LENGTHS = (11, 12)

#: Таймаут приёмного сокета: определяет, как быстро останавливается RX-поток.
RX_POLL_TIMEOUT_S = 0.05

#: Размер приёмного буфера сокета симулятора.
SOCKET_BUFFER_BYTES = 1 << 20


class SimState(Enum):
    """Состояние симулятора. `DEBUG` — мгновенное, между запросом и ответом 30 03."""

    IDLE = "простой"
    STREAMING = "поток телеметрии"
    DEBUG = "однократная развёртка"


@dataclass
class Faults:
    """Управляемое внесение сбоев. Все поля читаются на каждой отправке.

    Ничего из этого не является наблюдённым поведением прибора: это стимулы
    для будущих тестов сессии и отказоустойчивости (сценарии G1…G8 из KB_06).
    """

    frame_drop_probability: float = 0.0
    """Доля кадров телеметрии, которые не уходят в сеть."""

    response_delay_s: float = 0.0
    """Задержка перед ответом на команду — для проверки таймаутов сессии."""

    silent_until: float = 0.0
    """Момент `perf_counter`, до которого прибор молчит полностью."""

    bad_len: bool = False
    """Портить поле LEN в ответах: LEN не совпадает с фактической длиной."""

    garbage: bool = False
    """Отвечать мусором вместо корректного кадра."""

    def is_silent(self, now: float) -> bool:
        """True, если прибор сейчас в режиме молчания."""
        return now < self.silent_until


@dataclass
class SimStats:
    """Счётчики симулятора. Читаются из тестов после остановки."""

    requests: int = 0
    unknown_requests: int = 0
    rejected_requests: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    reboots: int = 0


@dataclass
class DeviceState:
    """Изменяемая конфигурация прибора: то, что пишется 0x20 и читается 0x10.

    Состояние настоящее, а не заглушка: команды записи меняют именно эти поля,
    а команды чтения отдают именно их.
    """

    version_raw: int
    serial: int
    speed_code: int
    channels: int
    fbg_per_channel: int
    peak_gap_ghz: int
    start_param: int
    step_param: int
    stop_param: int
    adc_step_param: int
    thresholds: list[int]
    gain_modes: list[int]
    gain_levels: list[int]
    saved_thresholds: list[int] | None = None

    @classmethod
    def factory(cls, profile: DeviceProfile) -> "DeviceState":
        """Заводская конфигурация — та, что прочитана с прибора SN 94401220."""
        return cls(
            version_raw=FACTORY_VERSION_RAW,
            serial=FACTORY_SERIAL,
            speed_code=FACTORY_SPEED_CODE,
            channels=profile.channels,
            fbg_per_channel=profile.fbg_per_channel,
            peak_gap_ghz=profile.peak_gap_ghz,
            start_param=profile.start_param,
            step_param=profile.step_param,
            stop_param=profile.stop_param,
            adc_step_param=profile.adc_step_param,
            thresholds=[FACTORY_THRESHOLD_AUTO] * profile.channels,
            gain_modes=[FACTORY_GAIN_MODE] * profile.channels,
            gain_levels=[FACTORY_GAIN_LEVEL] * profile.channels,
        )

    def channel_setup(self) -> list[tuple[int, int, int]]:
        """Тройки (порог, режим усиления, уровень) для ответа 10 06."""
        return list(zip(self.thresholds, self.gain_modes, self.gain_levels, strict=True))


# --------------------------------------------------------------------------------------
# Выдерживание темпа
# --------------------------------------------------------------------------------------


@dataclass
class PaceReport:
    """Фактический темп и джиттер отправки.

    Основные величины считаются онлайн, без массивов. Дополнительно можно
    попросить сохранять сами отклонения (`capacity` > 0): распределение
    отклонений имеет тяжёлый хвост от пауз планировщика, и одно
    среднеквадратичное отклонение его описывает плохо — 99 % кадров могут
    уходить с точностью в единицы микросекунд при отдельных выбросах
    в десятки миллисекунд. Перцентили показывают это честно, σ — нет.
    """

    frames: int = 0
    elapsed_s: float = 0.0
    resyncs: int = 0
    late_frames: int = 0
    """Кадры, ушедшие позже расчётного момента больше чем на период."""

    period_s: float = 0.0
    _dev_sum: float = 0.0
    _dev_sq: float = 0.0
    _dev_max: float = 0.0
    _samples: np.ndarray | None = None
    _sample_count: int = 0

    @property
    def rate_hz(self) -> float:
        """Фактический темп отправки, кадров/с."""
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def mean_deviation_us(self) -> float:
        """Среднее отклонение момента отправки от расчётного, мкс."""
        return self._dev_sum / self.frames * 1e6 if self.frames else 0.0

    @property
    def jitter_us(self) -> float:
        """Среднеквадратичное отклонение момента отправки от расчётного, мкс."""
        if self.frames == 0:
            return 0.0
        mean = self._dev_sum / self.frames
        variance = max(self._dev_sq / self.frames - mean * mean, 0.0)
        return variance**0.5 * 1e6

    @property
    def max_deviation_us(self) -> float:
        """Максимальное по модулю отклонение, мкс."""
        return self._dev_max * 1e6

    def percentile_us(self, q: float) -> float:
        """Перцентиль отклонения, мкс. Требует включённого сбора отклонений."""
        if self._samples is None or self._sample_count == 0:
            raise RuntimeError("отклонения не сохранялись: создайте Pacer с deviation_capacity > 0")
        return float(np.percentile(self._samples[: self._sample_count], q) * 1e6)

    def describe(self) -> str:
        """Однострочная сводка для вывода из теста."""
        text = (
            f"{self.frames} кадров за {self.elapsed_s:.2f} с → {self.rate_hz:.2f} Гц; "
            f"отклонение: среднее {self.mean_deviation_us:+.1f} мкс, "
            f"σ {self.jitter_us:.1f} мкс, макс {self.max_deviation_us:.0f} мкс; "
            f"опоздали дольше периода: {self.late_frames}; ресинхронизаций: {self.resyncs}"
        )
        if self._samples is not None and self._sample_count:
            text += (
                f"; перцентили p50 {self.percentile_us(50):.1f} "
                f"p99 {self.percentile_us(99):.1f} "
                f"p99.9 {self.percentile_us(99.9):.0f} мкс"
            )
        return text


class Pacer:
    """Выдерживание темпа отправки с точностью, недостижимой для `time.sleep`.

    При 2000 кадрах/с период равен 500 мкс. `time.sleep` промахивается на
    десятки-сотни микросекунд в Linux и до 15 мс в Windows, поэтому цикл
    `sleep(period)` дал бы кратно меньший темп: ошибка накапливается каждый шаг.

    Здесь момент отправки каждого кадра считается **от абсолютной базы**
    `base + n·period`, так что промах одного шага не смещает следующие. Остаток
    выбирается сном до `due − margin` и добирается коротким спином на
    `perf_counter`. `margin` калибруется по фактическому оверслипу `sleep`:
    если сон на этой машине грубее периода (Windows по умолчанию), margin
    окажется больше периода и цикл сам выродится в чистый спин — без
    отдельной ветки под операционную систему.

    Отставание больше `max_lag_periods` (пауза планировщика, сборка мусора)
    не догоняется залпом: база пересинхронизируется, потерянные периоды
    считаются в `resyncs`. Иначе после паузы симулятор выпалил бы очередь
    кадров и переполнил приёмный буфер, а нагрузочный тест померил бы
    размер `SO_RCVBUF`, а не приёмник.
    """

    __slots__ = (
        "_base",
        "_index",
        "_margin",
        "_started",
        "deviation_capacity",
        "max_lag_periods",
        "period",
        "report",
    )

    #: Запас к измеренному оверслипу сна.
    MARGIN_SAFETY = 1.2

    #: Нижняя граница запаса, чтобы спин не начинался слишком поздно.
    MIN_MARGIN_S = 50e-6

    def __init__(
        self,
        rate_hz: float,
        *,
        max_lag_periods: int = 50,
        deviation_capacity: int = 0,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError(f"темп {rate_hz} должен быть положительным")
        self.period = 1.0 / rate_hz
        self.max_lag_periods = max_lag_periods
        self.deviation_capacity = deviation_capacity
        self._margin: float | None = None
        self._base = 0.0
        self._started = 0.0
        self._index = 0
        self.report = PaceReport()

    @staticmethod
    def measure_sleep_overshoot(samples: int = 32, request_s: float = 500e-6) -> float:
        """Измеряет, насколько `time.sleep` промахивается мимо запрошенного срока."""
        worst = 0.0
        for _ in range(samples):
            started = time.perf_counter()
            time.sleep(request_s)
            worst = max(worst, time.perf_counter() - started - request_s)
        return worst

    def calibrate(self) -> float:
        """Подбирает запас на неточность сна. Вызывается один раз перед стартом."""
        overshoot = self.measure_sleep_overshoot()
        self._margin = max(overshoot * self.MARGIN_SAFETY, self.MIN_MARGIN_S)
        return self._margin

    @property
    def margin_s(self) -> float:
        """Текущий запас на неточность сна, секунды."""
        return self._margin if self._margin is not None else self.MIN_MARGIN_S

    def start(self) -> None:
        """Начинает отсчёт: сбрасывает базу, индекс и статистику."""
        if self._margin is None:
            self.calibrate()
        self._base = self._started = time.perf_counter()
        self._index = 0
        self.report = PaceReport(period_s=self.period)
        if self.deviation_capacity > 0:
            # Буфер выделяется заранее: аллокация в цикле с бюджетом 500 мкс недопустима.
            self.report._samples = np.empty(self.deviation_capacity, dtype=np.float64)

    def wait(self) -> None:
        """Ждёт момента отправки следующего кадра и обновляет статистику."""
        self._index += 1
        due = self._base + self._index * self.period
        margin = self.margin_s

        left = due - time.perf_counter() - margin
        if left > 0:
            time.sleep(left)
        now = time.perf_counter()
        while now < due:
            now = time.perf_counter()

        deviation = now - due
        report = self.report
        report.frames += 1
        report._dev_sum += deviation
        report._dev_sq += deviation * deviation
        report._dev_max = max(report._dev_max, abs(deviation))
        # От начала сеанса, а не от базы: при ресинхронизации база сдвигается,
        # и отсчёт от неё завысил бы темп — кадры-то уже посчитаны.
        report.elapsed_s = now - self._started
        if deviation > self.period:
            report.late_frames += 1

        samples = report._samples
        if samples is not None and report._sample_count < samples.size:
            samples[report._sample_count] = deviation
            report._sample_count += 1

        if deviation > self.max_lag_periods * self.period:
            self._base = now
            self._index = 0
            report.resyncs += 1


def _enable_high_resolution_timer() -> bool:
    """Просит Windows поднять разрешение системного таймера до 1 мс.

    Без этого `time.sleep` там квантуется 15.6 мс и спин занял бы весь период.
    На прочих системах ничего не делает.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
    except (OSError, AttributeError):  # pragma: no cover — только Windows
        return False


def _disable_high_resolution_timer() -> None:
    """Возвращает разрешение таймера Windows к умолчанию."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.WinDLL("winmm").timeEndPeriod(1)
    except (OSError, AttributeError):  # pragma: no cover — только Windows
        pass


# --------------------------------------------------------------------------------------
# Симулятор
# --------------------------------------------------------------------------------------


@dataclass
class DeviceSimulator:
    """UDP-симулятор прибора.

    `reply_to` обязателен: прибор отвечает на прописанный в нём адрес назначения,
    а не на source-порт запроса (KB_01). Сетевые адреса намеренно не заведены
    в `DeviceProfile`: тот описывает раскладку байтов, а не транспорт.

    `frame_rate_hz` перекрывает темп, расшифрованный из кода скорости в `30 02`.
    Нужен тестам: гонять всё подряд на 2000 Гц дорого.
    """

    profile: DeviceProfile
    scene: Scene
    reply_to: tuple[str, int]
    bind_to: tuple[str, int] = ("127.0.0.1", 0)
    frame_rate_hz: float | None = None
    faults: Faults = field(default_factory=Faults)
    seed: int = 0
    deviation_capacity: int = 0
    """Сколько отклонений темпа сохранять для перцентилей. 0 — не сохранять."""

    def __post_init__(self) -> None:
        self.state = DeviceState.factory(self.profile)
        self.stats = SimStats()
        self.sim_state = SimState.IDLE
        self.state_history: list[SimState] = [SimState.IDLE]
        self._rng = np.random.default_rng(self.seed)
        self._sock: socket.socket | None = None
        self._shutdown = threading.Event()
        self._streaming = threading.Event()
        self._lock = threading.Lock()
        self._rx_thread: threading.Thread | None = None
        self._tx_thread: threading.Thread | None = None
        self._encoder = MeasurementEncoder(self.profile)
        self._pacer = Pacer(self._effective_rate_hz(), deviation_capacity=self.deviation_capacity)
        self._timer_raised = False

    # --- Жизненный цикл ----------------------------------------------------------------

    def start(self) -> None:
        """Открывает сокет и запускает потоки приёма и передачи."""
        if self._sock is not None:
            raise RuntimeError("симулятор уже запущен")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_BYTES)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_BYTES)
        sock.settimeout(RX_POLL_TIMEOUT_S)
        sock.bind(self.bind_to)
        self._sock = sock
        self._shutdown.clear()
        self._streaming.clear()
        self._timer_raised = _enable_high_resolution_timer()
        self._pacer.calibrate()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="sim-rx", daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, name="sim-tx", daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()

    def stop(self) -> None:
        """Останавливает потоки и освобождает порт. Повторный вызов безвреден."""
        self._shutdown.set()
        self._streaming.clear()
        for thread in (self._tx_thread, self._rx_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        self._tx_thread = self._rx_thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._timer_raised:
            _disable_high_resolution_timer()
            self._timer_raised = False

    def __enter__(self) -> "DeviceSimulator":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        """Порт, на котором слушает симулятор. Осмыслен только после `start`."""
        if self._sock is None:
            raise RuntimeError("симулятор не запущен")
        return int(self._sock.getsockname()[1])

    @property
    def address(self) -> tuple[str, int]:
        """Адрес симулятора для отправки команд."""
        if self._sock is None:
            raise RuntimeError("симулятор не запущен")
        host, port = self._sock.getsockname()[:2]
        return str(host), int(port)

    @property
    def pace(self) -> PaceReport:
        """Отчёт о темпе последнего сеанса потоковой передачи."""
        return self._pacer.report

    @property
    def streaming(self) -> bool:
        """True, если сейчас идёт поток телеметрии."""
        return self._streaming.is_set()

    # --- Управление сбоями -------------------------------------------------------------

    def go_silent(self, seconds: float) -> None:
        """Прибор перестаёт отвечать и слать телеметрию на заданное время."""
        self.faults.silent_until = time.perf_counter() + seconds

    def reboot(self) -> None:
        """«Перезагрузка»: сброс настроек к заводским посреди работы.

        Поток останавливается, все настройки, заданные командами 0x20, теряются.
        """
        with self._lock:
            self.state = DeviceState.factory(self.profile)
            self.stats.reboots += 1
        self._streaming.clear()
        self._set_state(SimState.IDLE)

    # --- Автомат состояний -------------------------------------------------------------

    def _set_state(self, state: SimState) -> None:
        """Фиксирует переход состояния; история нужна тестам."""
        self.sim_state = state
        self.state_history.append(state)

    def _effective_rate_hz(self) -> float:
        """Темп потока: явно заданный или расшифрованный из кода скорости."""
        if self.frame_rate_hz is not None:
            return self.frame_rate_hz
        decoded = sim_decode_speed_code(self.state.speed_code)
        return float(decoded) if decoded else float(self.profile.sweep_speed_hz)

    # --- Приём и отправка --------------------------------------------------------------

    def _rx_loop(self) -> None:
        """Принимает команды и отвечает на них. Блокирующий сокет с таймаутом."""
        sock = self._sock
        assert sock is not None
        while not self._shutdown.is_set():
            try:
                request, _source = sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:  # сокет закрыт при остановке
                return
            self.stats.requests += 1
            for response in self._handle(request):
                self._send_response(response)

    def _send_response(self, payload: bytes) -> None:
        """Отправляет ответ с учётом внесённых сбоев."""
        faults = self.faults
        if faults.is_silent(time.perf_counter()):
            return
        if faults.response_delay_s > 0.0:
            time.sleep(faults.response_delay_s)
        if faults.garbage:
            payload = self._garbage()
        elif faults.bad_len:
            payload = self._corrupt_len(payload)
        sock = self._sock
        if sock is None:
            return
        # Гонка с остановкой: сокет мог закрыться между проверкой и отправкой.
        with contextlib.suppress(OSError):
            sock.sendto(payload, self.reply_to)

    def _garbage(self) -> bytes:
        """Мусорный ответ — сценарий G5 из KB_06."""
        return bytes(self._rng.integers(0, 256, size=16, dtype=np.uint8))

    def _corrupt_len(self, payload: bytes) -> bytes:
        """Портит поле LEN, оставляя всё остальное целым — сценарий G6."""
        width = self.profile.mode_len_width if payload[0] == SIM_ID_MODE else 2
        if len(payload) < 2 + width:
            return payload
        declared = int.from_bytes(payload[2 : 2 + width], "big")
        broken = (declared + 1) % (1 << (8 * width))
        return payload[:2] + broken.to_bytes(width, "big") + payload[2 + width :]

    def _tx_loop(self) -> None:
        """Гонит телеметрию, пока включён поток. В простое ждёт на событии."""
        while not self._shutdown.is_set():
            if not self._streaming.wait(RX_POLL_TIMEOUT_S):
                continue
            self._pacer.period = 1.0 / self._effective_rate_hz()
            self._pacer.start()
            while self._streaming.is_set() and not self._shutdown.is_set():
                self._pacer.wait()
                self._emit_frame()

    def _emit_frame(self) -> None:
        """Собирает и отправляет один кадр телеметрии."""
        faults = self.faults
        if faults.is_silent(time.perf_counter()):
            return
        if faults.frame_drop_probability > 0.0 and (
            self._rng.random() < faults.frame_drop_probability
        ):
            self.stats.frames_dropped += 1
            return
        self._encoder.update(self.scene.sample_freq_raw(), self.scene.sample_temp_raw())
        sock = self._sock
        if sock is None:
            return
        try:
            sock.sendto(self._encoder.frame, self.reply_to)
        except OSError:  # pragma: no cover — сокет закрыт гонкой с остановкой
            return
        self.stats.frames_sent += 1

    # --- Обработка команд --------------------------------------------------------------

    def _handle(self, request: bytes) -> list[bytes]:
        """Разбирает команду и возвращает список ответов (пустой — прибор молчит).

        Поле LEN запроса намеренно **не проверяется**: реакция прибора на
        неверный LEN — открытый вопрос N10 (сценарий G6 в KB_06). Придумывать
        её значило бы зафиксировать в тестах несуществующее поведение.
        """
        if len(request) < 3:
            self.stats.unknown_requests += 1
            return []
        ident, fc = request[0], request[1]
        with self._lock:
            if ident == SIM_ID_READ:
                return self._handle_read(fc)
            if ident == SIM_ID_WRITE:
                return self._handle_write(fc, request)
            if ident == SIM_ID_MODE:
                return self._handle_mode(fc, request)
        self.stats.unknown_requests += 1
        return []

    def _handle_read(self, fc: int) -> list[bytes]:
        """Команды чтения 0x10 — отдают текущее состояние, а не заводское."""
        state = self.state
        if fc == 0x01:
            return [encode_version(state.version_raw)]
        if fc == 0x03:
            return [encode_serial(state.serial)]
        if fc == 0x04:
            return [
                encode_module_params(
                    state.speed_code, state.channels, state.fbg_per_channel, state.peak_gap_ghz
                )
            ]
        if fc == 0x05:
            return [
                encode_sweep(
                    state.start_param, state.step_param, state.stop_param, state.adc_step_param
                )
            ]
        if fc == 0x06:
            return [encode_channel_setup(state.channel_setup())]
        self.stats.unknown_requests += 1
        return []

    def _handle_write(self, fc: int, request: bytes) -> list[bytes]:
        """Команды записи 0x20 — меняют состояние и подтверждаются ответом.

        Отказ (`00 00` вместо `00 01`) отдаётся на аргумент вне допустимого
        диапазона. Это **гипотеза**: настоящая реакция прибора на такой
        аргумент — открытый вопрос N10, сценарии G7 и G8 из KB_06.
        """
        if fc == 0x01:
            return [encode_write_ack(fc, self._apply_sweep(request))]
        if fc == 0x02:
            return [encode_write_ack(fc, self._apply_threshold(request))]
        if fc == 0x03:
            return [encode_write_ack(fc, self._apply_gain(request))]
        if fc == 0x04:
            return [encode_write_ack(fc, self._apply_peak_gap(request))]
        if fc == 0x06:
            # D4: прибор не отвечает на «сохранить пороги».
            self.state.saved_thresholds = list(self.state.thresholds)
            return []
        self.stats.unknown_requests += 1
        return []

    def _apply_sweep(self, request: bytes) -> bool:
        """20 01 — развёртка. Принимаются обе длины кадра из вопроса D3."""
        if len(request) not in SET_SWEEP_ACCEPTED_LENGTHS:
            self.stats.rejected_requests += 1
            return False
        start, step, stop, adc_step = struct.unpack(">4H", request[3:11])
        if start >= stop or step < 1 or adc_step < 1:
            self.stats.rejected_requests += 1
            return False
        self.state.start_param = start
        self.state.step_param = step
        self.state.stop_param = stop
        self.state.adc_step_param = adc_step
        return True

    def _apply_threshold(self, request: bytes) -> bool:
        """20 02 — порог канала. FF FF означает автоматический расчёт."""
        if len(request) < 6:
            self.stats.rejected_requests += 1
            return False
        channel = request[3]
        value = int.from_bytes(request[4:6], "big")
        if channel >= self.state.channels:
            self.stats.rejected_requests += 1
            return False
        if value != FACTORY_THRESHOLD_AUTO and value > self.profile.adc_max:
            self.stats.rejected_requests += 1
            return False
        self.state.thresholds[channel] = value
        return True

    def _apply_gain(self, request: bytes) -> bool:
        """20 03 — усиление канала: 00 0N автоматический режим, 80 0N ручной."""
        if len(request) < 6:
            self.stats.rejected_requests += 1
            return False
        channel, mode, level = request[3], request[4], request[5]
        if channel >= self.state.channels:
            self.stats.rejected_requests += 1
            return False
        if mode not in (0x00, SIM_GAIN_MANUAL_FLAG) or level > self.profile.gain_max_level:
            self.stats.rejected_requests += 1
            return False
        self.state.gain_modes[channel] = mode
        self.state.gain_levels[channel] = level
        return True

    def _apply_peak_gap(self, request: bytes) -> bool:
        """20 04 — минимальный интервал между пиками, ГГц, один байт."""
        if len(request) < 4:
            self.stats.rejected_requests += 1
            return False
        gap = request[3]
        if gap < 1:
            self.stats.rejected_requests += 1
            return False
        self.state.peak_gap_ghz = gap
        return True

    def _handle_mode(self, fc: int, request: bytes) -> list[bytes]:
        """Команды режимов 0x30."""
        width = self.profile.mode_len_width
        if fc == 0x01:
            self._streaming.clear()
            self._set_state(SimState.IDLE)
            return [encode_stop_ack(True, width)]
        if fc == 0x02:
            return self._start_stream(request)
        if fc == 0x03:
            return self._run_debug(width)
        if fc == 0x07:
            return self._read_raw_adc(request, width)
        self.stats.unknown_requests += 1
        return []

    def _start_stream(self, request: bytes) -> list[bytes]:
        """30 02 — старт потока. Подтверждения нет: сразу начинается телеметрия."""
        if len(request) >= 5:
            code = int.from_bytes(request[3:5], "big")
            if code != SIM_SPEED_KEEP_CURRENT:
                self.state.speed_code = code
        self._set_state(SimState.STREAMING)
        self._streaming.set()
        return []

    def _run_debug(self, width: int) -> list[bytes]:
        """30 03 — однократная развёртка: состояние возвращается в исходное.

        🔴 Раскладка тела — гипотеза, вопрос N14. См. `Scene.debug_payload`.
        """
        previous = self.sim_state
        self._set_state(SimState.DEBUG)
        payload = self.scene.debug_payload(self.state.gain_levels)
        self._set_state(previous)
        return [encode_debug(payload, width)]

    def _read_raw_adc(self, request: bytes, width: int) -> list[bytes]:
        """30 07 — сырые отсчёты АЦП одного канала.

        Номер канала вне диапазона оставляется без ответа: реакция прибора
        на такой запрос неизвестна (сценарий G7 в KB_06), а молчание —
        единственный вариант, который ничего не утверждает о приборе.
        """
        if len(request) < 6:
            self.stats.rejected_requests += 1
            return []
        channel = request[5]
        if channel >= self.state.channels:
            self.stats.rejected_requests += 1
            return []
        level = self.state.gain_levels[channel]
        spectrum = self.scene.spectrum(channel, level)
        return [encode_raw_adc(channel, self.state.gain_modes[channel], level, spectrum, width)]
