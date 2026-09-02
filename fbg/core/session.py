"""Сессия: автомат состояний, корреляция запрос-ответ, таймауты, retry, watchdog.

Сессия связывает `transport` (возит байты) и `codec` (понимает байты) и добавляет
то, чего нет ни у того, ни у другого: состояние связи, правило «одна команда
в полёте», повторы, слежение за живостью прибора и верификацию записи чтением.

Корреляция без sequence number
------------------------------
В протоколе нет ни контрольной суммы, ни номера последовательности (KB_02),
поэтому единственный признак «это ответ на мою команду» — пара `(ID, FC)`,
и то лишь пока ожидающий ровно один. Отсюда правило KB_05 №5: одна команда
в полёте. Второй одновременный запрос — ошибка вызывающего (`Busy`), а не
очередь: очередь скрыла бы от вызывающего тот факт, что он нарушил протокол,
и превратила бы отладку рассинхронизации в археологию.

Ответ, пришедший, когда ожидающего нет, или с чужой парой `(ID, FC)`,
отбрасывается и считается в `orphan_responses`. Это диагностика качества связи,
а не ошибка: чаще всего это наш собственный ответ, опоздавший после таймаута.

Потери против отказов
---------------------
Пропавшая датаграмма телеметрии **не является ошибкой сессии**: у потока нет
ни подтверждений, ни повторов, потеря в UDP штатна (R3 в KB_03). Сессия следит
только за темпом кадров и уходит в `Degraded`, когда поток пропал целиком.
Ошибка связи — это отсутствие ответа на команду, у которой ответ обязан быть.

Что сессия не делает
--------------------
Не разбирает телеметрию: сырые байты кадра уходят в колбэк `on_telemetry`,
разбор и децимация — работа `pipeline`. Не пишет файлов и не трогает Qt.
Тело ответа `30 03` разбирает кодек (N14 ✅ закрыт скринингом); сессия только
собирает его из датаграмм по объявленному LEN.

Колбэки вызываются из чужих потоков
-----------------------------------
`on_telemetry` — из потока-диспетчера транспорта, `on_state` и
`on_config_mismatch` — из потока watchdog либо из потока вызывающего.
Блокироваться в них надолго нельзя (см. контракт `tap` в `transport.py`).

Журнал пакетов — необязательный отвод, а не зависимость
-------------------------------------------------------
Проводка журнала (Р54) сделана двумя колбэками `log_rx` и `log_tx`, а не
объектом журнала: `fbg/core` не импортирует `fbg/io` и не должен (направление
зависимостей в KB_03). Сигнатуры совпадают с методами `fbg.io.packet_log`,
поэтому подключение выглядит как ``Session(..., log_rx=log.log_rx,
log_tx=log.log_tx)``, но сессия про журнал ничего не знает и без него
работает ровно как раньше.

`log_rx` зовётся **первой строкой** `_on_datagram`, до классификации: сырые
байты попадают в журнал до разбора всегда (KB_05 №3). `log_tx` зовётся из
`_send` после успешной отправки. Отказ любого из них считается в
`SessionStats.tap_errors` и обмен не прерывает (Р50).
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType

from fbg.core import codec
from fbg.core.endpoint import Endpoint
from fbg.core.frames import (
    AdcBlock,
    ChannelSetup,
    DebugResponse,
    GainSetting,
    ModuleParams,
    ParseResult,
    SweepConfig,
)
from fbg.core.profile import DeviceProfile
from fbg.core.transport import TransportStats, UdpTransport

#: Шаг опроса события ответа. Определяет, насколько фактический таймаут может
#: превысить заданный; на бюджет команд (сотни миллисекунд) влияния не имеет.
POLL_SLICE_S = 0.005

#: Из режимной группы 0x30 во время Streaming разрешён только Stop.
#:
#: Р62 подтверждён скринингом 01.09.2026: чтения 0x10 и записи 0x20 проходят
#: без остановки телеметрии, а 0x30 вытесняет поток. Keepalive в Streaming
#: по-прежнему не нужен: живость там проверяется самими кадрами телеметрии.
STREAMING_ALLOWED_MODE: frozenset[tuple[int, int]] = frozenset({(codec.ID_MODE, codec.FC_STOP)})

#: Команды, ответ на которые протоколом не предусмотрен.
#:
#: `20 06` — D4 подтверждён скринингом. `30 02` — старт потока:
#: подтверждения нет, сразу идёт телеметрия (KB_02).
NO_RESPONSE: frozenset[tuple[int, int]] = codec.NO_RESPONSE_COMMANDS | frozenset(
    {(codec.ID_MODE, codec.FC_STREAM)}
)

#: Ответы, не помещающиеся в одну датаграмму (D5 ✅ закрыт скринингом):
#: `30 07` — 5112 байт (4 датаграммы), `30 03` — 20430 байт (14 датаграмм).
LONG_RESPONSES: frozenset[tuple[int, int]] = frozenset(
    {(codec.ID_MODE, codec.FC_RAW_ADC), (codec.ID_MODE, codec.FC_DEBUG)}
)


# --------------------------------------------------------------------------------------
# Состояния и ошибки
# --------------------------------------------------------------------------------------


class SessionState(Enum):
    """Состояние связи с прибором."""

    DISCONNECTED = "не подключено"
    PROBING = "опрос конфигурации"
    IDLE = "готов к командам"
    STREAMING = "идёт поток телеметрии"
    DEBUG = "однократная развёртка"
    DEGRADED = "связь потеряна, идёт пробник"
    RECONNECTING = "переподключение с задержкой"


class SessionErrorKind(Enum):
    """Причина отказа операции сессии."""

    NOT_CONNECTED = "сессия не подключена"
    WRONG_STATE = "команда недопустима в текущем состоянии"
    BUSY = "предыдущая команда ещё не завершена"
    SEND_FAILED = "не удалось отправить датаграмму"
    TIMEOUT = "прибор не ответил за отведённое время"
    INCOMPLETE_RESPONSE = "ответ не собран целиком до таймаута добора"
    BAD_RESPONSE = "ответ пришёл, но не разобрался"
    DEVICE_REJECTED = "прибор ответил отказом"
    VERIFICATION_MISMATCH = "прочитанное значение не совпало с записанным"
    CANCELLED = "операция отменена"


@dataclass(frozen=True)
class SessionError:
    """Отказ операции: вид и человекочитаемое пояснение."""

    kind: SessionErrorKind
    message: str

    def __str__(self) -> str:
        return f"{self.kind.name}: {self.message}"


@dataclass(frozen=True)
class Result[T]:
    """Либо значение, либо ошибка. Ровно одно из двух заполнено.

    Ошибки сессии возвращаются, а не бросаются (KB_03): отсутствие ответа
    от прибора — штатная ситуация. Исключения остаются за программными багами.
    """

    value: T | None = None
    error: SessionError | None = None

    @property
    def ok(self) -> bool:
        """True, если операция удалась."""
        return self.error is None

    def unwrap(self) -> T:
        """Возвращает значение; при ошибке бросает. Для тестов и мест, где ошибка — баг."""
        if self.error is not None:
            raise RuntimeError(str(self.error))
        assert self.value is not None
        return self.value


def _ok[T](value: T) -> Result[T]:
    """Успешный результат."""
    return Result(value=value)


def _fail[T](kind: SessionErrorKind, message: str) -> Result[T]:
    """Неуспешный результат."""
    return Result(error=SessionError(kind, message))


# --------------------------------------------------------------------------------------
# Конфигурация прибора и параметры сессии
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceConfig:
    """Всё, что прибор рассказал о себе при опросе.

    Собирается пятью командами чтения. Хранится, чтобы после восстановления
    связи было с чем сравнить: прибор мог быть перезагружен по питанию
    и потерять всё, что мы в него записали.
    """

    version_raw: int
    serial: int
    module: ModuleParams
    sweep: SweepConfig
    channels: tuple[ChannelSetup, ...]

    @property
    def version(self) -> str:
        """Версия прошивки в человеческом виде: 410 → «4.10»."""
        return codec.format_version(self.version_raw)

    def differences(self, other: "DeviceConfig") -> tuple[str, ...]:
        """Перечисляет расхождения с другой конфигурацией, `self` — прежняя."""
        diffs: list[str] = []
        if self.version_raw != other.version_raw:
            diffs.append(f"версия прошивки: было {self.version}, стало {other.version}")
        if self.serial != other.serial:
            diffs.append(f"серийный номер: было {self.serial}, стало {other.serial}")
        if self.module != other.module:
            diffs.append(f"параметры модуля: было {self.module}, стало {other.module}")
        if self.sweep != other.sweep:
            diffs.append(f"развёртка: было {self.sweep}, стало {other.sweep}")
        if len(self.channels) != len(other.channels):
            diffs.append(
                f"число каналов в ответе 10 06: было {len(self.channels)}, "
                f"стало {len(other.channels)}"
            )
        for was, now in zip(self.channels, other.channels, strict=False):
            if was != now:
                diffs.append(f"канал {was.channel}: было {was}, стало {now}")
        return tuple(diffs)


@dataclass(frozen=True)
class SessionConfig:
    """Параметры поведения сессии, которых нет в `Endpoint`.

    Таймауты команд и число повторов живут в `Endpoint` — это сетевые
    настройки, которые пользователь правит в одном диалоге. Здесь остаётся
    политика: как часто щупать живость, когда считать связь потерянной,
    как долго ждать перед переподключением.

    ⚠️ Значения по умолчанию — **не наблюдения, а инженерные умолчания**
    (вопрос N8: есть ли keepalive у штатного ПО, неизвестно, захвата нет).
    Поэтому они параметры, а не константы: после фазы 1 скрининга правится
    умолчание, а не код.
    """

    keepalive_period_s: float = 2.0
    """Период keepalive `10 01` в состоянии Idle (N8)."""

    keepalive_failures_to_degrade: int = 3
    """Сколько keepalive подряд должны не ответить, чтобы уйти в Degraded (N8)."""

    stream_stall_floor_s: float = 0.5
    """Нижняя граница окна тишины в потоке."""

    stream_stall_periods: int = 20
    """Окно тишины в периодах развёртки; период берётся из прочитанных 10 04."""

    backoff_schedule: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    """Паузы между попытками переподключения; последняя — потолок."""

    retry_pause_s: float = 0.2
    """Пауза между повторами команды (KB_03)."""

    reassembly_timeout_s: float = 0.5
    """Сколько ждать очередной кусок длинного ответа после предыдущего (D5)."""

    watchdog_tick_s: float = 0.05
    """Шаг работы watchdog: с такой точностью замечаются сроки."""

    settle_before_readback_s: float = 0.05
    """Пауза перед read-back после команды без ответа (`20 06`)."""

    auto_reconnect: bool = True
    """Уходить ли в Reconnecting при неудачном опросе, вместо возврата в Disconnected."""

    def __post_init__(self) -> None:
        """Проверяет согласованность параметров: некорректные — баг вызывающего."""
        for name, value in (
            ("keepalive_period_s", self.keepalive_period_s),
            ("stream_stall_floor_s", self.stream_stall_floor_s),
            ("reassembly_timeout_s", self.reassembly_timeout_s),
            ("watchdog_tick_s", self.watchdog_tick_s),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} должен быть положительным")
        if self.keepalive_failures_to_degrade < 1:
            raise ValueError("keepalive_failures_to_degrade должен быть ≥ 1")
        if self.stream_stall_periods < 1:
            raise ValueError("stream_stall_periods должен быть ≥ 1")
        if not self.backoff_schedule:
            raise ValueError("backoff_schedule не может быть пустым")
        if any(pause < 0 for pause in self.backoff_schedule):
            raise ValueError("паузы backoff не могут быть отрицательными")
        if self.retry_pause_s < 0 or self.settle_before_readback_s < 0:
            raise ValueError("паузы не могут быть отрицательными")


# --------------------------------------------------------------------------------------
# Счётчики
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionStats:
    """Снимок счётчиков сессии."""

    commands: int = 0
    retries: int = 0
    timeouts: int = 0
    orphan_responses: int = 0
    """Ответы, пришедшие без ожидающего или с чужой парой (ID, FC)."""

    telemetry_frames: int = 0
    degraded_events: int = 0
    reconnect_attempts: int = 0
    verification_mismatches: int = 0
    keepalive_failures: int = 0
    incomplete_responses: int = 0
    tap_errors: int = 0
    """Отказы журнала пакетов. Сессию они не роняют (Р50), но и не молчат."""

    errors: dict[str, int] = field(default_factory=dict)
    """Исключения в потоке watchdog по имени класса — признак бага, а не связи."""


@dataclass
class _Counters:
    """Изменяемые счётчики.

    Блокировка не нужна: у каждого счётчика один писатель либо запись идёт
    под замком команды. `telemetry_frames` и `orphan_responses` пишет только
    поток-диспетчер транспорта; `commands`, `retries`, `timeouts`,
    `incomplete_responses`, `verification_mismatches` — тот, кто держит
    `_command_lock`; `degraded_events`, `reconnect_attempts`,
    `keepalive_failures` — только поток watchdog.
    """

    commands: int = 0
    retries: int = 0
    timeouts: int = 0
    orphan_responses: int = 0
    telemetry_frames: int = 0
    degraded_events: int = 0
    reconnect_attempts: int = 0
    verification_mismatches: int = 0
    keepalive_failures: int = 0
    incomplete_responses: int = 0
    tap_errors: int = 0


@dataclass
class _Pending:
    """Единственный ожидаемый ответ.

    `declared` не None означает, что идёт сборка длинного ответа: первая
    датаграмма объявила LEN больше собственной длины (D5).
    """

    ident: int
    fc: int
    reassemble: bool
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    buffer: bytearray = field(default_factory=bytearray)
    declared: int | None = None
    frame: bytes | None = None


# --------------------------------------------------------------------------------------
# Сессия
# --------------------------------------------------------------------------------------


class Session:
    """Обмен с прибором: состояние, корреляция, повторы, живость, верификация.

    Сессия владеет транспортом: создаёт его в конструкторе, открывает
    в `connect` и закрывает в `disconnect`. Отдельный поток watchdog живёт
    столько же, сколько открыт транспорт.

    Все команды синхронные: метод возвращается, когда ответ получен, разобран
    и (для записи) подтверждён чтением. Одновременный вызов из двух потоков —
    ошибка `Busy` у второго, а не очередь.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        profile: DeviceProfile | None = None,
        config: SessionConfig | None = None,
        *,
        on_telemetry: Callable[[bytes, float], None] | None = None,
        on_state: Callable[[SessionState, SessionState], None] | None = None,
        on_config_mismatch: Callable[[tuple[str, ...]], None] | None = None,
        log_rx: Callable[[bytes, float], None] | None = None,
        log_tx: Callable[[bytes, float], None] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._profile = profile or DeviceProfile()
        self._config = config or SessionConfig()
        self._on_telemetry = on_telemetry
        self._on_state = on_state
        self._on_config_mismatch = on_config_mismatch
        self._log_rx = log_rx
        self._log_tx = log_tx

        self._transport = UdpTransport(endpoint, self._on_datagram)
        self._counters = _Counters()
        self._errors: dict[str, int] = {}

        self._state = SessionState.DISCONNECTED
        self._state_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._lock_owner: str | None = None
        self._pending_lock = threading.Lock()
        self._pending: _Pending | None = None

        self._shutdown = threading.Event()
        self._watchdog: threading.Thread | None = None

        self._device: DeviceConfig | None = None
        self._mismatch: tuple[str, ...] = ()
        self._unconfirmed: set[str] = set()
        self._stream_interrupted = False

        self._last_ok_mono = 0.0
        self._last_frame_mono = 0.0
        self._keepalive_failures = 0
        self._state_before_degraded = SessionState.IDLE
        self._backoff_index = 0
        self._next_attempt_mono = 0.0

    # --- Состояние ---------------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        """Текущее состояние автомата."""
        with self._state_lock:
            return self._state

    @property
    def device_config(self) -> DeviceConfig | None:
        """Последняя успешно прочитанная конфигурация прибора."""
        return self._device

    @property
    def config_mismatch(self) -> tuple[str, ...]:
        """Расхождения конфигурации, замеченные после восстановления связи.

        Непустой кортеж означает, что прибор вернулся другим, чем уходил —
        вероятнее всего, был перезагружен по питанию. Сессия ничего
        не перезаписывает сама: решение принимает оператор.
        """
        return self._mismatch

    @property
    def stream_interrupted(self) -> bool:
        """True, если поток телеметрии прервался восстановлением связи."""
        return self._stream_interrupted

    @property
    def unconfirmed(self) -> frozenset[str]:
        """Значения, записанные без подтверждения read-back'ом."""
        return frozenset(self._unconfirmed)

    @property
    def profile(self) -> DeviceProfile:
        """Профиль, с которым разбираются кадры."""
        return self._profile

    @property
    def transport_stats(self) -> TransportStats:
        """Счётчики транспорта."""
        return self._transport.stats()

    @property
    def local_address(self) -> tuple[str, int]:
        """Фактический адрес приёма. Осмыслен только при открытом транспорте."""
        return self._transport.local_address

    @property
    def last_frame_mono(self) -> float:
        """Момент `perf_counter` последнего кадра телеметрии."""
        return self._last_frame_mono

    def stats(self) -> SessionStats:
        """Снимок счётчиков сессии."""
        counters = self._counters
        return SessionStats(
            commands=counters.commands,
            retries=counters.retries,
            timeouts=counters.timeouts,
            orphan_responses=counters.orphan_responses,
            telemetry_frames=counters.telemetry_frames,
            degraded_events=counters.degraded_events,
            reconnect_attempts=counters.reconnect_attempts,
            verification_mismatches=counters.verification_mismatches,
            keepalive_failures=counters.keepalive_failures,
            incomplete_responses=counters.incomplete_responses,
            tap_errors=counters.tap_errors,
            errors=dict(self._errors),
        )

    def _set_state(self, new: SessionState) -> None:
        """Меняет состояние и уведомляет подписчика вне блокировки."""
        with self._state_lock:
            old = self._state
            if old is new:
                return
            self._state = new
        if self._on_state is not None:
            self._on_state(old, new)

    # --- Жизненный цикл ----------------------------------------------------------------

    def connect(self) -> Result[DeviceConfig]:
        """Подключается и опрашивает прибор: Stop, затем пять команд чтения.

        `Stop` первым обязателен (KB_05 №6): прибор мог остаться в потоке
        с прошлого сеанса, и тогда любой ответ утонул бы в телеметрии.

        При неудаче опроса сессия либо уходит в `Reconnecting` и продолжает
        попытки в фоне (`SessionConfig.auto_reconnect`), либо возвращается
        в `Disconnected`, закрыв транспорт. Ошибка возвращается в обоих случаях.
        """
        if self.state is not SessionState.DISCONNECTED:
            return _fail(
                SessionErrorKind.WRONG_STATE,
                f"connect допустим только из Disconnected, сейчас {self.state.name}",
            )
        self._shutdown.clear()
        self._transport.open()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="fbg-watchdog", daemon=True
        )
        self._watchdog.start()

        self._set_state(SessionState.PROBING)
        probed = self._probe(with_stop=True)
        if probed.error is not None:
            if self._config.auto_reconnect:
                self._enter_reconnecting()
            else:
                self._teardown()
            return probed

        self._device = probed.value
        self._last_ok_mono = time.perf_counter()
        self._set_state(SessionState.IDLE)
        return probed

    def disconnect(self) -> None:
        """Останавливает прибор и закрывает связь. Повторный вызов безвреден.

        `Stop` уходит в `finally` (KB_05 №6): что бы ни случилось выше,
        прибор не должен остаться льющим поток в закрытый порт. Ошибка
        отправки здесь игнорируется намеренно — связи может уже не быть,
        а закрыть транспорт нужно в любом случае.
        """
        try:
            if self._transport.is_open:
                self._send(codec.build_stop())
        except (OSError, RuntimeError):
            pass
        finally:
            self._teardown()

    def close(self) -> None:
        """Синоним `disconnect` — для симметрии с транспортом."""
        self.disconnect()

    def _teardown(self) -> None:
        """Останавливает watchdog, закрывает транспорт, сбрасывает состояние."""
        self._shutdown.set()
        watchdog = self._watchdog
        if watchdog is not None and watchdog.is_alive():
            watchdog.join(timeout=self._config.watchdog_tick_s + 5.0)
        self._watchdog = None
        self._transport.close()
        with self._pending_lock:
            self._pending = None
        self._set_state(SessionState.DISCONNECTED)

    def __enter__(self) -> "Session":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

    # --- Приём датаграмм ---------------------------------------------------------------

    def _on_datagram(self, data: bytes, t_mono: float) -> None:
        """Колбэк транспорта: журнал, затем телеметрия потребителю или корреляция.

        Вызывается из потока-диспетчера транспорта. Разбора здесь нет:
        телеметрия уходит сырыми байтами (KB_05 №3 — журналу нужны байты
        до разбора), корреляция работает по двум первым байтам.

        Журнал стоит **первой строкой**, до `classify` и до любого ветвления
        (правило KB_05 №3): байты обязаны попасть в него раньше, чем кто-либо
        решит, что они означают, — иначе датаграмма, которую мы не сумели
        опознать, в журнал не попала бы вовсе, а именно она при отладке
        протокола интереснее всего (Р48).
        """
        self._tap(self._log_rx, data, t_mono)
        key = codec.classify(data)
        if key == (codec.ID_MODE, codec.FC_STREAM):
            self._last_frame_mono = t_mono
            self._counters.telemetry_frames += 1
            if self._on_telemetry is not None:
                self._on_telemetry(data, t_mono)
            return

        with self._pending_lock:
            pending = self._pending
            if pending is None:
                self._counters.orphan_responses += 1
                return
            if pending.declared is not None:
                self._append_fragment(pending, data)
                return
            if key != (pending.ident, pending.fc):
                self._counters.orphan_responses += 1
                return
            if pending.reassemble and self._starts_long_response(pending, data):
                self._append_fragment(pending, data)
                return
            pending.frame = data
            pending.done.set()

    def _tap(self, sink: Callable[[bytes, float], None] | None, data: bytes, t_mono: float) -> None:
        """Отдаёт байты журналу, если он подключён. Его отказ сессию не роняет.

        Журнал по отношению к обмену вторичен (Р50): диагностический модуль
        не имеет права остановить протокол. Поэтому здесь ловится `Exception`
        целиком — включая баг в самом журнале, — но не молча: отказ считается
        в `SessionStats.tap_errors`, потому что тихой потери в тракте быть
        не должно нигде (KB_05 №13).

        Дорогого здесь не делается ничего: контракт `log_rx` — положить байты
        в свою очередь и вернуться (KB_05 №23, №28).
        """
        if sink is None:
            return
        try:
            sink(data, t_mono)
        except Exception:
            self._counters.tap_errors += 1

    def _send(self, request: bytes) -> bool:
        """Отправляет датаграмму и отдаёт её журналу. False — сокет отверг.

        Журнал зовётся **после** успешной отправки: он ведёт запись провода,
        а команда, которую сокет не принял, на проводе не появилась. Отказ
        отправки при этом не теряется — его считает транспорт, а сессия
        возвращает отдельный `SEND_FAILED`.
        """
        if not self._transport.send(request):
            return False
        self._tap(self._log_tx, request, time.perf_counter())
        return True

    def _starts_long_response(self, pending: _Pending, data: bytes) -> bool:
        """True, если первая датаграмма объявила LEN больше собственной длины (D5).

        ✅ Вопрос D5 закрыт скринингом, и принятая здесь модель оказалась верной:
        первая датаграмма несёт заголовок с полным LEN, продолжения идут
        **без заголовка**, чистыми данными. `30 07` пришёл 5112 байтами
        четырьмя датаграммами (1472 · 1472 · 1472 · 696), `30 03` — 20430
        байтами по той же схеме.
        """
        width = self._len_width(pending.ident)
        if len(data) < 2 + width:
            return False
        declared = int.from_bytes(data[2 : 2 + width], "big")
        if declared <= len(data):
            return False
        pending.declared = declared
        return True

    def _append_fragment(self, pending: _Pending, data: bytes) -> None:
        """Дописывает кусок длинного ответа и продлевает срок добора."""
        pending.buffer += data
        pending.deadline = time.perf_counter() + self._config.reassembly_timeout_s
        assert pending.declared is not None
        if len(pending.buffer) >= pending.declared:
            pending.frame = bytes(pending.buffer)
            pending.done.set()

    def _len_width(self, ident: int) -> int:
        """Ширина поля LEN в ответе: 2 байта для 0x10 и 0x20, профиль — для 0x30."""
        return self._profile.mode_len_width if ident == codec.ID_MODE else codec.RESP_LEN_WIDTH

    # --- Обмен -------------------------------------------------------------------------

    def _acquire_command_lock(self, internal: bool) -> bool:
        """Берёт право на единственную команду в полёте.

        Пользовательская команда получает `Busy` мгновенно, если замок держит
        **другая пользовательская** команда: это нарушение правила KB_05 №5
        и ошибка вызывающего, которую нельзя прятать за очередью.

        Если же замок держит наша собственная служебная команда — keepalive
        или пробник, — вызывающий ничего не нарушал, и он ждёт её завершения,
        а не получает отказ. Иначе нажатие кнопки в UI изредка отбивалось бы
        сообщением про незавершённую команду, которой пользователь не давал.
        """
        if self._command_lock.acquire(blocking=False):
            self._lock_owner = "internal" if internal else "user"
            return True
        if internal or self._lock_owner != "internal":
            return False
        patience = self._endpoint.read_timeout_s + self._config.retry_pause_s
        if not self._command_lock.acquire(timeout=patience):
            return False
        self._lock_owner = "user"
        return True

    def _release_command_lock(self) -> None:
        """Отпускает замок команды."""
        self._lock_owner = None
        self._command_lock.release()

    def _check_state(self, ident: int, fc: int, internal: bool) -> SessionError | None:
        """Проверяет, допустима ли команда в текущем состоянии.

        Р62 подтверждено скринингом 01.09.2026, а не принято из осторожности:
        во время потока группы 0x10 и 0x20 проходят без пропусков телеметрии,
        тогда как режимная группа 0x30 вытесняет поток. Поэтому в Streaming
        запрещены только команды 0x30, кроме Stop ``30 01``.
        """
        state = self.state
        if state is SessionState.DISCONNECTED:
            return SessionError(SessionErrorKind.NOT_CONNECTED, "сессия не подключена")
        if internal:
            return None
        mode_forbidden = ident == codec.ID_MODE and (ident, fc) not in STREAMING_ALLOWED_MODE
        if state is SessionState.STREAMING and mode_forbidden:
            return SessionError(
                SessionErrorKind.WRONG_STATE,
                f"команда {ident:02X} {fc:02X} запрещена во время потока: "
                "режимная группа 0x30 вытесняет Streaming (Р62); разрешён только Stop 30 01",
            )
        if state in (SessionState.PROBING, SessionState.DEGRADED, SessionState.RECONNECTING):
            return SessionError(
                SessionErrorKind.WRONG_STATE,
                f"команда {ident:02X} {fc:02X} недопустима в состоянии {state.name}",
            )
        return None

    def _exchange(
        self, request: bytes, ident: int, fc: int, *, timeout: float, reassemble: bool
    ) -> Result[bytes]:
        """Одна попытка: отправить команду и дождаться сырых байт ответа."""
        pending = _Pending(
            ident=ident, fc=fc, reassemble=reassemble, deadline=time.perf_counter() + timeout
        )
        with self._pending_lock:
            self._pending = pending
        try:
            if not self._send(request):
                return _fail(
                    SessionErrorKind.SEND_FAILED,
                    f"сокет отверг команду {ident:02X} {fc:02X}; см. transport.stats().errors",
                )
            while not self._shutdown.is_set():
                with self._pending_lock:
                    deadline = pending.deadline
                    partial = pending.declared is not None
                    collected = len(pending.buffer)
                    declared = pending.declared
                left = deadline - time.perf_counter()
                if left <= 0:
                    if partial:
                        return _fail(
                            SessionErrorKind.INCOMPLETE_RESPONSE,
                            f"ответ {ident:02X} {fc:02X} собран не полностью: "
                            f"{collected} из объявленных {declared} байт (вопрос D5)",
                        )
                    return _fail(
                        SessionErrorKind.TIMEOUT,
                        f"нет ответа на {ident:02X} {fc:02X} за {timeout:.3f} с",
                    )
                if pending.done.wait(min(left, POLL_SLICE_S)):
                    assert pending.frame is not None
                    return _ok(pending.frame)
            return _fail(SessionErrorKind.CANCELLED, "сессия закрывается")
        finally:
            with self._pending_lock:
                self._pending = None

    def _request[T](
        self,
        request: bytes,
        ident: int,
        fc: int,
        parse: Callable[[bytes], ParseResult[T]],
        *,
        timeout: float,
        reassemble: bool = False,
        internal: bool = False,
        attempts: int | None = None,
    ) -> Result[T]:
        """Команда с ответом: проверка состояния, отправка, повторы, разбор.

        Повторяются таймауты, недобор длинного ответа и неразобравшийся ответ:
        всё это может быть следствием потери или порчи датаграммы, а чтения
        идемпотентны. Не повторяются отказ прибора и отказ сокета: первое —
        осознанный ответ, второе — локальная проблема, которая не рассосётся
        от повтора.
        """
        gate = self._check_state(ident, fc, internal)
        if gate is not None:
            return Result(error=gate)
        if not self._acquire_command_lock(internal):
            return _fail(
                SessionErrorKind.BUSY,
                f"команда {ident:02X} {fc:02X} отклонена: предыдущая ещё в полёте. "
                "Sequence number в протоколе отсутствует, поэтому одновременных "
                "команд быть не может (KB_05 №5)",
            )
        try:
            return self._request_locked(
                request,
                ident,
                fc,
                parse,
                timeout=timeout,
                reassemble=reassemble,
                attempts=attempts if attempts is not None else 1 + self._endpoint.retries,
            )
        finally:
            self._release_command_lock()

    def _request_locked[T](
        self,
        request: bytes,
        ident: int,
        fc: int,
        parse: Callable[[bytes], ParseResult[T]],
        *,
        timeout: float,
        reassemble: bool,
        attempts: int,
    ) -> Result[T]:
        """Тело `_request` под уже взятым замком команды."""
        last: Result[T] = _fail(SessionErrorKind.TIMEOUT, "нет попыток")
        for attempt in range(max(attempts, 1)):
            if attempt:
                self._counters.retries += 1
                if self._shutdown.wait(self._config.retry_pause_s):
                    return _fail(SessionErrorKind.CANCELLED, "сессия закрывается")
            self._counters.commands += 1
            raw = self._exchange(request, ident, fc, timeout=timeout, reassemble=reassemble)
            if raw.error is not None:
                if raw.error.kind is SessionErrorKind.TIMEOUT:
                    self._counters.timeouts += 1
                elif raw.error.kind is SessionErrorKind.INCOMPLETE_RESPONSE:
                    self._counters.incomplete_responses += 1
                else:
                    return Result(error=raw.error)
                last = Result(error=raw.error)
                continue
            assert raw.value is not None
            parsed = parse(raw.value)
            if parsed.ok:
                self._last_ok_mono = time.perf_counter()
                self._keepalive_failures = 0
                return _ok(parsed.unwrap())
            last = _fail(
                SessionErrorKind.BAD_RESPONSE,
                f"ответ {ident:02X} {fc:02X} не разобрался: {parsed.error}",
            )
        return last

    def _send_only(
        self, request: bytes, ident: int, fc: int, *, internal: bool = False
    ) -> Result[None]:
        """Команда без ответа: отправить и не ждать (D4 и старт потока).

        Подтверждения нет по протоколу, поэтому «успех» здесь означает ровно
        одно — датаграмма отдана ядру. Реально ли команда применилась,
        показывает только последующее чтение.
        """
        gate = self._check_state(ident, fc, internal)
        if gate is not None:
            return Result(error=gate)
        if not self._acquire_command_lock(internal):
            return _fail(
                SessionErrorKind.BUSY, f"команда {ident:02X} {fc:02X}: предыдущая в полёте"
            )
        try:
            self._counters.commands += 1
            if not self._send(request):
                return _fail(
                    SessionErrorKind.SEND_FAILED, f"сокет отверг команду {ident:02X} {fc:02X}"
                )
            return _ok(None)
        finally:
            self._release_command_lock()

    # --- Команды чтения ----------------------------------------------------------------

    def read_version(self, *, internal: bool = False) -> Result[int]:
        """10 01 — версия прошивки в сотых долях (410 = v4.10)."""
        return self._request(
            codec.build_read_version(),
            codec.ID_READ,
            codec.FC_VERSION,
            codec.parse_version,
            timeout=self._endpoint.read_timeout_s,
            internal=internal,
        )

    def read_serial(self, *, internal: bool = False) -> Result[int]:
        """10 03 — серийный номер."""
        return self._request(
            codec.build_read_serial(),
            codec.ID_READ,
            codec.FC_SERIAL,
            codec.parse_serial,
            timeout=self._endpoint.read_timeout_s,
            internal=internal,
        )

    def read_module_params(self, *, internal: bool = False) -> Result[ModuleParams]:
        """10 04 — скорость развёртки, каналы, решётки, интервал пиков."""
        return self._request(
            codec.build_read_module_params(),
            codec.ID_READ,
            codec.FC_MODULE_PARAMS,
            codec.parse_module_params,
            timeout=self._endpoint.read_timeout_s,
            internal=internal,
        )

    def read_sweep(self, *, internal: bool = False) -> Result[SweepConfig]:
        """10 05 — параметры развёртки."""
        return self._request(
            codec.build_read_sweep(),
            codec.ID_READ,
            codec.FC_SWEEP,
            lambda frame: codec.parse_sweep_params(frame, self._profile),
            timeout=self._endpoint.read_timeout_s,
            internal=internal,
        )

    def read_channel_setup(self, *, internal: bool = False) -> Result[tuple[ChannelSetup, ...]]:
        """10 06 — пороги и усиления всех каналов."""
        return self._request(
            codec.build_read_channel_setup(),
            codec.ID_READ,
            codec.FC_CHANNEL_SETUP,
            lambda frame: codec.parse_channel_setup(frame, self._profile),
            timeout=self._endpoint.read_timeout_s,
            internal=internal,
        )

    def refresh_config(self, *, internal: bool = False) -> Result[DeviceConfig]:
        """Перечитывает конфигурацию целиком и запоминает её как текущую."""
        probed = self._probe(with_stop=False, internal=internal)
        if probed.ok:
            self._device = probed.value
        return probed

    def _probe(self, *, with_stop: bool, internal: bool = True) -> Result[DeviceConfig]:
        """Опрос конфигурации: Stop, затем 10 01, 10 03, 10 04, 10 05, 10 06.

        ⚠️ Порядок выбран нами, а не подсмотрен у штатного ПО: вопрос N7
        открыт, захвата старта штатного ПО не существует. Логика порядка —
        от неизменного к изменяемому: сначала кто это (версия, серийный),
        потом что он умеет (модуль), потом как настроен (развёртка, каналы).
        """
        if with_stop:
            stopped = self._stop_command(internal=internal)
            if stopped.error is not None:
                return Result(error=stopped.error)
        version = self.read_version(internal=internal)
        if version.error is not None:
            return Result(error=version.error)
        serial = self.read_serial(internal=internal)
        if serial.error is not None:
            return Result(error=serial.error)
        module = self.read_module_params(internal=internal)
        if module.error is not None:
            return Result(error=module.error)
        sweep = self.read_sweep(internal=internal)
        if sweep.error is not None:
            return Result(error=sweep.error)
        channels = self.read_channel_setup(internal=internal)
        if channels.error is not None:
            return Result(error=channels.error)
        return _ok(
            DeviceConfig(
                version_raw=version.unwrap(),
                serial=serial.unwrap(),
                module=module.unwrap(),
                sweep=sweep.unwrap(),
                channels=channels.unwrap(),
            )
        )

    # --- Команды записи с верификацией -------------------------------------------------

    def _write(self, request: bytes, fc: int, *, internal: bool = False) -> Result[bool]:
        """Команда записи 0x20 с разбором подтверждения."""
        return self._request(
            request,
            codec.ID_WRITE,
            fc,
            codec.parse_write_ack,
            timeout=self._endpoint.write_timeout_s,
            internal=internal,
        )

    def _acknowledged(self, fc: int, ack: Result[bool]) -> SessionError | None:
        """Превращает подтверждение записи в ошибку, если прибор отказал."""
        if ack.error is not None:
            return ack.error
        if not ack.value:
            return SessionError(
                SessionErrorKind.DEVICE_REJECTED,
                f"прибор ответил отказом (00 00) на команду 20 {fc:02X}",
            )
        return None

    def _verified[T](self, key: str, expected: object, actual: object, value: T) -> Result[T]:
        """Сверяет записанное с прочитанным и ведёт список неподтверждённых.

        Read-back — часть контракта записи, а не опция: без него «успех»
        означает лишь то, что прибор ответил `00 01`, а не то, что значение
        применилось (KB_03).
        """
        if expected == actual:
            self._unconfirmed.discard(key)
            return _ok(value)
        self._unconfirmed.add(key)
        self._counters.verification_mismatches += 1
        return _fail(
            SessionErrorKind.VERIFICATION_MISMATCH,
            f"{key}: записано {expected}, прочитано {actual}; значение не подтверждено",
        )

    def set_sweep(self, sweep: SweepConfig) -> Result[SweepConfig]:
        """20 01 — задать развёртку, затем проверить чтением 10 05."""
        request = codec.build_set_sweep(sweep, self._profile)
        problem = self._acknowledged(codec.FC_SET_SWEEP, self._write(request, codec.FC_SET_SWEEP))
        if problem is not None:
            self._unconfirmed.add("sweep")
            return Result(error=problem)
        readback = self.read_sweep()
        if readback.error is not None:
            self._unconfirmed.add("sweep")
            return Result(error=readback.error)
        actual = readback.unwrap()
        self._remember_sweep(actual)
        expected = (sweep.start_param, sweep.step_param, sweep.stop_param, sweep.adc_step_param)
        got = (actual.start_param, actual.step_param, actual.stop_param, actual.adc_step_param)
        return self._verified("sweep", expected, got, actual)

    def set_threshold(self, channel: int, threshold: int | None) -> Result[ChannelSetup]:
        """20 02 — задать порог канала (None — авторасчёт), затем проверить 10 06."""
        request = codec.build_set_threshold(channel, threshold, self._profile)
        problem = self._acknowledged(
            codec.FC_SET_THRESHOLD, self._write(request, codec.FC_SET_THRESHOLD)
        )
        key = f"threshold:{channel}"
        if problem is not None:
            self._unconfirmed.add(key)
            return Result(error=problem)
        setup = self._readback_channel(channel, key)
        if setup.error is not None:
            return Result(error=setup.error)
        actual = setup.unwrap()
        return self._verified(key, threshold, actual.threshold, actual)

    def set_gain(self, channel: int, gain: GainSetting) -> Result[ChannelSetup]:
        """20 03 — задать усиление канала, затем проверить 10 06."""
        request = codec.build_set_gain(channel, gain, self._profile)
        problem = self._acknowledged(codec.FC_SET_GAIN, self._write(request, codec.FC_SET_GAIN))
        key = f"gain:{channel}"
        if problem is not None:
            self._unconfirmed.add(key)
            return Result(error=problem)
        setup = self._readback_channel(channel, key)
        if setup.error is not None:
            return Result(error=setup.error)
        actual = setup.unwrap()
        return self._verified(key, gain, actual.gain, actual)

    def set_peak_gap(self, gap_ghz: int) -> Result[int]:
        """20 04 — задать минимальный интервал между пиками, затем проверить 10 04."""
        request = codec.build_set_peak_gap(gap_ghz)
        problem = self._acknowledged(
            codec.FC_SET_PEAK_GAP, self._write(request, codec.FC_SET_PEAK_GAP)
        )
        if problem is not None:
            self._unconfirmed.add("peak_gap")
            return Result(error=problem)
        readback = self.read_module_params()
        if readback.error is not None:
            self._unconfirmed.add("peak_gap")
            return Result(error=readback.error)
        actual = readback.unwrap()
        self._remember_module(actual)
        return self._verified("peak_gap", gap_ghz, actual.peak_gap_ghz, actual.peak_gap_ghz)

    def save_thresholds(self) -> Result[tuple[ChannelSetup, ...]]:
        """20 06 — сохранить пороги. Ответа нет (D4), поэтому проверяем чтением.

        Read-back здесь подтверждает только то, что прибор жив и пороги
        в нём те же, что мы записывали. Что они действительно легли
        в энергонезависимую память, чтением не проверить: это видно лишь
        после отключения питания. D4 подтверждён скринингом 01.09.2026:
        ответа на 20 06 нет, а сохранённый порог переживает отключение питания.
        """
        sent = self._send_only(
            codec.build_save_thresholds(), codec.ID_WRITE, codec.FC_SAVE_THRESHOLDS
        )
        if sent.error is not None:
            return Result(error=sent.error)
        if self._shutdown.wait(self._config.settle_before_readback_s):
            return _fail(SessionErrorKind.CANCELLED, "сессия закрывается")
        readback = self.read_channel_setup()
        if readback.error is not None:
            self._unconfirmed.add("saved_thresholds")
            return Result(error=readback.error)
        actual = readback.unwrap()
        # Снимок берётся строго до `_remember_channels`: перезапись сначала
        # сделала бы сравнение тождественным и расхождение было бы невидимо.
        known = self._device.channels if self._device is not None else None
        self._remember_channels(actual)
        if known is None:
            self._unconfirmed.add("saved_thresholds")
            return _fail(
                SessionErrorKind.VERIFICATION_MISMATCH,
                "не с чем сравнивать: конфигурация каналов ещё не читалась",
            )
        expected = tuple(setup.threshold for setup in known)
        got = tuple(setup.threshold for setup in actual)
        return self._verified("saved_thresholds", expected, got, actual)

    def _readback_channel(self, channel: int, key: str) -> Result[ChannelSetup]:
        """Читает 10 06 и достаёт нужный канал, обновляя запомненную конфигурацию."""
        readback = self.read_channel_setup()
        if readback.error is not None:
            self._unconfirmed.add(key)
            return Result(error=readback.error)
        setups = readback.unwrap()
        self._remember_channels(setups)
        if channel >= len(setups):
            self._unconfirmed.add(key)
            return _fail(
                SessionErrorKind.BAD_RESPONSE,
                f"в ответе 10 06 {len(setups)} каналов, запрошен канал {channel}",
            )
        return _ok(setups[channel])

    # --- Режимы ------------------------------------------------------------------------

    def _stop_command(self, *, internal: bool = False) -> Result[bool]:
        """30 01 — остановить поток; отдельно от `stop_stream`, без смены состояния."""
        return self._request(
            codec.build_stop(),
            codec.ID_MODE,
            codec.FC_STOP,
            lambda frame: codec.parse_stop_ack(frame, self._profile),
            timeout=self._endpoint.read_timeout_s,
            internal=internal,
        )

    def start_stream(self, speed_hz: int | None = None) -> Result[None]:
        """30 02 — запустить поток телеметрии. Подтверждения протоколом не предусмотрено.

        Успех означает, что команда ушла. Что поток действительно пошёл,
        обнаружит watchdog: если кадров не будет дольше окна тишины,
        сессия уйдёт в `Degraded`.
        """
        if self.state is not SessionState.IDLE:
            return _fail(
                SessionErrorKind.WRONG_STATE,
                f"старт потока допустим только из Idle, сейчас {self.state.name}",
            )
        sent = self._send_only(codec.build_start_stream(speed_hz), codec.ID_MODE, codec.FC_STREAM)
        if sent.error is not None:
            return sent
        self._last_frame_mono = time.perf_counter()
        self._stream_interrupted = False
        self._set_state(SessionState.STREAMING)
        return _ok(None)

    def stop_stream(self) -> Result[bool]:
        """30 01 — остановить поток. Допустима и в Idle: остановка идемпотентна."""
        result = self._stop_command()
        if result.ok:
            self._set_state(SessionState.IDLE)
        return result

    def debug_once(self) -> Result[DebugResponse]:
        """30 03 — одна развёртка: блоки АЦП всех каналов сразу.

        ✅ Раскладка тела подтверждена скринингом (N14 закрыт), кодек её
        разбирает. Сессия собирает ответ целиком по объявленному LEN
        (20430 байт, 14 датаграмм) и отдаёт разобранные блоки.

        ⚠️ Команда порождает **два** ответа с разными парами (ID, FC):
        непосредственно перед `30 03` прибор шлёт отдельной датаграммой кадр
        телеметрии `30 02`. Он уходит в `on_telemetry` наравне с потоковым
        и корреляцию не ломает — телеметрия отбирается в `_on_datagram`
        до всякой корреляции. На это есть отдельный тест.
        """
        if self.state is not SessionState.IDLE:
            return _fail(
                SessionErrorKind.WRONG_STATE,
                f"отладочная развёртка допустима только из Idle, сейчас {self.state.name}",
            )
        self._set_state(SessionState.DEBUG)
        try:
            return self._request(
                codec.build_debug_once(),
                codec.ID_MODE,
                codec.FC_DEBUG,
                lambda frame: codec.parse_debug_once(frame, self._profile),
                timeout=self._endpoint.read_timeout_s,
                reassemble=True,
                internal=True,
            )
        finally:
            self._set_state(SessionState.IDLE)

    def read_raw_adc(self, channel: int) -> Result[AdcBlock]:
        """30 07 — сырые отсчёты АЦП канала; длинный ответ собирается по LEN (D5)."""
        return self._request(
            codec.build_read_raw_adc(channel, self._profile),
            codec.ID_MODE,
            codec.FC_RAW_ADC,
            lambda frame: codec.parse_raw_adc(frame, self._profile),
            timeout=self._endpoint.read_timeout_s,
            reassemble=True,
        )

    # --- Запоминание прочитанного ------------------------------------------------------

    def _remember_module(self, module: ModuleParams) -> None:
        """Обновляет запомненные параметры модуля."""
        if self._device is not None:
            self._device = DeviceConfig(
                version_raw=self._device.version_raw,
                serial=self._device.serial,
                module=module,
                sweep=self._device.sweep,
                channels=self._device.channels,
            )

    def _remember_sweep(self, sweep: SweepConfig) -> None:
        """Обновляет запомненную развёртку."""
        if self._device is not None:
            self._device = DeviceConfig(
                version_raw=self._device.version_raw,
                serial=self._device.serial,
                module=self._device.module,
                sweep=sweep,
                channels=self._device.channels,
            )

    def _remember_channels(self, channels: tuple[ChannelSetup, ...]) -> None:
        """Обновляет запомненную настройку каналов."""
        if self._device is not None:
            self._device = DeviceConfig(
                version_raw=self._device.version_raw,
                serial=self._device.serial,
                module=self._device.module,
                sweep=self._device.sweep,
                channels=channels,
            )

    # --- Watchdog ----------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        """Следит за живостью прибора и ведёт восстановление связи."""
        tick = self._config.watchdog_tick_s
        while not self._shutdown.is_set():
            try:
                self._watchdog_tick()
            # Watchdog не имеет права умереть: без него отказ связи некому заметить.
            except Exception as exc:
                name = type(exc).__name__
                self._errors[name] = self._errors.get(name, 0) + 1
            self._shutdown.wait(tick)

    def _watchdog_tick(self) -> None:
        """Один шаг watchdog: что делать, зависит от состояния."""
        state = self.state
        if state is SessionState.IDLE:
            self._keepalive()
        elif state is SessionState.STREAMING:
            self._check_stream_alive()
        elif state is SessionState.DEGRADED:
            self._probe_degraded()
        elif state is SessionState.RECONNECTING:
            self._reconnect_step()

    def _keepalive(self) -> None:
        """Keepalive `10 01` в Idle: N таймаутов подряд → Degraded.

        Период и порог — параметры (`SessionConfig`), а не константы: есть ли
        keepalive у штатного ПО и с каким периодом, неизвестно (вопрос N8).
        Успешная пользовательская команда сдвигает срок: если по линии и так
        идёт обмен, лишний запрос не нужен.
        """
        now = time.perf_counter()
        if now - self._last_ok_mono < self._config.keepalive_period_s:
            return
        if not self._acquire_command_lock(internal=True):
            return
        try:
            result = self._request_locked(
                codec.build_read_version(),
                codec.ID_READ,
                codec.FC_VERSION,
                codec.parse_version,
                timeout=self._endpoint.read_timeout_s,
                reassemble=False,
                attempts=1,
            )
        finally:
            self._release_command_lock()
        if result.ok:
            self._keepalive_failures = 0
            return
        self._keepalive_failures += 1
        self._counters.keepalive_failures += 1
        if self._keepalive_failures >= self._config.keepalive_failures_to_degrade:
            self._to_degraded()

    def _check_stream_alive(self) -> None:
        """Следит за темпом кадров: тишина дольше окна → Degraded.

        Keepalive в потоке не шлётся — вместо него признаком живости служит
        сама телеметрия. Окно — `max(500 мс, 20 периодов развёртки)`, период
        берётся из прочитанных `10 04`, а не из константы. Если скорость
        расшифровать не удалось, остаётся только нижняя граница.
        """
        limit = self._config.stream_stall_floor_s
        device = self._device
        if device is not None and device.module.speed_hz:
            limit = max(limit, self._config.stream_stall_periods / device.module.speed_hz)
        if time.perf_counter() - self._last_frame_mono > limit:
            self._to_degraded()

    def _to_degraded(self) -> None:
        """Переводит сессию в Degraded, запомнив состояние, из которого ушли."""
        self._state_before_degraded = self.state
        self._counters.degraded_events += 1
        self._keepalive_failures = 0
        self._set_state(SessionState.DEGRADED)

    def _liveness_probe(self) -> bool:
        """Пробник: Stop и версия, по одной попытке. True — прибор отвечает."""
        if not self._acquire_command_lock(internal=True):
            return False
        try:
            stopped = self._request_locked(
                codec.build_stop(),
                codec.ID_MODE,
                codec.FC_STOP,
                lambda frame: codec.parse_stop_ack(frame, self._profile),
                timeout=self._endpoint.read_timeout_s,
                reassemble=False,
                attempts=1,
            )
            if not stopped.ok:
                return False
            version = self._request_locked(
                codec.build_read_version(),
                codec.ID_READ,
                codec.FC_VERSION,
                codec.parse_version,
                timeout=self._endpoint.read_timeout_s,
                reassemble=False,
                attempts=1,
            )
            return version.ok
        finally:
            self._release_command_lock()

    def _probe_degraded(self) -> None:
        """Degraded: пробник. Успех — восстановление, неудача — Reconnecting."""
        if self._liveness_probe():
            self._after_recovery()
        else:
            self._enter_reconnecting()

    def _enter_reconnecting(self) -> None:
        """Переводит в Reconnecting и назначает первую паузу backoff."""
        self._backoff_index = 0
        self._next_attempt_mono = time.perf_counter() + self._config.backoff_schedule[0]
        self._set_state(SessionState.RECONNECTING)

    def _reconnect_step(self) -> None:
        """Одна попытка переподключения по расписанию backoff.

        Пауза не спится целиком: watchdog просыпается каждый тик и сверяется
        со сроком. Поэтому отмена (`disconnect`) отрабатывает за тик, а не
        за десять секунд.
        """
        if time.perf_counter() < self._next_attempt_mono:
            return
        self._counters.reconnect_attempts += 1
        if self._liveness_probe():
            self._after_recovery()
            return
        schedule = self._config.backoff_schedule
        self._backoff_index = min(self._backoff_index + 1, len(schedule) - 1)
        self._next_attempt_mono = time.perf_counter() + schedule[self._backoff_index]

    def _after_recovery(self) -> None:
        """Связь вернулась: перечитать конфигурацию и сравнить с прежней.

        Прибор мог быть перезагружен по питанию и потерять всё, что мы в него
        записали, поэтому конфигурация не перезаписывается молча: расхождение
        попадает в `config_mismatch` и в колбэк, а решение принимает оператор.

        Состояние после восстановления — всегда `Idle`, даже если уходили
        из `Streaming`. Пробник содержит `Stop`, то есть поток остановлен
        нашей же командой; объявить состояние `Streaming` значило бы, что
        автомат врёт. Факт обрыва потока виден в `stream_interrupted`,
        перезапуск — решение вызывающего, а не сессии.
        """
        previous = self._state_before_degraded
        fresh = self._probe(with_stop=False)
        if fresh.error is not None:
            self._enter_reconnecting()
            return
        known = self._device
        config = fresh.unwrap()
        self._device = config
        self._last_ok_mono = time.perf_counter()
        self._keepalive_failures = 0
        if known is not None:
            diffs = known.differences(config)
            if diffs:
                self._mismatch = diffs
                if self._on_config_mismatch is not None:
                    self._on_config_mismatch(diffs)
        if previous is SessionState.STREAMING:
            self._stream_interrupted = True
        self._set_state(SessionState.IDLE)
