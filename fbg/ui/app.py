"""Сборка приложения: кто кого создаёт, в каком порядке и кто кого гасит.

Модуль **не импортирует Qt**, и это главное решение чата. Всё, что здесь
происходит, — порядок создания, сверка профиля с прибором и порядок остановки, —
проверяется тестами без окна. Виджетам остаётся звать методы контроллера
и читать `snapshot()` по таймеру.

Порядок создания
----------------
1. **Журнал пакетов** — раньше всех, до подключения: он обязан записать сам
   обмен подключения. Цена — `unknown` в шапке первого файла первого запуска
   (см. докстринг `fbg.io.config`).
2. **Pipeline** — единственный потребитель телеметрии, кольцо истории.
3. **Session** — с `on_telemetry` от pipeline и `log_rx`/`log_tx` от журнала
   (Р54). Ядро не знает про `fbg/io`: это два колбэка, а не объект журнала.
4. **Recorder** — по запросу, уже после `Probing`: тогда его шапка верна
   с первого байта.

Первое, что делает приложение после `connect()`
-----------------------------------------------
Сверяет геометрию профиля с тем, что сообщил прибор ответами `10 04` и `10 05`.
`Session` профиль не обновляет намеренно — профиль нужен и без живого прибора,
при переразборе журнала, — поэтому звать `config.profile_from_device` обязано
приложение (найдено в чате №9).

Расхождение **не перезаписывается молча**: оно означает либо другой прибор,
либо испорченный файл настроек. Принять геометрию прибора можно явным
`apply_device_profile()`, и только при закрытой связи: кольцо истории, буфер
кадра и фильтр телеметрии журнала уже построены по старому профилю,
и подменить его на живых объектах нельзя — их надо пересобрать.

Порядок остановки
-----------------
Stop прибору → сессия → recorder → журнал → pipeline. Сначала замолкает
источник, потом закрывается связь, потом писатели дописывают хвосты, и только
затем гаснет тракт, из кольца которого они тянут. Отказ любого шага
не мешает остальным: каждый идёт в своём `try`, а список отказов возвращается
вызывающему (правило KB_05 №6 — `Stop` в `finally` — здесь тоже действует).
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType

import numpy as np

from fbg.core import calibration
from fbg.core.calibration import Sensor, SensorReading
from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting, SweepConfig
from fbg.core.pipeline import Pipeline, UiSnapshot
from fbg.core.session import (
    DeviceConfig,
    Result,
    Session,
    SessionError,
    SessionErrorKind,
    SessionState,
    StreamRecoveryOutcome,
)
from fbg.io import config as config_module
from fbg.io.config import PROFILE_DEVICE_FIELDS, AppConfig
from fbg.io.packet_log import Direction, PacketLog, PacketRecord, filter_records
from fbg.io.recorder import Recorder, RecorderConfig, RecorderStats
from fbg.ui import texts
from fbg.ui.models import (
    AppSnapshot,
    ProfileDifference,
    SensorHistorySnapshot,
    SpectrumModel,
    spectrum_model,
)

#: Сколько сообщений держать для панели. Ограничение обязательно: колбэки
#: сессии приходят из чужих потоков, и неограниченный список рос бы вечно.
NOTICE_LIMIT = 50

#: История графика физических величин считается на частоте UI. Минуты
#: достаточно для оперативной диагностики и это всего 600×120 чисел при 10 Гц.
SENSOR_HISTORY_POINTS = 600


@dataclass(frozen=True)
class ShutdownFailure:
    """Компонент, который не смог закрыться, и почему."""

    step: str
    error: str

    def __str__(self) -> str:
        return f"{self.step}: {self.error}"


class AppController:
    """Приложение без окна: владеет ядром, знает порядок и сверяет профиль.

    Использование::

        result = config_module.load()
        controller = AppController(result.config, config_path=result.path)
        controller.start()
        controller.connect()
        ...
        controller.shutdown()

    Команды, которые могут ждать сеть секунды, UI-поток не блокируют:
    подключение и спектр выполняются обычными `threading.Thread`, а окно
    забирает их состояние существующим таймером через `snapshot()` (KB_05 №34).
    Одновременно команда всё равно только одна — это гарантирует Session.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        config_path: Path | None = None,
        issues: Sequence[object] = (),
    ) -> None:
        self._config = config
        self._config_path = config_path
        """Куда сохранять настройки. None — не сохранять вовсе (тесты)."""

        self._notices: deque[str] = deque(maxlen=NOTICE_LIMIT)
        for issue in issues:
            self._notices.append(str(issue))

        self._sensors, sensor_issues = config_module.load_sensors(config.calibration_path)
        for issue in sensor_issues:
            self._notices.append(str(issue))
        self._sensor_version = 0
        self._sensor_readings: dict[str, SensorReading] = {}
        self._sensor_last_ui_seq: int | None = None
        self._sensor_history_t: deque[float] = deque(maxlen=SENSOR_HISTORY_POINTS)
        self._sensor_history_values: deque[tuple[float, ...]] = deque(
            maxlen=SENSOR_HISTORY_POINTS
        )

        # Привязка монотонных меток к настенным часам снимается один раз (Р45):
        # иначе колонка времени в панели журнала дёргалась бы вслед за NTP.
        self._wall_offset = time.time() - time.perf_counter()

        self._connect_thread: threading.Thread | None = None
        self._connect_result: Result[DeviceConfig] | None = None
        self._recorder: Recorder | None = None
        self._last_recorder_stats: RecorderStats | None = None
        self._recording_started_mono: float | None = None
        self._last_recording_elapsed_s = 0.0
        self._trace_positions: tuple[tuple[int, int], ...] = ()
        self._trace_history_s = 5.0
        self._last_device: DeviceConfig | None = None
        self._last_error: SessionError | None = None
        self._last_spectrum: SpectrumModel | None = None
        self._spectrum_version = 0
        self._spectrum_thread: threading.Thread | None = None
        self._spectrum_stop = threading.Event()
        self._spectrum_continuous = False
        self._spectrum_period_s = 1.0
        self._spectrum_completed_mono: deque[float] = deque(maxlen=8)
        self._spectrum_actual_period_s: float | None = None
        self._profile_mismatch: tuple[ProfileDifference, ...] = ()

        self._packet_log: PacketLog
        self._pipeline: Pipeline
        self._session: Session
        self._build()

    # --- Составные части ---------------------------------------------------------------

    def _build(self) -> None:
        """Создаёт журнал, тракт и сессию по текущему `AppConfig`."""
        profile = self._config.profile
        self._packet_log = PacketLog(profile, self._config.packet_log_config(), on_error=self.note)
        self._pipeline = Pipeline(profile, self._config.pipeline)
        self._session = Session(
            self._config.endpoint,
            profile,
            self._config.session,
            on_telemetry=self._pipeline.on_telemetry,
            on_state=self._on_state,
            on_config_mismatch=self._on_config_mismatch,
            on_stream_gap=self._on_stream_gap,
            log_rx=self._packet_log.log_rx,
            log_tx=self._packet_log.log_tx,
        )

    @property
    def config(self) -> AppConfig:
        """Текущие настройки."""
        return self._config

    @property
    def config_path(self) -> Path | None:
        """Файл настроек. None — сохранение выключено."""
        return self._config_path

    @property
    def session(self) -> Session:
        """Сессия. Панелям нужна для команд, снимки читаются через `snapshot()`."""
        return self._session

    @property
    def pipeline(self) -> Pipeline:
        """Приёмный тракт."""
        return self._pipeline

    @property
    def packet_log(self) -> PacketLog:
        """Журнал пакетов."""
        return self._packet_log

    @property
    def recorder(self) -> Recorder | None:
        """Писатель измерений. None — запись не идёт."""
        self._reap_failed_recorder()
        return self._recorder

    @property
    def wall_offset(self) -> float:
        """Сдвиг `time.time() − perf_counter()`, снятый один раз при старте."""
        return self._wall_offset

    @property
    def notices(self) -> tuple[str, ...]:
        """Сообщения приложения: замечания настроек, смены состояния, отказы."""
        return tuple(self._notices)

    @property
    def profile_mismatch(self) -> tuple[ProfileDifference, ...]:
        """Расхождения геометрии профиля с прибором. Пусто — расхождений нет."""
        return self._profile_mismatch

    @property
    def sensors(self) -> tuple[Sensor, ...]:
        """Текущий сохранённый набор датчиков."""
        return self._sensors

    def replace_sensors(self, sensors: Sequence[Sensor]) -> None:
        """Проверяет и атомарно сохраняет весь набор датчиков.

        Конфигурация маленькая и меняется человеком, поэтому один файл и один
        валидируемый набор проще и надёжнее частичных операций на диске.
        """
        new = tuple(sensors)
        problems = calibration.validate_sensors(new)
        if problems:
            raise ValueError("; ".join(problems))
        config_module.save_sensors(new, self._config.calibration_path)
        self._sensors = new
        self._sensor_version += 1
        self._sensor_readings = {}
        self._sensor_last_ui_seq = None
        self._sensor_history_t.clear()
        self._sensor_history_values.clear()

    def upsert_sensor(self, sensor: Sensor, *, previous_id: str | None = None) -> None:
        """Добавляет датчик либо заменяет выбранный, затем сохраняет набор."""
        replacement: list[Sensor] = []
        replaced = False
        target = previous_id or sensor.id
        for current in self._sensors:
            if current.id == target:
                replacement.append(sensor)
                replaced = True
            else:
                replacement.append(current)
        if not replaced:
            replacement.append(sensor)
        self.replace_sensors(replacement)

    def delete_sensor(self, sensor_id: str) -> None:
        """Удаляет датчик; ссылки компенсации проверит `validate_sensors`."""
        remaining = tuple(sensor for sensor in self._sensors if sensor.id != sensor_id)
        if len(remaining) == len(self._sensors):
            return
        self.replace_sensors(remaining)

    @property
    def is_recording(self) -> bool:
        """Идёт ли запись измерений."""
        self._reap_failed_recorder()
        return self._recorder is not None

    def note(self, message: str) -> None:
        """Добавляет сообщение. Зовётся в том числе из чужих потоков."""
        self._notices.append(message)

    def clear_notices(self) -> None:
        """Убирает накопленные сообщения — по кнопке панели."""
        self._notices.clear()

    # --- Колбэки ядра ------------------------------------------------------------------

    def _on_state(self, old: SessionState, new: SessionState) -> None:
        """Смена состояния сессии. Вызывается из watchdog — ничего дорогого."""
        if new is SessionState.DEGRADED and old is SessionState.STREAMING:
            self.note(texts.STREAM_CONNECTION_LOST)
        elif new is SessionState.STREAMING:
            outcome = self._session.stream_recovery_outcome
            if outcome is StreamRecoveryOutcome.RESUMED:
                self.note(texts.STREAM_RECOVERED_RESUMED)
            elif outcome is StreamRecoveryOutcome.RESTARTED:
                self.note(texts.STREAM_RECOVERED_RESTARTED)
        if new in (SessionState.DEGRADED, SessionState.RECONNECTING):
            self.note(f"состояние: {old.name} → {new.name}")

    def _on_config_mismatch(self, differences: tuple[str, ...]) -> None:
        """Конфигурация прибора изменилась после восстановления связи."""
        for difference in differences:
            self.note(f"конфигурация прибора изменилась: {difference}")

    def _on_stream_gap(self, t_mono_from: float, t_mono_to: float) -> None:
        """Передаёт сетевой разрыв активному Recorder без остановки записи.

        Колбэк приходит из диспетчера транспорта **до** первого кадра после
        паузы. `Recorder.mark_gap` только кладёт границы в свою очередь и не
        делает файловый I/O, поэтому бюджет приёмного потока не нарушается.
        """
        recorder = self._recorder
        if recorder is not None:
            recorder.mark_gap(t_mono_from, t_mono_to)

    # --- Жизненный цикл ----------------------------------------------------------------

    def start(self) -> None:
        """Поднимает журнал и тракт. Связь при этом не открывается.

        Журнал стартует до подключения намеренно: он обязан записать сам
        обмен подключения, включая `Stop`, который уходит первым.
        """
        self._packet_log.start()
        self._pipeline.start()

    def connect(self) -> Result[DeviceConfig]:
        """Подключается и сразу сверяет геометрию профиля с прибором.

        Успех означает, что прибор опрошен. Сверка профиля успеха не отменяет:
        расхождение — это сообщение оператору, а не отказ подключения.

        Вызов **синхронный** и на молчащем приборе занимает секунды. Окну
        нужен `connect_async`.
        """
        self._profile_mismatch = ()
        result = self._session.connect()
        self._last_error = result.error
        if result.error is not None or result.value is None:
            return result
        device = result.value
        self._last_device = device
        self._reconcile_profile(device)
        # Ожидаемый темп кадров известен только теперь: скорость развёртки
        # приходит ответом 10 04 и может быть изменена командой 20 01.
        self._pipeline.set_expected_rate(device.module.speed_hz)
        return result

    def connect_async(self) -> bool:
        """Запускает подключение в отдельном потоке. False — уже идёт.

        Почему поток. `Probing` — это `Stop` плюс пять чтений, каждое
        с таймаутом и повторами: на молчащем приборе окно замерло бы
        на десяток секунд, и человек за стендом убил бы приложение через
        диспетчер задач ровно тогда, когда диагностика полезнее всего.

        Почему **обычный** `threading.Thread`, а не `QThread`. Запрет касается
        второго слоя управления жизненным циклом поверх наших потоков; здесь
        его нет. Поток зовёт `connect()` и завершается, а результат забирает
        тот же таймер 10 Гц, которым UI и так читает снимки, — приём тот же,
        что и с `pipeline.snapshot()`.

        Повторный вызов во время подключения игнорируется: сессия ответила бы
        `WRONG_STATE`, и пользователь получил бы отказ на кнопку, которую
        нажал один раз по делу.
        """
        thread = self._connect_thread
        if thread is not None and thread.is_alive():
            return False
        self._connect_result = None
        self._connect_thread = threading.Thread(
            target=self._connect_worker, name="fbg-connect", daemon=True
        )
        self._connect_thread.start()
        return True

    def _connect_worker(self) -> None:
        """Тело потока подключения. Исключение здесь — сообщение, а не падение."""
        try:
            self._connect_result = self.connect()
        except Exception as exc:
            self.note(f"подключение прервано: {type(exc).__name__}: {exc}")
            self._connect_result = Result(error=SessionError(SessionErrorKind.CANCELLED, str(exc)))

    @property
    def is_connecting(self) -> bool:
        """Идёт ли подключение в фоне."""
        thread = self._connect_thread
        return thread is not None and thread.is_alive()

    def take_connect_result(self) -> Result[DeviceConfig] | None:
        """Забирает результат фонового подключения. None — ещё не готов.

        Зовётся таймером окна. Результат отдаётся ровно один раз, чтобы
        панель не показывала один и тот же отказ на каждом такте.
        """
        if self.is_connecting:
            return None
        result = self._connect_result
        self._connect_result = None
        return result

    def join_connect(self, timeout: float = 15.0) -> None:
        """Дожидается фонового подключения. Нужен остановке и тестам.

        Отменить `Probing` на полпути нельзя — команда уже в полёте, — поэтому
        закрытие ждёт поток, а не бросает его: иначе `Stop` при остановке ушёл
        бы одновременно с чтением конфигурации.
        """
        thread = self._connect_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def disconnect(self) -> None:
        """Закрывает связь, сначала завершая периодический спектр."""
        self.stop_spectrum_continuous()
        self._session.disconnect()

    def start_stream(self) -> Result[None]:
        """Запускает поток телеметрии."""
        return self._session.start_stream()

    def stop_stream(self) -> Result[bool]:
        """Останавливает поток телеметрии."""
        return self._session.stop_stream()

    @property
    def spectrum_busy(self) -> bool:
        """Идёт ли сейчас одиночный либо периодический спектральный цикл."""
        thread = self._spectrum_thread
        return thread is not None and thread.is_alive()

    @property
    def spectrum_running(self) -> bool:
        """Идёт ли именно периодическое снятие спектра."""
        return self._spectrum_continuous and self.spectrum_busy

    def _finish_mode_command(self, was_streaming: bool) -> Result[None]:
        """Выходит из 30 07/30 03 через Stop; при прежнем потоке выполняет Р63."""
        stopped = self._session.stop_stream()
        if stopped.error is not None:
            return Result(error=stopped.error)
        if not was_streaming:
            return Result(value=None)
        refreshed = self._session.refresh_config()
        if refreshed.error is not None:
            return Result(error=refreshed.error)
        return self._session.start_stream()

    def take_spectrum(self, channel: int, threshold_adc: int = 3000) -> Result[SpectrumModel]:
        """Один снимок 30 07; при прежнем Streaming автоматически восстанавливает поток."""
        if self.is_recording:
            raise RuntimeError("снимать спектр во время записи измерений нельзя")
        if self._session.state not in (SessionState.IDLE, SessionState.STREAMING):
            raise RuntimeError(
                f"спектр доступен только из Idle/Streaming, сейчас {self._session.state.name}"
            )
        self._validate_live_channel(channel)
        was_streaming = self._session.state is SessionState.STREAMING
        if was_streaming:
            stopped = self._session.stop_stream()
            if stopped.error is not None:
                return Result(error=stopped.error)
        cleanup: Result[None]
        try:
            raw = self._session.read_raw_adc(channel)
        finally:
            cleanup = self._finish_mode_command(was_streaming)
        if raw.error is not None:
            if cleanup.error is not None:
                self.note(f"спектр: после ошибки 30 07 не удалось выйти из режима: {cleanup.error}")
            return Result(error=raw.error)
        if cleanup.error is not None:
            return Result(error=cleanup.error)
        assert raw.value is not None
        model = spectrum_model(raw.value, self._config.profile, threshold_adc)
        self._last_spectrum = model
        self._spectrum_version += 1
        return Result(value=model)

    def take_spectrum_async(self, channel: int, threshold_adc: int = 3000) -> bool:
        """Запускает один снимок без блокировки UI; результат попадёт в следующий snapshot."""
        if self.is_recording:
            raise RuntimeError("снимать спектр во время записи измерений нельзя")
        if self._session.state not in (SessionState.IDLE, SessionState.STREAMING):
            raise RuntimeError(
                f"спектр доступен только из Idle/Streaming, сейчас {self._session.state.name}"
            )
        self._validate_live_channel(channel)
        if self.spectrum_busy:
            return False
        self._spectrum_continuous = False
        self._spectrum_stop.clear()
        self._spectrum_thread = threading.Thread(
            target=self._spectrum_once_worker,
            args=(channel, threshold_adc),
            name="fbg-spectrum-once",
            daemon=True,
        )
        self._spectrum_thread.start()
        return True

    def _spectrum_once_worker(self, channel: int, threshold_adc: int) -> None:
        try:
            result = self.take_spectrum(channel, threshold_adc)
            if result.error is not None:
                self.note(f"спектр: {result.error}")
        except Exception as exc:
            self.note(f"спектр: {type(exc).__name__}: {exc}")

    def take_debug_spectra(self, threshold_adc: int = 3000) -> Result[tuple[SpectrumModel, ...]]:
        """Один 30 03 по всем каналам; UI пока использует 30 07 как основной путь."""
        if self.is_recording:
            raise RuntimeError("режим отладки во время записи измерений нельзя запускать")
        if self._session.state not in (SessionState.IDLE, SessionState.STREAMING):
            raise RuntimeError(
                "режим отладки доступен только из Idle/Streaming, "
                f"сейчас {self._session.state.name}"
            )
        was_streaming = self._session.state is SessionState.STREAMING
        if was_streaming:
            stopped = self._session.stop_stream()
            if stopped.error is not None:
                return Result(error=stopped.error)
        cleanup: Result[None]
        try:
            debug = self._session.debug_once()
        finally:
            cleanup = self._finish_mode_command(was_streaming)
        if debug.error is not None:
            if cleanup.error is not None:
                self.note(
                    f"отладка: после ошибки 30 03 не удалось выйти из режима: {cleanup.error}"
                )
            return Result(error=debug.error)
        if cleanup.error is not None:
            return Result(error=cleanup.error)
        assert debug.value is not None
        models = tuple(
            spectrum_model(block, self._config.profile, threshold_adc)
            for block in debug.value.blocks
        )
        return Result(value=models)

    def start_spectrum_continuous(
        self, channel: int, period_s: float, threshold_adc: int = 3000
    ) -> bool:
        """Периодически делает отдельные 30 07; новый цикл не накладывается на предыдущий."""
        if self.is_recording:
            raise RuntimeError("непрерывный спектр нельзя запускать во время записи измерений")
        if self._session.state not in (SessionState.IDLE, SessionState.STREAMING):
            raise RuntimeError(
                "непрерывный спектр доступен только из Idle/Streaming, "
                f"сейчас {self._session.state.name}"
            )
        self._validate_live_channel(channel)
        if period_s <= 0.0:
            raise ValueError("период спектра должен быть положительным")
        if self.spectrum_busy:
            return False
        self._spectrum_period_s = period_s
        self._spectrum_completed_mono.clear()
        self._spectrum_actual_period_s = None
        self._spectrum_stop.clear()
        self._spectrum_continuous = True
        self._spectrum_thread = threading.Thread(
            target=self._spectrum_loop,
            args=(channel, threshold_adc),
            name="fbg-spectrum",
            daemon=True,
        )
        self._spectrum_thread.start()
        return True

    def _spectrum_loop(self, channel: int, threshold_adc: int) -> None:
        """Последовательные снимки с периодом между началами циклов."""
        while not self._spectrum_stop.is_set():
            started = time.perf_counter()
            try:
                result = self.take_spectrum(channel, threshold_adc)
                if result.error is not None:
                    self.note(f"спектр: {result.error}")
                else:
                    completed = time.perf_counter()
                    self._spectrum_completed_mono.append(completed)
                    if len(self._spectrum_completed_mono) >= 2:
                        first = self._spectrum_completed_mono[0]
                        last = self._spectrum_completed_mono[-1]
                        self._spectrum_actual_period_s = (last - first) / (
                            len(self._spectrum_completed_mono) - 1
                        )
            except Exception as exc:
                self.note(f"спектр: {type(exc).__name__}: {exc}")
            left = self._spectrum_period_s - (time.perf_counter() - started)
            if left > 0.0:
                self._spectrum_stop.wait(left)

    def stop_spectrum_continuous(self) -> None:
        """Останавливает периодический режим и дожидается любого текущего цикла."""
        self._spectrum_stop.set()
        thread = self._spectrum_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=15.0)
            if thread.is_alive():
                raise RuntimeError("поток спектра не завершился за 15 секунд")
        self._spectrum_thread = None
        self._spectrum_continuous = False

    # --- Настройка прибора -------------------------------------------------------------

    def _require_configurable_profile(self) -> None:
        """Не пишет геометрию, пока профиль приложения расходится с прибором."""
        if self._profile_mismatch:
            raise RuntimeError(
                "настройка прибора заблокирована: сначала примите его геометрию и переподключитесь"
            )

    def _live_channel_count(self) -> int:
        """Число каналов по живому ответу 10 04, а не по профилю (R14)."""
        device = self._session.device_config
        if device is None:
            raise RuntimeError("настройка канала доступна только после опроса прибора")
        return device.module.channels

    def _validate_live_channel(self, channel: int) -> None:
        count = self._live_channel_count()
        if not 0 <= channel < count:
            raise ValueError(
                f"номер канала {channel} вне диапазона 0…{count - 1}; "
                "прибор не проверяет канал и может испортить настройки других каналов (R14)"
            )

    def _remember_command_result[T](self, label: str, result: Result[T]) -> Result[T]:
        """Сохраняет отказ для строки состояния и сообщает его оператору."""
        self._last_error = result.error
        if result.error is not None:
            self.note(f"{label}: {result.error}")
        return result

    def set_threshold(self, channel: int, threshold: int | None) -> Result[ChannelSetup]:
        """20 02 + обязательный read-back 10 06. Разрешено и в Streaming (Р62)."""
        self._require_configurable_profile()
        self._validate_live_channel(channel)
        if threshold is not None and not 0 <= threshold <= self._config.profile.adc_max:
            raise ValueError(
                f"порог {threshold} вне диапазона 0…{self._config.profile.adc_max}; "
                "авторежим задаётся только значением None"
            )
        return self._remember_command_result(
            f"порог канала {channel + 1}", self._session.set_threshold(channel, threshold)
        )

    def set_gain(self, channel: int, manual: bool, level: int) -> Result[ChannelSetup]:
        """20 03 + обязательный read-back 10 06. Разрешено и в Streaming (Р62)."""
        self._require_configurable_profile()
        self._validate_live_channel(channel)
        if not 0 <= level <= self._config.profile.gain_max_level:
            raise ValueError(
                f"уровень усиления {level} вне диапазона 0…{self._config.profile.gain_max_level}"
            )
        gain = GainSetting(manual=manual, level=level)
        return self._remember_command_result(
            f"усиление канала {channel + 1}", self._session.set_gain(channel, gain)
        )

    def _cache_live_geometry(self) -> None:
        """Кэширует подтверждённые 10 04/10 05 в `AppConfig` после записи."""
        device = self._session.device_config
        if device is None:
            return
        profile = config_module.profile_from_device(
            self._config.profile, device.module, device.sweep
        )
        self._last_device = device
        if profile == self._config.profile:
            return
        self._config = self._config.with_profile(profile)
        self._save()

    def set_peak_gap(self, gap_ghz: int) -> Result[int]:
        """20 04 + обязательный read-back 10 04."""
        self._require_configurable_profile()
        if not 1 <= gap_ghz <= 0xFF:
            raise ValueError(f"интервал пиков {gap_ghz} ГГц вне диапазона 1…255")
        result = self._remember_command_result(
            "интервал пиков", self._session.set_peak_gap(gap_ghz)
        )
        if result.ok:
            self._cache_live_geometry()
        return result

    def set_sweep(
        self, start_param: int, step_param: int, stop_param: int, adc_step_param: int
    ) -> Result[SweepConfig]:
        """20 01 + read-back; при новой геометрии пересобирает тракт.

        Сама команда 0x20 разрешена в Streaming по Р62. Но границы развёртки
        входят в `DeviceProfile`, которым pipeline валидирует телеметрию. После
        подтверждённого изменения старый профиль использовать нельзя, поэтому
        сессия останавливается, профиль сохраняется и тракт пересобирается.
        Следующее подключение уже работает с новыми границами.
        """
        self._require_configurable_profile()
        if self.is_recording:
            raise RuntimeError("развёртку нельзя менять во время записи измерений")
        for name, value in (
            ("start_param", start_param),
            ("step_param", step_param),
            ("stop_param", stop_param),
            ("adc_step_param", adc_step_param),
        ):
            if not 0 <= value <= 0xFFFF:
                raise ValueError(f"{name}={value} не помещается в 16 бит")
        if start_param >= stop_param:
            raise ValueError(
                "нарушен инвариант развёртки: start_param должен быть меньше stop_param"
            )
        if step_param < 1 or adc_step_param < 1:
            raise ValueError("шаг развёртки и шаг АЦП должны быть не меньше 1 ГГц")
        sweep = SweepConfig.from_params(
            start_param, step_param, stop_param, adc_step_param, self._config.profile
        )
        result = self._remember_command_result("развёртка", self._session.set_sweep(sweep))
        if result.ok:
            self._rebuild_after_sweep_change()
        return result

    def _rebuild_after_sweep_change(self) -> None:
        """Пересобирает parser/log/session, если 10 05 подтвердил новые границы."""
        device = self._session.device_config
        if device is None:
            return
        updated = config_module.profile_from_device(
            self._config.profile, device.module, device.sweep
        )
        self._last_device = device
        if updated == self._config.profile:
            return
        self._session.disconnect()
        self._config = self._config.with_profile(updated)
        self._save()
        self._rebuild()
        self.note(
            "развёртка применена и подтверждена; тракт пересобран под новые границы, "
            "подключитесь снова"
        )

    def save_thresholds(self) -> Result[tuple[ChannelSetup, ...]]:
        """20 06 без ответа, задержка, затем read-back 10 06 (D4 подтверждён)."""
        self._require_configurable_profile()
        return self._remember_command_result("сохранение порогов", self._session.save_thresholds())

    def configure_recording(
        self,
        *,
        directory: Path,
        decimation: int,
        fbg_limit: int | None,
    ) -> RecorderConfig:
        """Сохраняет настройки следующей записи.

        Менять их посреди открытого файла нельзя: шапка уже зафиксировала
        `decimation` и `fbg_written`. Панель поэтому сначала останавливает
        запись, затем меняет параметры и только потом создаёт новый Recorder.
        """
        if self.is_recording:
            raise RuntimeError("настройки записи нельзя менять во время записи")
        recorder = replace(
            self._config.recorder,
            directory=directory,
            decimation=decimation,
            fbg_limit=fbg_limit,
        )
        self._config = replace(self._config, recorder=recorder)
        self._save()
        return recorder

    def set_measurement_trace_request(
        self,
        positions: Sequence[tuple[int, int]],
        history_s: float,
    ) -> None:
        """Задаёт, какую историю включать в следующий `AppSnapshot`.

        Хранится только маленький запрос пользователя. Само кольцо остаётся
        внутри pipeline, а `snapshot()` получает из него копию выбранных линий
        через `Pipeline.trace_history` (Р36).
        """
        if history_s <= 0:
            raise ValueError(f"history_s={history_s} должен быть положительным")
        selected = tuple(positions)
        for channel, position in selected:
            if not 0 <= channel < self._config.profile.channels:
                raise ValueError(
                    f"канал {channel} вне диапазона 0…{self._config.profile.channels - 1}"
                )
            if not 0 <= position < self._config.profile.fbg_per_channel:
                raise ValueError(
                    f"позиция {position} вне диапазона 0…{self._config.profile.fbg_per_channel - 1}"
                )
        self._trace_positions = selected
        self._trace_history_s = history_s

    def _recording_elapsed_s(self) -> float:
        """Длительность текущей либо последней записи для панели."""
        started = self._recording_started_mono
        if started is None:
            return self._last_recording_elapsed_s
        return max(0.0, time.perf_counter() - started)

    def _remember_recorder_end(self, recorder: Recorder) -> None:
        """Сохраняет итоговый снимок записи после штатной остановки или отказа."""
        self._last_recorder_stats = recorder.stats
        started = self._recording_started_mono
        if started is not None:
            self._last_recording_elapsed_s = max(0.0, time.perf_counter() - started)
        self._recording_started_mono = None

    def _reap_failed_recorder(self) -> None:
        """Отцепляет Recorder, который сам остановился из-за ошибки файла.

        Ошибка диска завершает только поток записи (Р47). После его завершения
        активного писателя уже нет, поэтому `recording` обязан стать False,
        но статистика и причина остаются на панели до следующего старта.
        """
        recorder = self._recorder
        if recorder is None:
            return
        stats = recorder.stats
        if stats.error is None or recorder.is_running:
            return
        self._remember_recorder_end(recorder)
        self._recorder = None

    def start_recording(self) -> Recorder:
        """Начинает запись измерений. Повторный вызов возвращает того же писателя.

        Recorder создаётся здесь, а не в `_build`: его шапка обязана нести
        серийный номер и прошивку, а они известны только после `Probing`.
        """
        self._reap_failed_recorder()
        if self._recorder is not None:
            return self._recorder
        if self.spectrum_busy:
            raise RuntimeError("запись нельзя запускать во время снятия спектра")
        recorder = Recorder(self._pipeline, self._config.recorder_config(), on_error=self.note)
        recorder.start()
        self._recorder = recorder
        self._last_recorder_stats = None
        self._recording_started_mono = time.perf_counter()
        self._last_recording_elapsed_s = 0.0
        return recorder

    def stop_recording(self) -> None:
        """Останавливает запись, дописав хвост кольца. Повторный вызов безвреден."""
        self._reap_failed_recorder()
        recorder = self._recorder
        if recorder is not None:
            recorder.stop()
            self._remember_recorder_end(recorder)
            self._recorder = None

    # --- Сверка профиля с прибором -----------------------------------------------------

    def _reconcile_profile(self, device: DeviceConfig) -> None:
        """Сравнивает геометрию настроек с тем, что сообщил прибор.

        Совпало — запоминаем идентификацию прибора и сохраняем настройки:
        со следующего запуска шапка журнала верна с первого байта.

        Разошлось — **ничего не пишем**. Другая геометрия означает либо другой
        прибор, либо испорченный файл настроек, и оба случая требуют человека:
        молчаливая перезапись скрыла бы подмену прибора, а молчаливое
        игнорирование оставило бы кольцо и буферы кадра на чужих размерах.
        """
        updated = config_module.profile_from_device(
            self._config.profile, device.module, device.sweep
        )
        differences = tuple(
            ProfileDifference(name, getattr(self._config.profile, name), getattr(updated, name))
            for name in sorted(PROFILE_DEVICE_FIELDS)
            if getattr(self._config.profile, name) != getattr(updated, name)
        )
        self._profile_mismatch = differences
        if differences:
            for difference in differences:
                self.note(str(difference))
            return
        self._remember_device(device)

    def _remember_device(self, device: DeviceConfig) -> None:
        """Запоминает идентификацию прибора в настройках и сохраняет их."""
        updated = self._config.with_device(serial=device.serial, firmware=device.version)
        if updated == self._config:
            return
        self._config = updated
        self._save()

    def apply_device_profile(self) -> None:
        """Принимает геометрию прибора: правит настройки и пересобирает тракт.

        Допустима только при закрытой связи и без идущей записи. Причина
        не в осторожности: кольцо истории, буфер разбора кадра и фильтр
        телеметрии журнала созданы по старому профилю, а курсор writer'а
        указывает в старое кольцо. Подменить профиль на живых объектах —
        значит оставить их с чужими размерами, поэтому объекты создаются
        заново, и после этого нужно подключиться ещё раз.
        """
        if self._session.state is not SessionState.DISCONNECTED:
            raise RuntimeError("принять геометрию прибора можно только при закрытой связи")
        if self.is_recording:
            raise RuntimeError("принять геометрию прибора можно только при остановленной записи")
        device = self._last_device
        if device is None or not self._profile_mismatch:
            return
        updated = config_module.profile_from_device(
            self._config.profile, device.module, device.sweep
        )
        self._config = self._config.with_profile(updated).with_device(
            serial=device.serial, firmware=device.version
        )
        self._save()
        self._profile_mismatch = ()
        self._rebuild()

    def set_endpoint(self, endpoint: Endpoint) -> None:
        """Меняет сетевые настройки. Тоже только при закрытой связи."""
        if self._session.state is not SessionState.DISCONNECTED:
            raise RuntimeError("адреса правятся только при закрытой связи")
        if endpoint == self._config.endpoint:
            return
        self._config = replace(self._config, endpoint=endpoint)
        self._save()
        self._rebuild()

    def _rebuild(self) -> None:
        """Пересобирает журнал, тракт и сессию под новый `AppConfig`.

        Журнал при этом начинает новый файл — и это не побочный эффект,
        а желаемое: прежний файл описан своей шапкой, а у нового шапка другая.
        """
        running = self._pipeline.is_running or self._packet_log.is_running
        self._pipeline.stop()
        self._packet_log.close()
        self._build()
        if running:
            self.start()

    def _save(self) -> None:
        """Сохраняет настройки. Отказ записи — сообщение, а не остановка работы.

        `config_module.save` намеренно не глотает `OSError`: не записавшиеся
        настройки — отказ, о котором пользователь обязан узнать. Но узнать
        он должен сообщением, а не потерей уже установленной связи.
        """
        if self._config_path is None:
            return
        try:
            config_module.save(self._config, self._config_path)
        except OSError as exc:
            self.note(f"настройки не сохранены: {exc}")

    # --- Снимок для UI -----------------------------------------------------------------

    def _sensor_snapshot(
        self,
        ui_snapshot: UiSnapshot | None,
    ) -> tuple[tuple[SensorReading, ...], SensorHistorySnapshot | None]:
        """Считает калибровку только по опубликованному UI-кадру (Р75).

        Метод вызывается из `snapshot()`, то есть около 10 Гц из GUI, а не из
        `Pipeline.on_telemetry` на 2 кГц. При повторном чтении того же кадра
        расчёт и история не дублируются.
        """
        if not self._sensors or ui_snapshot is None:
            return (), None

        ui = ui_snapshot
        seq = int(ui.seq)
        if seq != self._sensor_last_ui_seq:
            readings = calibration.evaluate_all(
                self._sensors,
                ui.wavelength_nm,
            )
            self._sensor_readings = readings
            self._sensor_last_ui_seq = seq
            self._sensor_history_t.append(float(ui.t_mono))
            self._sensor_history_values.append(
                tuple(readings[sensor.id].value for sensor in self._sensors)
            )

        ordered = tuple(self._sensor_readings[sensor.id] for sensor in self._sensors)
        if not self._sensor_history_t:
            history = None
        else:
            history = SensorHistorySnapshot(
                t_mono=np.fromiter(self._sensor_history_t, dtype=np.float64),
                sensor_ids=tuple(sensor.id for sensor in self._sensors),
                values=np.asarray(tuple(self._sensor_history_values), dtype=np.float64),
            )
        return ordered, history

    def snapshot(
        self,
        *,
        include_trace_history: bool = True,
        include_sensor_data: bool = True,
    ) -> AppSnapshot:
        """Всё, что панели читают за один такт таймера.

        Собирается целиком здесь, чтобы панель не ходила по живым объектам
        ядра и не могла случайно удержать кольцо (Р36).
        """
        self._reap_failed_recorder()
        session = self._session
        recorder = self._recorder
        recorder_stats = recorder.stats if recorder is not None else self._last_recorder_stats
        ui_snapshot = self._pipeline.snapshot()
        trace_history = (
            self._pipeline.trace_history(self._trace_positions, self._trace_history_s)
            if include_trace_history
            else None
        )
        if include_sensor_data:
            sensor_readings, sensor_history = self._sensor_snapshot(ui_snapshot)
        else:
            sensor_readings, sensor_history = (), None
        return AppSnapshot(
            endpoint=self._config.endpoint,
            profile=self._config.profile,
            state=session.state,
            device_model=self._config.device_model,
            serial=self._config.serial,
            firmware=self._config.firmware,
            device=session.device_config,
            stream_interrupted=session.stream_interrupted,
            stream_recovery_outcome=session.stream_recovery_outcome,
            config_mismatch=session.config_mismatch,
            unconfirmed=session.unconfirmed,
            profile_mismatch=self._profile_mismatch,
            last_error=self._last_error,
            notices=self.notices,
            session=session.stats(),
            transport=session.transport_stats,
            metrics=self._pipeline.metrics(),
            ui=ui_snapshot,
            trace_history=trace_history,
            log=self._packet_log.stats,
            recorder_config=self._config.recorder,
            recorder=recorder_stats,
            recording_elapsed_s=self._recording_elapsed_s(),
            spectrum=self._last_spectrum,
            spectrum_version=self._spectrum_version,
            spectrum_busy=self.spectrum_busy,
            spectrum_running=self.spectrum_running,
            spectrum_period_s=self._spectrum_period_s,
            spectrum_actual_period_s=self._spectrum_actual_period_s,
            last_spectrum_max_adc=(
                None if self._last_spectrum is None else self._last_spectrum.max_adc
            ),
            last_spectrum_saturated_points=(
                None if self._last_spectrum is None else self._last_spectrum.saturated_points
            ),
            sensors=self._sensors,
            sensor_readings=sensor_readings,
            sensor_history=sensor_history,
            sensor_version=self._sensor_version,
            connected=session.state is not SessionState.DISCONNECTED,
            recording=recorder is not None,
            connecting=self.is_connecting,
        )

    def packet_records(
        self,
        *,
        direction: Direction | None = None,
        id_fc: tuple[int, int] | None = None,
        limit: int | None = None,
    ) -> tuple[PacketRecord, ...]:
        """Снимок кольца журнала с фильтром. Копия, а не само кольцо."""
        records = self._packet_log.snapshot(limit)
        if direction is None and id_fc is None:
            return records
        return filter_records(records, direction=direction, id_fc=id_fc)

    def export_packets(
        self,
        path: Path,
        *,
        direction: Direction | None = None,
        id_fc: tuple[int, int] | None = None,
    ) -> int:
        """Выгружает кольцо журнала в файл с фильтром. Возвращает число записей."""
        return self._packet_log.export(path, direction=direction, id_fc=id_fc)

    # --- Остановка ---------------------------------------------------------------------

    def shutdown(self) -> tuple[ShutdownFailure, ...]:
        """Гасит всё в правильном порядке. Возвращает список отказов.

        Порядок: Stop прибору → сессия → recorder → журнал → pipeline.
        Сначала замолкает источник, затем закрывается связь, затем писатели
        дописывают хвосты — им нужно кольцо, которое ещё живо, — и последним
        гаснет сам тракт.

        Отказ одного шага не мешает остальным закрыться: каждый идёт в своём
        `try`. Молча отказ при этом не проходит — он попадает в возвращаемый
        список и в сообщения.
        """
        failures: list[ShutdownFailure] = []
        # Подключение могло остаться в полёте: команда уже отправлена, и Stop
        # не должен уйти одновременно с чтением конфигурации.
        self._step("фоновое подключение", self.join_connect, failures)
        self._step("спектр", self.stop_spectrum_continuous, failures)
        self._step("Stop прибору", self._stop_device, failures)
        self._step("сессия", self._session.disconnect, failures)
        self._step("запись измерений", self.stop_recording, failures)
        self._step("журнал пакетов", self._packet_log.stop, failures)
        self._step("приёмный тракт", self._pipeline.stop, failures)
        return tuple(failures)

    def _stop_device(self) -> None:
        """Отправляет `Stop`, если связь ещё жива (правило KB_05 №6).

        Дублирует `Stop` из `Session.disconnect`, и намеренно: там он уходит
        в `finally` без разбора ответа, здесь — как обычная команда, отказ
        которой виден. Прибор, оставшийся льющим поток, — это R6.
        """
        if self._session.state is SessionState.DISCONNECTED:
            return
        result = self._session.stop_stream()
        if result.error is not None:
            raise RuntimeError(str(result.error))

    def _step(self, name: str, action: Callable[[], None], failures: list[ShutdownFailure]) -> None:
        """Выполняет один шаг остановки, не давая ему уронить остальные."""
        try:
            action()
        except Exception as exc:
            failure = ShutdownFailure(name, f"{type(exc).__name__}: {exc}")
            failures.append(failure)
            self.note(str(failure))

    def __enter__(self) -> "AppController":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()
