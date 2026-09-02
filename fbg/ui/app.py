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

from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting, SweepConfig
from fbg.core.pipeline import Pipeline
from fbg.core.session import (
    DeviceConfig,
    Result,
    Session,
    SessionError,
    SessionErrorKind,
    SessionState,
)
from fbg.io import config as config_module
from fbg.io.config import PROFILE_DEVICE_FIELDS, AppConfig
from fbg.io.packet_log import Direction, PacketLog, PacketRecord, filter_records
from fbg.io.recorder import Recorder
from fbg.ui.models import AppSnapshot, ProfileDifference

#: Сколько сообщений держать для панели. Ограничение обязательно: колбэки
#: сессии приходят из чужих потоков, и неограниченный список рос бы вечно.
NOTICE_LIMIT = 50


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

    Все методы зовутся из потока UI. Чужие потоки трогают только `_notices`
    (через колбэки сессии) — это `deque` с ограничением, и `append` на нём
    атомарен.
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

        # Привязка монотонных меток к настенным часам снимается один раз (Р45):
        # иначе колонка времени в панели журнала дёргалась бы вслед за NTP.
        self._wall_offset = time.time() - time.perf_counter()

        self._connect_thread: threading.Thread | None = None
        self._connect_result: Result[DeviceConfig] | None = None
        self._recorder: Recorder | None = None
        self._last_device: DeviceConfig | None = None
        self._last_error: SessionError | None = None
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
    def is_recording(self) -> bool:
        """Идёт ли запись измерений."""
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
        if new in (SessionState.DEGRADED, SessionState.RECONNECTING):
            self.note(f"состояние: {old.name} → {new.name}")

    def _on_config_mismatch(self, differences: tuple[str, ...]) -> None:
        """Конфигурация прибора изменилась после восстановления связи."""
        for difference in differences:
            self.note(f"конфигурация прибора изменилась: {difference}")

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
        """Закрывает связь. `Stop` уходит внутри `Session.disconnect` (правило №6)."""
        self._session.disconnect()

    def start_stream(self) -> Result[None]:
        """Запускает поток телеметрии."""
        return self._session.start_stream()

    def stop_stream(self) -> Result[bool]:
        """Останавливает поток телеметрии."""
        return self._session.stop_stream()

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
        if self._recorder is not None:
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

    def start_recording(self) -> Recorder:
        """Начинает запись измерений. Повторный вызов возвращает того же писателя.

        Recorder создаётся здесь, а не в `_build`: его шапка обязана нести
        серийный номер и прошивку, а они известны только после `Probing`.
        """
        if self._recorder is not None:
            return self._recorder
        recorder = Recorder(self._pipeline, self._config.recorder_config(), on_error=self.note)
        recorder.start()
        self._recorder = recorder
        return recorder

    def stop_recording(self) -> None:
        """Останавливает запись, дописав хвост кольца. Повторный вызов безвреден."""
        recorder = self._recorder
        self._recorder = None
        if recorder is not None:
            recorder.stop()

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
        if self._recorder is not None:
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

    def snapshot(self) -> AppSnapshot:
        """Всё, что панели читают за один такт таймера.

        Собирается целиком здесь, чтобы панель не ходила по живым объектам
        ядра и не могла случайно удержать кольцо (Р36).
        """
        session = self._session
        recorder = self._recorder
        return AppSnapshot(
            endpoint=self._config.endpoint,
            profile=self._config.profile,
            state=session.state,
            device_model=self._config.device_model,
            serial=self._config.serial,
            firmware=self._config.firmware,
            device=session.device_config,
            stream_interrupted=session.stream_interrupted,
            config_mismatch=session.config_mismatch,
            unconfirmed=session.unconfirmed,
            profile_mismatch=self._profile_mismatch,
            last_error=self._last_error,
            notices=self.notices,
            session=session.stats(),
            transport=session.transport_stats,
            metrics=self._pipeline.metrics(),
            ui=self._pipeline.snapshot(),
            log=self._packet_log.stats,
            recorder=None if recorder is None else recorder.stats,
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
