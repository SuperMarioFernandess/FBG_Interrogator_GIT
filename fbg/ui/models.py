"""Что показывают панели — отдельно от того, чем они это рисуют.

Модуль не импортирует Qt. Это не аккуратность ради аккуратности: окно
в песочнице не открывается и визуальную приёмку делает человек, поэтому
всё, что можно проверить без окна, обязано быть проверяемым без окна.
Виджету остаётся расставить готовые строки по ячейкам.

Вход — **снимок** (`AppSnapshot`), а не живые объекты ядра. UI читает снимки
по таймеру и ничего не держит (Р36): ни кольца истории, ни кольца журнала.
"""

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting
from fbg.core.pipeline import PipelineMetrics, UiSnapshot
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.core.session import DeviceConfig, SessionError, SessionState, SessionStats
from fbg.core.transport import TransportStats
from fbg.io.packet_log import Direction, PacketLogStats, PacketRecord, format_hex, format_id_fc
from fbg.io.recorder import RecorderStats
from fbg.ui import texts
from fbg.ui.texts import Tone

#: Сколько байт датаграммы показывать в ячейке hex. Обрезается только показ.
HEX_DISPLAY_BYTES = 48

#: Столбцы таблицы журнала. Порядок совпадает с файлом (`packet_log.COLUMNS`):
#: hex стоит **до** расшифровки, потому что байты первичны (KB_05 №3).
LOG_COLUMN_COUNT = len(texts.LOG_COLUMNS)


# --------------------------------------------------------------------------------------
# Снимок приложения
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileDifference:
    """Одно поле, в котором профиль настроек разошёлся с прибором."""

    field: str
    configured: object
    device: object

    def __str__(self) -> str:
        return texts.PROFILE_MISMATCH_ROW.format(
            field=self.field, configured=self.configured, device=self.device
        )


@dataclass(frozen=True)
class AppSnapshot:
    """Всё, что панели читают за один такт таймера.

    Собирается контроллером; панели не ходят к сессии, кольцу и журналу сами.
    Так UI не может случайно оказаться держателем ссылки на кольцо и не может
    сделать в чужом потоке ничего дорогого.
    """

    endpoint: Endpoint
    profile: DeviceProfile
    state: SessionState

    device_model: str = texts.UNKNOWN
    serial: int | None = None
    firmware: str | None = None

    device: DeviceConfig | None = None
    stream_interrupted: bool = False
    config_mismatch: tuple[str, ...] = ()
    unconfirmed: frozenset[str] = frozenset()
    profile_mismatch: tuple[ProfileDifference, ...] = ()
    last_error: SessionError | None = None
    notices: tuple[str, ...] = ()

    session: SessionStats = field(default_factory=SessionStats)
    transport: TransportStats = field(default_factory=TransportStats)
    metrics: PipelineMetrics | None = None
    ui: UiSnapshot | None = None
    log: PacketLogStats | None = None
    recorder: RecorderStats | None = None

    connected: bool = False
    recording: bool = False
    connecting: bool = False
    """Идёт фоновое подключение: команда уже в полёте, кнопки заблокированы."""


# --------------------------------------------------------------------------------------
# Форматирование чисел
# --------------------------------------------------------------------------------------


def _int(value: int | None) -> str:
    """Целое либо прочерк. Пробел разделяет разряды — числа тут крупные."""
    return texts.UNKNOWN if value is None else f"{value:,}".replace(",", " ")


def _float(value: float | None, digits: int = 2) -> str:
    """Дробное либо прочерк; NaN тоже прочерк — это «нет значения»."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return texts.UNKNOWN
    return f"{value:.{digits}f}"


def _nm(ghz: float | None) -> str:
    """Длина волны по частоте. Четыре знака — как в файле измерений."""
    if ghz is None or ghz <= 0:
        return texts.UNKNOWN
    return f"{C_NM_GHZ / ghz:.4f}"


def format_gain(gain: GainSetting) -> str:
    """Усиление словами: режим и уровень (KB_02: `00 0N` авто, `80 0N` вручную)."""
    mode = "вручную" if gain.manual else "авто"
    return f"{mode}, уровень {gain.level}"


def format_threshold(setup: ChannelSetup) -> str:
    """Порог словами. `None` — прибор считает его сам (`FFFF`)."""
    if setup.threshold is None:
        return "авто (FFFF)"
    return str(setup.threshold)


# --------------------------------------------------------------------------------------
# Состояние сессии
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StateView:
    """Индикатор состояния: подпись, тон и уточнение под ней."""

    label: str
    tone: Tone
    detail: str = ""

    @property
    def text(self) -> str:
        """Подпись вместе с уточнением, одной строкой."""
        return f"{self.label} — {self.detail}" if self.detail else self.label


def state_view(snapshot: AppSnapshot) -> StateView:
    """Строит индикатор состояния сессии.

    Показываются все семь состояний автомата. `Degraded` и `Reconnecting`
    получают тон `WARN`, а не тон `Disconnected`: связь потеряна, но её
    восстанавливают, и это разные новости для человека за стендом.
    """
    label, tone = texts.STATE_LABELS[snapshot.state]
    details: list[str] = []
    if snapshot.stream_interrupted:
        details.append(texts.STREAM_INTERRUPTED)
    if snapshot.config_mismatch:
        details.append(texts.CONFIG_MISMATCH)
    return StateView(label=label, tone=tone, detail="; ".join(details))


def status_line(snapshot: AppSnapshot) -> str:
    """Строка состояния окна: состояние, темп, журнал, запись."""
    parts = [texts.STATE_LABELS[snapshot.state][0]]
    if snapshot.metrics is not None and snapshot.metrics.frame_rate_hz > 0:
        parts.append(f"{snapshot.metrics.frame_rate_hz:.1f} Гц")
    if snapshot.log is not None:
        parts.append(f"журнал: {_int(snapshot.log.records_in)}")
    if snapshot.recorder is not None:
        parts.append(f"запись: {_int(snapshot.recorder.rows)} строк")
    return " · ".join(parts)


def profile_mismatch_lines(snapshot: AppSnapshot) -> tuple[str, ...]:
    """Сообщение о расхождении геометрии профиля с прибором.

    Пустой кортеж — расхождения нет. Молчаливой перезаписи настроек не бывает:
    расхождение означает либо другой прибор, либо испорченный файл.
    """
    if not snapshot.profile_mismatch:
        return ()
    lines = [texts.profile_mismatch_headline(len(snapshot.profile_mismatch))]
    lines += [str(difference) for difference in snapshot.profile_mismatch]
    lines.append(texts.PROFILE_MISMATCH_HINT)
    return tuple(lines)


# --------------------------------------------------------------------------------------
# Панель прибора
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InfoRow:
    """Строка «подпись — значение». Всё только чтение."""

    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class InfoSection:
    """Группа строк панели прибора."""

    title: str
    rows: tuple[InfoRow, ...]


def _device_section(snapshot: AppSnapshot) -> InfoSection:
    """Идентификация прибора и текущее состояние связи."""
    device = snapshot.device
    serial = snapshot.serial if device is None else device.serial
    firmware = snapshot.firmware if device is None else device.version
    rows = (
        InfoRow(texts.ROW_MODEL, snapshot.device_model),
        InfoRow(texts.ROW_SERIAL, texts.UNKNOWN if serial is None else str(serial)),
        InfoRow(texts.ROW_FIRMWARE, firmware or texts.UNKNOWN),
        InfoRow(texts.ROW_STATE, state_view(snapshot).text),
    )
    return InfoSection(texts.SECTION_DEVICE, rows)


def _sweep_section(snapshot: AppSnapshot) -> InfoSection:
    """Развёртка: параметры прибора, их частоты и длины волн.

    Показываются значения **прибора**, если он опрошен, и профиля, если нет.
    Смешивать нельзя: ровно на этом расхождении и держится сверка геометрии.
    """
    profile = snapshot.profile
    device = snapshot.device
    if device is None:
        start_param, stop_param = profile.start_param, profile.stop_param
        step_param, adc_step = profile.step_param, profile.adc_step_param
        start_ghz, stop_ghz = profile.start_ghz, profile.stop_ghz
        speed: int | None = profile.sweep_speed_hz
        gap: int | None = profile.peak_gap_ghz
    else:
        sweep = device.sweep
        start_param, stop_param = sweep.start_param, sweep.stop_param
        step_param, adc_step = sweep.step_param, sweep.adc_step_param
        start_ghz, stop_ghz = sweep.start_ghz, sweep.stop_ghz
        speed = device.module.speed_hz
        gap = device.module.peak_gap_ghz
    points = (start_ghz - stop_ghz) // adc_step + 1 if adc_step else None
    rows = (
        InfoRow(texts.ROW_SPEED, _int(speed)),
        InfoRow(
            texts.ROW_START,
            f"параметр {start_param} · {_int(start_ghz)} ГГц · {_nm(start_ghz)} нм",
        ),
        InfoRow(
            texts.ROW_STOP,
            f"параметр {stop_param} · {_int(stop_ghz)} ГГц · {_nm(stop_ghz)} нм",
        ),
        InfoRow(texts.ROW_STEP, _int(step_param)),
        InfoRow(texts.ROW_ADC_STEP, _int(adc_step)),
        InfoRow(texts.ROW_ADC_POINTS, _int(points)),
        InfoRow(texts.ROW_PEAK_GAP, _int(gap)),
    )
    return InfoSection(texts.SECTION_SWEEP, rows)


def _channels_section(snapshot: AppSnapshot) -> InfoSection:
    """Каналы: сколько их, сколько решёток и как каждый настроен."""
    device = snapshot.device
    rows = [
        InfoRow(
            texts.ROW_CHANNELS,
            _int(snapshot.profile.channels if device is None else device.module.channels),
        ),
        InfoRow(
            texts.ROW_FBG,
            _int(
                snapshot.profile.fbg_per_channel
                if device is None
                else device.module.fbg_per_channel
            ),
        ),
    ]
    if device is not None:
        for setup in device.channels:
            # Номер канала в протоколе 0-based, человеку показывается 1-based:
            # шильдик и штатное ПО нумеруют каналы с единицы (KB_01).
            note = "не подтверждено" if f"threshold:{setup.channel}" in snapshot.unconfirmed else ""
            rows.append(
                InfoRow(
                    f"Канал {setup.channel + 1}",
                    f"{texts.ROW_THRESHOLD} {format_threshold(setup)} · "
                    f"{texts.ROW_GAIN} {format_gain(setup.gain)}",
                    note,
                )
            )
    return InfoSection(texts.SECTION_CHANNELS, tuple(rows))


def _quality_section(snapshot: AppSnapshot) -> InfoSection:
    """Качество связи: то, ради чего панель смотрят во время работы."""
    stats = snapshot.session
    transport = snapshot.transport
    metrics = snapshot.metrics
    rows = [
        InfoRow(texts.ROW_FRAME_RATE, _float(None if metrics is None else metrics.frame_rate_hz)),
        InfoRow(
            texts.ROW_EXPECTED_RATE, _float(None if metrics is None else metrics.expected_rate_hz)
        ),
        InfoRow(
            texts.ROW_LOSS,
            texts.UNKNOWN
            if metrics is None or metrics.loss_estimate is None
            else f"{metrics.loss_estimate * 100:.2f} %",
            texts.LOSS_IS_AN_ESTIMATE,
        ),
        InfoRow(texts.ROW_FRAMES, _int(None if metrics is None else metrics.frames)),
        InfoRow(texts.ROW_PARSE_ERRORS, _int(None if metrics is None else metrics.parse_errors)),
        InfoRow(texts.ROW_TIMEOUTS, _int(stats.timeouts)),
        InfoRow(texts.ROW_RETRIES, _int(stats.retries)),
        InfoRow(texts.ROW_ORPHANS, _int(stats.orphan_responses)),
        InfoRow(texts.ROW_TAP_ERRORS, _int(stats.tap_errors)),
        InfoRow(texts.ROW_INCOMPLETE, _int(stats.incomplete_responses)),
        InfoRow(texts.ROW_DEGRADED, _int(stats.degraded_events)),
        InfoRow(texts.ROW_RECONNECTS, _int(stats.reconnect_attempts)),
        InfoRow(texts.ROW_MISMATCHES, _int(stats.verification_mismatches)),
        InfoRow(texts.ROW_SENT, _int(transport.commands_sent)),
        InfoRow(texts.ROW_RECEIVED, _int(transport.datagrams_received)),
        InfoRow(texts.ROW_FOREIGN, _int(transport.foreign_datagrams)),
        InfoRow(texts.ROW_QUEUE_DROPPED, _int(transport.dropped_queue_full)),
        InfoRow(texts.ROW_QUEUE_PEAK, _int(transport.queue_peak)),
        InfoRow(texts.ROW_ICMP, _int(transport.icmp_resets)),
    ]
    if metrics is not None:
        history = f"{_int(metrics.history_used)} / {_int(metrics.history_frames)}"
        rows.append(InfoRow(texts.ROW_HISTORY, history))
        rows.append(InfoRow(texts.ROW_EVICTED, _int(metrics.evicted), texts.EVICTED_IS_NOT_LOSS))
    return InfoSection(texts.SECTION_QUALITY, tuple(rows))


def _log_section(snapshot: AppSnapshot) -> InfoSection | None:
    """Состояние журнала пакетов: он тоже часть картины связи."""
    log = snapshot.log
    if log is None:
        return None
    rows = (
        InfoRow(texts.ROW_LOG_IN, _int(log.records_in)),
        InfoRow(texts.ROW_LOG_WRITTEN, _int(log.records_written)),
        InfoRow(texts.ROW_LOG_LOST, _int(log.lost_records)),
        InfoRow(texts.ROW_LOG_DROPPED, _int(log.dropped_queue_full)),
        InfoRow(texts.ROW_LOG_DECODE_ERRORS, _int(log.decode_errors)),
        InfoRow(texts.ROW_LOG_FILE, texts.UNKNOWN if log.path is None else str(log.path)),
    )
    return InfoSection(texts.SECTION_LOG, rows)


def device_sections(snapshot: AppSnapshot) -> tuple[InfoSection, ...]:
    """Полная модель панели прибора. Всё только чтение (настройка — чат №11)."""
    sections = [
        _device_section(snapshot),
        _sweep_section(snapshot),
        _channels_section(snapshot),
        _quality_section(snapshot),
    ]
    log_section = _log_section(snapshot)
    if log_section is not None:
        sections.append(log_section)
    return tuple(sections)


# --------------------------------------------------------------------------------------
# Панель журнала пакетов
# --------------------------------------------------------------------------------------


def local_time(t_mono: float, wall_offset: float) -> str:
    """Локальное время `HH:MM:SS.mmm` по метке `perf_counter`.

    Тот же вид, что в колонке `t_local` файла журнала, и то же соглашение:
    привязка к настенным часам снимается **один раз** (Р45), поэтому колонка
    согласована с `t_mono` и не дёргается вслед за NTP. Совпадение с файлом
    закреплено отдельным тестом — иначе панель и файл разъедутся молча.
    """
    wall = t_mono + wall_offset
    whole = math.floor(wall)
    milliseconds = int((wall - whole) * 1000)
    return time.strftime("%H:%M:%S", time.localtime(whole)) + f".{milliseconds:03d}"


def format_hex_cell(data: bytes, limit: int = HEX_DISPLAY_BYTES) -> str:
    """Байты для ячейки таблицы: полностью либо с явной пометкой об обрезке.

    Обрезается **только показ**. Ответ `30 03` — 20430 байт, то есть 61 КБ
    в одной ячейке: таблица на таком встаёт, а прочитать его глазами всё
    равно нельзя. В файле журнала и в экспорте байты лежат целиком
    и не обрезаются никогда (KB_05 №3), и пометка об этом говорит.
    """
    if len(data) <= limit:
        return format_hex(data)
    return format_hex(data[:limit]) + texts.hex_truncated_suffix(len(data))


def packet_cell(record: PacketRecord, column: int, wall_offset: float) -> str:
    """Значение одной ячейки таблицы журнала.

    По ячейке, а не строкой целиком: таблица спрашивает только видимые
    строки, и hex длинного ответа не форматируется, пока на него не смотрят.
    """
    match column:
        case 0:
            return str(record.seq)
        case 1:
            return record.direction.value
        case 2:
            return f"{record.t_mono:.6f}"
        case 3:
            return local_time(record.t_mono, wall_offset)
        case 4:
            return str(len(record.data))
        case 5:
            return format_id_fc(record.data)
        case 6:
            return format_hex_cell(record.data)
        case 7:
            return record.decoded
    raise IndexError(f"столбца {column} в таблице журнала нет")


def packet_row(record: PacketRecord, wall_offset: float) -> tuple[str, ...]:
    """Вся строка таблицы. Нужна экспорту в тестах и проверке форматирования."""
    return tuple(packet_cell(record, column, wall_offset) for column in range(LOG_COLUMN_COUNT))


def id_fc_choices(records: Sequence[PacketRecord]) -> tuple[tuple[int, int], ...]:
    """Пары (ID, FC), встретившиеся в записях, по возрастанию.

    Список строится по факту, а не по `codec.KNOWN_COMMANDS`: в журнале
    интереснее всего то, чего в списке известных команд нет.
    """
    seen = {record.id_fc for record in records if record.id_fc is not None}
    return tuple(sorted(seen))


def format_id_fc_pair(pair: tuple[int, int]) -> str:
    """Пара (ID, FC) для выпадающего списка: «30 02»."""
    return f"{pair[0]:02X} {pair[1]:02X}"


def parse_id_fc_pair(text: str) -> tuple[int, int]:
    """Обратно из «30 02» в пару чисел.

    Нужна потому, что данные элемента списка Qt хранит через `QVariant`
    и **типы Python не сохраняет**: кортеж возвращается списком, а `StrEnum` —
    обычной строкой. Поэтому в списках хранится строка, а разбирают её здесь,
    в проверяемом без окна месте.
    """
    ident, _, fc = text.partition(" ")
    return int(ident, 16), int(fc, 16)


def direction_choices() -> tuple[Direction, ...]:
    """Направления для фильтра. `NOTE` — события журнала, а не трафик."""
    return (Direction.TX, Direction.RX, Direction.NOTE)


def export_suggested_name(now: float | None = None) -> str:
    """Имя файла по умолчанию для экспорта журнала. ASCII, как и сам файл."""
    stamp = time.localtime(time.time() if now is None else now)
    return time.strftime("packets_export_%Y%m%d_%H%M%S.log", stamp)


def export_path(directory: Path, now: float | None = None) -> Path:
    """Полный путь экспорта по умолчанию."""
    return directory / export_suggested_name(now)
