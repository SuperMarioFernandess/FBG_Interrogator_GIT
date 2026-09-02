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

import numpy as np

from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting
from fbg.core.pipeline import PipelineMetrics, TraceHistorySnapshot, UiSnapshot
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.core.session import DeviceConfig, SessionError, SessionState, SessionStats
from fbg.core.transport import TransportStats
from fbg.io.packet_log import Direction, PacketLogStats, PacketRecord, format_hex, format_id_fc
from fbg.io.recorder import RecorderConfig, RecorderStats, format_rows, row_format
from fbg.ui import texts
from fbg.ui.texts import Tone

#: Сколько байт датаграммы показывать в ячейке hex. Обрезается только показ.
HEX_DISPLAY_BYTES = 48

#: Столбцы таблицы журнала. Порядок совпадает с файлом (`packet_log.COLUMNS`):
#: hex стоит **до** расшифровки, потому что байты первичны (KB_05 №3).
LOG_COLUMN_COUNT = len(texts.LOG_COLUMNS)

#: Минимальный вертикальный диапазон графика Δλ. Два пикометра по длине
#: волны (0.002 нм) сопоставимы с паспортной повторяемостью прибора ±2 пм и
#: не дают pyqtgraph схлопнуть диапазон на идеально ровной линии.
GRAPH_MIN_SPAN_NM = 0.002

#: Запас сверху и снизу от видимого диапазона, чтобы экстремум не лежал
#: непосредственно на рамке графика.
GRAPH_RANGE_PADDING = 0.10

#: Оценка объёма показывается для стандартных десяти минут из требования чата.
RECORDING_ESTIMATE_SECONDS = 600.0


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
    trace_history: TraceHistorySnapshot | None = None
    log: PacketLogStats | None = None
    recorder_config: RecorderConfig | None = None
    recorder: RecorderStats | None = None
    recording_elapsed_s: float = 0.0

    last_spectrum_max_adc: int | None = None
    last_spectrum_saturated_points: int | None = None
    """Сводка последнего спектра; `None` означает, что спектра в приложении ещё не было."""

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
# Панель измерения: график, таблица, запись
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class SlotRef:
    """Одна позиция телеметрии: физический канал и слот, оба 0-based.

    Это **не датчик** (Р30). Идентификатор нужен только для выбора кривой
    и строки таблицы; физическая решётка появится этажом выше, в калибровке.
    """

    channel: int
    position: int


def slot_token(slot: SlotRef) -> str:
    """Строка для `QVariant`: типы Python там не сохраняются (KB_05 №36)."""
    return f"{slot.channel}:{slot.position}"


def parse_slot_token(token: str) -> SlotRef:
    """Обратное преобразование строки элемента Qt в ссылку на слот."""
    try:
        channel_text, position_text = token.split(":", 1)
        channel = int(channel_text)
        position = int(position_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"неверный идентификатор позиции {token!r}") from exc
    if channel < 0 or position < 0:
        raise ValueError(f"неверный идентификатор позиции {token!r}")
    return SlotRef(channel, position)


def available_slots(profile: DeviceProfile) -> tuple[SlotRef, ...]:
    """Все слоты профиля в порядке канал → позиция."""
    return tuple(
        SlotRef(channel, position)
        for channel in range(profile.channels)
        for position in range(profile.fbg_per_channel)
    )


@dataclass(frozen=True)
class GraphTrace:
    """Одна выбранная линия графика в координатах Δλ(t)."""

    slot: SlotRef
    delta_nm: np.ndarray
    baseline_nm: float | None
    latest_nm: float | None
    valid_points: int


@dataclass(frozen=True)
class MeasurementGraphModel:
    """Готовые данные графика из копии истории pipeline."""

    t_s: np.ndarray
    """Время относительно последнего кадра: правый край всегда 0 с."""

    traces: tuple[GraphTrace, ...]
    y_min_nm: float
    y_max_nm: float
    history_span_s: float


def _visible_y_range(values: Sequence[np.ndarray]) -> tuple[float, float]:
    """Диапазон Y только по конечным точкам видимых выбранных линий.

    Сначала каждая линия уже приведена к Δλ относительно своего первого
    валидного значения. Это сохраняет пикометровую динамику даже если две
    решётки лежат на 1545 и 1551 нм. Затем диапазон строится только по тем
    линиям, которые действительно выбраны пользователем.
    """
    finite_blocks = [block[np.isfinite(block)] for block in values]
    finite_blocks = [block for block in finite_blocks if block.size]
    if not finite_blocks:
        half = GRAPH_MIN_SPAN_NM / 2.0
        return -half, half
    low = min(float(block.min()) for block in finite_blocks)
    high = max(float(block.max()) for block in finite_blocks)
    span = high - low
    if span < GRAPH_MIN_SPAN_NM:
        center = (low + high) / 2.0
        half = GRAPH_MIN_SPAN_NM / 2.0
        return center - half, center + half
    padding = span * GRAPH_RANGE_PADDING
    return low - padding, high + padding


def measurement_graph_model(
    snapshot: AppSnapshot,
    selected: Sequence[SlotRef],
) -> MeasurementGraphModel:
    """Строит Δλ(t) выбранных слотов из `TraceHistorySnapshot`.

    Базой каждой линии служит **первое валидное значение в видимой истории**.
    Так абсолютный уровень около 1550 нм не съедает пикометровые изменения,
    при этом единица оси остаётся нанометром. NaN вычитается как NaN и тем
    самым остаётся разрывом — никакого заполнения последним значением нет.
    """
    history = snapshot.trace_history
    if history is None or history.frames == 0:
        low, high = _visible_y_range(())
        return MeasurementGraphModel(np.empty(0), (), low, high, 0.0)

    columns = {SlotRef(*pair): index for index, pair in enumerate(history.positions)}
    t_s = history.t_mono - history.t_mono[-1]
    traces: list[GraphTrace] = []
    deltas: list[np.ndarray] = []
    for slot in selected:
        column = columns.get(slot)
        if column is None:
            continue
        absolute = history.wavelength_nm[:, column]
        finite = np.flatnonzero(np.isfinite(absolute))
        if finite.size:
            baseline = float(absolute[int(finite[0])])
            latest = float(absolute[int(finite[-1])])
            delta = absolute - baseline
            valid_points = int(finite.size)
        else:
            baseline = None
            latest = None
            delta = np.full(absolute.shape, np.nan, dtype=np.float64)
            valid_points = 0
        deltas.append(delta)
        traces.append(GraphTrace(slot, delta, baseline, latest, valid_points))
    low, high = _visible_y_range(deltas)
    return MeasurementGraphModel(t_s, tuple(traces), low, high, history.span_s)


@dataclass(frozen=True)
class MeasurementTableModel:
    """Последний кадр как таблица канал × позиция, без привязки к датчикам."""

    wavelength_nm: np.ndarray
    valid: np.ndarray
    case_temp_c: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.wavelength_nm.shape[0])

    @property
    def positions(self) -> int:
        return int(self.wavelength_nm.shape[1])


def measurement_table_model(snapshot: AppSnapshot) -> MeasurementTableModel:
    """Строит полную таблицу 4×30 (или геометрию текущего профиля) одним снимком."""
    shape = (snapshot.profile.channels, snapshot.profile.fbg_per_channel)
    if snapshot.ui is None:
        wavelength = np.full(shape, np.nan, dtype=np.float64)
        temperature = np.full(snapshot.profile.channels, np.nan, dtype=np.float64)
    else:
        wavelength = snapshot.ui.wavelength_nm.copy()
        temperature = snapshot.ui.case_temp_c.copy()
    valid = np.isfinite(wavelength)
    return MeasurementTableModel(wavelength, valid, temperature)


@dataclass(frozen=True)
class RecordingPanelModel:
    """Состояние и прогноз записи, которые панель показывает одним тактом."""

    active: bool
    directory: Path
    decimation: int
    fbg_limit: int | None
    path: Path | None
    rows: int
    bytes_written: int
    elapsed_s: float
    gaps: int
    lost_frames: int
    pending_gap: int
    error: str | None
    estimated_bytes_10m: int
    estimated_max_bytes_10m: int

    @property
    def has_gaps(self) -> bool:
        """True и для уже записанного, и для ещё не сброшенного `# GAP`."""
        return self.gaps > 0 or self.pending_gap > 0 or self.lost_frames > 0


def _estimated_row_bytes(
    snapshot: AppSnapshot,
    config: RecorderConfig,
    *,
    all_valid: bool,
) -> int:
    """Длина характерной CSV-строки тем же форматтером, что использует Recorder."""
    channels = snapshot.profile.channels
    fbg_written = (
        snapshot.profile.fbg_per_channel
        if config.fbg_limit is None
        else min(config.fbg_limit, snapshot.profile.fbg_per_channel)
    )
    if all_valid:
        nm = np.full((channels, fbg_written), 1550.0, dtype=np.float64)
    elif snapshot.ui is None:
        nm = np.full((channels, fbg_written), np.nan, dtype=np.float64)
    else:
        nm = snapshot.ui.wavelength_nm[:, :fbg_written].copy()
    if snapshot.ui is None:
        temp = np.full(channels, 20.0, dtype=np.float64)
    else:
        temp = snapshot.ui.case_temp_c.copy()
        temp[~np.isfinite(temp)] = 20.0
    row = format_rows(
        row_format(channels, fbg_written),
        np.asarray([1_000_000], dtype=np.int64),
        np.asarray([1_000_000.0], dtype=np.float64),
        np.asarray([1_700_000_000.0], dtype=np.float64),
        nm[np.newaxis, :, :],
        temp[np.newaxis, :],
    )
    return len(row.encode("ascii"))


def estimate_recording_bytes(
    snapshot: AppSnapshot,
    config: RecorderConfig,
    *,
    duration_s: float = RECORDING_ESTIMATE_SECONDS,
    all_valid: bool = False,
) -> int:
    """Оценивает размер CSV заранее по темпу, децимации и ширине строки.

    Для обычной оценки используется текущая заполненность последнего кадра:
    `nan` в ASCII заметно короче `1550.0000`, поэтому реальная линия с 2–3
    найденными пиками даёт около 660 МБ за 10 минут при 2 кГц, а полностью
    заполненные 120 позиций заметно больше. `all_valid=True` даёт верхнюю
    оценку для выбранного `fbg_limit`.
    """
    if duration_s < 0:
        raise ValueError("duration_s не может быть отрицательной")
    expected = None if snapshot.metrics is None else snapshot.metrics.expected_rate_hz
    rate_hz = float(expected or snapshot.profile.sweep_speed_hz)
    rows = rate_hz * duration_s / config.decimation
    return round(rows * _estimated_row_bytes(snapshot, config, all_valid=all_valid))


def recording_panel_model(snapshot: AppSnapshot) -> RecordingPanelModel:
    """Собирает состояние записи и прогноз из одного `AppSnapshot`."""
    config = snapshot.recorder_config
    if config is None:
        # В нормальном AppController поле всегда есть; умолчание нужно только
        # для лёгких синтетических снимков тестов модели.
        config = RecorderConfig(directory=Path("data"))
    stats = snapshot.recorder
    return RecordingPanelModel(
        active=snapshot.recording,
        directory=config.directory,
        decimation=config.decimation,
        fbg_limit=config.fbg_limit,
        path=None if stats is None else stats.path,
        rows=0 if stats is None else stats.rows,
        bytes_written=0 if stats is None else stats.bytes_written,
        elapsed_s=snapshot.recording_elapsed_s,
        gaps=0 if stats is None else stats.gaps,
        lost_frames=0 if stats is None else stats.lost_frames,
        pending_gap=0 if stats is None else stats.pending_gap,
        error=None if stats is None else stats.error,
        estimated_bytes_10m=estimate_recording_bytes(snapshot, config),
        estimated_max_bytes_10m=estimate_recording_bytes(snapshot, config, all_valid=True),
    )


# --------------------------------------------------------------------------------------
# Панель настройки прибора
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelConfigModel:
    """Редактируемые настройки одного физического канала."""

    channel: int
    threshold: int | None
    gain: GainSetting
    threshold_unconfirmed: bool = False
    gain_unconfirmed: bool = False


@dataclass(frozen=True)
class SweepEditModel:
    """Поля команды 20 01 и производные величины для предварительного просмотра."""

    start_param: int
    step_param: int
    stop_param: int
    adc_step_param: int
    start_ghz: int
    stop_ghz: int
    start_nm: float
    stop_nm: float
    adc_points: int
    unconfirmed: bool = False


@dataclass(frozen=True)
class DeviceConfigModel:
    """Вся модель панели настройки, построенная из одного `AppSnapshot`."""

    enabled: bool
    sweep_enabled: bool
    channel_count: int
    channels: tuple[ChannelConfigModel, ...]
    peak_gap_ghz: int | None
    peak_gap_unconfirmed: bool
    sweep: SweepEditModel | None
    saved_thresholds_unconfirmed: bool
    adc_max: int
    gain_max_level: int
    last_spectrum_max_adc: int | None
    last_spectrum_saturated_points: int | None


def validate_channel(channel: int, channel_count: int) -> int:
    """Проверяет 0-based номер **реального** канала (R14)."""
    if not 0 <= channel < channel_count:
        raise ValueError(
            f"номер канала {channel} вне диапазона 0…{channel_count - 1}; "
            "прибор не проверяет канал и может испортить настройки других каналов (R14)"
        )
    return channel


def threshold_value(auto: bool, value: int, adc_max: int) -> int | None:
    """Значение для 20 02: `None` кодирует FFFF, число строго 0…adc_max."""
    if auto:
        return None
    if not 0 <= value <= adc_max:
        raise ValueError(
            f"порог {value} вне диапазона 0…{adc_max}; FFFF доступен только как режим «авто»"
        )
    return value


def gain_value(manual: bool, level: int, gain_max_level: int) -> GainSetting:
    """Значение для 20 03 с жёсткой проверкой уровня."""
    if not 0 <= level <= gain_max_level:
        raise ValueError(f"уровень усиления {level} вне диапазона 0…{gain_max_level}")
    return GainSetting(manual=manual, level=level)


def sweep_edit_model(
    profile: DeviceProfile,
    start_param: int,
    step_param: int,
    stop_param: int,
    adc_step_param: int,
    *,
    unconfirmed: bool = False,
) -> SweepEditModel:
    """Проверяет поля 20 01 и считает точки АЦП и границы в нм до отправки."""
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
            "нарушен инвариант развёртки: Start-частота должна быть больше Stop-частоты "
            "(то есть start_param < stop_param)"
        )
    if step_param < 1 or adc_step_param < 1:
        raise ValueError("шаг развёртки и шаг АЦП должны быть не меньше 1 ГГц")

    start_ghz = profile.param_to_ghz(start_param)
    stop_ghz = profile.param_to_ghz(stop_param)
    adc_points = (start_ghz - stop_ghz) // adc_step_param + 1
    return SweepEditModel(
        start_param=start_param,
        step_param=step_param,
        stop_param=stop_param,
        adc_step_param=adc_step_param,
        start_ghz=start_ghz,
        stop_ghz=stop_ghz,
        start_nm=C_NM_GHZ / start_ghz,
        stop_nm=C_NM_GHZ / stop_ghz,
        adc_points=adc_points,
        unconfirmed=unconfirmed,
    )


def device_config_model(snapshot: AppSnapshot) -> DeviceConfigModel:
    """Строит модель редактирования только из подтверждённого снимка прибора.

    Каналы берутся из `DeviceConfig.module.channels`, а не из профиля: R14
    показал, что ошибочный номер не отвергается железом и портит реальные
    настройки. До успешного опроса панель поэтому заблокирована целиком.
    """
    device = snapshot.device
    enabled = (
        device is not None
        and snapshot.state in (SessionState.IDLE, SessionState.STREAMING)
        and not snapshot.connecting
        and not snapshot.profile_mismatch
    )
    if device is None:
        return DeviceConfigModel(
            enabled=False,
            sweep_enabled=False,
            channel_count=0,
            channels=(),
            peak_gap_ghz=None,
            peak_gap_unconfirmed="peak_gap" in snapshot.unconfirmed,
            sweep=None,
            saved_thresholds_unconfirmed="saved_thresholds" in snapshot.unconfirmed,
            adc_max=snapshot.profile.adc_max,
            gain_max_level=snapshot.profile.gain_max_level,
            last_spectrum_max_adc=snapshot.last_spectrum_max_adc,
            last_spectrum_saturated_points=snapshot.last_spectrum_saturated_points,
        )

    channel_count = device.module.channels
    channels = tuple(
        ChannelConfigModel(
            channel=setup.channel,
            threshold=setup.threshold,
            gain=setup.gain,
            threshold_unconfirmed=f"threshold:{setup.channel}" in snapshot.unconfirmed,
            gain_unconfirmed=f"gain:{setup.channel}" in snapshot.unconfirmed,
        )
        for setup in device.channels
        if 0 <= setup.channel < channel_count
    )
    sweep = sweep_edit_model(
        snapshot.profile,
        device.sweep.start_param,
        device.sweep.step_param,
        device.sweep.stop_param,
        device.sweep.adc_step_param,
        unconfirmed="sweep" in snapshot.unconfirmed,
    )
    return DeviceConfigModel(
        enabled=enabled,
        sweep_enabled=enabled and not snapshot.recording,
        channel_count=channel_count,
        channels=channels,
        peak_gap_ghz=device.module.peak_gap_ghz,
        peak_gap_unconfirmed="peak_gap" in snapshot.unconfirmed,
        sweep=sweep,
        saved_thresholds_unconfirmed="saved_thresholds" in snapshot.unconfirmed,
        adc_max=snapshot.profile.adc_max,
        gain_max_level=snapshot.profile.gain_max_level,
        last_spectrum_max_adc=snapshot.last_spectrum_max_adc,
        last_spectrum_saturated_points=snapshot.last_spectrum_saturated_points,
    )


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
