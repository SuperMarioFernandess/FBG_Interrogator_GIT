"""Модели данных протокола.

Неизменяемые модели — `dataclass(frozen=True)`. Кадр телеметрии `MeasurementFrame` —
обычный класс с массивами numpy: при 2000 кадрах/с его буферы переиспользуются
вызывающей стороной (KB_05, раздел «Стиль»).

Ошибки разбора возвращаются как `ParseResult`, а не выбрасываются: разбор чужих
байтов — штатная ситуация, а не программный баг (KB_05).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from fbg.core.profile import C_NM_GHZ, DeviceProfile

# --------------------------------------------------------------------------------------
# Результат разбора
# --------------------------------------------------------------------------------------


class ParseErrorKind(Enum):
    """Причина отказа при разборе кадра."""

    TOO_SHORT = "кадр короче минимально возможного"
    LEN_MISMATCH = "поле LEN не совпадает с фактической длиной кадра"
    WRONG_COMMAND = "кадр относится к другой команде"
    UNKNOWN_COMMAND = "неизвестная пара (ID, FC)"
    BAD_VALUE = "поле содержит недопустимое значение"
    AMBIGUOUS_UNITS = "не удалось определить единицы частоты"


@dataclass(frozen=True)
class ParseError:
    """Отказ разбора: вид и человекочитаемое пояснение."""

    kind: ParseErrorKind
    message: str

    def __str__(self) -> str:
        return f"{self.kind.name}: {self.message}"


@dataclass(frozen=True)
class ParseResult[T]:
    """Либо разобранное значение, либо ошибка. Ровно одно из двух заполнено."""

    value: T | None = None
    error: ParseError | None = None

    @property
    def ok(self) -> bool:
        """True, если разбор удался."""
        return self.error is None

    def unwrap(self) -> T:
        """Возвращает значение; при ошибке бросает исключение.

        Предназначено для тестов и для мест, где ошибка означала бы баг.
        В рабочем коде проверяйте `ok` и читайте `error`.
        """
        if self.error is not None:
            raise ValueError(str(self.error))
        assert self.value is not None
        return self.value


def ok[T](value: T) -> ParseResult[T]:
    """Успешный результат разбора."""
    return ParseResult(value=value)


def fail[T](kind: ParseErrorKind, message: str) -> ParseResult[T]:
    """Неуспешный результат разбора."""
    return ParseResult(error=ParseError(kind, message))


# --------------------------------------------------------------------------------------
# Модели ответов на команды чтения
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleParams:
    """Ответ 10 04 — параметры модуля."""

    speed_code: int
    """Сырой код скорости развёртки, как пришёл в кадре."""

    speed_hz: int | None
    """Расшифрованная скорость, Гц. None — код не удалось расшифровать."""

    channels: int
    fbg_per_channel: int
    peak_gap_ghz: int


@dataclass(frozen=True)
class SweepConfig:
    """Ответ 10 05 — параметры развёртки. Хранит и сырые параметры, и ГГц."""

    start_param: int
    step_param: int
    stop_param: int
    adc_step_param: int
    start_ghz: int
    stop_ghz: int

    @property
    def adc_points(self) -> int:
        """Число точек АЦП, соответствующее этой развёртке."""
        return (self.start_ghz - self.stop_ghz) // self.adc_step_param + 1

    @classmethod
    def from_params(
        cls,
        start_param: int,
        step_param: int,
        stop_param: int,
        adc_step_param: int,
        profile: DeviceProfile,
    ) -> "SweepConfig":
        """Собирает конфигурацию из сырых параметров, досчитывая частоты."""
        return cls(
            start_param=start_param,
            step_param=step_param,
            stop_param=stop_param,
            adc_step_param=adc_step_param,
            start_ghz=profile.param_to_ghz(start_param),
            stop_ghz=profile.param_to_ghz(stop_param),
        )


@dataclass(frozen=True)
class GainSetting:
    """Усиление канала: режим и уровень.

    Кодировка (KB_02): `00 0N` — автоматический режим, текущий уровень N;
    `80 0N` — ручной режим, уровень N. Уровень 0 — минимум, 5 — максимум.
    """

    manual: bool
    level: int

    def to_bytes(self) -> bytes:
        """Двухбайтовое представление для команды 20 03."""
        return bytes([0x80 if self.manual else 0x00, self.level])


@dataclass(frozen=True)
class ChannelSetup:
    """Порог и усиление одного канала (ответ 10 06)."""

    channel: int
    """Номер канала, 0-based: канал 1 прибора — это 0."""

    threshold: int | None
    """Порог 0…16383 либо None — автоматический расчёт по форме спектра."""

    gain: GainSetting

    @property
    def threshold_auto(self) -> bool:
        """True, если порог рассчитывается прибором автоматически."""
        return self.threshold is None


@dataclass(frozen=True)
class AdcBlock:
    """Ответ 30 07 — сырые отсчёты АЦП одного канала."""

    channel: int
    """Номер канала, 0-based."""

    gain: GainSetting
    adc: np.ndarray
    """Отсчёты АЦП, dtype uint16, длина — сколько пришло в кадре."""

    @property
    def points(self) -> int:
        """Число пришедших отсчётов."""
        return int(self.adc.size)


# --------------------------------------------------------------------------------------
# Кадр телеметрии
# --------------------------------------------------------------------------------------


class MeasurementFrame:
    """Разобранный кадр телеметрии 30 02.

    Изменяемый контейнер с переиспользуемыми буферами: при 2000 кадрах/с
    вызывающая сторона держит один экземпляр и передаёт его в `parse_measurement`
    параметром `out`. Сам кодек состояния не имеет — буфер принадлежит вызывающему.

    `freq_ghz` — частоты в ГГц, форма (каналы, решётки). `NaN` означает
    «пик не найден или значение не прошло валидацию» и никогда не заменяется
    последним известным значением (KB_05 №7).
    """

    __slots__ = (
        "case_temp_c",
        "freq_divisor",
        "freq_ghz",
        "index_mismatches",
        "missing",
        "t_mono",
    )

    def __init__(self, channels: int, fbg_per_channel: int) -> None:
        self.t_mono: float = 0.0
        self.freq_ghz: np.ndarray = np.full((channels, fbg_per_channel), np.nan, dtype=np.float64)
        self.case_temp_c: np.ndarray = np.full(channels, np.nan, dtype=np.float64)
        self.freq_divisor: int = 0
        """Делитель единиц частоты, фактически применённый к этому кадру."""
        self.missing: int = 0
        """Сколько позиций получили NaN."""
        self.index_mismatches: int = 0
        """Сколько байт индекса не совпало с ожидаемым порядком 00…1D."""

    @classmethod
    def for_profile(cls, profile: DeviceProfile) -> "MeasurementFrame":
        """Создаёт буфер под конфигурацию из профиля."""
        return cls(profile.channels, profile.fbg_per_channel)

    def wavelength_nm(self) -> np.ndarray:
        """Длины волн, нм. NaN сохраняется в тех же позициях, что и в `freq_ghz`."""
        return C_NM_GHZ / self.freq_ghz


@dataclass(frozen=True)
class DebugResponse:
    """Ответ 30 03 — одиночная развёртка в отладочном режиме.

    Байтовая раскладка тела ответа неизвестна: в KB_02 она описана словами
    «частоты + ADC всех каналов», числового примера нет, захвата нет.
    Поэтому проверяется только заголовок, а тело отдаётся сырым.
    См. открытый вопрос N14.
    """

    payload: bytes
