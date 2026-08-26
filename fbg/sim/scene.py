"""Физическая модель того, что «видит» прибор: решётки, дрейф, шум, спектр АЦП.

Опирается на раздел «Физика» KB_01:

    λ [нм] = 299792458 / f [ГГц]
    FBG FWHM ≈ 0.2 нм ≈ 25 ГГц ≈ 12–13 отсчётов АЦП при шаге 2 ГГц
    Температурный коэффициент FBG ≈ 10 пм/°C
    Повторяемость ±2 пм

Сцена не знает ни про сеть, ни про протокол: она отдаёт сырые поля кадра
и массив отсчётов АЦП. Байты из этого делает `fbg.sim.encode`.

Параметризованные неизвестные:
  * `divisor` — единицы поля частоты (D1), обе гипотезы равноправны;
  * `missing_raw` — код «пик не найден» (N3), **не факт о приборе**;
  * знаковость и масштаб температуры — из `DeviceProfile` (N2, N2b).
"""

from dataclasses import dataclass, field

import numpy as np

from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.sim.encode import MISSING_STIMULUS, encode_measurement, nm_to_raw

#: Пересчёт ширины по уровню половины в среднеквадратичную для гауссова пика.
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

#: Ширина пика FBG по уровню половины, ГГц (KB_01: 0.2 нм ≈ 25 ГГц).
DEFAULT_FWHM_GHZ = 25.0

#: Температурный коэффициент решётки, пм/°C (KB_01).
DEFAULT_TEMP_COEFF_PM_PER_C = 10.0

#: Джиттер положения пика, пм — паспортная повторяемость ±2 пм (KB_01).
DEFAULT_JITTER_PM = 2.0

#: Коэффициенты пересчёта АЦП → мощность по уровням усиления 0…5 (KB_02).
#: 🟡 Для версии 2000 Гц применимость не подтверждена — открытый вопрос N6.
#: Здесь используются только как отношение амплитуд между уровнями усиления.
ADC_POWER_COEFF: tuple[float, ...] = (
    2.36161e-5,  # уровень 0 — минимум усиления
    1.50849e-5,
    1.01289e-5,
    6.4699e-6,
    4.356e-6,
    2.9059e-6,  # уровень 5 — максимум усиления
)

#: Амплитуда пика в отсчётах АЦП при максимальном усилении. Согласована со
#: шкалой панели спектра штатного ПО: 0…12000 единиц по вертикали (KB_01).
DEFAULT_PEAK_ADC = 10000.0

#: Шумовая полка спектра и её среднеквадратичное отклонение, отсчёты АЦП.
DEFAULT_FLOOR_ADC = 200.0
DEFAULT_FLOOR_NOISE_ADC = 30.0


@dataclass
class Grating:
    """Одна решётка Брэгга: где стоит, на какой длине волны и как реагирует на нагрев.

    `position` — 0-based индекс позиции в канале, как байт индекса в кадре.
    `temperature_c` — температура именно этой решётки: нагрев одной из двух
    решёток канала — типовой сценарий скрининга.
    """

    channel: int
    position: int
    wavelength_nm: float
    temperature_c: float = 25.0
    temp_coeff_pm_per_c: float = DEFAULT_TEMP_COEFF_PM_PER_C
    enabled: bool = True


@dataclass
class Scene:
    """Физическая обстановка на входе прибора.

    `divisor` — гипотеза единиц поля частоты (D1). `missing_raw` — сырой код,
    который ставится в свободные позиции: кодировка «пик не найден» неизвестна
    (N3), поэтому это параметр сцены, а не константа и не факт о приборе.
    """

    profile: DeviceProfile
    gratings: list[Grating] = field(default_factory=list)
    divisor: int = 10
    missing_raw: int = MISSING_STIMULUS
    case_temp_c: float = 25.0
    reference_temp_c: float = 25.0
    jitter_pm: float = DEFAULT_JITTER_PM
    fwhm_ghz: float = DEFAULT_FWHM_GHZ
    peak_adc: float = DEFAULT_PEAK_ADC
    floor_adc: float = DEFAULT_FLOOR_ADC
    floor_noise_adc: float = DEFAULT_FLOOR_NOISE_ADC
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._disconnected: set[int] = set()
        self._freq_buf = np.empty(
            (self.profile.channels, self.profile.fbg_per_channel), dtype=np.uint32
        )
        self._temp_buf = np.empty(self.profile.channels, dtype=np.int64)
        for grating in self.gratings:
            self._check_placement(grating)

    # --- Управление составом сцены -----------------------------------------------------

    def _check_placement(self, grating: Grating) -> None:
        """Позиция решётки обязана существовать в конфигурации прибора."""
        if not 0 <= grating.channel < self.profile.channels:
            raise ValueError(f"канал {grating.channel} вне диапазона 0…{self.profile.channels - 1}")
        if not 0 <= grating.position < self.profile.fbg_per_channel:
            raise ValueError(
                f"позиция {grating.position} вне диапазона 0…{self.profile.fbg_per_channel - 1}"
            )

    def add(self, grating: Grating) -> None:
        """Добавляет решётку в сцену."""
        self._check_placement(grating)
        self.gratings.append(grating)

    def find(self, channel: int, position: int) -> Grating:
        """Возвращает решётку по каналу и позиции; отсутствие — ошибка вызывающего."""
        for grating in self.gratings:
            if grating.channel == channel and grating.position == position:
                return grating
        raise ValueError(f"в сцене нет решётки на канале {channel}, позиции {position}")

    def heat(self, channel: int, position: int, temperature_c: float) -> None:
        """Задаёт температуру одной решётки — сценарий нагрева при скрининге."""
        self.find(channel, position).temperature_c = temperature_c

    def disconnect_channel(self, channel: int) -> None:
        """Отключает оптическую линию: все позиции канала становятся пустыми."""
        if not 0 <= channel < self.profile.channels:
            raise ValueError(f"канал {channel} вне диапазона 0…{self.profile.channels - 1}")
        self._disconnected.add(channel)

    def connect_channel(self, channel: int) -> None:
        """Возвращает линию канала на место."""
        self._disconnected.discard(channel)

    def is_connected(self, channel: int) -> bool:
        """True, если линия канала подключена."""
        return channel not in self._disconnected

    # --- Физика ------------------------------------------------------------------------

    def _is_visible(self, grating: Grating) -> bool:
        """True, если прибор реально увидит эту решётку.

        Проверка поштучная, а не сборка списка видимых: `sample_freq_raw`
        вызывается 2000 раз в секунду, и список на каждый кадр — лишняя
        аллокация в цикле с бюджетом 500 мкс.
        """
        return grating.enabled and grating.channel not in self._disconnected

    def visible(self) -> list[Grating]:
        """Список видимых решёток. Для холодных путей: спектр, отладка, тесты."""
        return [grating for grating in self.gratings if self._is_visible(grating)]

    def wavelength_of(self, grating: Grating, *, noise: bool = True) -> float:
        """Текущая длина волны решётки: дрейф от температуры плюс джиттер, нм."""
        shift_pm = (grating.temperature_c - self.reference_temp_c) * grating.temp_coeff_pm_per_c
        if noise and self.jitter_pm > 0.0:
            shift_pm += float(self._rng.normal(0.0, self.jitter_pm))
        return grating.wavelength_nm + shift_pm * 1e-3

    def sample_freq_raw(self, *, noise: bool = True) -> np.ndarray:
        """Сырые поля частоты всего кадра, форма (каналы, решётки), dtype uint32.

        Буфер переиспользуется между вызовами: при 2000 кадрах/с аллокация
        на кадр — лишняя работа. Решётка вне развёртки прибора невидима
        и даёт `missing_raw`, как и свободная позиция.
        """
        self._freq_buf[...] = self.missing_raw
        low_ghz, high_ghz = self.profile.stop_ghz, self.profile.start_ghz
        for grating in self.gratings:
            if not self._is_visible(grating):
                continue
            freq_ghz = C_NM_GHZ / self.wavelength_of(grating, noise=noise)
            if not low_ghz <= freq_ghz <= high_ghz:
                continue
            self._freq_buf[grating.channel, grating.position] = round(freq_ghz * self.divisor)
        return self._freq_buf

    def sample_temp_raw(self) -> np.ndarray:
        """Сырые поля температуры корпуса, форма (каналы,).

        Температура одинакова во всех каналах: корпус один. Масштаб и знаковость —
        из профиля (вопросы N2 и N2b).
        """
        self._temp_buf[...] = round(self.case_temp_c / self.profile.case_temp_scale)
        return self._temp_buf

    def freq_axis_ghz(self) -> np.ndarray:
        """Частотная ось развёртки, ГГц: от `start_ghz` вниз шагом `adc_step_param`."""
        return (
            self.profile.start_ghz
            - np.arange(self.profile.adc_points, dtype=np.float64) * self.profile.adc_step_param
        )

    def spectrum(self, channel: int, gain_level: int) -> np.ndarray:
        """Спектр АЦП одного канала: гауссовы пики над шумовой полкой, 14 бит.

        Амплитуда зависит от уровня усиления по отношению коэффициентов
        пересчёта АЦП → мощность из KB_02 (при фиксированной оптической
        мощности отсчёт обратно пропорционален коэффициенту).
        """
        if not 0 <= gain_level < len(ADC_POWER_COEFF):
            raise ValueError(f"уровень усиления {gain_level} вне диапазона 0…5")

        axis = self.freq_axis_ghz()
        spectrum = self._rng.normal(self.floor_adc, self.floor_noise_adc, axis.size)

        if channel not in self._disconnected:
            sigma = self.fwhm_ghz * FWHM_TO_SIGMA
            amplitude = self.peak_adc * ADC_POWER_COEFF[-1] / ADC_POWER_COEFF[gain_level]
            for grating in self.visible():
                if grating.channel != channel:
                    continue
                center = C_NM_GHZ / self.wavelength_of(grating, noise=False)
                spectrum += amplitude * np.exp(-0.5 * ((axis - center) / sigma) ** 2)

        return np.clip(spectrum, 0.0, self.profile.adc_max).astype(np.uint16)

    def debug_payload(self, gain_levels: list[int]) -> bytes:
        """Тело ответа 30 03 — 🔴 **гипотеза**, вопрос N14 открыт.

        В KB_02 про тело известно только словесное описание «частоты + ADC всех
        каналов» и оценка размера ≈ 21 КБ из KB_01. Числового примера нет ни
        в PDF, ни в захватах: порядок блоков, наличие подзаголовков и то,
        повторяется ли внутри кадр телеметрии, — неизвестны.

        Здесь собирается ровно то, что описано словами: кадр телеметрии 30 02
        целиком, затем блоки АЦП всех каналов подряд без подзаголовков. Размер
        совпадает с оценкой KB_01 (494 + 4·2551·2 = 20902 байта), но это
        совпадение по размеру, а не подтверждение раскладки. Ни один тест
        не проверяет эту раскладку как факт о приборе; `codec.parse_debug_once`
        по-прежнему отдаёт тело сырым.
        """
        head = encode_measurement(self.profile, self.sample_freq_raw(), self.sample_temp_raw())
        blocks = [
            self.spectrum(channel, gain_levels[channel]).astype(">u2").tobytes()
            for channel in range(self.profile.channels)
        ]
        return head + b"".join(blocks)


def scene_two_gratings(profile: DeviceProfile, divisor: int) -> tuple[np.ndarray, np.ndarray]:
    """Эталонный статический стимул для сохранённого вектора телеметрии.

    Это не физическая сцена, а фиксированный набор сырых полей, по которому
    порождён `tests/vectors/measurement_synthetic.hex`. Шума и дрейфа здесь нет
    намеренно: вектор обязан быть побайтово воспроизводимым.

    Канал 1 — три решётки на 1545, 1550 и 1555 нм; канал 2 — одна на 1560 нм.
    Канал 3, позиция 1 — значение вне диапазона развёртки (проверка валидации).
    Канал 4, позиция 1 — все единицы FF FF FF (тоже вне диапазона).
    Температура корпуса одинакова во всех каналах: 25.00 °C.
    """
    channels, fbg = profile.channels, profile.fbg_per_channel
    freq_raw = np.full((channels, fbg), MISSING_STIMULUS, dtype=np.uint32)

    freq_raw[0, 0] = nm_to_raw(1545.0, divisor)
    freq_raw[0, 1] = nm_to_raw(1550.0, divisor)
    freq_raw[0, 2] = nm_to_raw(1555.0, divisor)
    freq_raw[1, 0] = nm_to_raw(1560.0, divisor)
    # 300000 вне обеих гипотез: больше 196250 и меньше 1911500.
    freq_raw[2, 0] = 300000
    freq_raw[3, 0] = 0xFFFFFF

    temp_raw = np.full(channels, round(25.00 / profile.case_temp_scale), dtype=np.int32)
    return freq_raw, temp_raw
