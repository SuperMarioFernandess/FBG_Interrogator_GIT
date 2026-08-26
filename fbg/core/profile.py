"""Профиль прибора: параметры и все спорные места протокола — данными, не кодом.

Правило KB_05 №8: ничего спорного не должно быть константой в коде. Любое значение,
которое может измениться после скрининга (KB_04), живёт здесь отдельным полем.

Значения по умолчанию соответствуют FBG-Interrogator GC-97001C-03-01-A-F,
SN 94401220, прошивка 4.10 — то есть ответам прибора, записанным в KB_02.
"""

from dataclasses import dataclass, field, replace

# Скорость света в вакууме, нм·ГГц: λ[нм] = C_NM_GHZ / f[ГГц].
#
# Внимание: в KB_01 эта формула записана с константой 299792.458 — это значение
# для частоты в ТГц, а не в ГГц. Проверка по таблице из того же KB_01:
#   299792458 / 196250 = 1527.60 нм · 299792458 / 191150 = 1568.36 нм,
# что совпадает с паспортными границами развёртки. Подтверждается и оценкой
# квантования оттуда же: 1550² · 1 ГГц / 299792458 = 8.01 пм.
C_NM_GHZ = 299_792_458.0

# Гипотезы единиц 3-байтового поля частоты в кадре телеметрии (KB_04, D1).
#   1  — гипотеза A: поле хранит целые ГГц (191150 … 196250)
#   10 — гипотеза B: поле хранит десятые доли ГГц (1911500 … 1962500)
# Диапазоны не пересекаются, поэтому по первому валидному кадру гипотеза
# определяется однозначно (см. detect_freq_divisor в codec.py).
FREQ_DIVISOR_CANDIDATES: tuple[int, ...] = (1, 10)

# Допустимые длины кадра команды 20 01 (KB_04, D3).
SET_SWEEP_FRAME_LENGTHS: tuple[int, ...] = (11, 12)


@dataclass(frozen=True)
class DeviceProfile:
    """Все параметры прибора и все параметризованные неизвестные протокола.

    Поля разделены на три группы: проверенная конфигурация, проверенные
    константы протокола и спорные места. У каждого спорного поля в комментарии
    стоит номер вопроса из KB_04 — при закрытии вопроса правится это поле,
    а не код.
    """

    # --- Конфигурация прибора, ✅ прочитана командами 10 04 и 10 05 ---
    channels: int = 4
    fbg_per_channel: int = 30
    sweep_speed_hz: int = 2000
    peak_gap_ghz: int = 30

    # Параметры развёртки хранятся в «сырых» единицах прибора, как в 10 05.
    start_param: int = 1
    step_param: int = 2
    stop_param: int = 5101
    adc_step_param: int = 2

    # --- Проверенные константы протокола ---
    # Частота [ГГц] = freq_ref_ghz − параметр. Проверено: 1 → 196250, 5101 → 191150.
    freq_ref_ghz: int = 196251
    # АЦП 14 бит: порог задаётся в этом диапазоне либо threshold_auto.
    adc_max: int = 16383
    threshold_auto: int = 0xFFFF
    gain_max_level: int = 5

    # --- Спорные места протокола (KB_04) ---

    # D1: единицы поля частоты. None — определять автоматически по кадру.
    freq_divisor: int | None = None

    # N2: масштаб поля температуры корпуса. Гипотеза 0.01 °C из скриншота 25.00°C.
    case_temp_scale: float = 0.01

    # N2 (побочный вопрос, обозначен в сводке как N2b): знаковое ли поле
    # температуры. Прибор работает от −15 °C, поэтому по умолчанию знаковое.
    case_temp_signed: bool = True

    # D3: длина кадра команды 20 01 на проводе — 11 или 12 байт.
    set_sweep_frame_len: int = 12

    # D3 (вторая половина вопроса): значение, записываемое в поле LEN этой
    # команды. None — LEN равен фактической длине кадра, как во всех остальных
    # проверенных командах. Пара (11, 12) воспроизводит строку из KB_02,
    # взятую из PDF, где LEN=0x0C при 11 байтах кадра.
    set_sweep_len_field: int | None = None

    # N3: сырые коды поля частоты, которые прибор использует как маркер
    # «пик не найден». Пусто — работает только валидация по диапазону
    # (правило KB_05 №9). Ни один маркер не выдуман: коды впишутся сюда
    # после фазы 6.3 скрининга.
    peak_missing_codes: frozenset[int] = field(default_factory=frozenset)

    # Ширина поля LEN в ответах ID=0x30. В KB_02 помечена 🟡: единственный
    # пример — ответ на Stop (30 01 00 00 00 08 00 01), где LEN занимает
    # 4 байта. Для ответов 0x10 и 0x20 ширина 2 байта проверена на 5 ответах
    # и живёт константой в codec.py.
    mode_len_width: int = 4

    def __post_init__(self) -> None:
        """Проверяет внутреннюю согласованность профиля.

        Некорректный профиль — программная ошибка (KB_05: исключения только
        для багов), поэтому здесь ValueError, а не Result.
        """
        if self.channels < 1:
            raise ValueError(f"channels должно быть ≥ 1, получено {self.channels}")
        if self.fbg_per_channel < 1:
            raise ValueError(f"fbg_per_channel должно быть ≥ 1, получено {self.fbg_per_channel}")
        if self.start_param >= self.stop_param:
            raise ValueError(
                "нарушен инвариант развёртки start_param < stop_param: "
                f"{self.start_param} ≥ {self.stop_param}"
            )
        if self.step_param < 1 or self.adc_step_param < 1:
            raise ValueError("шаг развёртки должен быть ≥ 1")
        if self.freq_divisor is not None and self.freq_divisor not in FREQ_DIVISOR_CANDIDATES:
            raise ValueError(
                f"freq_divisor={self.freq_divisor} вне гипотез {FREQ_DIVISOR_CANDIDATES}"
            )
        if self.set_sweep_frame_len not in SET_SWEEP_FRAME_LENGTHS:
            raise ValueError(
                f"set_sweep_frame_len={self.set_sweep_frame_len} "
                f"вне допустимых {SET_SWEEP_FRAME_LENGTHS}"
            )
        if self.mode_len_width < 1:
            raise ValueError("mode_len_width должно быть ≥ 1")

    # --- Расчётные величины ---

    @property
    def start_ghz(self) -> int:
        """Начальная частота развёртки, ГГц (для профиля по умолчанию 196250)."""
        return self.freq_ref_ghz - self.start_param

    @property
    def stop_ghz(self) -> int:
        """Конечная частота развёртки, ГГц (для профиля по умолчанию 191150)."""
        return self.freq_ref_ghz - self.stop_param

    @property
    def adc_points(self) -> int:
        """Число точек АЦП в одной развёртке (для профиля по умолчанию 2551)."""
        return (self.start_ghz - self.stop_ghz) // self.adc_step_param + 1

    @property
    def frame_size(self) -> int:
        """Полный размер кадра телеметрии 30 02 в байтах (по умолчанию 494).

        Раскладка (KB_02, гипотеза N4): заголовок 6 байт, затем на каждый канал
        30 групп «индекс(1) + частота(3)» и 2 байта температуры.
        """
        return 6 + self.channels * (self.fbg_per_channel * 4 + 2)

    @property
    def channel_bytes(self) -> int:
        """Размер блока одного канала в кадре телеметрии (по умолчанию 122)."""
        return self.fbg_per_channel * 4 + 2

    def freq_raw_bounds(self, divisor: int) -> tuple[int, int]:
        """Границы сырого поля частоты для заданной гипотезы единиц.

        Возвращает (нижняя, верхняя) в сырых единицах поля. Значение вне этих
        границ считается отсутствующим пиком (правило KB_05 №9).
        """
        return self.stop_ghz * divisor, self.start_ghz * divisor

    def param_to_ghz(self, param: int) -> int:
        """Пересчитывает параметр развёртки в частоту, ГГц."""
        return self.freq_ref_ghz - param

    def ghz_to_param(self, ghz: int) -> int:
        """Пересчитывает частоту в ГГц в параметр развёртки."""
        return self.freq_ref_ghz - ghz

    def with_freq_divisor(self, divisor: int) -> "DeviceProfile":
        """Возвращает копию профиля с зафиксированной гипотезой единиц частоты.

        Используется сессией после автодетекта по первому кадру: дальше разбор
        идёт по известному делителю, без пересчёта гипотез на каждом кадре.
        """
        return replace(self, freq_divisor=divisor)
