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

# Единицы 3-байтового поля частоты в кадре телеметрии (KB_04, D1 — ✅ закрыт).
#   1  — гипотеза A: поле хранит целые ГГц
#   10 — ✅ факт: поле хранит десятые доли ГГц
# Скрининг 27.08.2026 показал 1D 9C C0 = 1 940 674 → 194 067.4 ГГц → 1544.785 нм,
# что совпало с решёткой стенда 1544.80. Гипотеза A даёт значения вне
# физического диапазона прибора вообще.
#
# Автодетект остаётся в коде страховкой — диапазоны гипотез не пересекаются, —
# но значением по умолчанию является 10, а не None.
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

    # База развёртки: Частота [ГГц] = sweep_base_ghz − параметр.
    #
    # ✅ 196250, а не 196251 из PDF (KB_04, D8). Доказательство — файл вендора
    # Spec_*.txt, снятый штатным ПО при параметрах 0x0000 / 0x13EC: он содержит
    # ровно 2551 строку с частотами от 191150 до 196250 ГГц. 196250 − 0 = 196250
    # и 196250 − 5100 = 191150 сходятся точно, тогда как 196251 − 0 = 196251
    # выше паспортного максимума прибора.
    #
    # Следствие: заводские параметры 0x0001 / 0x13ED дают 196249 / 191149,
    # а не круглые 196250 / 191150. Это нормально — прибор хранит параметр,
    # а не частоту, и круглые границы получаются при параметрах 0 / 5100.
    #
    # Поле, а не константа: разница в 1 ГГц ≈ 8 пм systematic, и если найдётся
    # прибор с другой базой, правится профиль, а не код (KB_05 №8).
    sweep_base_ghz: int = 196250
    # АЦП 14 бит: порог задаётся в этом диапазоне либо threshold_auto.
    adc_max: int = 16383
    threshold_auto: int = 0xFFFF
    gain_max_level: int = 5

    # N6 ✅ скрининг 30.08.2026: тёмное смещение и коэффициенты пересчёта
    # P[dBm] = 10*log10((ADC - adc_dark_offset) * X[gain]).
    # Это физические параметры конкретного семейства прибора, поэтому живут
    # в профиле, а не константами UI (KB_05: физические числа — profile.py).
    adc_dark_offset: int = 68
    gain_power_coefficients: tuple[float, ...] = (
        2.36161e-5,
        1.50849e-5,
        1.01289e-5,
        6.4699e-6,
        4.356e-6,
        2.9059e-6,
    )

    # --- Спорные места протокола (KB_04) ---

    # D1 ✅ закрыт скринингом: поле хранит десятые доли ГГц.
    # None оставлено допустимым значением — тогда единицы определяются
    # автоматически по первому кадру (см. detect_freq_divisor в codec.py).
    freq_divisor: int | None = 10

    # N2 ✅ закрыт скринингом: масштаб 0.01 °C. Сырое 1685 → 16.85 °C,
    # одинаково во всех четырёх каналах — корпус один.
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

    # N3 ✅ закрыт скринингом: «пик не найден» кодируется сырым 0x000000,
    # байт индекса при этом сохраняется и продолжает нумерацию 00…1D.
    # Наблюдение прямое: из 30 позиций канала 1 прибор заполнил две, остальные
    # 28 содержали нули при корректных байтах индекса.
    #
    # Ноль и так не попадает в диапазон stop_ghz…start_ghz, то есть маркер
    # дублирует валидацию по диапазону (KB_05 №9). Он оставлен явным, потому
    # что это разные утверждения: одно про физику, другое про кодировку.
    peak_missing_codes: frozenset[int] = field(default_factory=lambda: frozenset({0}))

    # D9 ✅ закрыт скринингом: индекс 0 массива АЦП соответствует **Stop**
    # (191150 ГГц, 1568.36 нм), последний индекс — Start (196250 ГГц, 1527.60 нм).
    # То есть частота возрастает с индексом, а длина волны убывает.
    # Прежняя интуиция «от Start к Stop» была обратной и ошибочной.
    # Источник: первая колонка файла вендора Spec_*.txt идёт от 191150 вверх
    # с шагом 2 ГГц, 2551 строка; корреляция с нашим разбором 0.9997.
    adc_index_ascending_freq: bool = True

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
        if not 0 <= self.adc_dark_offset < self.adc_max:
            raise ValueError(
                f"adc_dark_offset={self.adc_dark_offset} вне диапазона 0…{self.adc_max - 1}"
            )
        expected_gain_levels = self.gain_max_level + 1
        if len(self.gain_power_coefficients) != expected_gain_levels:
            raise ValueError(
                "число gain_power_coefficients должно совпадать с уровнями усиления: "
                f"ожидалось {expected_gain_levels}, получено {len(self.gain_power_coefficients)}"
            )
        if any(coefficient <= 0.0 for coefficient in self.gain_power_coefficients):
            raise ValueError("gain_power_coefficients должны быть положительными")
        if self.sweep_base_ghz - self.stop_param < 1:
            raise ValueError(
                f"sweep_base_ghz={self.sweep_base_ghz} меньше stop_param={self.stop_param}: "
                "нижняя граница развёртки получилась неположительной"
            )

    # --- Расчётные величины ---

    @property
    def start_ghz(self) -> int:
        """Верхняя частота развёртки, ГГц.

        Для заводских параметров (1 / 5101) это 196249, а не круглые 196250:
        база равна 196250, и круглая граница получается при параметре 0.
        """
        return self.sweep_base_ghz - self.start_param

    @property
    def stop_ghz(self) -> int:
        """Нижняя частота развёртки, ГГц (для заводских параметров 191149)."""
        return self.sweep_base_ghz - self.stop_param

    @property
    def adc_points(self) -> int:
        """Число точек АЦП в одной развёртке (для профиля по умолчанию 2551)."""
        return (self.start_ghz - self.stop_ghz) // self.adc_step_param + 1

    def adc_index_to_ghz(self, index: float) -> float:
        """Частоту, ГГц, по индексу в массиве АЦП (вопрос D9 ✅ закрыт).

        Индекс 0 соответствует `stop_ghz`, последний — `start_ghz`: частота
        **возрастает** с индексом, длина волны убывает. Ориентация проверена
        по файлу вендора `Spec_*.txt`, первая колонка которого идёт от 191150
        вверх с шагом 2 ГГц.

        Аргумент дробный намеренно: положение пика в массиве определяется
        интерполяцией и целым индексом не является (наблюдённое 1455.7).
        """
        offset = index * self.adc_step_param
        if self.adc_index_ascending_freq:
            return self.stop_ghz + offset
        return self.start_ghz - offset

    def ghz_to_adc_index(self, freq_ghz: float) -> float:
        """Дробный индекс в массиве АЦП по частоте, ГГц. Обратна `adc_index_to_ghz`."""
        if self.adc_index_ascending_freq:
            return (freq_ghz - self.stop_ghz) / self.adc_step_param
        return (self.start_ghz - freq_ghz) / self.adc_step_param

    def adc_index_to_nm(self, index: float) -> float:
        """Длину волны, нм, по индексу в массиве АЦП.

        Сверка со скринингом: индекс 1455.7 при параметрах 0 / 5100 даёт
        1544.84 нм, а курсор GUI штатного ПО на том же пике показал 1544.850.
        """
        return C_NM_GHZ / self.adc_index_to_ghz(index)

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
        return self.sweep_base_ghz - param

    def ghz_to_param(self, ghz: int) -> int:
        """Пересчитывает частоту в ГГц в параметр развёртки."""
        return self.sweep_base_ghz - ghz

    def with_freq_divisor(self, divisor: int) -> "DeviceProfile":
        """Возвращает копию профиля с зафиксированной гипотезой единиц частоты.

        Используется сессией после автодетекта по первому кадру: дальше разбор
        идёт по известному делителю, без пересчёта гипотез на каждом кадре.
        """
        return replace(self, freq_divisor=divisor)
