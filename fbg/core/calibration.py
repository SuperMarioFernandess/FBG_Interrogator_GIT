"""Калибровка датчиков: длина волны → физическая величина.

Рабочая модель приложения записана относительно физически осмысленной
опорной точки::

    значение = значение₀ + k₁·(λ − λ₀) + k₂·(λ − λ₀)²

Здесь `λ₀` — `Sensor.expected_nm`: та же длина волны, вокруг которой датчик
ищется в кадре. Для температурной решётки это даёт коэффициенты порядка
`25 °C`, `100 °C/нм`, `0`, а не разность двух чисел порядка 150 тысяч.

Это также ближе к разобранной структуре штатного ПО: в `FBGInfo` поле
`Coeff0` оказалось длиной волны, а не свободным членом абсолютного полинома
(N23). Точная семантика вендорских `Coeff1`/`Coeff2` всё ещё не доказана;
новая форма — решение нашего приложения, а не утверждение о формуле вендора.

плюс температурная компенсация со ссылкой на другой датчик, плюс пределы
`UpLimit` / `DownLimit`, плюс тип датчика из десяти.

Модуль состоит из чистых функций и неизменяемых данных: ни файлов, ни часов,
ни состояния. Хранение — JSON, и читает его `fbg/io/config.py`; здесь есть
только преобразование «объект ↔ словарь», тоже чистое.

Почему датчик не привязан к позиции в кадре
-------------------------------------------
Решение Р30, и это главное отличие от модели вендора. У вендора датчик
адресуется парой (Channel, Sequence), то есть номером позиции в кадре.
Скрининг 27.08.2026 показал, что позиции — **слоты, заполняемые по возрастанию
λ по мере обнаружения**, а не номера решёток: из четырёх решёток стенда прибор
распознал две, и 1551.51 нм легла в позицию 1, будучи четвёртой решёткой линии.
Пропадёт первая решётка — вторая переедет в позицию 0, и калибровка, привязанная
к номеру позиции, начнёт считать данные одного датчика по коэффициентам другого.
Причём молча: числа останутся правдоподобными.

Поэтому датчик описывается **ожидаемой длиной волны и окном поиска**, а
сопоставление делается поиском ближайшего пика внутри окна. Канал при этом
остаётся привязкой: канал — это физическое волокно, а не слот.

По той же причине заменена и ссылка температурной компенсации. У вендора это
`TC_Ch` + `TC_Seq` — то есть ровно та привязка к позиции, которую запрещает Р30.
Здесь компенсация ссылается на **идентификатор другого датчика** того же набора.

Ширина окна — рабочий диапазон, а не разброс решётки
----------------------------------------------------
Окно обязано покрывать то, куда датчик уедет **в работе**, а не паспортный
разброс изготовления решётки. Иначе первый же реальный нагрев выведет пик
за окно, и датчик пропадёт именно тогда, когда он интересен. Арифметика
(KB_01, раздел «Физика»)::

    температурный коэффициент FBG   ≈ 10 пм/°C
    рабочий диапазон −15…+55 °C     → 0.70 нм полного хода → окно ±0.35 нм
    тензодатчик ≈ 1.2 пм/µε, ±1000 µε → 1.2 нм в каждую сторону

Сверху окно ограничено соседями: минимальный интервал пиков у прибора —
30 ГГц, а это 0.24 нм при 1550 нм, то есть у плотно набитой линии окно физически
не может быть шире ±0.12 нм. На стенде решётки 1544.80 и 1551.50 разнесены
на 6.7 нм, и запас там любой. Проверяет это `validate_sensors`: пересечение
окон двух датчиков одного канала — ошибка конфигурации, и узнать о ней надо
один раз при загрузке, а не по странным числам через месяц.

Три исхода поиска, и все различимы
----------------------------------
============================  =========================================
Что произошло                 Что возвращается
============================  =========================================
ровно один пик в окне         значение, `status = OK`
ни одного пика в окне         `NaN`, `status = PEAK_NOT_FOUND`
больше одного пика в окне     `NaN`, `status = AMBIGUOUS`
============================  =========================================

Два пика в окне дают `NaN`, а не ближайший из них, и это осознанный выбор.
«Ближайший» — монетка: если настоящая решётка датчика пропала (на стенде это
рядовое событие, прибор распознаёт две из четырёх), а в окне остался соседний
или паразитный пик, «ближайший» вернёт чужую длину волны под именем этого
датчика. Ровно тот класс отказа, ради которого написан Р30, только этажом
выше. Сколько пиков было кандидатами, видно в `SensorReading.candidates` —
диагностика остаётся, догадка не делается (правило KB_05 №7).

Пределы `UpLimit` / `DownLimit` работают иначе: значение за пределом
**возвращается** и помечается `OUT_OF_LIMITS`. Предел описывает объявленный
диапазон датчика, а не достоверность арифметики; обнулить его в `NaN` значило
бы сделать выход за диапазон неотличимым от «пик не найден», а это разные
события (KB_05 №24).

`NaN` на входе даёт `NaN` на выходе везде. Если компенсация нужна, а опорный
датчик не измерен, результат `NaN` со `status = REFERENCE_MISSING`:
некомпенсированное значение под видом компенсированного — выдумка.

⚠️ Гипотеза о формуле компенсации
---------------------------------
Точный вид формулы у вендора **неизвестен**: из `FBGInfo` известны только имена
полей `TC_Coff` и `TC_Base`, самого выражения ни в одном источнике нет
(вопрос N22 в KB_04). Здесь принято::

    значение = значение₀ + k₁·(λ − λ₀) + k₂·(λ − λ₀)²
               + TC_Coff · (опорное − TC_Base)

Знак выбран «плюс», и это не догадка о вендоре, а решение сделать догадку
ненужной: `TC_Coff` — знаковое число, которое вводит пользователь, поэтому
вычитание представимо отрицательным коэффициентом. Что действительно
неизвестно и параметром не лечится — берётся ли у опорного датчика его
**калиброванное значение** (принято здесь) или сырая длина волны.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import pairwise

import numpy as np

#: Ключ, под которым набор датчиков лежит в JSON-файле калибровок.
SENSORS_KEY = "sensors"


class SensorType(IntEnum):
    """Тип датчика. Номера — из таблицы `FBGInfo` штатного ПО (KB_01)."""

    TEMPERATURE = 0
    DISPLACEMENT_MM = 1
    ACCELEROMETER_MG = 2
    STRAIN_UE = 3
    STEEL_STRESS_MPA = 4
    PRESSURE_RING_KN = 5
    INCLINOMETER = 6
    TEMP_COMPENSATION_PM = 7
    SETTLEMENT_MM = 8
    OTHER = 9


#: Единицы измерения по типу датчика. Пустая строка — единица **неизвестна**,
#: а не отсутствует: в KB_01 у инклинометра и у типа «другое» единица не указана,
#: и подставлять сюда правдоподобные «градусы» значило бы выдумать факт о приборе
#: (правило №10). UI в этом случае показывает значение без единицы.
UNITS: dict[SensorType, str] = {
    SensorType.TEMPERATURE: "°C",
    SensorType.DISPLACEMENT_MM: "мм",
    SensorType.ACCELEROMETER_MG: "mg",
    SensorType.STRAIN_UE: "µε",
    SensorType.STEEL_STRESS_MPA: "МПа",
    SensorType.PRESSURE_RING_KN: "кН",
    SensorType.INCLINOMETER: "",
    SensorType.TEMP_COMPENSATION_PM: "пм",
    SensorType.SETTLEMENT_MM: "мм",
    SensorType.OTHER: "",
}


class ReadingStatus(StrEnum):
    """Исход вычисления одного датчика."""

    OK = "ok"
    PEAK_NOT_FOUND = "peak_not_found"
    """Ни один пик канала не попал в окно. Штатная ситуация, не ошибка."""

    AMBIGUOUS = "ambiguous"
    """В окно попало больше одного пика: датчик не опознан однозначно."""

    REFERENCE_MISSING = "reference_missing"
    """Нужна компенсация, но опорный датчик не измерен."""

    OUT_OF_LIMITS = "out_of_limits"
    """Значение посчитано, но вышло за `up_limit` / `down_limit`."""


class FitKind(StrEnum):
    """Форма подгонки опорных точек."""

    LINEAR = "linear"
    QUADRATIC = "quadratic"


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """Одна воспроизводимая опорная точка калибровки."""

    wavelength_nm: float
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm опорной точки должна быть положительной")
        if not math.isfinite(self.value):
            raise ValueError("value опорной точки должно быть конечным")


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    """Результат подгонки относительно выбранного `reference_nm`."""

    kind: FitKind
    reference_nm: float
    value0: float
    k1: float
    k2: float
    rms: float
    max_abs_residual: float
    points: int


@dataclass(frozen=True)
class TempCompensation:
    """Температурная компенсация со ссылкой на другой датчик набора.

    Поля соответствуют `TC_Coff` и `TC_Base` таблицы `FBGInfo`. Пара
    `TC_Ch` + `TC_Seq` заменена на `reference` — идентификатор датчика:
    адресация по номеру позиции запрещена решением Р30.
    """

    reference: str
    """Идентификатор опорного датчика в том же наборе."""

    coeff: float
    """`TC_Coff`: сколько единиц величины на единицу опорного значения."""

    base: float
    """`TC_Base`: опорное значение, при котором компенсация равна нулю."""


@dataclass(frozen=True)
class Sensor:
    """Описание одного датчика: где искать его пик и как считать величину.

    Номер канала **0-based**, как в `codec` и как индекс в массиве кадра:
    канал 1 прибора — это 0. Позиции внутри канала не адресуются вовсе (Р30).
    """

    id: str
    """Идентификатор внутри набора: на него ссылается компенсация."""

    name: str
    """Человеческое имя для UI и заголовков. Может быть на русском."""

    channel: int
    """Номер канала прибора, 0-based."""

    type: SensorType

    expected_nm: float
    """Центр окна поиска и одновременно `λ₀` калибровочной кривой."""

    window_nm: float
    """Полуширина окна поиска. Рабочий диапазон датчика, не разброс решётки."""

    value0: float = 0.0
    """Физическая величина при `λ = expected_nm`."""

    k1: float = 0.0
    """Линейная чувствительность: единиц величины на нм."""

    k2: float = 0.0
    """Квадратичный член: единиц величины на нм²."""

    calibration_points: tuple[CalibrationPoint, ...] = ()
    """Сохранённые исходные точки, из которых получены коэффициенты."""

    up_limit: float | None = None
    down_limit: float | None = None

    compensation: TempCompensation | None = None

    def __post_init__(self) -> None:
        """Проверяет описание датчика: некорректное — баг вызывающего (KB_05)."""
        if not self.id:
            raise ValueError("id датчика не может быть пустым")
        if self.channel < 0:
            raise ValueError(f"{self.id}: channel={self.channel} не может быть отрицательным")
        if not math.isfinite(self.expected_nm) or self.expected_nm <= 0:
            raise ValueError(f"{self.id}: expected_nm={self.expected_nm} должна быть положительной")
        if not math.isfinite(self.window_nm) or self.window_nm <= 0:
            raise ValueError(f"{self.id}: window_nm={self.window_nm} должно быть положительным")
        for name, value in (
            ("value0", self.value0),
            ("k1", self.k1),
            ("k2", self.k2),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{self.id}: коэффициент {name}={value} должен быть конечным")
        if (
            self.up_limit is not None
            and self.down_limit is not None
            and self.up_limit < self.down_limit
        ):
            raise ValueError(
                f"{self.id}: up_limit={self.up_limit} ниже down_limit={self.down_limit}"
            )
        if self.compensation is not None and self.compensation.reference == self.id:
            raise ValueError(f"{self.id}: компенсация не может ссылаться на сам датчик")

    @property
    def unit(self) -> str:
        """Единица измерения. Пустая строка — единица неизвестна (см. `UNITS`)."""
        return UNITS[self.type]

    @property
    def low_nm(self) -> float:
        """Нижняя граница окна поиска, нм."""
        return self.expected_nm - self.window_nm

    @property
    def high_nm(self) -> float:
        """Верхняя граница окна поиска, нм."""
        return self.expected_nm + self.window_nm


@dataclass(frozen=True, slots=True)
class SensorReading:
    """Результат по одному датчику за один кадр."""

    sensor_id: str
    status: ReadingStatus

    wavelength_nm: float
    """Найденная длина волны. `NaN`, если пик не найден или неоднозначен."""

    value: float
    """Физическая величина. `NaN` всюду, кроме `OK` и `OUT_OF_LIMITS`."""

    position: int
    """Позиция пика в кадре, из которой он взят. −1, если пик не найден.

    Только диагностика: связывать датчик с позицией запрещено (Р30). Поле
    существует ровно затем, чтобы дрейф слотов было **видно**.
    """

    candidates: int
    """Сколько пиков канала попало в окно."""

    @property
    def ok(self) -> bool:
        """True, если значение посчитано (включая выход за пределы)."""
        return self.status in (ReadingStatus.OK, ReadingStatus.OUT_OF_LIMITS)


def _missing(sensor_id: str, status: ReadingStatus, candidates: int = 0) -> SensorReading:
    """Результат без значения."""
    return SensorReading(
        sensor_id=sensor_id,
        status=status,
        wavelength_nm=math.nan,
        value=math.nan,
        position=-1,
        candidates=candidates,
    )


def match_peak(
    wavelengths_nm: np.ndarray, expected_nm: float, window_nm: float
) -> tuple[int, int, float]:
    """Ищет пик датчика среди длин волн одного канала.

    Возвращает `(позиция, число кандидатов, длина волны)`. Позиция −1 и `NaN`
    означают, что однозначного пика нет: либо кандидатов ноль, либо больше
    одного. Решение по каждому случаю принимает `evaluate` — здесь только поиск.

    `NaN` в массиве отсеиваются сами: сравнение с `NaN` ложно, и позиция без
    найденного пика кандидатом не становится.
    """
    values = np.asarray(wavelengths_nm, dtype=np.float64)
    distance = np.abs(values - expected_nm)
    inside = np.flatnonzero(distance <= window_nm)
    if inside.size != 1:
        return -1, int(inside.size), math.nan
    position = int(inside[0])
    return position, 1, float(values[position])


def apply_curve(sensor: Sensor, wavelength_nm: float) -> float:
    """Кривая относительно опоры без компенсации и без проверки пределов.

    `NaN` на входе даёт `NaN` на выходе: правило KB_05 №7 действует и здесь.
    """
    if not math.isfinite(wavelength_nm):
        return math.nan
    delta_nm = wavelength_nm - sensor.expected_nm
    return sensor.value0 + sensor.k1 * delta_nm + sensor.k2 * delta_nm * delta_nm


def fit_calibration(
    points: Sequence[CalibrationPoint],
    reference_nm: float,
    *,
    kind: FitKind = FitKind.LINEAR,
) -> CalibrationFit:
    """Подгоняет сохранённые точки в координате `(λ − reference_nm)`.

    Прямая — рабочий вариант по умолчанию и требует минимум две точки.
    Парабола разрешена только от **четырёх** точек: три точки определяют её
    точно и дают нулевую невязку по построению, что выглядит как проверенная
    идеальная калибровка, хотя проверено ничего не было (KB_05 №39).
    """
    if not math.isfinite(reference_nm) or reference_nm <= 0.0:
        raise ValueError("reference_nm должна быть положительной")
    required = 4 if kind is FitKind.QUADRATIC else 2
    if len(points) < required:
        name = "параболы" if kind is FitKind.QUADRATIC else "прямой"
        raise ValueError(f"для {name} нужно не менее {required} опорных точек")

    wavelengths = np.fromiter((point.wavelength_nm for point in points), dtype=np.float64)
    values = np.fromiter((point.value for point in points), dtype=np.float64)
    x = wavelengths - reference_nm
    degree = 2 if kind is FitKind.QUADRATIC else 1
    required_distinct = degree + 1
    if np.unique(wavelengths).size < required_distinct:
        raise ValueError(
            f"для подгонки степени {degree} нужно не менее {required_distinct} "
            "различных длин волн"
        )
    coefficients = np.polyfit(x, values, degree)
    if degree == 2:
        k2, k1, value0 = (float(value) for value in coefficients)
    else:
        k1, value0 = (float(value) for value in coefficients)
        k2 = 0.0
    predicted = value0 + k1 * x + k2 * x * x
    residual = values - predicted
    rms = float(np.sqrt(np.mean(residual * residual)))
    max_abs = float(np.max(np.abs(residual)))
    return CalibrationFit(kind, reference_nm, value0, k1, k2, rms, max_abs, len(points))


def _classify_limits(sensor: Sensor, value: float) -> ReadingStatus:
    """OK или OUT_OF_LIMITS: значение за пределом не теряется, а помечается."""
    if sensor.up_limit is not None and value > sensor.up_limit:
        return ReadingStatus.OUT_OF_LIMITS
    if sensor.down_limit is not None and value < sensor.down_limit:
        return ReadingStatus.OUT_OF_LIMITS
    return ReadingStatus.OK


def evaluate(
    sensor: Sensor,
    wavelengths_nm: np.ndarray,
    *,
    reference_value: float = math.nan,
) -> SensorReading:
    """Считает величину одного датчика по длинам волн **его канала**.

    `reference_value` нужен только датчику с компенсацией: это значение
    опорного датчика. Если оно `NaN`, а компенсация задана, результат —
    `NaN` со статусом `REFERENCE_MISSING`: некомпенсированное значение под
    видом компенсированного было бы выдумкой.
    """
    position, candidates, found_nm = match_peak(
        wavelengths_nm, sensor.expected_nm, sensor.window_nm
    )
    if position < 0:
        status = ReadingStatus.AMBIGUOUS if candidates > 1 else ReadingStatus.PEAK_NOT_FOUND
        return _missing(sensor.id, status, candidates)

    value = apply_curve(sensor, found_nm)
    if sensor.compensation is not None:
        if not math.isfinite(reference_value):
            return SensorReading(
                sensor_id=sensor.id,
                status=ReadingStatus.REFERENCE_MISSING,
                wavelength_nm=found_nm,
                value=math.nan,
                position=position,
                candidates=candidates,
            )
        value += sensor.compensation.coeff * (reference_value - sensor.compensation.base)

    return SensorReading(
        sensor_id=sensor.id,
        status=_classify_limits(sensor, value),
        wavelength_nm=found_nm,
        value=value,
        position=position,
        candidates=candidates,
    )


def evaluate_all(sensors: Sequence[Sensor], wavelengths_nm: np.ndarray) -> dict[str, SensorReading]:
    """Считает весь набор датчиков по кадру длин волн формы (каналы, позиции).

    Порядок работы в два прохода, потому что компенсация ссылается на другой
    датчик: сначала считаются все датчики без компенсации, потом те, у кого
    она есть. Цепочки ссылок глубже одного уровня запрещены `validate_sensors`,
    поэтому рекурсии и поиска циклов здесь нет и не нужно.

    Датчик на несуществующем канале получает `PEAK_NOT_FOUND`, а не исключение:
    кадр мог прийти от прибора с меньшим числом каналов, и это диагностика
    конфигурации, а не баг вызывающего.
    """
    frame = np.asarray(wavelengths_nm, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"ожидался массив формы (каналы, позиции), получен ndim={frame.ndim}")

    readings: dict[str, SensorReading] = {}
    deferred: list[Sensor] = []

    for sensor in sensors:
        if sensor.compensation is not None:
            deferred.append(sensor)
            continue
        if not 0 <= sensor.channel < frame.shape[0]:
            readings[sensor.id] = _missing(sensor.id, ReadingStatus.PEAK_NOT_FOUND)
            continue
        readings[sensor.id] = evaluate(sensor, frame[sensor.channel])

    for sensor in deferred:
        if not 0 <= sensor.channel < frame.shape[0]:
            readings[sensor.id] = _missing(sensor.id, ReadingStatus.PEAK_NOT_FOUND)
            continue
        assert sensor.compensation is not None
        reference = readings.get(sensor.compensation.reference)
        reference_value = math.nan if reference is None else reference.value
        readings[sensor.id] = evaluate(
            sensor, frame[sensor.channel], reference_value=reference_value
        )

    return readings


def validate_sensors(sensors: Sequence[Sensor]) -> tuple[str, ...]:
    """Проверяет набор целиком и возвращает список замечаний.

    Пустой кортеж — набор согласован. Проверяются четыре вещи, и все четыре
    сломались бы иначе молча, уже на данных:

    * повторяющийся `id` — компенсация сослалась бы неизвестно на что;
    * ссылка компенсации в пустоту;
    * ссылка на датчик, у которого своя компенсация: цепочка глубже одного
      уровня в модели вендора не встречается, а `evaluate_all` её не считает;
    * пересечение окон двух датчиков одного канала — при таком наборе
      `AMBIGUOUS` становится не аномалией, а нормой, и датчики перестают
      читаться вовсе.
    """
    problems: list[str] = []
    seen: dict[str, Sensor] = {}
    for sensor in sensors:
        if sensor.id in seen:
            problems.append(f"идентификатор {sensor.id!r} встречается больше одного раза")
            continue
        seen[sensor.id] = sensor

    for sensor in seen.values():
        if sensor.compensation is None:
            continue
        target = seen.get(sensor.compensation.reference)
        if target is None:
            problems.append(
                f"{sensor.id}: компенсация ссылается на {sensor.compensation.reference!r}, "
                "которого нет в наборе"
            )
        elif target.compensation is not None:
            problems.append(
                f"{sensor.id}: компенсация ссылается на {target.id!r}, у которого своя "
                "компенсация; цепочки глубже одного уровня не поддерживаются"
            )

    ordered = sorted(seen.values(), key=lambda s: (s.channel, s.expected_nm))
    for left, right in pairwise(ordered):
        if left.channel != right.channel:
            continue
        if left.high_nm >= right.low_nm:
            problems.append(
                f"канал {left.channel}: окна {left.id!r} "
                f"({left.low_nm:.4f}…{left.high_nm:.4f} нм) и {right.id!r} "
                f"({right.low_nm:.4f}…{right.high_nm:.4f} нм) пересекаются"
            )
    return tuple(problems)


# --------------------------------------------------------------------------------------
# JSON: чистое преобразование «объект ↔ словарь». Файлы читает fbg/io/config.py
# --------------------------------------------------------------------------------------


def sensor_to_json(sensor: Sensor) -> dict[str, object]:
    """Словарь, готовый к `json.dump`. Необязательные поля опускаются."""
    data: dict[str, object] = {
        "id": sensor.id,
        "name": sensor.name,
        "channel": sensor.channel,
        "type": int(sensor.type),
        "expected_nm": sensor.expected_nm,
        "window_nm": sensor.window_nm,
        "value0": sensor.value0,
        "k1": sensor.k1,
        "k2": sensor.k2,
        "calibration_points": [
            {"wavelength_nm": point.wavelength_nm, "value": point.value}
            for point in sensor.calibration_points
        ],
    }
    if sensor.up_limit is not None:
        data["up_limit"] = sensor.up_limit
    if sensor.down_limit is not None:
        data["down_limit"] = sensor.down_limit
    if sensor.compensation is not None:
        data["compensation"] = {
            "reference": sensor.compensation.reference,
            "coeff": sensor.compensation.coeff,
            "base": sensor.compensation.base,
        }
    return data


def _optional_number(source: Mapping[str, object], key: str) -> float | None:
    """Читает число, отвергая bool: в JSON `true` числом не является."""
    if key not in source or source[key] is None:
        return None
    value = source[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"поле {key!r}: ожидалось число, получено {value!r}")
    return float(value)


def _number(source: Mapping[str, object], key: str, default: float) -> float:
    """То же, но с умолчанием: отсутствующее поле не ошибка."""
    value = _optional_number(source, key)
    return default if value is None else value


def sensor_from_json(source: Mapping[str, object]) -> Sensor:
    """Собирает датчик из словаря. Некорректное описание — `ValueError`.

    Ошибку ловит вызывающий и превращает в замечание для пользователя: файл
    калибровок правят руками, и одна испорченная запись не должна отменять
    остальные.
    """
    legacy = sorted({"c0", "c1", "c2"}.intersection(source))
    if legacy:
        raise ValueError(
            "устаревшая абсолютная форма калибровки "
            f"({', '.join(legacy)}); автоматическая миграция не выполняется"
        )

    raw_id = source.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError(f"поле 'id': ожидалась непустая строка, получено {raw_id!r}")

    raw_name = source.get("name", raw_id)
    if not isinstance(raw_name, str):
        raise ValueError(f"{raw_id}: поле 'name' должно быть строкой")

    raw_channel = source.get("channel")
    if isinstance(raw_channel, bool) or not isinstance(raw_channel, int):
        raise ValueError(f"{raw_id}: поле 'channel' должно быть целым, получено {raw_channel!r}")

    raw_type = source.get("type")
    if isinstance(raw_type, bool) or not isinstance(raw_type, int):
        raise ValueError(f"{raw_id}: поле 'type' должно быть целым, получено {raw_type!r}")
    try:
        sensor_type = SensorType(raw_type)
    except ValueError as exc:
        raise ValueError(f"{raw_id}: неизвестный тип датчика {raw_type}") from exc

    expected = _optional_number(source, "expected_nm")
    window = _optional_number(source, "window_nm")
    if expected is None or window is None:
        raise ValueError(f"{raw_id}: обязательны поля 'expected_nm' и 'window_nm'")

    compensation: TempCompensation | None = None
    raw_comp = source.get("compensation")
    if raw_comp is not None:
        if not isinstance(raw_comp, Mapping):
            raise ValueError(f"{raw_id}: поле 'compensation' должно быть объектом")
        reference = raw_comp.get("reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"{raw_id}: 'compensation.reference' должно быть непустой строкой")
        compensation = TempCompensation(
            reference=reference,
            coeff=_number(raw_comp, "coeff", 0.0),
            base=_number(raw_comp, "base", 0.0),
        )

    calibration_points: tuple[CalibrationPoint, ...] = ()
    raw_points = source.get("calibration_points", [])
    if not isinstance(raw_points, list):
        raise ValueError(f"{raw_id}: поле 'calibration_points' должно быть массивом")
    parsed_points: list[CalibrationPoint] = []
    for index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"{raw_id}: calibration_points[{index}] должно быть объектом")
        wavelength_nm = _optional_number(raw_point, "wavelength_nm")
        value = _optional_number(raw_point, "value")
        if wavelength_nm is None or value is None:
            raise ValueError(
                f"{raw_id}: calibration_points[{index}] требует wavelength_nm и value"
            )
        parsed_points.append(CalibrationPoint(wavelength_nm, value))
    calibration_points = tuple(parsed_points)

    return Sensor(
        id=raw_id,
        name=raw_name,
        channel=raw_channel,
        type=sensor_type,
        expected_nm=expected,
        window_nm=window,
        value0=_number(source, "value0", 0.0),
        k1=_number(source, "k1", 0.0),
        k2=_number(source, "k2", 0.0),
        calibration_points=calibration_points,
        up_limit=_optional_number(source, "up_limit"),
        down_limit=_optional_number(source, "down_limit"),
        compensation=compensation,
    )


def sensors_to_json(sensors: Iterable[Sensor]) -> dict[str, object]:
    """Полное содержимое файла калибровок."""
    return {SENSORS_KEY: [sensor_to_json(sensor) for sensor in sensors]}
