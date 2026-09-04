"""Тесты калибровки: кривая, компенсация, пределы, поиск датчика по длине волны.

Модуль чистый, прибора и файлов здесь нет вовсе. Проверяется в основном
не арифметика (она в одну строку), а **решения**: что происходит, когда пик
не найден, когда их два, когда опорный датчик не измерен и когда значение
вышло за объявленный предел. Каждое из четырёх — отдельное событие, и тесты
следят за тем, чтобы они оставались различимыми (KB_05 №24).

⚠️ Формула температурной компенсации — **гипотеза** (вопрос N22 в KB_04):
из таблицы `FBGInfo` штатного ПО известны только имена полей `TC_Coff`
и `TC_Base`, самого выражения нет ни в одном источнике. Тесты ниже фиксируют
поведение **нашего кода**, а не поведение вендора (KB_05 №12).
"""

import math

import numpy as np
import pytest

from fbg.core.calibration import (
    UNITS,
    CalibrationPoint,
    FitKind,
    ReadingStatus,
    Sensor,
    SensorType,
    TempCompensation,
    apply_curve,
    evaluate,
    evaluate_all,
    fit_calibration,
    match_peak,
    sensor_from_json,
    sensor_to_json,
    sensors_to_json,
    validate_sensors,
)

#: Длины волн решёток стенда, ✅ скрининг 27.08.2026 (KB_01).
STAND_NM_1 = 1544.787
STAND_NM_2 = 1551.505


def temperature_sensor(**overrides: object) -> Sensor:
    """Датчик температуры на первой решётке стенда: 0.1 °C на пикометр."""
    params: dict[str, object] = {
        "id": "T1",
        "name": "Температура балки",
        "channel": 0,
        "type": SensorType.TEMPERATURE,
        "expected_nm": STAND_NM_1,
        "window_nm": 0.35,
        "value0": 0.0,
        "k1": 100.0,
    }
    params.update(overrides)
    return Sensor(**params)  # type: ignore[arg-type]


def frame(*rows: list[float]) -> np.ndarray:
    """Кадр длин волн формы (каналы, позиции)."""
    return np.array(rows, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Кривая
# --------------------------------------------------------------------------------------


def test_линейная_кривая() -> None:
    """`value0 + k1·(λ − λ0)` считается точно."""
    sensor = temperature_sensor(expected_nm=2.0, value0=1.0, k1=2.0, k2=0.0)
    assert apply_curve(sensor, 3.0) == pytest.approx(3.0)


def test_квадратичная_кривая() -> None:
    """Третий член `k2·(λ − λ0)²` участвует наравне с остальными."""
    sensor = temperature_sensor(expected_nm=2.0, value0=1.0, k1=2.0, k2=3.0)
    assert apply_curve(sensor, 3.0) == pytest.approx(1.0 + 2.0 + 3.0)


def test_кривая_на_реальной_длине_волны_стенда() -> None:
    """Коэффициенты подобраны так, что 1544.787 нм даёт 0 °C — проверка масштаба."""
    sensor = temperature_sensor()
    assert apply_curve(sensor, STAND_NM_1) == pytest.approx(0.0, abs=1e-6)
    # 10 пм сдвига — это 1 °C при коэффициенте 100 °C/нм.
    assert apply_curve(sensor, STAND_NM_1 + 0.010) == pytest.approx(1.0, abs=1e-6)


def test_опорная_форма_не_требует_большого_свободного_члена() -> None:
    """Температурная решётка описывается физическими 25 °C и 100 °C/нм."""
    sensor = temperature_sensor(expected_nm=1544.80, value0=25.0, k1=100.0, k2=0.0)
    assert apply_curve(sensor, 1544.80) == pytest.approx(25.0)
    assert apply_curve(sensor, 1544.81) == pytest.approx(26.0)


def test_fit_прямой_по_трем_точкам_показывает_невязку() -> None:
    """KB_05 №39: прямая — default; третья точка проверяет, а не определяет её."""
    points = (
        CalibrationPoint(1544.80, 20.0),
        CalibrationPoint(1545.00, 40.2),
        CalibrationPoint(1545.20, 59.8),
    )
    fit = fit_calibration(points, 1544.80)
    assert fit.kind is FitKind.LINEAR
    assert fit.k2 == 0.0
    assert fit.k1 == pytest.approx(99.5, rel=1e-3)
    assert fit.rms > 0.0
    assert fit.max_abs_residual > 0.0


def test_fit_параболы_требует_четыре_точки() -> None:
    """Три точки не дают ложную «идеальную» параболу с нулевой невязкой."""
    points = (
        CalibrationPoint(1544.80, 20.0),
        CalibrationPoint(1545.00, 40.0),
        CalibrationPoint(1545.20, 60.0),
    )
    with pytest.raises(ValueError, match="не менее 4"):
        fit_calibration(points, 1544.80, kind=FitKind.QUADRATIC)


def test_fit_параболы_по_четырем_точкам() -> None:
    """Явно выбранный квадратичный член восстанавливается от четырёх точек."""
    reference = 1550.0
    xs = (-0.3, -0.1, 0.1, 0.3)
    points = tuple(
        CalibrationPoint(reference + x, 5.0 + 2.0 * x + 3.0 * x * x) for x in xs
    )
    fit = fit_calibration(points, reference, kind=FitKind.QUADRATIC)
    assert fit.value0 == pytest.approx(5.0)
    assert fit.k1 == pytest.approx(2.0)
    assert fit.k2 == pytest.approx(3.0)
    assert fit.rms < 1e-10


def test_nan_на_входе_кривой_даёт_nan() -> None:
    """Правило KB_05 №7 действует и в калибровке: данные не додумываются."""
    assert math.isnan(apply_curve(temperature_sensor(), math.nan))


def test_все_десять_типов_датчиков_известны() -> None:
    """Десять типов из таблицы `FBGInfo` (KB_01), и у каждого есть строка единиц."""
    assert [int(t) for t in SensorType] == list(range(10))
    assert set(UNITS) == set(SensorType)


def test_все_десять_типов_считаются_одинаково() -> None:
    """Тип — ярлык для UI: на арифметику он не влияет ни у одного из десяти."""
    for sensor_type in SensorType:
        sensor = temperature_sensor(id=f"S{int(sensor_type)}", type=sensor_type, value0=5.0, k1=0.0)
        reading = evaluate(sensor, np.array([STAND_NM_1]))
        assert reading.status is ReadingStatus.OK
        assert reading.value == pytest.approx(5.0)


def test_единица_неизвестного_типа_пустая_а_не_выдуманная() -> None:
    """У инклинометра и «другого» единица в KB_01 не указана — и не выдумывается.

    Подставить сюда правдоподобные «градусы» значило бы завести факт о приборе,
    которого нет ни в одном источнике (правило №10).
    """
    assert UNITS[SensorType.INCLINOMETER] == ""
    assert UNITS[SensorType.OTHER] == ""
    assert UNITS[SensorType.STRAIN_UE] == "µε"


# --------------------------------------------------------------------------------------
# Поиск пика: три исхода
# --------------------------------------------------------------------------------------


def test_ровно_один_пик_в_допуске() -> None:
    """Единственный кандидат — он и есть датчик."""
    row = np.array([STAND_NM_1, STAND_NM_2, math.nan])
    position, candidates, found = match_peak(row, STAND_NM_1, 0.35)
    assert (position, candidates) == (0, 1)
    assert found == pytest.approx(STAND_NM_1)


def test_пик_найден_не_в_нулевой_позиции() -> None:
    """Позиции — слоты, а не номера решёток (Р30): датчик ищется, а не адресуется.

    Ровно случай стенда: 1551.505 нм лежит в позиции 1, будучи четвёртой
    решёткой линии, и калибровка обязана его найти именно поиском.
    """
    row = np.array([STAND_NM_1, STAND_NM_2] + [math.nan] * 28)
    position, candidates, found = match_peak(row, STAND_NM_2, 0.35)
    assert (position, candidates) == (1, 1)
    assert found == pytest.approx(STAND_NM_2)


def test_датчик_переезжает_вслед_за_слотом() -> None:
    """Пропала первая решётка — вторая переехала в позицию 0, датчик тот же.

    Это и есть причина Р30: привязка к номеру позиции здесь начала бы считать
    данные одного датчика по коэффициентам другого, причём молча.
    """
    sensor = temperature_sensor(id="T2", expected_nm=STAND_NM_2, value0=0.0, k1=1.0)
    было = evaluate(sensor, np.array([STAND_NM_1, STAND_NM_2]))
    стало = evaluate(sensor, np.array([STAND_NM_2, math.nan]))
    assert (было.position, стало.position) == (1, 0)
    assert было.value == pytest.approx(стало.value)


def test_ни_одного_пика_в_допуске_это_nan_а_не_ошибка() -> None:
    """Кадр без пиков штатен: на стенде из четырёх решёток распознаются две."""
    reading = evaluate(temperature_sensor(), np.array([1560.0, math.nan]))
    assert reading.status is ReadingStatus.PEAK_NOT_FOUND
    assert math.isnan(reading.value) and math.isnan(reading.wavelength_nm)
    assert reading.position == -1 and reading.candidates == 0


def test_два_пика_в_допуске_дают_nan_а_не_ближайший() -> None:
    """Неоднозначность — не повод угадать.

    «Ближайший» вернул бы правдоподобное число: если настоящая решётка датчика
    пропала, а в окне остался соседний или паразитный пик, значение оказалось бы
    чужим под именем этого датчика. Число кандидатов при этом сохраняется —
    диагностика остаётся, догадка не делается.
    """
    row = np.array([STAND_NM_1, STAND_NM_1 + 0.1])
    reading = evaluate(temperature_sensor(), row)
    assert reading.status is ReadingStatus.AMBIGUOUS
    assert math.isnan(reading.value) and math.isnan(reading.wavelength_nm)
    assert reading.candidates == 2 and reading.position == -1


def test_ближайший_не_выбирается_даже_при_явно_разном_расстоянии() -> None:
    """Один кандидат вплотную, второй у самого края окна — всё равно NaN.

    Тест охраняет именно решение, а не арифметику: соблазн «взять ближайший»
    сильнее всего там, где расстояния различаются в десятки раз.
    """
    row = np.array([STAND_NM_1 + 0.001, STAND_NM_1 + 0.349])
    assert evaluate(temperature_sensor(), row).status is ReadingStatus.AMBIGUOUS


def test_граница_допуска_включительная() -> None:
    """Пик ровно на границе окна считается попавшим."""
    sensor = temperature_sensor(expected_nm=1550.0, window_nm=0.5)
    assert evaluate(sensor, np.array([1550.5])).status is ReadingStatus.OK
    assert evaluate(sensor, np.array([1550.5 + 1e-9])).status is ReadingStatus.PEAK_NOT_FOUND


def test_nan_позиции_кандидатами_не_становятся() -> None:
    """Кадр, где прибор не нашёл пиков, — это 28 NaN из 30 (скрининг)."""
    row = np.full(30, math.nan)
    row[7] = STAND_NM_1
    position, candidates, _ = match_peak(row, STAND_NM_1, 0.35)
    assert (position, candidates) == (7, 1)


def test_допуск_покрывает_рабочий_диапазон_а_не_разброс_решётки() -> None:
    """Окно ±0.35 нм держит весь рабочий диапазон датчика температуры.

    Арифметика из KB_01: FBG ≈ 10 пм/°C, прибор работает −15…+55 °C, то есть
    0.70 нм полного хода. Окно, выбранное по паспортному разбросу решётки
    (±0.05 нм), потеряло бы датчик при первом же реальном нагреве — и потерял
    бы он его именно тогда, когда датчик интересен.
    """
    sensor = temperature_sensor(expected_nm=1548.0, window_nm=0.35)
    узкий = temperature_sensor(expected_nm=1548.0, window_nm=0.05)
    for градусы in (-35.0, 0.0, 35.0):
        нагретая = 1548.0 + градусы * 0.010
        assert evaluate(sensor, np.array([нагретая])).status is ReadingStatus.OK
    assert evaluate(узкий, np.array([1548.0 + 35.0 * 0.010])).status is ReadingStatus.PEAK_NOT_FOUND


# --------------------------------------------------------------------------------------
# Пределы
# --------------------------------------------------------------------------------------


def test_выход_за_верхний_предел_помечается_но_значение_остаётся() -> None:
    """Предел описывает диапазон датчика, а не достоверность арифметики.

    Обнулить значение в NaN значило бы сделать выход за диапазон неотличимым
    от «пик не найден», а это разные события (KB_05 №24).
    """
    sensor = temperature_sensor(value0=1200.0, k1=1.0, up_limit=1000.0)
    reading = evaluate(sensor, np.array([STAND_NM_1]))
    assert reading.status is ReadingStatus.OUT_OF_LIMITS
    assert reading.value == pytest.approx(1200.0)
    assert reading.ok


def test_выход_за_нижний_предел() -> None:
    """Симметрично верхнему."""
    sensor = temperature_sensor(value0=1000.0, k1=1.0, down_limit=2000.0)
    assert evaluate(sensor, np.array([STAND_NM_1])).status is ReadingStatus.OUT_OF_LIMITS


def test_внутри_пределов_статус_ok() -> None:
    """Пределы, в которые значение укладывается, статуса не меняют."""
    sensor = temperature_sensor(value0=1550.0, k1=1.0, down_limit=1500.0, up_limit=1600.0)
    assert evaluate(sensor, np.array([STAND_NM_1])).status is ReadingStatus.OK


def test_перевёрнутые_пределы_отвергаются_при_создании() -> None:
    """Несогласованное описание датчика — баг вызывающего, значит ValueError."""
    with pytest.raises(ValueError, match="up_limit"):
        temperature_sensor(up_limit=1.0, down_limit=2.0)


# --------------------------------------------------------------------------------------
# Температурная компенсация
# --------------------------------------------------------------------------------------


def test_компенсация_сдвигает_значение_на_отклонение_опорного() -> None:
    """⚠️ Гипотеза N22: `значение + TC_Coff · (опорное − TC_Base)`."""
    sensor = temperature_sensor(
        id="S1",
        value0=0.0,
        k1=0.0,
        k2=0.0,
        compensation=TempCompensation(reference="T1", coeff=-2.0, base=20.0),
    )
    reading = evaluate(sensor, np.array([STAND_NM_1]), reference_value=25.0)
    assert reading.value == pytest.approx(-10.0)


def test_знак_компенсации_задаётся_коэффициентом_а_не_кодом() -> None:
    """Вычитание представимо отрицательным `TC_Coff`, поэтому знак не догадка.

    Ровно это делает неизвестность формулы у вендора безвредной: какой бы знак
    он ни использовал, пользователь вводит знаковое число. Проверяются
    отклонения от некомпенсированного значения — они обязаны быть равными
    по модулю и противоположными по знаку.
    """
    row = np.array([STAND_NM_1])
    без = evaluate(temperature_sensor(id="0", k1=0.0), row).value
    вверх = evaluate(
        temperature_sensor(id="A", k1=0.0, compensation=TempCompensation("T1", 1.5, 20.0)),
        row,
        reference_value=30.0,
    ).value
    вниз = evaluate(
        temperature_sensor(id="B", k1=0.0, compensation=TempCompensation("T1", -1.5, 20.0)),
        row,
        reference_value=30.0,
    ).value
    assert вверх - без == pytest.approx(15.0)
    assert вниз - без == pytest.approx(-15.0)


def test_компенсация_на_опорном_значении_равном_базе_ничего_не_меняет() -> None:
    """При `опорное == TC_Base` компенсация равна нулю по построению."""
    comp = TempCompensation(reference="T1", coeff=7.0, base=20.0)
    sensor = temperature_sensor(id="S1", value0=3.0, k1=0.0, compensation=comp)
    assert evaluate(sensor, np.array([STAND_NM_1]), reference_value=20.0).value == pytest.approx(
        3.0
    )


def test_опорный_датчик_не_измерен_даёт_nan_а_не_некомпенсированное() -> None:
    """Некомпенсированное значение под видом компенсированного — выдумка."""
    comp = TempCompensation(reference="T1", coeff=1.0, base=0.0)
    sensor = temperature_sensor(id="S1", compensation=comp)
    reading = evaluate(sensor, np.array([STAND_NM_1]), reference_value=math.nan)
    assert reading.status is ReadingStatus.REFERENCE_MISSING
    assert math.isnan(reading.value)
    # Длина волны при этом найдена и сохранена: сырое от калибровки не зависит.
    assert reading.wavelength_nm == pytest.approx(STAND_NM_1)


def test_компенсация_не_может_ссылаться_на_себя() -> None:
    """Самоссылка — баг описания, а не рабочий случай."""
    with pytest.raises(ValueError, match="сам датчик"):
        temperature_sensor(compensation=TempCompensation(reference="T1", coeff=1.0, base=0.0))


# --------------------------------------------------------------------------------------
# Набор целиком
# --------------------------------------------------------------------------------------


def test_evaluate_all_считает_опорный_раньше_зависимого() -> None:
    """Порядок в списке значения не имеет: зависимые считаются вторым проходом."""
    reference = temperature_sensor(id="T1", expected_nm=STAND_NM_1, value0=0.0, k1=0.0, k2=0.0)
    dependent = temperature_sensor(
        id="S1",
        expected_nm=STAND_NM_2,
        value0=0.0,
        k1=0.0,
        k2=0.0,
        compensation=TempCompensation(reference="T1", coeff=3.0, base=-1.0),
    )
    readings = evaluate_all([dependent, reference], frame([STAND_NM_1, STAND_NM_2]))
    assert readings["T1"].value == pytest.approx(0.0)
    assert readings["S1"].value == pytest.approx(3.0)


def test_evaluate_all_разводит_датчики_по_каналам() -> None:
    """Канал — физическое волокно, привязка к нему законна (в отличие от позиции)."""
    первый = temperature_sensor(id="A", channel=0, expected_nm=1545.0, value0=0.0, k1=1.0)
    второй = temperature_sensor(id="B", channel=1, expected_nm=1545.0, value0=0.0, k1=1.0)
    readings = evaluate_all([первый, второй], frame([1545.0, math.nan], [1545.2, math.nan]))
    assert readings["A"].value == pytest.approx(0.0)
    assert readings["B"].value == pytest.approx(0.2)


def test_датчик_на_несуществующем_канале_не_роняет_расчёт() -> None:
    """Кадр мог прийти от прибора с меньшим числом каналов — это диагностика."""
    sensor = temperature_sensor(id="A", channel=7)
    readings = evaluate_all([sensor], frame([STAND_NM_1]))
    assert readings["A"].status is ReadingStatus.PEAK_NOT_FOUND


def test_потеря_опорного_датчика_гасит_зависимый_но_не_остальные() -> None:
    """Пропал опорный пик — зависимый стал NaN, независимый считается дальше."""
    reference = temperature_sensor(id="T1", expected_nm=STAND_NM_1)
    dependent = temperature_sensor(
        id="S1",
        expected_nm=STAND_NM_2,
        value0=0.0,
        k1=1.0,
        compensation=TempCompensation(reference="T1", coeff=1.0, base=0.0),
    )
    independent = temperature_sensor(id="S2", expected_nm=STAND_NM_2, value0=0.0, k1=1.0)
    readings = evaluate_all([reference, dependent, independent], frame([math.nan, STAND_NM_2]))
    assert readings["T1"].status is ReadingStatus.PEAK_NOT_FOUND
    assert readings["S1"].status is ReadingStatus.REFERENCE_MISSING
    assert readings["S2"].value == pytest.approx(0.0)


def test_кадр_без_единого_пика_даёт_nan_всем() -> None:
    """Рядовая ситуация стенда: линия отключена, распознано ноль решёток."""
    sensors = [temperature_sensor(id="A"), temperature_sensor(id="B", expected_nm=STAND_NM_2)]
    readings = evaluate_all(sensors, frame([math.nan] * 30))
    assert all(math.isnan(reading.value) for reading in readings.values())
    assert all(reading.status is ReadingStatus.PEAK_NOT_FOUND for reading in readings.values())


def test_evaluate_all_требует_двумерного_кадра() -> None:
    """Массив чужой формы — баг вызывающего (KB_05, таблица ошибок)."""
    with pytest.raises(ValueError, match="ndim"):
        evaluate_all([temperature_sensor()], np.array([1545.0]))


# --------------------------------------------------------------------------------------
# Проверка набора
# --------------------------------------------------------------------------------------


def test_согласованный_набор_замечаний_не_даёт() -> None:
    """Решётки стенда разнесены на 6.7 нм — запас при окне 0.35 нм любой."""
    sensors = [
        temperature_sensor(id="A", expected_nm=STAND_NM_1),
        temperature_sensor(id="B", expected_nm=STAND_NM_2),
    ]
    assert validate_sensors(sensors) == ()


def test_пересечение_окон_одного_канала_ловится_при_загрузке() -> None:
    """Иначе `AMBIGUOUS` стал бы нормой и датчики перестали бы читаться вовсе."""
    sensors = [
        temperature_sensor(id="A", expected_nm=1545.0, window_nm=0.5),
        temperature_sensor(id="B", expected_nm=1545.6, window_nm=0.5),
    ]
    problems = validate_sensors(sensors)
    assert len(problems) == 1 and "пересекаются" in problems[0]


def test_окна_на_разных_каналах_пересекаться_вправе() -> None:
    """Разные волокна — разные пики; совпадение длин волн там нормально."""
    sensors = [
        temperature_sensor(id="A", channel=0, expected_nm=1545.0, window_nm=0.5),
        temperature_sensor(id="B", channel=1, expected_nm=1545.0, window_nm=0.5),
    ]
    assert validate_sensors(sensors) == ()


def test_окно_шире_соседства_решёток_ловится() -> None:
    """Минимальный интервал пиков прибора — 30 ГГц, это 0.24 нм при 1550 нм.

    Значит у плотно набитой линии окно не может быть шире ±0.12 нм, и набор
    с окнами ±0.2 нм на таком шаге неработоспособен по построению.
    """
    sensors = [
        temperature_sensor(id="A", expected_nm=1550.00, window_nm=0.2),
        temperature_sensor(id="B", expected_nm=1550.24, window_nm=0.2),
    ]
    assert validate_sensors(sensors) != ()


def test_ссылка_компенсации_в_пустоту_ловится() -> None:
    """Иначе датчик молча считался бы вечным `REFERENCE_MISSING`."""
    sensor = temperature_sensor(
        id="S1", compensation=TempCompensation(reference="нет такого", coeff=1.0, base=0.0)
    )
    problems = validate_sensors([sensor])
    assert len(problems) == 1 and "нет в наборе" in problems[0]


def test_цепочка_компенсаций_глубже_одного_уровня_ловится() -> None:
    """`evaluate_all` считает в два прохода и цепочку не разрешает — это названо."""
    a = temperature_sensor(id="A", expected_nm=1545.0)
    b = temperature_sensor(id="B", expected_nm=1550.0, compensation=TempCompensation("A", 1.0, 0.0))
    c = temperature_sensor(id="C", expected_nm=1555.0, compensation=TempCompensation("B", 1.0, 0.0))
    problems = validate_sensors([a, b, c])
    assert len(problems) == 1 and "цепочки" in problems[0]


def test_повторяющийся_идентификатор_ловится() -> None:
    """Компенсация сослалась бы неизвестно на который из двух."""
    sensors = [temperature_sensor(id="A"), temperature_sensor(id="A", expected_nm=1560.0)]
    problems = validate_sensors(sensors)
    assert any("больше одного раза" in problem for problem in problems)


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------


def test_круговорот_json_сохраняет_датчик_целиком() -> None:
    """Всё, что влияет на расчёт, включая исходные точки, переживает JSON."""
    sensor = temperature_sensor(
        id="S1",
        value0=25.0,
        k2=1e-6,
        calibration_points=(
            CalibrationPoint(STAND_NM_1, 25.0),
            CalibrationPoint(STAND_NM_1 + 0.1, 35.0),
        ),
        up_limit=100.0,
        down_limit=-50.0,
        compensation=TempCompensation(reference="T1", coeff=-1.5, base=20.0),
    )
    assert sensor_from_json(sensor_to_json(sensor)) == sensor


def test_старая_абсолютная_форма_json_явно_отвергается() -> None:
    """c0/c1/c2 нельзя молча истолковать как value0/k1/k2."""
    with pytest.raises(ValueError, match="устаревшая абсолютная форма"):
        sensor_from_json(
            {
                "id": "S1",
                "channel": 0,
                "type": 0,
                "expected_nm": 1545.0,
                "window_nm": 0.3,
                "c0": -154475.0,
                "c1": 100.0,
                "c2": 0.0,
            }
        )


def test_необязательные_поля_в_json_не_пишутся() -> None:
    """Файл калибровок читают глазами: пустые ключи в нём только мешают."""
    data = sensor_to_json(temperature_sensor())
    assert "up_limit" not in data and "compensation" not in data


def test_имя_датчика_остаётся_русским() -> None:
    """Ограничение ASCII касается файлов данных и журнала (Р44), а не этого.

    Файл калибровок читает приложение, а не `numpy.genfromtxt` и не Excel,
    поэтому имя датчика может быть человеческим.
    """
    restored = sensor_from_json(sensor_to_json(temperature_sensor(name="Свая №3")))
    assert restored.name == "Свая №3"


def test_отсутствие_имени_подставляет_идентификатор() -> None:
    """Минимальное описание датчика не должно требовать дублирования."""
    restored = sensor_from_json(
        {"id": "S1", "channel": 0, "type": 0, "expected_nm": 1545.0, "window_nm": 0.3}
    )
    assert restored.name == "S1" and restored.k1 == 0.0


def test_неизвестный_тип_датчика_отвергается() -> None:
    """Типов ровно десять; одиннадцатый означает опечатку, а не новый датчик."""
    with pytest.raises(ValueError, match="неизвестный тип"):
        sensor_from_json(
            {"id": "S1", "channel": 0, "type": 10, "expected_nm": 1545.0, "window_nm": 0.3}
        )


def test_true_числом_не_считается() -> None:
    """В Python `True` — это `int`, и без явной проверки прошло бы как 1."""
    with pytest.raises(ValueError, match="ожидалось число"):
        sensor_from_json(
            {"id": "S1", "channel": 0, "type": 0, "expected_nm": True, "window_nm": 0.3}
        )


def test_обязательные_поля_названы() -> None:
    """Без ожидаемой длины волны и окна датчик не описан вовсе (Р30)."""
    with pytest.raises(ValueError, match="expected_nm"):
        sensor_from_json({"id": "S1", "channel": 0, "type": 0})


def test_sensors_to_json_кладёт_набор_под_ключ() -> None:
    """Обёртка объектом, а не голый массив: файлу нужно место под версию потом."""
    data = sensors_to_json([temperature_sensor()])
    assert list(data) == ["sensors"] and len(data["sensors"]) == 1  # type: ignore[arg-type]


def test_подгонка_отвергает_повтор_одной_длины_волны() -> None:
    points = (
        CalibrationPoint(1544.8, 20.0),
        CalibrationPoint(1544.8, 40.0),
    )
    with pytest.raises(ValueError, match="различных длин волн"):
        fit_calibration(points, 1544.8)
