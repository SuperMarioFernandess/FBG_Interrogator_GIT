"""Тесты кодека протокола.

Позитивные тесты команд чтения — на пяти ответах, снятых с прибора (KB_02).
Кадр телеметрии тестируется на синтетическом векторе: настоящего захвата
не существует, см. предупреждение в `tests/synthetic.py`.
"""

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.frames import GainSetting, MeasurementFrame, ParseErrorKind, SweepConfig
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from tests.synthetic import (
    MISSING_STIMULUS,
    encode_measurement,
    load_vectors,
    nm_to_raw,
    scene_two_gratings,
)

VEC = load_vectors()
SYNTH = load_vectors("measurement_synthetic.hex")


@pytest.fixture
def profile() -> DeviceProfile:
    """Профиль прибора по умолчанию: 4 канала, 30 решёток, 2000 Гц."""
    return DeviceProfile()


# ======================================================================================
# Профиль
# ======================================================================================


def test_профиль_считает_производные_величины(profile: DeviceProfile) -> None:
    """Расчётные величины совпадают с KB_01."""
    assert profile.start_ghz == 196250
    assert profile.stop_ghz == 191150
    assert profile.adc_points == 2551
    assert profile.frame_size == 494
    assert profile.channel_bytes == 122


def test_профиль_переводит_параметры_в_частоту(profile: DeviceProfile) -> None:
    """Пересчёт параметр ↔ частота обратим и совпадает с проверенными значениями."""
    assert profile.param_to_ghz(1) == 196250
    assert profile.param_to_ghz(5101) == 191150
    assert profile.ghz_to_param(191150) == 5101


def test_константа_скорости_света_согласована_с_границами_развёртки() -> None:
    """λ на границах развёртки совпадает с паспортными 1527.60 и 1568.36 нм.

    Это же проверяет, что константа задана в нм·ГГц, а не в нм·ТГц:
    в KB_01 формула записана с константой для ТГц, что противоречит
    приведённой там же таблице.
    """
    start_nm = C_NM_GHZ / 196250
    stop_nm = C_NM_GHZ / 191150
    assert start_nm == pytest.approx(1527.60, abs=0.01)
    assert stop_nm == pytest.approx(1568.36, abs=0.01)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"channels": 0}, "channels"),
        ({"fbg_per_channel": 0}, "fbg_per_channel"),
        ({"start_param": 9000}, "start_param < stop_param"),
        ({"step_param": 0}, "шаг развёртки"),
        ({"freq_divisor": 3}, "freq_divisor"),
        ({"set_sweep_frame_len": 10}, "set_sweep_frame_len"),
    ],
)
def test_профиль_отвергает_несогласованные_значения(kwargs: dict, fragment: str) -> None:
    """Некорректный профиль — программная ошибка, поэтому ValueError."""
    with pytest.raises(ValueError, match=fragment):
        DeviceProfile(**kwargs)


def test_профиль_фиксирует_делитель_копией(profile: DeviceProfile) -> None:
    """`with_freq_divisor` возвращает копию, исходный профиль не меняется."""
    fixed = profile.with_freq_divisor(10)
    assert fixed.freq_divisor == 10
    assert profile.freq_divisor is None


# ======================================================================================
# Сборка команд — сверка с байтами из KB_02
# ======================================================================================


@pytest.mark.parametrize(
    ("builder", "vector"),
    [
        (codec.build_read_version, "req_version"),
        (codec.build_read_serial, "req_serial"),
        (codec.build_read_module_params, "req_module_params"),
        (codec.build_read_sweep, "req_sweep"),
        (codec.build_read_channel_setup, "req_channel_setup"),
        (codec.build_save_thresholds, "cmd_save_thresholds"),
        (codec.build_stop, "cmd_stop"),
        (codec.build_debug_once, "cmd_debug_once"),
    ],
)
def test_команда_без_аргументов_совпадает_с_вектором(builder, vector: str) -> None:
    """Команды без аргументов собираются байт в байт как в KB_02."""
    assert builder() == VEC[vector]


def test_старт_потока_без_скорости_использует_текущую_настройку() -> None:
    """`speed_hz=None` кодируется как 00 00 — «оставить настройку прибора»."""
    assert codec.build_start_stream() == VEC["cmd_start_stream_current"]


def test_старт_потока_с_явной_скоростью() -> None:
    """2000 Гц кодируются кодом 0x00CA, подтверждённым на приборе."""
    assert codec.build_start_stream(2000) == bytes.fromhex("300206 00CA 00".replace(" ", ""))


def test_старт_потока_отвергает_скорость_вне_таблицы() -> None:
    """Правило кодирования произвольных скоростей неизвестно — не выдумываем."""
    with pytest.raises(ValueError, match="таблице подтверждённых кодов"):
        codec.build_start_stream(1500)


@pytest.mark.parametrize(
    ("channel", "threshold", "vector"),
    [
        (2, 1200, "cmd_set_threshold_ch3_1200"),
        (2, None, "cmd_set_threshold_ch3_auto"),
    ],
)
def test_команда_порога(
    channel: int, threshold: int | None, vector: str, profile: DeviceProfile
) -> None:
    """Порог: число 0…16383 либо None для автоматического расчёта."""
    assert codec.build_set_threshold(channel, threshold, profile) == VEC[vector]


@pytest.mark.parametrize(
    ("gain", "vector"),
    [
        (GainSetting(manual=True, level=2), "cmd_set_gain_ch4_manual2"),
        (GainSetting(manual=False, level=0), "cmd_set_gain_ch4_auto"),
    ],
)
def test_команда_усиления(gain: GainSetting, vector: str, profile: DeviceProfile) -> None:
    """Усиление: 80 0N — вручную, 00 0N — автоматически."""
    assert codec.build_set_gain(3, gain, profile) == VEC[vector]


@pytest.mark.parametrize(
    ("gap", "vector"), [(30, "cmd_set_peak_gap_30"), (40, "cmd_set_peak_gap_40")]
)
def test_команда_интервала_пиков(gap: int, vector: str) -> None:
    """Интервал пиков занимает один байт."""
    assert codec.build_set_peak_gap(gap) == VEC[vector]


def test_команда_чтения_сырых_ацп(profile: DeviceProfile) -> None:
    """30 07 адресует канал 0-based последним байтом."""
    assert codec.build_read_raw_adc(0, profile) == VEC["cmd_read_raw_adc_ch1"]
    assert codec.build_read_raw_adc(3, profile) == bytes.fromhex("300706000003")


# --- D3: длина кадра команды 20 01 ---


def test_развёртка_профиль_по_умолчанию_даёт_12_байт(profile: DeviceProfile) -> None:
    """D3, гипотеза 12: самосогласованный кадр, LEN равен фактической длине."""
    config = SweepConfig.from_params(1, 2, 5101, 2, profile)
    frame = codec.build_set_sweep(config, profile)
    assert len(frame) == 12
    assert frame == bytes.fromhex("20010C 0001 0002 13ED 0002 00".replace(" ", ""))
    assert frame[2] == len(frame)


def test_развёртка_гипотеза_11_байт_воспроизводит_строку_из_kb02() -> None:
    """D3, гипотеза 11: 11 байт при LEN=0x0C — ровно как в PDF производителя."""
    profile = DeviceProfile(set_sweep_frame_len=11, set_sweep_len_field=12)
    config = SweepConfig.from_params(1, 2, 5101, 2, profile)
    frame = codec.build_set_sweep(config, profile)
    assert len(frame) == 11
    assert frame == VEC["cmd_set_sweep_pdf"]


def test_развёртка_11_байт_без_подмены_len_самосогласована() -> None:
    """Если LEN не подменять, при 11 байтах он равен 11."""
    profile = DeviceProfile(set_sweep_frame_len=11)
    frame = codec.build_set_sweep(SweepConfig.from_params(1, 2, 5101, 2, profile), profile)
    assert len(frame) == 11
    assert frame[2] == 11


# --- Негативные случаи сборки ---


@pytest.mark.parametrize("channel", [-1, 4, 99])
def test_сборка_отвергает_канал_за_границей(channel: int, profile: DeviceProfile) -> None:
    """Номер канала обязан быть ограничен вызывающим: выход за границы — баг."""
    with pytest.raises(ValueError, match="номер канала"):
        codec.build_set_threshold(channel, 1000, profile)
    with pytest.raises(ValueError, match="номер канала"):
        codec.build_set_gain(channel, GainSetting(manual=False, level=0), profile)
    with pytest.raises(ValueError, match="номер канала"):
        codec.build_read_raw_adc(channel, profile)


def test_сборка_отвергает_порог_вне_разрядности_ацп(profile: DeviceProfile) -> None:
    """АЦП 14-битный: 16384 уже вне диапазона (негативный сценарий G8)."""
    with pytest.raises(ValueError, match="вне диапазона"):
        codec.build_set_threshold(0, 16384, profile)


def test_сборка_отвергает_уровень_усиления_вне_диапазона(profile: DeviceProfile) -> None:
    """Уровни усиления 0…5."""
    with pytest.raises(ValueError, match="уровень усиления"):
        codec.build_set_gain(0, GainSetting(manual=True, level=6), profile)


def test_сборка_развёртки_проверяет_инвариант(profile: DeviceProfile) -> None:
    """Start обязан быть меньше Stop в параметрах, иначе развёртка вывернута."""
    with pytest.raises(ValueError, match="start_param < stop_param"):
        codec.build_set_sweep(SweepConfig.from_params(5101, 2, 1, 2, profile), profile)


def test_сборка_отвергает_интервал_пиков_вне_байта() -> None:
    """Поле интервала пиков однобайтовое."""
    with pytest.raises(ValueError, match="не помещается в один байт"):
        codec.build_set_peak_gap(300)


# ======================================================================================
# Кодирование скорости развёртки
# ======================================================================================


@pytest.mark.parametrize(
    ("hz", "code"),
    [
        (1, 0x000A),
        (3, 0x001E),
        (100, 0x0065),
        (200, 0x00C9),
        (500, 0x01F5),
        (1000, 0x0066),
        (2000, 0x00CA),
        (4000, 0x0192),
    ],
)
def test_скорость_кодируется_и_декодируется(hz: int, code: int) -> None:
    """Все восемь подтверждённых кодов из KB_02, в обе стороны."""
    assert codec.encode_sweep_speed(hz) == code
    assert codec.decode_sweep_speed(code) == hz


def test_декодер_скорости_использует_формулу_для_кода_вне_таблицы() -> None:
    """Формула `F = (code // 10) · 10^(code % 10)` — запасной путь декодера."""
    assert codec.decode_sweep_speed(0x0032) == 5  # 50 → M=5, E=0
    assert codec.decode_sweep_speed(302) == 3000  # M=30, E=2


def test_декодер_скорости_возвращает_none_на_бессмысленном_коде() -> None:
    """Нулевой код и код с нулевой мантиссой скорости не задают."""
    assert codec.decode_sweep_speed(0) is None
    assert codec.decode_sweep_speed(5) is None


def test_версия_форматируется() -> None:
    """410 → '4.10', ведущий ноль в младшей части сохраняется."""
    assert codec.format_version(410) == "4.10"
    assert codec.format_version(101) == "1.01"


# ======================================================================================
# Разбор ответов — пять реальных векторов с прибора
# ======================================================================================


def test_разбор_версии_прошивки() -> None:
    """10 01: 0x0000019A = 410 = v4.10."""
    raw = codec.parse_version(VEC["resp_version"]).unwrap()
    assert raw == 410
    assert codec.format_version(raw) == "4.10"


def test_разбор_серийного_номера() -> None:
    """10 03: 0x05A072C4 = 94401220 — совпадает с шильдиком."""
    assert codec.parse_serial(VEC["resp_serial"]).unwrap() == 94401220


def test_разбор_параметров_модуля() -> None:
    """10 04: 2000 Гц, 4 канала, 30 решёток, интервал пиков 30 ГГц."""
    params = codec.parse_module_params(VEC["resp_module_params"]).unwrap()
    assert params.speed_code == 0x00CA
    assert params.speed_hz == 2000
    assert params.channels == 4
    assert params.fbg_per_channel == 30
    assert params.peak_gap_ghz == 30


def test_разбор_параметров_развёртки(profile: DeviceProfile) -> None:
    """10 05: параметры 1/2/5101/2 дают 196250…191150 ГГц и 2551 точку АЦП."""
    sweep = codec.parse_sweep_params(VEC["resp_sweep"], profile).unwrap()
    assert (sweep.start_param, sweep.step_param) == (1, 2)
    assert (sweep.stop_param, sweep.adc_step_param) == (5101, 2)
    assert sweep.start_ghz == 196250
    assert sweep.stop_ghz == 191150
    assert sweep.adc_points == 2551


def test_разбор_порогов_и_усилений(profile: DeviceProfile) -> None:
    """10 06: четыре канала, порог авто, усиление авто уровня 5."""
    setups = codec.parse_channel_setup(VEC["resp_channel_setup"], profile).unwrap()
    assert len(setups) == 4
    for index, setup in enumerate(setups):
        assert setup.channel == index
        assert setup.threshold is None
        assert setup.threshold_auto
        assert setup.gain == GainSetting(manual=False, level=5)


def test_разбор_подтверждения_записи() -> None:
    """20 FC 00 06 00 01 — успех, 00 00 — отказ."""
    assert codec.parse_write_ack(VEC["ack_write_ok"]).unwrap() is True
    assert codec.parse_write_ack(VEC["ack_write_fail"]).unwrap() is False


def test_разбор_подтверждения_остановки(profile: DeviceProfile) -> None:
    """30 01: поле LEN шириной 4 байта, значение 8 = полная длина кадра."""
    assert codec.parse_stop_ack(VEC["resp_stop_ack"], profile).unwrap() is True


# ======================================================================================
# Негативные случаи разбора
# ======================================================================================


def test_короткий_кадр(profile: DeviceProfile) -> None:
    """Кадр короче заголовка отвергается как TOO_SHORT."""
    result = codec.parse_version(b"\x10\x01")
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.TOO_SHORT

    result_mode = codec.parse_stop_ack(b"\x30\x01\x00", profile)
    assert result_mode.error is not None
    assert result_mode.error.kind is ParseErrorKind.TOO_SHORT


def test_len_не_совпадает_с_длиной() -> None:
    """LEN обязан быть равен полной длине кадра."""
    broken = bytes.fromhex("100100FF000001 9A".replace(" ", ""))
    result = codec.parse_version(broken)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_ответ_другой_команды() -> None:
    """Парсер версии не примет ответ серийного номера."""
    result = codec.parse_version(VEC["resp_serial"])
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.WRONG_COMMAND


def test_неизвестная_пара_id_fc(profile: DeviceProfile) -> None:
    """Пары (10, 02) в списке известных команд нет."""
    result = codec.parse_any(bytes.fromhex("10020400"), profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.UNKNOWN_COMMAND


def test_мусор_вместо_кадра(profile: DeviceProfile) -> None:
    """Сценарий G5: прибору отправили мусор, в ответ пришёл мусор."""
    result = codec.parse_any(bytes.fromhex("AABBCCDD"), profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.UNKNOWN_COMMAND
    assert codec.parse_any(b"\xaa", profile).error.kind is ParseErrorKind.TOO_SHORT


def test_порог_вне_разрядности_в_ответе(profile: DeviceProfile) -> None:
    """Порог 0x4000 = 16384 не помещается в 14 бит АЦП."""
    broken = bytes.fromhex("10060008" + "4000" + "0005")
    result = codec.parse_channel_setup(broken, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.BAD_VALUE


def test_недопустимая_кодировка_усиления(profile: DeviceProfile) -> None:
    """Старший байт усиления бывает только 00 или 80."""
    broken = bytes.fromhex("10060008" + "FFFF" + "4005")
    result = codec.parse_channel_setup(broken, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.BAD_VALUE


def test_тело_порогов_не_кратно_каналу(profile: DeviceProfile) -> None:
    """4 байта на канал: тело в 6 байт нацело не делится."""
    broken = bytes.fromhex("1006000A" + "FFFF0005" + "FFFF")
    result = codec.parse_channel_setup(broken, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_вывернутая_развёртка_в_ответе(profile: DeviceProfile) -> None:
    """Start ≥ Stop в параметрах нарушает инвариант развёртки."""
    broken = bytes.fromhex("1005000C" + "13ED" + "0002" + "0001" + "0002")
    result = codec.parse_sweep_params(broken, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.BAD_VALUE


def test_неизвестный_код_результата_записи() -> None:
    """Код результата бывает только 0 или 1."""
    broken = bytes.fromhex("20020006" + "0007")
    result = codec.parse_write_ack(broken)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.BAD_VALUE


# ======================================================================================
# Автодетект единиц частоты (D1)
# ======================================================================================


@pytest.mark.parametrize("divisor", [1, 10])
def test_автодетект_различает_обе_гипотезы(divisor: int, profile: DeviceProfile) -> None:
    """Диапазоны гипотез не пересекаются, поэтому детектор однозначен."""
    freq_raw, _ = scene_two_gratings(profile, divisor)
    assert codec.detect_freq_divisor(freq_raw, profile) == divisor


def test_автодетект_возвращает_none_когда_нечего_детектировать(profile: DeviceProfile) -> None:
    """Кадр без единого валидного пика единицы определить не позволяет."""
    freq_raw = np.zeros((profile.channels, profile.fbg_per_channel), dtype=np.uint32)
    assert codec.detect_freq_divisor(freq_raw, profile) is None


def test_диапазоны_гипотез_не_пересекаются(profile: DeviceProfile) -> None:
    """Верх гипотезы A ниже низа гипотезы B — иначе автодетект был бы неоднозначен."""
    _, high_a = profile.freq_raw_bounds(1)
    low_b, _ = profile.freq_raw_bounds(10)
    assert high_a < low_b


# ======================================================================================
# Кадр телеметрии — синтетический вектор по гипотезе N4
# ======================================================================================


def test_синтетический_вектор_не_разошёлся_с_генератором(profile: DeviceProfile) -> None:
    """Сохранённый вектор совпадает с тем, что порождает `tests/synthetic.py`."""
    generated = encode_measurement(profile, *scene_two_gratings(profile, 10))
    assert generated == SYNTH["measurement_divisor10"]


@pytest.mark.parametrize("divisor", [1, 10])
def test_разбор_телеметрии_на_обеих_гипотезах(divisor: int, profile: DeviceProfile) -> None:
    """Кадр разбирается одинаково по структуре при любой гипотезе единиц."""
    freq_raw, temp_raw = scene_two_gratings(profile, divisor)
    frame = encode_measurement(profile, freq_raw, temp_raw)

    result = codec.parse_measurement(frame, profile, t_mono=12.5).unwrap()

    assert result.freq_divisor == divisor
    assert result.t_mono == 12.5
    assert result.index_mismatches == 0
    assert result.freq_ghz.shape == (4, 30)
    # Валидны только четыре позиции: три в канале 1 и одна в канале 2.
    assert result.missing == 116
    assert np.count_nonzero(np.isfinite(result.freq_ghz)) == 4


@pytest.mark.parametrize(("divisor", "tolerance_nm"), [(1, 0.01), (10, 0.001)])
def test_длины_волн_совпадают_со_сценой(
    divisor: int, tolerance_nm: float, profile: DeviceProfile
) -> None:
    """Восстановленные λ совпадают с заложенными в сцену, с точностью квантования.

    Допуск разный не случайно: при гипотезе A шаг поля 1 ГГц даёт ~8 пм,
    при гипотезе B — 0.8 пм. Это и есть довод в пользу B из KB_01.
    """
    freq_raw, temp_raw = scene_two_gratings(profile, divisor)
    frame = encode_measurement(profile, freq_raw, temp_raw)
    waves = codec.parse_measurement(frame, profile).unwrap().wavelength_nm()

    assert waves[0, 0] == pytest.approx(1545.0, abs=tolerance_nm)
    assert waves[0, 1] == pytest.approx(1550.0, abs=tolerance_nm)
    assert waves[0, 2] == pytest.approx(1555.0, abs=tolerance_nm)
    assert waves[1, 0] == pytest.approx(1560.0, abs=tolerance_nm)


def test_пик_не_найден_даёт_nan(profile: DeviceProfile) -> None:
    """Значение вне stop_ghz…start_ghz — NaN, а не последнее известное."""
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    frame = encode_measurement(profile, freq_raw, temp_raw)
    result = codec.parse_measurement(frame, profile).unwrap()

    assert np.isnan(result.freq_ghz[0, 3])  # нули на пустой позиции
    assert np.isnan(result.freq_ghz[2, 0])  # 300000 — вне обеих гипотез
    assert np.isnan(result.freq_ghz[3, 0])  # FF FF FF
    assert np.isnan(result.wavelength_nm()[0, 3])


def test_граничные_значения_диапазона_валидны(profile: DeviceProfile) -> None:
    """Границы stop_ghz и start_ghz включаются, соседние значения — нет."""
    channels, fbg = profile.channels, profile.fbg_per_channel
    freq_raw = np.full((channels, fbg), MISSING_STIMULUS, dtype=np.uint32)
    low, high = profile.freq_raw_bounds(10)
    freq_raw[0, 0] = low
    freq_raw[0, 1] = high
    freq_raw[0, 2] = low - 1
    freq_raw[0, 3] = high + 1
    temp_raw = np.zeros(channels, dtype=np.int32)

    frame = encode_measurement(profile, freq_raw, temp_raw)
    freq = codec.parse_measurement(frame, profile.with_freq_divisor(10)).unwrap().freq_ghz

    assert freq[0, 0] == pytest.approx(profile.stop_ghz)
    assert freq[0, 1] == pytest.approx(profile.start_ghz)
    assert np.isnan(freq[0, 2])
    assert np.isnan(freq[0, 3])


def test_нарушенный_порядок_индексов_даёт_nan(profile: DeviceProfile) -> None:
    """Байт индекса не совпал с ожидаемым — доверять позиции нельзя."""
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    indices = np.tile(np.arange(profile.fbg_per_channel, dtype=np.uint8), (profile.channels, 1))
    indices[0, 1] = 0x1F

    frame = encode_measurement(profile, freq_raw, temp_raw, indices=indices)
    result = codec.parse_measurement(frame, profile).unwrap()

    assert result.index_mismatches == 1
    assert np.isnan(result.freq_ghz[0, 1])
    assert np.isfinite(result.freq_ghz[0, 0])


def test_явный_маркер_отсутствующего_пика(profile: DeviceProfile) -> None:
    """Если скрининг найдёт маркер N3, он задаётся полем профиля.

    Проверяем механизм: валидное по диапазону значение, объявленное маркером,
    отбраковывается. Само значение выбрано произвольно — это тест механизма,
    а не утверждение о приборе.
    """
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    marker = int(freq_raw[0, 1])
    frame = encode_measurement(profile, freq_raw, temp_raw)

    tuned = DeviceProfile(freq_divisor=10, peak_missing_codes=frozenset({marker}))
    result = codec.parse_measurement(frame, tuned).unwrap()

    assert np.isnan(result.freq_ghz[0, 1])
    assert np.isfinite(result.freq_ghz[0, 0])


def test_температура_корпуса(profile: DeviceProfile) -> None:
    """Масштаб 0.01 °C — гипотеза N2, задаётся полем профиля."""
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    frame = encode_measurement(profile, freq_raw, temp_raw)
    result = codec.parse_measurement(frame, profile).unwrap()
    assert result.case_temp_c == pytest.approx(np.full(4, 25.0))


def test_температура_корпуса_отрицательная() -> None:
    """Прибор работает от −15 °C, поэтому поле по умолчанию знаковое."""
    profile = DeviceProfile(freq_divisor=10)
    freq_raw, _ = scene_two_gratings(profile, 10)
    temp_raw = np.full(profile.channels, -1250, dtype=np.int32)
    frame = encode_measurement(profile, freq_raw, temp_raw)
    result = codec.parse_measurement(frame, profile).unwrap()
    assert result.case_temp_c[0] == pytest.approx(-12.5)


def test_другой_масштаб_температуры_меняет_только_профиль() -> None:
    """Смена гипотезы N2 не требует правки кода."""
    profile = DeviceProfile(freq_divisor=10, case_temp_scale=0.1)
    freq_raw, _ = scene_two_gratings(DeviceProfile(), 10)
    frame = encode_measurement(profile, freq_raw, np.full(4, 250, dtype=np.int32))
    result = codec.parse_measurement(frame, profile).unwrap()
    assert result.case_temp_c[0] == pytest.approx(25.0)


def test_кадр_без_валидных_пиков_не_даёт_угадывать_единицы(profile: DeviceProfile) -> None:
    """Единицы неизвестны и определить их нечем — это ошибка, а не догадка."""
    channels, fbg = profile.channels, profile.fbg_per_channel
    freq_raw = np.zeros((channels, fbg), dtype=np.uint32)
    frame = encode_measurement(profile, freq_raw, np.zeros(channels, dtype=np.int32))

    result = codec.parse_measurement(frame, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.AMBIGUOUS_UNITS


def test_кадр_без_пиков_разбирается_при_известном_делителе(profile: DeviceProfile) -> None:
    """С зафиксированным делителем пустой кадр — это просто 120 значений NaN."""
    channels, fbg = profile.channels, profile.fbg_per_channel
    freq_raw = np.zeros((channels, fbg), dtype=np.uint32)
    frame = encode_measurement(profile, freq_raw, np.zeros(channels, dtype=np.int32))

    result = codec.parse_measurement(frame, profile.with_freq_divisor(10)).unwrap()
    assert result.missing == channels * fbg
    assert np.isnan(result.freq_ghz).all()


def test_короткий_кадр_телеметрии(profile: DeviceProfile) -> None:
    """Обрезанная датаграмма не совпадает с LEN и отбраковывается."""
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    frame = encode_measurement(profile, freq_raw, temp_raw)

    result = codec.parse_measurement(frame[:200], profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_неверный_len_в_кадре_телеметрии(profile: DeviceProfile) -> None:
    """Длина верная, но LEN другой — кадр не наш."""
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    frame = encode_measurement(profile, freq_raw, temp_raw, len_field=490)

    result = codec.parse_measurement(frame, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_буфер_кадра_переиспользуется(profile: DeviceProfile) -> None:
    """При 2000 кадрах/с буфер принадлежит вызывающему, кодек остаётся чистым."""
    buffer = MeasurementFrame.for_profile(profile)
    freq_raw, temp_raw = scene_two_gratings(profile, 10)

    first = codec.parse_measurement(
        encode_measurement(profile, freq_raw, temp_raw), profile, t_mono=1.0, out=buffer
    ).unwrap()
    assert first is buffer

    freq_raw[0, 5] = nm_to_raw(1540.0, 10)
    second = codec.parse_measurement(
        encode_measurement(profile, freq_raw, temp_raw), profile, t_mono=2.0, out=buffer
    ).unwrap()

    assert second is buffer
    assert buffer.t_mono == 2.0
    assert buffer.wavelength_nm()[0, 5] == pytest.approx(1540.0, abs=0.001)


def test_буфер_чужой_формы_отвергается(profile: DeviceProfile) -> None:
    """Буфер не под тот профиль — программная ошибка вызывающего."""
    buffer = MeasurementFrame(2, 10)
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    frame = encode_measurement(profile, freq_raw, temp_raw)

    with pytest.raises(ValueError, match="буфер out"):
        codec.parse_measurement(frame, profile, out=buffer)


def test_профиль_другой_конфигурации_меняет_размер_кадра() -> None:
    """Число каналов и решёток — данные профиля, а не константы кода."""
    profile = DeviceProfile(channels=2, fbg_per_channel=8, freq_divisor=10)
    assert profile.frame_size == 6 + 2 * (8 * 4 + 2)

    freq_raw = np.full((2, 8), MISSING_STIMULUS, dtype=np.uint32)
    freq_raw[0, 0] = nm_to_raw(1550.0, 10)
    frame = encode_measurement(profile, freq_raw, np.zeros(2, dtype=np.int32))

    result = codec.parse_measurement(frame, profile).unwrap()
    assert result.freq_ghz.shape == (2, 8)
    assert result.wavelength_nm()[0, 0] == pytest.approx(1550.0, abs=0.001)


# ======================================================================================
# Сырые отсчёты АЦП и отладочный режим
# ======================================================================================


def _adc_frame(
    profile: DeviceProfile, channel: int, points: int, gain: bytes = b"\x00\x05"
) -> bytes:
    """Собирает синтетический ответ 30 07: LEN(4) Канал(2) Усиление(2) ADC(2)×N."""
    body = channel.to_bytes(2, "big") + gain
    body += b"".join((value % 16384).to_bytes(2, "big") for value in range(points))
    total = 2 + profile.mode_len_width + len(body)
    return bytes([0x30, 0x07]) + total.to_bytes(profile.mode_len_width, "big") + body


def test_разбор_сырых_ацп(profile: DeviceProfile) -> None:
    """Число отсчётов берётся из фактической длины кадра."""
    frame = _adc_frame(profile, channel=1, points=profile.adc_points)
    block = codec.parse_raw_adc(frame, profile).unwrap()

    assert block.channel == 1
    assert block.gain == GainSetting(manual=False, level=5)
    assert block.points == 2551
    assert block.adc.dtype == np.uint16
    assert block.adc[0] == 0
    assert block.adc[10] == 10


def test_разбор_сырых_ацп_отвергает_канал_за_границей(profile: DeviceProfile) -> None:
    """Сценарий G7: ответ про канал 6 на четырёхканальном приборе."""
    frame = _adc_frame(profile, channel=5, points=4)
    result = codec.parse_raw_adc(frame, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.BAD_VALUE


def test_разбор_сырых_ацп_отвергает_нечётное_тело(profile: DeviceProfile) -> None:
    """Отсчёт занимает 2 байта, нечётный хвост означает потерю."""
    frame = _adc_frame(profile, channel=0, points=4) + b"\x00"
    frame = frame[:2] + len(frame).to_bytes(profile.mode_len_width, "big") + frame[6:]
    result = codec.parse_raw_adc(frame, profile)
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_отладочный_ответ_отдаёт_тело_сырым(profile: DeviceProfile) -> None:
    """Раскладка тела 30 03 неизвестна (вопрос N14) — проверяется только заголовок."""
    payload = bytes(range(20))
    total = 2 + profile.mode_len_width + len(payload)
    frame = bytes([0x30, 0x03]) + total.to_bytes(profile.mode_len_width, "big") + payload

    result = codec.parse_debug_once(frame, profile).unwrap()
    assert result.payload == payload


# ======================================================================================
# Диспетчер
# ======================================================================================


def test_классификация_кадра() -> None:
    """Пара (ID, FC) читается из первых двух байт."""
    assert codec.classify(VEC["resp_version"]) == (0x10, 0x01)
    assert codec.classify(b"\x30") is None


@pytest.mark.parametrize(
    ("vector", "expected"),
    [
        ("resp_version", 410),
        ("resp_serial", 94401220),
        ("ack_write_ok", True),
        ("resp_stop_ack", True),
    ],
)
def test_диспетчер_выбирает_нужный_парсер(
    vector: str, expected: object, profile: DeviceProfile
) -> None:
    """`parse_any` нужен приёмному тракту: до разбора неизвестно, что пришло."""
    assert codec.parse_any(VEC[vector], profile).unwrap() == expected


def test_диспетчер_разбирает_телеметрию(profile: DeviceProfile) -> None:
    """Кадр 30 02 приходит по тому же сокету, что и ответы на команды."""
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    frame = encode_measurement(profile, freq_raw, temp_raw)

    result = codec.parse_any(frame, profile, t_mono=3.0).unwrap()
    assert isinstance(result, MeasurementFrame)
    assert result.t_mono == 3.0


def test_все_известные_команды_перечислены() -> None:
    """В KB_02 их четырнадцать: 5 чтения, 5 записи, 4 режима."""
    assert len(codec.KNOWN_COMMANDS) == 14
    assert (0x20, 0x06) in codec.NO_RESPONSE_COMMANDS
