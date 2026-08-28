"""Кодек протокола: сборка команд и разбор ответов.

Только чистые функции. Ни сети, ни файлов, ни внутреннего состояния, ни времени:
метка времени кадра телеметрии приходит параметром от того, кто принял датаграмму.

Разделение ответственности за ошибки (KB_05):
  * `build_*` бросают ValueError — некорректный аргумент здесь означает баг
    вызывающего кода: номер канала обязан быть ограничен на уровне UI;
  * `parse_*` возвращают `ParseResult` — испорченный кадр из сети штатная
    ситуация, а не баг.

Байты читаются только через `int.from_bytes` и `struct` (KB_05 №1).
"""

import struct

import numpy as np

from fbg.core.frames import (
    AdcBlock,
    ChannelSetup,
    DebugResponse,
    GainSetting,
    MeasurementFrame,
    ModuleParams,
    ParseErrorKind,
    ParseResult,
    SweepConfig,
    UndocumentedResponse,
    fail,
    ok,
)
from fbg.core.profile import FREQ_DIVISOR_CANDIDATES, DeviceProfile

# --------------------------------------------------------------------------------------
# Байтовые константы протокола (KB_02)
# --------------------------------------------------------------------------------------

ID_READ = 0x10
ID_WRITE = 0x20
ID_MODE = 0x30

FC_VERSION = 0x01
#: Недокументированная команда, ✅ обнаружена скринингом (KB_04, N17).
#: Смысл полей ответа неизвестен; штатное ПО использует её как основной опрос
#: конфигурации и как keepalive вместо 10 05 и 10 06.
FC_UNDOCUMENTED = 0x02
FC_SERIAL = 0x03
FC_MODULE_PARAMS = 0x04
FC_SWEEP = 0x05
FC_CHANNEL_SETUP = 0x06

FC_SET_SWEEP = 0x01
FC_SET_THRESHOLD = 0x02
FC_SET_GAIN = 0x03
FC_SET_PEAK_GAP = 0x04
FC_SAVE_THRESHOLDS = 0x06

FC_STOP = 0x01
FC_STREAM = 0x02
FC_DEBUG = 0x03
FC_RAW_ADC = 0x07

#: Ширина поля LEN в ответах 0x10 и 0x20 — ✅ проверена на пяти ответах прибора.
#: Для ответов 0x30 ширина спорная и живёт в профиле (`mode_len_width`).
RESP_LEN_WIDTH = 2

#: Байт ручного режима усиления в старшем байте поля.
GAIN_MANUAL_FLAG = 0x80

#: Длина ответа-подтверждения на любую команду записи: 20 FC 00 06 SS SS.
WRITE_ACK_LEN = 6

#: Полный список известных пар (ID, FC) — KB_02, «Полный список известных (ID, FC)».
#: Пятнадцатая пара — (0x10, 0x02): команда в PDF отсутствует, но реально
#: существует и отвечает (KB_04, N17).
KNOWN_COMMANDS: frozenset[tuple[int, int]] = frozenset(
    {
        (ID_READ, FC_VERSION),
        (ID_READ, FC_UNDOCUMENTED),
        (ID_READ, FC_SERIAL),
        (ID_READ, FC_MODULE_PARAMS),
        (ID_READ, FC_SWEEP),
        (ID_READ, FC_CHANNEL_SETUP),
        (ID_WRITE, FC_SET_SWEEP),
        (ID_WRITE, FC_SET_THRESHOLD),
        (ID_WRITE, FC_SET_GAIN),
        (ID_WRITE, FC_SET_PEAK_GAP),
        (ID_WRITE, FC_SAVE_THRESHOLDS),
        (ID_MODE, FC_STOP),
        (ID_MODE, FC_STREAM),
        (ID_MODE, FC_DEBUG),
        (ID_MODE, FC_RAW_ADC),
    }
)

#: Команды, на которые прибор не отвечает (KB_04, D4 — гипотеза).
NO_RESPONSE_COMMANDS: frozenset[tuple[int, int]] = frozenset({(ID_WRITE, FC_SAVE_THRESHOLDS)})

#: Коды скорости развёртки, ✅ подтверждено на приборе для 2000 Гц (код 0x00CA).
#: Схема `code = M·10 + E`, где `F = M · 10^E` Гц.
SWEEP_SPEED_CODES: dict[int, int] = {
    1: 0x000A,
    3: 0x001E,
    100: 0x0065,
    200: 0x00C9,
    500: 0x01F5,
    1000: 0x0066,
    2000: 0x00CA,
    4000: 0x0192,
}

#: Код «использовать текущую настройку прибора» в команде 30 02.
SPEED_KEEP_CURRENT = 0x0000


# --------------------------------------------------------------------------------------
# Скорость развёртки
# --------------------------------------------------------------------------------------


def decode_sweep_speed(code: int) -> int | None:
    """Расшифровывает код скорости развёртки в герцы.

    Основной путь — таблица подтверждённых кодов, запасной — формула
    `F = (code // 10) · 10^(code % 10)` (KB_02). Возвращает None, если код
    нулевой или формула даёт бессмысленный результат.
    """
    for hz, known in SWEEP_SPEED_CODES.items():
        if known == code:
            return hz
    if code <= 0:
        return None
    mantissa, exponent = divmod(code, 10)
    if mantissa == 0:
        return None
    return mantissa * 10**exponent


def encode_sweep_speed(speed_hz: int) -> int:
    """Кодирует скорость развёртки в код команды 30 02.

    Только по таблице. Формула `code = M·10 + E` неоднозначна в обратную
    сторону: 2000 Гц представимы и как 202 (M=20, E=2), и как 23 (M=2, E=3).
    Прибор подтверждён на 202, правило выбора мантиссы неизвестно, поэтому
    для скорости вне таблицы поведение не выдумывается.
    """
    try:
        return SWEEP_SPEED_CODES[speed_hz]
    except KeyError:
        raise ValueError(
            f"скорость {speed_hz} Гц отсутствует в таблице подтверждённых кодов "
            f"{sorted(SWEEP_SPEED_CODES)}; правило кодирования произвольных "
            "скоростей неизвестно (KB_04, D2 закрыт только для этих восьми)"
        ) from None


def format_version(raw: int) -> str:
    """Форматирует сырое значение версии прошивки: 410 → '4.10'."""
    major, minor = divmod(raw, 100)
    return f"{major}.{minor:02d}"


# --------------------------------------------------------------------------------------
# Сборка команд чтения (ID = 0x10)
# --------------------------------------------------------------------------------------


def _read_request(fc: int) -> bytes:
    """Собирает запрос чтения: ID, FC, LEN=4 и байт заполнения."""
    return bytes([ID_READ, fc, 0x04, 0x00])


def build_read_version() -> bytes:
    """10 01 04 00 — версия прошивки."""
    return _read_request(FC_VERSION)


def build_read_serial() -> bytes:
    """10 03 04 00 — серийный номер."""
    return _read_request(FC_SERIAL)


def build_read_module_params() -> bytes:
    """10 04 04 00 — скорость, каналы, решётки, интервал пиков."""
    return _read_request(FC_MODULE_PARAMS)


def build_read_sweep() -> bytes:
    """10 05 04 00 — параметры развёртки."""
    return _read_request(FC_SWEEP)


def build_read_channel_setup() -> bytes:
    """10 06 04 00 — пороги и усиления всех каналов."""
    return _read_request(FC_CHANNEL_SETUP)


def build_read_undocumented() -> bytes:
    """10 02 04 00 — недокументированная команда (KB_04, N17).

    В PDF производителя её нет; обнаружена скринингом, где штатное ПО
    отправляет её при старте третьей и повторяет в простое как keepalive.
    Смысл ответа неизвестен, поэтому в наш пробник и watchdog команда
    **не включена** — сборка и разбор существуют, чтобы можно было
    воспроизвести обмен и опознать ответ в журнале, а не чтобы им пользоваться.
    """
    return _read_request(FC_UNDOCUMENTED)


# --------------------------------------------------------------------------------------
# Сборка команд записи (ID = 0x20)
# --------------------------------------------------------------------------------------


def _check_channel(channel: int, profile: DeviceProfile) -> None:
    """Проверяет 0-based номер канала. Выход за границы — баг вызывающего."""
    if not 0 <= channel < profile.channels:
        raise ValueError(
            f"номер канала {channel} вне диапазона 0…{profile.channels - 1} "
            "(нумерация 0-based: канал 1 прибора — это 0)"
        )


def build_set_sweep(config: SweepConfig, profile: DeviceProfile) -> bytes:
    """20 01 — задать развёртку.

    Длина кадра и значение поля LEN берутся из профиля: вопрос D3 не закрыт.
    Профиль по умолчанию (12, None) даёт самосогласованный кадр из 12 байт
    с LEN=0x0C. Профиль (11, 12) воспроизводит строку из KB_02, где LEN=0x0C
    стоит при 11 байтах кадра — так этот кадр выглядит в PDF производителя.
    """
    for name, value in (
        ("start_param", config.start_param),
        ("step_param", config.step_param),
        ("stop_param", config.stop_param),
        ("adc_step_param", config.adc_step_param),
    ):
        if not 0 <= value <= 0xFFFF:
            raise ValueError(f"{name}={value} не помещается в 16 бит")
    if config.start_param >= config.stop_param:
        raise ValueError(
            "нарушен инвариант развёртки start_param < stop_param: "
            f"{config.start_param} ≥ {config.stop_param}"
        )
    if config.step_param < 1 or config.adc_step_param < 1:
        raise ValueError("шаг развёртки должен быть ≥ 1")

    frame_len = profile.set_sweep_frame_len
    len_field = (
        profile.set_sweep_len_field if profile.set_sweep_len_field is not None else frame_len
    )
    head = bytes([ID_WRITE, FC_SET_SWEEP, len_field])
    body = struct.pack(
        ">4H",
        config.start_param,
        config.step_param,
        config.stop_param,
        config.adc_step_param,
    )
    padding = bytes(frame_len - len(head) - len(body))
    return head + body + padding


def build_set_threshold(channel: int, threshold: int | None, profile: DeviceProfile) -> bytes:
    """20 02 — задать порог канала. `threshold=None` — автоматический расчёт."""
    _check_channel(channel, profile)
    if threshold is None:
        value = profile.threshold_auto
    elif not 0 <= threshold <= profile.adc_max:
        raise ValueError(
            f"порог {threshold} вне диапазона 0…{profile.adc_max}; "
            "для автоматического расчёта передайте None"
        )
    else:
        value = threshold
    return bytes([ID_WRITE, FC_SET_THRESHOLD, 0x06, channel]) + struct.pack(">H", value)


def build_set_gain(channel: int, gain: GainSetting, profile: DeviceProfile) -> bytes:
    """20 03 — задать усиление канала."""
    _check_channel(channel, profile)
    if not 0 <= gain.level <= profile.gain_max_level:
        raise ValueError(f"уровень усиления {gain.level} вне диапазона 0…{profile.gain_max_level}")
    return bytes([ID_WRITE, FC_SET_GAIN, 0x06, channel]) + gain.to_bytes()


def build_set_peak_gap(gap_ghz: int) -> bytes:
    """20 04 — задать минимальный интервал между пиками, ГГц."""
    if not 1 <= gap_ghz <= 0xFF:
        raise ValueError(f"интервал пиков {gap_ghz} ГГц не помещается в один байт")
    return bytes([ID_WRITE, FC_SET_PEAK_GAP, 0x04, gap_ghz])


def build_save_thresholds() -> bytes:
    """20 06 04 00 — сохранить пороги в энергонезависимой памяти.

    По гипотезе D4 прибор на эту команду не отвечает; корреляцией ответа
    занимается сессия, см. NO_RESPONSE_COMMANDS.
    """
    return bytes([ID_WRITE, FC_SAVE_THRESHOLDS, 0x04, 0x00])


# --------------------------------------------------------------------------------------
# Сборка команд режимов (ID = 0x30)
# --------------------------------------------------------------------------------------


def build_stop() -> bytes:
    """30 01 06 00 00 00 — остановить поток."""
    return bytes([ID_MODE, FC_STOP, 0x06, 0x00, 0x00, 0x00])


def build_start_stream(speed_hz: int | None = None) -> bytes:
    """30 02 — запустить поток телеметрии.

    `speed_hz=None` — код 0x0000, то есть «использовать текущую настройку прибора».
    """
    code = SPEED_KEEP_CURRENT if speed_hz is None else encode_sweep_speed(speed_hz)
    return bytes([ID_MODE, FC_STREAM, 0x06]) + struct.pack(">H", code) + bytes(1)


def build_debug_once() -> bytes:
    """30 03 06 00 00 00 — одиночная развёртка в отладочном режиме."""
    return bytes([ID_MODE, FC_DEBUG, 0x06, 0x00, 0x00, 0x00])


def build_read_raw_adc(channel: int, profile: DeviceProfile) -> bytes:
    """30 07 — запросить сырые отсчёты АЦП одного канала."""
    _check_channel(channel, profile)
    return bytes([ID_MODE, FC_RAW_ADC, 0x06, 0x00, 0x00, channel])


# --------------------------------------------------------------------------------------
# Разбор: общая часть
# --------------------------------------------------------------------------------------


def classify(frame: bytes) -> tuple[int, int] | None:
    """Возвращает пару (ID, FC) кадра или None, если кадр слишком короткий."""
    if len(frame) < 2:
        return None
    return frame[0], frame[1]


def _check_frame[T](
    frame: bytes,
    ident: int,
    fc: int,
    len_width: int,
    expected_len: int | None = None,
) -> ParseResult[T] | None:
    """Проверяет заголовок и поле LEN. Возвращает ошибку либо None, если всё цело.

    LEN во всех проверенных ответах равен полной длине кадра, включая заголовок.
    """
    min_len = 2 + len_width
    if len(frame) < min_len:
        return fail(
            ParseErrorKind.TOO_SHORT,
            f"кадр {len(frame)} байт, минимум для заголовка {min_len}",
        )
    if frame[0] != ident or frame[1] != fc:
        return fail(
            ParseErrorKind.WRONG_COMMAND,
            f"ожидалось {ident:02X} {fc:02X}, получено {frame[0]:02X} {frame[1]:02X}",
        )
    declared = int.from_bytes(frame[2 : 2 + len_width], "big")
    if declared != len(frame):
        return fail(
            ParseErrorKind.LEN_MISMATCH,
            f"LEN={declared}, фактическая длина кадра {len(frame)}",
        )
    if expected_len is not None and len(frame) != expected_len:
        return fail(
            ParseErrorKind.LEN_MISMATCH,
            f"для этой команды ожидается кадр {expected_len} байт, получено {len(frame)}",
        )
    return None


def _parse_gain(raw: bytes, profile: DeviceProfile) -> GainSetting | None:
    """Разбирает двухбайтовое поле усиления. None — недопустимая кодировка."""
    mode, level = raw[0], raw[1]
    if mode not in (0x00, GAIN_MANUAL_FLAG):
        return None
    if level > profile.gain_max_level:
        return None
    return GainSetting(manual=mode == GAIN_MANUAL_FLAG, level=level)


# --------------------------------------------------------------------------------------
# Разбор ответов на команды чтения
# --------------------------------------------------------------------------------------


def parse_version(frame: bytes) -> ParseResult[int]:
    """10 01 — версия прошивки, сотые доли (410 = v4.10)."""
    problem: ParseResult[int] | None = _check_frame(frame, ID_READ, FC_VERSION, RESP_LEN_WIDTH, 8)
    if problem is not None:
        return problem
    return ok(int.from_bytes(frame[4:8], "big"))


def parse_serial(frame: bytes) -> ParseResult[int]:
    """10 03 — серийный номер."""
    problem: ParseResult[int] | None = _check_frame(frame, ID_READ, FC_SERIAL, RESP_LEN_WIDTH, 8)
    if problem is not None:
        return problem
    return ok(int.from_bytes(frame[4:8], "big"))


def parse_module_params(frame: bytes) -> ParseResult[ModuleParams]:
    """10 04 — скорость развёртки, число каналов, решёток и интервал пиков."""
    problem: ParseResult[ModuleParams] | None = _check_frame(
        frame, ID_READ, FC_MODULE_PARAMS, RESP_LEN_WIDTH, 12
    )
    if problem is not None:
        return problem
    speed_code, channels, fbg, gap = struct.unpack(">4H", frame[4:12])
    if channels < 1 or fbg < 1:
        return fail(
            ParseErrorKind.BAD_VALUE,
            f"каналов {channels}, решёток {fbg} — должно быть ≥ 1",
        )
    return ok(
        ModuleParams(
            speed_code=speed_code,
            speed_hz=decode_sweep_speed(speed_code),
            channels=channels,
            fbg_per_channel=fbg,
            peak_gap_ghz=gap,
        )
    )


def parse_sweep_params(frame: bytes, profile: DeviceProfile) -> ParseResult[SweepConfig]:
    """10 05 — параметры развёртки, с пересчётом в ГГц по профилю."""
    problem: ParseResult[SweepConfig] | None = _check_frame(
        frame, ID_READ, FC_SWEEP, RESP_LEN_WIDTH, 12
    )
    if problem is not None:
        return problem
    start, step, stop, adc_step = struct.unpack(">4H", frame[4:12])
    if start >= stop:
        return fail(
            ParseErrorKind.BAD_VALUE,
            f"нарушен инвариант start_param < stop_param: {start} ≥ {stop}",
        )
    if step < 1 or adc_step < 1:
        return fail(ParseErrorKind.BAD_VALUE, f"шаг развёртки {step}/{adc_step} должен быть ≥ 1")
    return ok(SweepConfig.from_params(start, step, stop, adc_step, profile))


def parse_channel_setup(
    frame: bytes, profile: DeviceProfile
) -> ParseResult[tuple[ChannelSetup, ...]]:
    """10 06 — пороги и усиления. Число каналов берётся из длины кадра."""
    problem: ParseResult[tuple[ChannelSetup, ...]] | None = _check_frame(
        frame, ID_READ, FC_CHANNEL_SETUP, RESP_LEN_WIDTH
    )
    if problem is not None:
        return problem
    payload = frame[4:]
    if len(payload) == 0 or len(payload) % 4 != 0:
        return fail(
            ParseErrorKind.LEN_MISMATCH,
            f"тело {len(payload)} байт не делится на 4 байта на канал",
        )

    setups: list[ChannelSetup] = []
    for index in range(len(payload) // 4):
        chunk = payload[index * 4 : index * 4 + 4]
        raw_threshold = int.from_bytes(chunk[0:2], "big")
        if raw_threshold == profile.threshold_auto:
            threshold = None
        elif raw_threshold > profile.adc_max:
            return fail(
                ParseErrorKind.BAD_VALUE,
                f"канал {index}: порог {raw_threshold} вне диапазона 0…{profile.adc_max}",
            )
        else:
            threshold = raw_threshold
        gain = _parse_gain(chunk[2:4], profile)
        if gain is None:
            return fail(
                ParseErrorKind.BAD_VALUE,
                f"канал {index}: недопустимое поле усиления {chunk[2]:02X} {chunk[3]:02X}",
            )
        setups.append(ChannelSetup(channel=index, threshold=threshold, gain=gain))
    return ok(tuple(setups))


def parse_undocumented(frame: bytes) -> ParseResult[UndocumentedResponse]:
    """10 02 — недокументированный ответ (KB_04, N17): тело отдаётся как есть.

    Наблюдён единственный вариант: LEN = 0x0014 = 20 байт, тело
    `05 DC | 0A 80 | 00 00 × 6`. Значения полей не интерпретируются —
    что они означают, неизвестно, а догадка в коде хуже пробела (KB_05).

    Длина кадра не фиксируется числом 20: единственный наблюдённый ответ
    основанием для константы не является, проверяется лишь то, что тело
    делится на 16-битные слова.
    """
    problem: ParseResult[UndocumentedResponse] | None = _check_frame(
        frame, ID_READ, FC_UNDOCUMENTED, RESP_LEN_WIDTH
    )
    if problem is not None:
        return problem
    payload = frame[2 + RESP_LEN_WIDTH :]
    if len(payload) % 2 != 0:
        return fail(
            ParseErrorKind.LEN_MISMATCH,
            f"тело {len(payload)} байт не делится на 16-битные слова",
        )
    words = struct.unpack(f">{len(payload) // 2}H", payload)
    return ok(UndocumentedResponse(payload=payload, words=words))


# --------------------------------------------------------------------------------------
# Разбор ответов на команды записи и режимов
# --------------------------------------------------------------------------------------


def parse_write_ack(frame: bytes) -> ParseResult[bool]:
    """20 FC 00 06 00 SS — подтверждение команды записи. True — успех."""
    if len(frame) < 2:
        return fail(ParseErrorKind.TOO_SHORT, f"кадр {len(frame)} байт")
    if frame[0] != ID_WRITE:
        return fail(ParseErrorKind.WRONG_COMMAND, f"ожидался ID 20, получен {frame[0]:02X}")
    problem: ParseResult[bool] | None = _check_frame(
        frame, ID_WRITE, frame[1], RESP_LEN_WIDTH, WRITE_ACK_LEN
    )
    if problem is not None:
        return problem
    status = int.from_bytes(frame[4:6], "big")
    if status not in (0, 1):
        return fail(ParseErrorKind.BAD_VALUE, f"код результата {status}, ожидалось 0 или 1")
    return ok(status == 1)


def parse_stop_ack(frame: bytes, profile: DeviceProfile) -> ParseResult[bool]:
    """30 01 00 00 00 08 00 01 — подтверждение остановки потока."""
    width = profile.mode_len_width
    problem: ParseResult[bool] | None = _check_frame(frame, ID_MODE, FC_STOP, width, 2 + width + 2)
    if problem is not None:
        return problem
    status = int.from_bytes(frame[2 + width :], "big")
    if status not in (0, 1):
        return fail(ParseErrorKind.BAD_VALUE, f"код результата {status}, ожидалось 0 или 1")
    return ok(status == 1)


#: Заголовок блока АЦП внутри тела: Канал(2) + Усиление(2).
#: Одинаков в ответах 30 07 и 30 03 — ✅ обе раскладки подтверждены скринингом.
ADC_BLOCK_HEADER = 4


def _parse_adc_block(
    block: bytes, offset: int, points: int, profile: DeviceProfile, where: str
) -> ParseResult[AdcBlock]:
    """Разбирает Канал(2) Усиление(2) ADC(2)×points начиная с `offset`.

    Общая часть ответов `30 07` (один канал) и `30 03` (все каналы подряд):
    заголовок блока в них одинаковый, что и подтвердил скрининг.
    `where` попадает в текст ошибки, чтобы по сообщению было видно,
    в каком из двух ответов и в каком блоке нашлась проблема.
    """
    channel = int.from_bytes(block[offset : offset + 2], "big")
    if not 0 <= channel < profile.channels:
        return fail(
            ParseErrorKind.BAD_VALUE,
            f"{where}: номер канала {channel} вне диапазона 0…{profile.channels - 1}",
        )
    gain = _parse_gain(block[offset + 2 : offset + ADC_BLOCK_HEADER], profile)
    if gain is None:
        return fail(
            ParseErrorKind.BAD_VALUE,
            f"{where}: недопустимое поле усиления {block[offset + 2]:02X} {block[offset + 3]:02X}",
        )
    adc = np.frombuffer(block, dtype=">u2", count=points, offset=offset + ADC_BLOCK_HEADER).astype(
        np.uint16
    )
    return ok(AdcBlock(channel=channel, gain=gain, adc=adc))


def parse_raw_adc(frame: bytes, profile: DeviceProfile) -> ParseResult[AdcBlock]:
    """30 07 — сырые отсчёты АЦП канала: LEN(4) Канал(2) Усиление(2) ADC(2)×N.

    Число отсчётов берётся из фактической длины и не сверяется с `profile.adc_points`:
    сборка длинного ответа из нескольких датаграмм — вопрос D5 и зона сессии,
    сюда кадр приходит уже собранным.
    """
    width = profile.mode_len_width
    header = 2 + width + 4
    problem: ParseResult[AdcBlock] | None = _check_frame(frame, ID_MODE, FC_RAW_ADC, width)
    if problem is not None:
        return problem
    if len(frame) < header:
        return fail(
            ParseErrorKind.TOO_SHORT,
            f"кадр {len(frame)} байт, минимум {header} для заголовка с каналом и усилением",
        )
    body = len(frame) - header
    if body % 2 != 0:
        return fail(ParseErrorKind.LEN_MISMATCH, f"тело АЦП {body} байт не делится на 2")

    return _parse_adc_block(frame, 2 + width, body // 2, profile, "30 07")


def parse_debug_once(frame: bytes, profile: DeviceProfile) -> ParseResult[DebugResponse]:
    """30 03 — одиночная развёртка по всем каналам сразу.

    Раскладка ✅ подтверждена скринингом (N14 закрыт)::

        30 03 | LEN(4) | [ Канал(2) | Усиление(2) | ADC(2) × 2551 ] × N_каналов

    Число точек берётся из профиля, а не из длины кадра: в отличие от `30 07`,
    здесь блоки идут подряд без разделителей, и без известного размера блока
    границы каналов определить нечем.

    ⚠️ Частот в этом ответе нет. Кадр телеметрии приходит отдельной
    датаграммой `30 02` перед `30 03` и разбирается `parse_measurement`.
    """
    width = profile.mode_len_width
    problem: ParseResult[DebugResponse] | None = _check_frame(frame, ID_MODE, FC_DEBUG, width)
    if problem is not None:
        return problem

    payload = frame[2 + width :]
    points = profile.adc_points
    block_size = ADC_BLOCK_HEADER + points * 2
    if len(payload) % block_size != 0 or not payload:
        return fail(
            ParseErrorKind.LEN_MISMATCH,
            f"тело {len(payload)} байт не делится на блок канала {block_size} байт "
            f"(заголовок {ADC_BLOCK_HEADER} + {points} отсчётов по 2 байта)",
        )
    count = len(payload) // block_size
    if count != profile.channels:
        return fail(
            ParseErrorKind.LEN_MISMATCH,
            f"тело содержит {count} блоков каналов, профиль описывает {profile.channels}",
        )

    blocks: list[AdcBlock] = []
    for index in range(count):
        parsed = _parse_adc_block(
            payload, index * block_size, points, profile, f"30 03, блок {index}"
        )
        if parsed.error is not None:
            return fail(parsed.error.kind, parsed.error.message)
        assert parsed.value is not None
        blocks.append(parsed.value)
    return ok(DebugResponse(blocks=tuple(blocks), payload=payload))


# --------------------------------------------------------------------------------------
# Кадр телеметрии
# --------------------------------------------------------------------------------------


def detect_freq_divisor(freq_raw: np.ndarray, profile: DeviceProfile) -> int | None:
    """Определяет единицы поля частоты по одному кадру (KB_04, D1).

    Диапазоны гипотез не пересекаются (для заводской развёртки 191149…196249
    против 1911490…1962490), поэтому побеждает та, под которую подошло больше
    значений. None — не подошло ни одно значение ни под одну гипотезу,
    единицы остаются неизвестными.

    ✅ Вопрос закрыт скринингом в пользу делителя 10, и профиль по умолчанию
    его и содержит. Функция остаётся страховкой: она нужна, если прибор
    окажется с другой прошивкой, а не для того, чтобы гадать при каждом кадре.
    """
    best: int | None = None
    best_count = 0
    for divisor in FREQ_DIVISOR_CANDIDATES:
        low, high = profile.freq_raw_bounds(divisor)
        count = int(np.count_nonzero((freq_raw >= low) & (freq_raw <= high)))
        if count > best_count:
            best, best_count = divisor, count
    return best


def parse_measurement(
    frame: bytes,
    profile: DeviceProfile,
    t_mono: float = 0.0,
    out: MeasurementFrame | None = None,
) -> ParseResult[MeasurementFrame]:
    """30 02 — кадр телеметрии. Разбор векторный, без цикла по 120 полям.

    Раскладка — гипотеза N4: на каждый канал 30 групп «индекс(1) + частота(3)»,
    затем 2 байта температуры корпуса.

    Позиция получает NaN, если частота вышла за `stop_ghz … start_ghz`, попала
    в `profile.peak_missing_codes` или байт индекса не совпал с ожидаемым
    порядком 00…1D. Последнее известное значение не подставляется никогда.

    `t_mono` проставляет тот, кто принял датаграмму: у кодека нет часов.
    `out` — переиспользуемый буфер вызывающего; кодек состояния не хранит.
    """
    expected = profile.frame_size
    problem: ParseResult[MeasurementFrame] | None = _check_frame(
        frame, ID_MODE, FC_STREAM, profile.mode_len_width, expected
    )
    if problem is not None:
        return problem

    channels = profile.channels
    fbg = profile.fbg_per_channel
    header = 2 + profile.mode_len_width
    slots_bytes = fbg * 4

    body = np.frombuffer(frame, dtype=np.uint8, count=expected - header, offset=header)
    block = body.reshape(channels, profile.channel_bytes)
    slots = block[:, :slots_bytes].reshape(channels, fbg, 4)

    # Трёхбайтовое поле частоты: дополняем нулевым старшим байтом и
    # переинтерпретируем как big-endian uint32. Ручная арифметика запрещена (KB_05 №1).
    padded = np.zeros((channels, fbg, 4), dtype=np.uint8)
    padded[:, :, 1:] = slots[:, :, 1:]
    freq_raw = padded.view(">u4").reshape(channels, fbg)

    divisor = profile.freq_divisor
    if divisor is None:
        divisor = detect_freq_divisor(freq_raw, profile)
    if divisor is None:
        return fail(
            ParseErrorKind.AMBIGUOUS_UNITS,
            "ни одно значение частоты не попало в диапазоны гипотез D1; "
            "единицы поля определить нельзя, кадр не разбирается",
        )

    low, high = profile.freq_raw_bounds(divisor)
    valid = (freq_raw >= low) & (freq_raw <= high)
    if profile.peak_missing_codes:
        markers = np.fromiter(profile.peak_missing_codes, dtype=np.uint32)
        valid &= ~np.isin(freq_raw, markers)

    expected_index = np.arange(fbg, dtype=np.uint8)
    index_ok = slots[:, :, 0] == expected_index
    valid &= index_ok

    temp_dtype = ">i2" if profile.case_temp_signed else ">u2"
    temp_raw = np.ascontiguousarray(block[:, slots_bytes:]).view(temp_dtype).reshape(channels)

    target = out if out is not None else MeasurementFrame(channels, fbg)
    if target.freq_ghz.shape != (channels, fbg):
        raise ValueError(
            f"буфер out имеет форму {target.freq_ghz.shape}, ожидалась {(channels, fbg)}"
        )
    target.t_mono = t_mono
    target.freq_divisor = divisor
    target.freq_ghz[...] = np.where(valid, freq_raw / divisor, np.nan)
    target.case_temp_c[...] = temp_raw * profile.case_temp_scale
    target.missing = int(np.count_nonzero(~valid))
    target.index_mismatches = int(np.count_nonzero(~index_ok))
    return ok(target)


# --------------------------------------------------------------------------------------
# Диспетчер
# --------------------------------------------------------------------------------------


def parse_any(
    frame: bytes,
    profile: DeviceProfile,
    t_mono: float = 0.0,
) -> ParseResult[object]:
    """Разбирает любой ответ прибора, выбирая парсер по паре (ID, FC).

    Нужен приёмному тракту, который до разбора не знает, что пришло:
    телеметрия и ответ на команду идут по одному сокету.
    """
    key = classify(frame)
    if key is None:
        return fail(ParseErrorKind.TOO_SHORT, f"кадр {len(frame)} байт, нет пары (ID, FC)")
    if key not in KNOWN_COMMANDS:
        return fail(
            ParseErrorKind.UNKNOWN_COMMAND,
            f"пара ({key[0]:02X}, {key[1]:02X}) отсутствует в списке известных команд",
        )

    ident, fc = key
    if ident == ID_WRITE:
        return parse_write_ack(frame)
    if ident == ID_READ:
        if fc == FC_VERSION:
            return parse_version(frame)
        if fc == FC_UNDOCUMENTED:
            return parse_undocumented(frame)
        if fc == FC_SERIAL:
            return parse_serial(frame)
        if fc == FC_MODULE_PARAMS:
            return parse_module_params(frame)
        if fc == FC_SWEEP:
            return parse_sweep_params(frame, profile)
        return parse_channel_setup(frame, profile)
    if fc == FC_STOP:
        return parse_stop_ack(frame, profile)
    if fc == FC_STREAM:
        return parse_measurement(frame, profile, t_mono)
    if fc == FC_DEBUG:
        return parse_debug_once(frame, profile)
    return parse_raw_adc(frame, profile)
