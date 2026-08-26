"""Сборка байтов «со стороны прибора»: ответы на команды и кадр телеметрии.

⚠️ Этот модуль **намеренно не импортирует `fbg.core.codec`.** Байтовые константы,
раскладки и таблица кодов скорости продублированы здесь литералами из
`KB_02_protocol.md`. Если бы симулятор собирал ответы вызовами кодека наоборот,
интеграционные тесты проверяли бы сам кодек на согласованность с собой:
общая ошибка в понимании протокола осталась бы невидимой в обе стороны.
Расхождение между этим модулем и кодеком — сигнал, ради которого он и написан.

⚠️ Раскладка кадра телеметрии — гипотеза N4 (KB_04), выведенная расчётом из
размера 494 байта. Захвата не существует, KB_06 пуст.
"""

import struct

import numpy as np

from fbg.core.profile import C_NM_GHZ, DeviceProfile

# --------------------------------------------------------------------------------------
# Байтовые константы протокола — дубликат KB_02, см. предупреждение в шапке модуля
# --------------------------------------------------------------------------------------

SIM_ID_READ = 0x10
SIM_ID_WRITE = 0x20
SIM_ID_MODE = 0x30

#: Ширина поля LEN в ответах 0x10 и 0x20 — ✅ проверена на пяти ответах прибора.
SIM_READ_LEN_WIDTH = 2

#: Длина ответа-подтверждения на команду записи: 20 FC 00 06 SS SS.
SIM_WRITE_ACK_LEN = 6

#: Коды скорости развёртки (KB_02): code = M·10 + E, где F = M·10^E Гц.
SIM_SWEEP_SPEED_CODES: dict[int, int] = {
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
SIM_SPEED_KEEP_CURRENT = 0x0000

#: Сырой код, которым в эталонных сценах изображается отсутствующий пик.
#: Это **стимул теста, а не факт о приборе**: вопрос N3 открыт.
MISSING_STIMULUS = 0x000000

#: Максимум трёхбайтового поля частоты.
FREQ_FIELD_MAX = 0xFFFFFF


def sim_decode_speed_code(code: int) -> int | None:
    """Расшифровывает код скорости развёртки в герцы; None — код нераспознан.

    Реализовано независимо от `codec.decode_sweep_speed` — см. шапку модуля.
    """
    for hz, known in SIM_SWEEP_SPEED_CODES.items():
        if known == code:
            return hz
    if code <= 0:
        return None
    mantissa, exponent = divmod(code, 10)
    if mantissa == 0:
        return None
    return mantissa * 10**exponent


# --------------------------------------------------------------------------------------
# Пересчёт физических величин в сырые поля
# --------------------------------------------------------------------------------------


def ghz_to_raw(freq_ghz: float, divisor: int) -> int:
    """Переводит частоту в ГГц в сырое поле кадра для заданной гипотезы единиц."""
    return round(freq_ghz * divisor)


def nm_to_raw(wavelength_nm: float, divisor: int) -> int:
    """Переводит длину волны в нм в сырое поле кадра."""
    return ghz_to_raw(C_NM_GHZ / wavelength_nm, divisor)


# --------------------------------------------------------------------------------------
# Ответы на команды чтения (ID = 0x10)
# --------------------------------------------------------------------------------------


def _read_response(fc: int, payload: bytes) -> bytes:
    """Собирает ответ чтения: 10 FC LEN(2) payload, LEN = полная длина кадра."""
    total = 2 + SIM_READ_LEN_WIDTH + len(payload)
    return bytes([SIM_ID_READ, fc]) + struct.pack(">H", total) + payload


def encode_version(version_raw: int) -> bytes:
    """10 01 — версия прошивки в сотых долях: 410 → v4.10."""
    return _read_response(0x01, struct.pack(">I", version_raw))


def encode_serial(serial: int) -> bytes:
    """10 03 — серийный номер."""
    return _read_response(0x03, struct.pack(">I", serial))


def encode_module_params(speed_code: int, channels: int, fbg: int, peak_gap_ghz: int) -> bytes:
    """10 04 — скорость развёртки, каналы, решётки, интервал пиков."""
    return _read_response(0x04, struct.pack(">4H", speed_code, channels, fbg, peak_gap_ghz))


def encode_sweep(start: int, step: int, stop: int, adc_step: int) -> bytes:
    """10 05 — параметры развёртки в сырых единицах прибора."""
    return _read_response(0x05, struct.pack(">4H", start, step, stop, adc_step))


def encode_channel_setup(setups: list[tuple[int, int, int]]) -> bytes:
    """10 06 — пороги и усиления: на канал (порог, режим усиления, уровень).

    `порог` — сырое 16-битное значение: 0…16383 либо 0xFFFF для авторасчёта.
    `режим` — 0x00 автоматический, 0x80 ручной.
    """
    payload = b"".join(
        struct.pack(">H", threshold) + bytes([mode, level]) for threshold, mode, level in setups
    )
    return _read_response(0x06, payload)


# --------------------------------------------------------------------------------------
# Ответы на команды записи (ID = 0x20) и режимов (ID = 0x30)
# --------------------------------------------------------------------------------------


def encode_write_ack(fc: int, success: bool) -> bytes:
    """20 FC 00 06 00 SS — подтверждение команды записи."""
    return bytes([SIM_ID_WRITE, fc]) + struct.pack(">HH", SIM_WRITE_ACK_LEN, int(success))


def _mode_response(fc: int, payload: bytes, len_width: int, len_field: int | None = None) -> bytes:
    """Собирает ответ режима: 30 FC LEN(len_width) payload."""
    total = 2 + len_width + len(payload)
    declared = total if len_field is None else len_field
    return bytes([SIM_ID_MODE, fc]) + declared.to_bytes(len_width, "big") + payload


def encode_stop_ack(success: bool, len_width: int, len_field: int | None = None) -> bytes:
    """30 01 00 00 00 08 00 01 — подтверждение остановки потока."""
    return _mode_response(0x01, struct.pack(">H", int(success)), len_width, len_field)


def encode_raw_adc(
    channel: int,
    gain_mode: int,
    gain_level: int,
    adc: np.ndarray,
    len_width: int,
    len_field: int | None = None,
) -> bytes:
    """30 07 — сырые отсчёты АЦП канала: LEN Канал(2) Усиление(2) ADC(2)×N."""
    payload = struct.pack(">H", channel) + bytes([gain_mode, gain_level])
    payload += adc.astype(">u2").tobytes()
    return _mode_response(0x07, payload, len_width, len_field)


def encode_debug(payload: bytes, len_width: int, len_field: int | None = None) -> bytes:
    """30 03 — одиночная развёртка в отладочном режиме.

    🔴 Раскладка тела — открытый вопрос N14. Тело здесь собирается **по гипотезе**:
    словесное описание из KB_02 «частоты + ADC всех каналов» плюс оценка размера
    ≈ 21 КБ из KB_01. Числового примера нет ни в PDF, ни в захватах. Тело
    формирует вызывающий (`scene.debug_payload`), этот модуль только надевает
    заголовок. Ни один тест не проверяет раскладку тела как факт.
    """
    return _mode_response(0x03, payload, len_width, len_field)


# --------------------------------------------------------------------------------------
# Кадр телеметрии (30 02) — гипотеза N4
# --------------------------------------------------------------------------------------


def encode_measurement(
    profile: DeviceProfile,
    freq_raw: np.ndarray,
    temp_raw: np.ndarray,
    indices: np.ndarray | None = None,
    len_field: int | None = None,
) -> bytes:
    """Собирает кадр 30 02 из сырых полей — прямой побайтовый путь.

    Медленный и буквальный: пишется поле за полем, как читается таблица
    раскладки в KB_02. Используется для эталонных векторов и тестов.
    Горячий путь симулятора — `MeasurementEncoder`, который патчит
    предсобранный буфер; отдельный тест сверяет их между собой.

    `indices` позволяет подделать байты индекса для негативных тестов,
    `len_field` — записать в LEN значение, отличное от фактической длины.
    """
    channels, fbg = profile.channels, profile.fbg_per_channel
    if indices is None:
        indices = np.tile(np.arange(fbg, dtype=np.uint8), (channels, 1))

    body = bytearray()
    for channel in range(channels):
        for position in range(fbg):
            body.append(int(indices[channel, position]))
            body += int(freq_raw[channel, position]).to_bytes(3, "big")
        body += int(temp_raw[channel]).to_bytes(2, "big", signed=profile.case_temp_signed)

    total = 2 + profile.mode_len_width + len(body)
    length = total if len_field is None else len_field
    head = bytes([SIM_ID_MODE, 0x02]) + length.to_bytes(profile.mode_len_width, "big")
    return head + bytes(body)


class MeasurementEncoder:
    """Предсобранный кадр 30 02 с патчем только изменяющихся байтов.

    Между кадрами меняются лишь 120 трёхбайтовых полей частоты и 4 поля
    температуры — заголовок, LEN и байты индекса постоянны. Пересборка всех
    494 байт заново при 2000 кадрах/с — лишняя работа в цикле с бюджетом 500 мкс,
    поэтому буфер собирается один раз, а дальше патчится через numpy-view.

    Буфер принадлежит энкодеру и переиспользуется: `frame` возвращает ссылку
    на него, а не копию. Вызывающий обязан отправить кадр до следующего `update`.
    """

    __slots__ = ("_buf", "_freq_view", "_stage_bytes", "_stage_freq", "_temp_dtype", "_temp_view")

    def __init__(self, profile: DeviceProfile, len_field: int | None = None) -> None:
        channels, fbg = profile.channels, profile.fbg_per_channel
        header = 2 + profile.mode_len_width
        slots_bytes = fbg * 4

        template = encode_measurement(
            profile,
            np.zeros((channels, fbg), dtype=np.uint32),
            np.zeros(channels, dtype=np.int32),
            len_field=len_field,
        )
        self._buf = bytearray(template)

        view = np.frombuffer(memoryview(self._buf), dtype=np.uint8)
        block = view[header:].reshape(channels, profile.channel_bytes)
        # Трёхбайтовое поле частоты — байты 1..3 каждой четвёрки; байт 0 это индекс.
        self._freq_view = block[:, :slots_bytes].reshape(channels, fbg, 4)[:, :, 1:]
        self._temp_view = block[:, slots_bytes:]

        # Промежуточный буфер: сырое поле частоты как big-endian uint32, из которого
        # берутся три младших байта. Ручная арифметика запрещена (KB_05 №1).
        self._stage_freq = np.zeros((channels, fbg), dtype=">u4")
        self._stage_bytes = self._stage_freq.view(np.uint8).reshape(channels, fbg, 4)
        self._temp_dtype = ">i2" if profile.case_temp_signed else ">u2"

    def update(self, freq_raw: np.ndarray, temp_raw: np.ndarray) -> None:
        """Записывает новые частоты и температуры в предсобранный буфер."""
        if freq_raw.max() > FREQ_FIELD_MAX:
            raise ValueError(
                f"сырое поле частоты {int(freq_raw.max())} не помещается в три байта "
                f"(максимум {FREQ_FIELD_MAX})"
            )
        self._stage_freq[...] = freq_raw
        self._freq_view[...] = self._stage_bytes[:, :, 1:]
        self._temp_view[...] = (
            np.asarray(temp_raw).astype(self._temp_dtype).view(np.uint8).reshape(-1, 2)
        )

    @property
    def frame(self) -> bytearray:
        """Текущий кадр. Ссылка на внутренний буфер, не копия."""
        return self._buf

    def to_bytes(self) -> bytes:
        """Копия текущего кадра — для тестов и сравнения с эталонным вектором."""
        return bytes(self._buf)
