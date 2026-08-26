"""Загрузка эталонных векторов и генерация синтетических кадров.

⚠️ ВНИМАНИЕ. Кадр телеметрии 30 02 здесь **синтетический**. Он собран по
гипотезе N4 (KB_04): на канал 30 групп «индекс(1) + частота(3)», затем
2 байта температуры. Реального захвата кадра телеметрии не существует —
KB_06 пуст, скрининг не проводился. Как только появится настоящий кадр,
эти генераторы заменяются вектором с прибора.

Кодировка «пик не найден» (вопрос N3) тоже неизвестна. В сценах ниже
отсутствующий пик изображается нулями — это **стимул теста, а не факт
о приборе**. Разбор на такую кодировку не опирается: он отбраковывает
значения по диапазону, поэтому и нули, и FF FF FF, и любой мусор
одинаково дают NaN.
"""

from pathlib import Path

import numpy as np

from fbg.core.profile import C_NM_GHZ, DeviceProfile

VECTORS_DIR = Path(__file__).parent / "vectors"

#: Сырой код, которым в сценах изображается отсутствующий пик. Не факт о приборе.
MISSING_STIMULUS = 0x000000


def load_vectors(name: str = "real_device.hex") -> dict[str, bytes]:
    """Читает файл векторов вида «имя = байты в hex»."""
    vectors: dict[str, bytes] = {}
    for line in (VECTORS_DIR / name).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        vectors[key.strip()] = bytes.fromhex(value.replace(" ", ""))
    return vectors


def ghz_to_raw(freq_ghz: float, divisor: int) -> int:
    """Переводит частоту в ГГц в сырое поле кадра для заданной гипотезы единиц."""
    return round(freq_ghz * divisor)


def nm_to_raw(wavelength_nm: float, divisor: int) -> int:
    """Переводит длину волны в нм в сырое поле кадра."""
    return ghz_to_raw(C_NM_GHZ / wavelength_nm, divisor)


def encode_measurement(
    profile: DeviceProfile,
    freq_raw: np.ndarray,
    temp_raw: np.ndarray,
    indices: np.ndarray | None = None,
    len_field: int | None = None,
) -> bytes:
    """Собирает кадр 30 02 из сырых полей.

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
    head = bytes([0x30, 0x02]) + length.to_bytes(profile.mode_len_width, "big")
    return head + bytes(body)


def scene_two_gratings(profile: DeviceProfile, divisor: int) -> tuple[np.ndarray, np.ndarray]:
    """Типовая сцена: несколько занятых позиций, остальные пустые.

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
