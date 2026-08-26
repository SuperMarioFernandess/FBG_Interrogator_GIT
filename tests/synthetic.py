"""Тонкая обёртка над примитивами симулятора плюс загрузчик эталонных векторов.

Сами примитивы кодирования живут в `fbg/sim/`: они нужны симулятору в рабочем
коде, а импорт из `tests/` в `fbg/` недопустим. Здесь остаётся только то,
что относится к тестам, — чтение файлов векторов, — и реэкспорт, чтобы
`tests/test_codec.py` не переписывался вслед за переездом.

⚠️ ВНИМАНИЕ. Кадр телеметрии 30 02 **синтетический**. Он собран по гипотезе
N4 (KB_04): на канал 30 групп «индекс(1) + частота(3)», затем 2 байта
температуры. Реального захвата кадра телеметрии не существует — KB_06 пуст,
скрининг не проводился. Как только появится настоящий кадр, эти генераторы
заменяются вектором с прибора.

Кодировка «пик не найден» (вопрос N3) тоже неизвестна. В сценах ниже
отсутствующий пик изображается нулями — это **стимул теста, а не факт
о приборе**. Разбор на такую кодировку не опирается: он отбраковывает
значения по диапазону, поэтому и нули, и FF FF FF, и любой мусор
одинаково дают NaN.
"""

from pathlib import Path

from fbg.sim.encode import MISSING_STIMULUS, encode_measurement, ghz_to_raw, nm_to_raw
from fbg.sim.scene import scene_two_gratings

__all__ = [
    "MISSING_STIMULUS",
    "VECTORS_DIR",
    "encode_measurement",
    "ghz_to_raw",
    "load_vectors",
    "nm_to_raw",
    "scene_two_gratings",
]

VECTORS_DIR = Path(__file__).parent / "vectors"


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
