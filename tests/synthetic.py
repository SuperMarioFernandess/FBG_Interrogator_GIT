"""Тонкая обёртка над примитивами симулятора плюс загрузчик эталонных векторов.

Сами примитивы кодирования живут в `fbg/sim/`: они нужны симулятору в рабочем
коде, а импорт из `tests/` в `fbg/` недопустим. Здесь остаётся только то,
что относится к тестам, — чтение файлов векторов, — и реэкспорт, чтобы
`tests/test_codec.py` не переписывался вслед за переездом.

Два файла векторов телеметрии, и разница между ними принципиальна:

  * `measurement_real.hex` — ✅ кадр со стенда, скрининг 27.08.2026.
    Канал 1 и температура прочитаны из захвата, каналы 2…4 реконструированы
    (см. шапку самого файла). Порождается `scene_real_capture`.
  * `measurement_synthetic.hex` — ⚠️ синтетика. Захвата кадра с четырьмя
    заполненными каналами, значением вне диапазона и `FF FF FF` не существует
    и не будет: такой кадр прибор не выдаёт. Вектор нужен для проверки
    валидации, и он остаётся синтетическим намеренно.

Раскладка кадра (N4) и кодировка «пик не найден» нулями (N3) после скрининга
подтверждены и синтетикой больше не являются. Разбор при этом на кодировку
не опирается: он отбраковывает значения и по диапазону тоже, поэтому и нули,
и `FF FF FF`, и любой мусор одинаково дают NaN.
"""

from pathlib import Path

from fbg.sim.encode import MISSING_STIMULUS, encode_measurement, ghz_to_raw, nm_to_raw
from fbg.sim.scene import scene_real_capture, scene_two_gratings

__all__ = [
    "MISSING_STIMULUS",
    "VECTORS_DIR",
    "encode_measurement",
    "ghz_to_raw",
    "load_vectors",
    "nm_to_raw",
    "scene_real_capture",
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
