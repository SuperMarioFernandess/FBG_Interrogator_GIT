"""Дымовые тесты каркаса.

Проверяют, что пакет импортируется и структура на месте. Нужны в том числе
для того, чтобы `pytest` не возвращал код 5 («не собрано ни одного теста»),
который CI трактует как падение.

Удалять после появления настоящих тестов не нужно: проверка импорта ловит
опечатки в структуре пакетов раньше, чем это сделает любой модульный тест.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGES = ["fbg", "fbg.core", "fbg.io", "fbg.sim", "fbg.ui"]

#: Модули, которые обязаны импортироваться без Qt. Кроме ядра сюда входит
#: часть `fbg/ui`: сборка приложения, модели панелей, тексты и диагностика —
#: это порядок действий и подстановка строк, а не окно, и проверяются они
#: тестами без экрана.
QT_FREE_MODULES = [
    "fbg.core",
    "fbg.io",
    "fbg.ui.app",
    "fbg.ui.models",
    "fbg.ui.texts",
    "fbg.ui.diagnostics",
]

#: Корень репозитория: подпроцессу нужен путь к пакету.
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", PACKAGES)
def test_пакет_импортируется(name: str) -> None:
    """Каждый пакет проекта импортируется без ошибок."""
    assert importlib.import_module(name) is not None


def test_ядро_не_тянет_qt() -> None:
    """`fbg.core`, `fbg.io` и логика UI не должны зависеть от PySide6 и pandas.

    Это архитектурное требование (KB_03): тесты ядра обязаны проходить
    в окружении без Qt, иначе CI придётся тащить 100 МБ зависимостей ради
    разбора байтов.

    Проверка ушла в **подпроцесс** в чате №10, и это не украшение. С появлением
    `fbg/ui` в том же прогоне живут тесты, которые Qt импортируют намеренно,
    и проверка «`PySide6` нет в `sys.modules`» внутри процесса стала зависеть
    от порядка сборки файлов: она зеленела бы ровно до тех пор, пока
    алфавитный порядок ставит `test_smoke` раньше `test_ui_widgets`.
    Отдельный интерпретатор от порядка не зависит вовсе.
    """
    program = (
        "import sys\n"
        f"for name in {QT_FREE_MODULES!r}:\n"
        "    __import__(name)\n"
        "banned = [name for name in ('PySide6', 'pandas') if name in sys.modules]\n"
        "sys.exit('импортировано лишнее: ' + ', '.join(banned) if banned else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
