"""Дымовые тесты каркаса.

Проверяют, что пакет импортируется и структура на месте. Нужны в том числе
для того, чтобы `pytest` не возвращал код 5 («не собрано ни одного теста»),
который CI трактует как падение.

Удалять после появления настоящих тестов не нужно: проверка импорта ловит
опечатки в структуре пакетов раньше, чем это сделает любой модульный тест.
"""

import importlib

import pytest

PACKAGES = ["fbg", "fbg.core", "fbg.io", "fbg.sim", "fbg.ui"]


@pytest.mark.parametrize("name", PACKAGES)
def test_пакет_импортируется(name: str) -> None:
    """Каждый пакет проекта импортируется без ошибок."""
    assert importlib.import_module(name) is not None


def test_ядро_не_тянет_qt() -> None:
    """`fbg.core` и `fbg.io` не должны зависеть от PySide6.

    Это архитектурное требование (KB_03): тесты ядра обязаны проходить
    в окружении без Qt, иначе CI придётся тащить 100 МБ зависимостей
    ради разбора байтов.
    """
    import sys

    for name in ("fbg.core", "fbg.io"):
        importlib.import_module(name)

    assert "PySide6" not in sys.modules
