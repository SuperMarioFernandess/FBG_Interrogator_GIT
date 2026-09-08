"""Общие механизмы доков и сохранения Qt-раскладки интерфейса.

Файл намеренно живёт в ``fbg.ui``: ``QMainWindow.saveState()`` возвращает
``QByteArray`` и тем самым принадлежит Qt-слою. ``fbg.io`` не должен знать
о раскладке окна и по-прежнему импортируется без PySide6 (KB_05 №35).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtWidgets import QDockWidget, QFrame, QMainWindow, QScrollArea, QWidget

UI_LAYOUT_FILENAME = "fbg_ui_layout.json"
UI_LAYOUT_VERSION = 1
DEFAULT_UI_PERIOD_MS = 100
MIN_UI_PERIOD_MS = 50
MAX_UI_PERIOD_MS = 2_000

_UNLOCKED_FEATURES = (
    QDockWidget.DockWidgetFeature.DockWidgetClosable
    | QDockWidget.DockWidgetFeature.DockWidgetMovable
    | QDockWidget.DockWidgetFeature.DockWidgetFloatable
)


@dataclass(frozen=True)
class UiLayoutState:
    """Сохранённое состояние всех вкладок и общие UI-настройки."""

    docks: dict[str, QByteArray]
    locked: bool = False
    period_ms: int = DEFAULT_UI_PERIOD_MS


def layout_path_for_config(config_path: Path | None) -> Path | None:
    """Возвращает соседний с конфигурацией путь раскладки.

    В тестах ``config_path=None`` означает отсутствие файлового состояния вообще:
    окно не должно неожиданно писать в текущий каталог тестового процесса.
    """

    if config_path is None:
        return None
    return config_path.with_name(UI_LAYOUT_FILENAME)


def _quarantine_path(path: Path) -> Path:
    target = path.with_name(path.name + ".bad")
    suffix = 2
    while target.exists():
        target = path.with_name(f"{path.name}.bad.{suffix}")
        suffix += 1
    return target


def quarantine_layout(path: Path) -> Path:
    """Откладывает непонятный файл, не затирая предыдущие ``*.bad``."""

    target = _quarantine_path(path)
    path.replace(target)
    return target


def load_layout(path: Path) -> UiLayoutState:
    """Читает раскладку или бросает ``ValueError`` с причиной.

    Любая ошибка трактуется верхним уровнем одинаково: раскладка по умолчанию,
    исходный файл в карантин. Частичное восстановление опасно тем же, чем
    частичное чтение будущей версии обычной конфигурации (KB_05 №33).
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"файл раскладки не читается: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("корень файла раскладки должен быть объектом")
    if raw.get("version") != UI_LAYOUT_VERSION:
        raise ValueError(f"неподдерживаемая версия раскладки: {raw.get('version')!r}")

    period_ms = raw.get("period_ms", DEFAULT_UI_PERIOD_MS)
    if type(period_ms) is not int or not MIN_UI_PERIOD_MS <= period_ms <= MAX_UI_PERIOD_MS:
        raise ValueError(f"period_ms должен быть целым {MIN_UI_PERIOD_MS}…{MAX_UI_PERIOD_MS}")
    locked = raw.get("locked", False)
    if type(locked) is not bool:
        raise ValueError("locked должен быть true/false")

    encoded = raw.get("docks")
    if not isinstance(encoded, dict) or not encoded:
        raise ValueError("секция docks отсутствует или пуста")
    docks: dict[str, QByteArray] = {}
    for name, value in encoded.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError("docks должен содержать пары строк имя→base64")
        try:
            decoded = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"docks.{name}: неверный base64") from exc
        if not decoded:
            raise ValueError(f"docks.{name}: пустое состояние")
        docks[name] = QByteArray(decoded)
    return UiLayoutState(docks=docks, locked=locked, period_ms=period_ms)


def save_layout(path: Path, state: UiLayoutState) -> Path:
    """Атомарно сохраняет раскладку как JSON с base64-состояниями Qt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": UI_LAYOUT_VERSION,
        "period_ms": state.period_ms,
        "locked": state.locked,
        "docks": {
            name: base64.b64encode(bytes(value)).decode("ascii")
            for name, value in sorted(state.docks.items())
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
        newline="\n",
    )
    os.replace(temporary, path)
    return path


def scrollable(widget: QWidget) -> QScrollArea:
    """Оборачивает длинную форму в прокрутку без изменения её логики."""

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


class DockTab(QMainWindow):
    """Вложенное главное окно одной вкладки с 2–4 крупными доками."""

    layout_key = ""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )
        self._docks: list[QDockWidget] = []
        self._dock_defaults: dict[QDockWidget, Qt.DockWidgetArea] = {}
        self._layout_locked = False

    @property
    def dock_widgets(self) -> tuple[QDockWidget, ...]:
        """Все доки вкладки в стабильном порядке."""

        return tuple(self._docks)

    def add_panel_dock(
        self,
        title: str,
        widget: QWidget,
        object_name: str,
        area: Qt.DockWidgetArea,
    ) -> QDockWidget:
        """Добавляет крупную область с именем, нужным ``saveState``."""

        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(_UNLOCKED_FEATURES)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self._docks.append(dock)
        self._dock_defaults[dock] = area
        return dock

    def _apply_default_splits(self) -> None:
        """Дополнительное деление дефолтной раскладки. Переопределяется панелью."""

    def reset_layout(self) -> None:
        """Возвращает все доки из floating/hidden в стартовую раскладку."""

        for dock in self._docks:
            dock.setFloating(False)
            dock.show()
            self.removeDockWidget(dock)
        for dock in self._docks:
            self.addDockWidget(self._dock_defaults[dock], dock)
        self._apply_default_splits()
        self.set_layout_locked(self._layout_locked)

    def set_layout_locked(self, locked: bool) -> None:
        """Запрещает или разрешает перенос, отделение и закрытие доков."""

        self._layout_locked = bool(locked)
        features = (
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures if locked else _UNLOCKED_FEATURES
        )
        for dock in self._docks:
            dock.setFeatures(features)

    def needs_refresh(self, *, active: bool) -> bool:
        """Нужен ли панели UI-такт сейчас.

        Неактивная вкладка обычно скрыта, но её floating-док остаётся отдельным
        видимым окном. Проверка по фактической видимости не даёт такому графику
        замереть только потому, что родительская вкладка не выбрана.
        """

        if active:
            return True
        # ``isVisible()`` учитывает скрытого родителя-QTabWidget и потому
        # возвращает False у отделённого дока, когда пользователь переключил
        # вкладку. ``isHidden()`` отражает именно явное закрытие самого дока.
        return any(dock.isFloating() and not dock.isHidden() for dock in self._docks)
