"""Панель информации о приборе: что он о себе сообщил и как идёт связь.

Всё только чтение. Настройка прибора — чат №11; здесь ни одного поля ввода
нет намеренно, чтобы панель нельзя было перепутать с диалогом настроек.

Модель строит `fbg.ui.models.device_sections` — без Qt и с тестами. Виджет
раскладывает её по дереву и старается не перестраивать его без нужды: дерево,
пересобираемое десять раз в секунду, теряет прокрутку и раскрытые группы,
то есть смотреть на него невозможно ровно тогда, когда это нужно.
"""

from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from fbg.ui import models
from fbg.ui.app import AppController
from fbg.ui.models import AppSnapshot, InfoSection

#: Заголовки колонок дерева.
COLUMNS: tuple[str, ...] = ("Параметр", "Значение", "Примечание")


def sections_shape(sections: tuple[InfoSection, ...]) -> tuple[tuple[str, int], ...]:
    """Форма модели: заголовки групп и число строк в каждой.

    По ней решается, перестраивать дерево или обновить текст на месте.
    Форма меняется редко — когда прибор опрошен и появились строки каналов, —
    а значения меняются каждый такт.
    """
    return tuple((section.title, len(section.rows)) for section in sections)


class DeviceInfoPanel(QWidget):
    """Прошивка, серийный номер, развёртка, каналы и качество связи."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._shape: tuple[tuple[str, int], ...] = ()

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)

        self.refresh(controller.snapshot())

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Обновляет дерево по снимку. Зовётся таймером окна, 10 Гц."""
        sections = models.device_sections(snapshot)
        shape = sections_shape(sections)
        if shape != self._shape:
            self._rebuild(sections)
            self._shape = shape
            return
        for group_index, section in enumerate(sections):
            group = self.tree.topLevelItem(group_index)
            if group is None:
                continue
            for row_index, row in enumerate(section.rows):
                item = group.child(row_index)
                if item is None:
                    continue
                if item.text(1) != row.value:
                    item.setText(1, row.value)
                if item.text(2) != row.note:
                    item.setText(2, row.note)

    def _rebuild(self, sections: tuple[InfoSection, ...]) -> None:
        """Пересобирает дерево целиком — только когда изменилась форма модели."""
        expanded = {
            self.tree.topLevelItem(index).text(0)  # type: ignore[union-attr]
            for index in range(self.tree.topLevelItemCount())
            if self.tree.topLevelItem(index) is not None
            and self.tree.topLevelItem(index).isExpanded()  # type: ignore[union-attr]
        }
        self.tree.clear()
        for section in sections:
            group = QTreeWidgetItem([section.title, "", ""])
            for row in section.rows:
                QTreeWidgetItem(group, [row.label, row.value, row.note])
            self.tree.addTopLevelItem(group)
            group.setExpanded(not expanded or section.title in expanded)
