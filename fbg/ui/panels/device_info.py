"""Вкладка прибора: наблюдение и настройка в трёх независимых доках."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem, QWidget

from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.docking import DockTab, scrollable
from fbg.ui.models import AppSnapshot, InfoSection
from fbg.ui.panels.device_config import DeviceConfigPanel

COLUMNS: tuple[str, ...] = ("Параметр", "Значение", "Примечание")


def sections_shape(sections: tuple[InfoSection, ...]) -> tuple[tuple[str, int], ...]:
    """Форма модели: заголовки групп и число строк в каждой."""

    return tuple((section.title, len(section.rows)) for section in sections)


class _SectionTree(QTreeWidget):
    """Дерево секций с обновлением значений на месте и бережным rebuild."""

    def __init__(self) -> None:
        super().__init__()
        self._shape: tuple[tuple[str, int], ...] = ()
        self.setColumnCount(len(COLUMNS))
        self.setHeaderLabels(list(COLUMNS))
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

    def update_sections(self, sections: tuple[InfoSection, ...]) -> None:
        """Меняет только текст, если состав строк прежний."""

        shape = sections_shape(sections)
        if shape != self._shape:
            self._rebuild(sections)
            self._shape = shape
            return
        for group_index, section in enumerate(sections):
            group = self.topLevelItem(group_index)
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

    def _selection_key(self) -> tuple[str, str] | None:
        items = self.selectedItems()
        if not items:
            return None
        item = items[0]
        parent = item.parent()
        if parent is None:
            return item.text(0), ""
        return parent.text(0), item.text(0)

    def _restore_selection(self, key: tuple[str, str] | None) -> None:
        if key is None:
            return
        for group_index in range(self.topLevelItemCount()):
            group = self.topLevelItem(group_index)
            if group is None or group.text(0) != key[0]:
                continue
            if not key[1]:
                group.setSelected(True)
                return
            for row_index in range(group.childCount()):
                child = group.child(row_index)
                if child.text(0) == key[1]:
                    child.setSelected(True)
                    return

    def _rebuild(self, sections: tuple[InfoSection, ...]) -> None:
        """Редкий rebuild сохраняет раскрытие, прокрутку и выделенную строку."""

        expanded = {
            item.text(0)
            for index in range(self.topLevelItemCount())
            if (item := self.topLevelItem(index)) is not None and item.isExpanded()
        }
        scroll = self.verticalScrollBar().value()
        selected = self._selection_key()
        self.blockSignals(True)
        try:
            self.clear()
            for section in sections:
                group = QTreeWidgetItem([section.title, "", ""])
                for row in section.rows:
                    QTreeWidgetItem(group, [row.label, row.value, row.note])
                self.addTopLevelItem(group)
                group.setExpanded(not expanded or section.title in expanded)
            self._restore_selection(selected)
            self.verticalScrollBar().setValue(scroll)
        finally:
            self.blockSignals(False)


class DeviceInfoPanel(DockTab):
    """Паспорт/развёртка, качество связи и единственное место записи в прибор."""

    layout_key = "device"

    def __init__(
        self,
        controller: AppController,
        config_panel: DeviceConfigPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.config_panel = config_panel

        self.summary_tree = _SectionTree()
        self.quality_tree = _SectionTree()
        # Совместимость с прежними тестами/ручными скриптами: ``tree`` теперь
        # означает наблюдательное дерево паспорта, а динамика вынесена отдельно.
        self.tree = self.summary_tree

        self.summary_dock = self.add_panel_dock(
            "Паспорт и развёртка",
            self.summary_tree,
            "device.summary",
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.quality_dock = self.add_panel_dock(
            texts.SECTION_QUALITY,
            self.quality_tree,
            "device.quality",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.config_dock = self.add_panel_dock(
            texts.TAB_DEVICE_CONFIG,
            scrollable(config_panel),
            "device.config",
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.config_dock.setStyleSheet("QDockWidget::title { font-weight: bold; }")
        self.reset_layout()
        self.refresh(controller.snapshot())

    def _apply_default_splits(self) -> None:
        self.splitDockWidget(
            self.quality_dock,
            self.config_dock,
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.summary_dock, self.quality_dock],
            [620, 430],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.quality_dock, self.config_dock],
            [330, 330],
            Qt.Orientation.Vertical,
        )

    @staticmethod
    def _partition(
        sections: tuple[InfoSection, ...],
    ) -> tuple[tuple[InfoSection, ...], tuple[InfoSection, ...]]:
        dynamic_titles = {texts.SECTION_QUALITY, texts.SECTION_LOG}
        summary = tuple(section for section in sections if section.title not in dynamic_titles)
        quality = tuple(section for section in sections if section.title in dynamic_titles)
        return summary, quality

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Статика и счётчики обновляются независимо; настройка не пересоздаётся."""

        summary, quality = self._partition(models.device_sections(snapshot))
        self.summary_tree.update_sections(summary)
        self.quality_tree.update_sections(quality)
        self.config_panel.refresh(snapshot)
