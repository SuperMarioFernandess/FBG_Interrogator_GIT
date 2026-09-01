"""Панель журнала пакетов: таблица, фильтры по направлению и паре, экспорт.

Кольцо и снимок готовы в `fbg/io/packet_log.py` и читаются так же, как UI
читает pipeline: по таймеру, неизменяемой копией. Панель кольца **не держит** —
у неё в руках кортеж, отданный `snapshot()`, и он не меняется у неё под руками
(Р36). Это проверяется тестом: после обновления в журнал добавляются записи,
и число строк таблицы не меняется до следующего такта.

Обновление по таймеру, а не по событию. При 2000 Гц событийная перерисовка
утопила бы UI: журнал видит все датаграммы, включая телеметрию, если она
включена. Такт — 10 Гц, как у остальных панелей.

Таблица форматирует **ячейки**, а не строки: `QTableView` спрашивает только
видимые, и hex ответа `30 03` (20430 байт) не превращается в 61 КБ текста,
пока на него не смотрят. А когда смотрят — показ обрезается с явной пометкой,
потому что в файле и в экспорте байты лежат целиком (KB_05 №3).
"""

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from fbg.io.packet_log import Direction, PacketRecord
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.models import AppSnapshot


class PacketTableModel(QAbstractTableModel):
    """Таблица записей журнала поверх неизменяемого снимка кольца."""

    def __init__(self, wall_offset: float, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._records: tuple[PacketRecord, ...] = ()
        self._wall_offset = wall_offset

    @property
    def records(self) -> tuple[PacketRecord, ...]:
        """Показанные записи. Копия кольца, а не само кольцо."""
        return self._records

    def set_records(self, records: tuple[PacketRecord, ...]) -> None:
        """Заменяет содержимое таблицы новым снимком."""
        if records == self._records:
            return
        self.beginResetModel()
        self._records = records
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802 — Qt
        """Число строк."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._records)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802 — Qt
        """Число колонок."""
        if parent is not None and parent.isValid():
            return 0
        return models.LOG_COLUMN_COUNT

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        """Значение ячейки. Форматируется по запросу, только для видимых строк."""
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        record = self._records[index.row()]
        return models.packet_cell(record, index.column(), self._wall_offset)

    def headerData(  # noqa: N802 — Qt
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        """Заголовки колонок."""
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return texts.LOG_COLUMNS[section]


class PacketLogPanel(QWidget):
    """Журнал обмена: таблица, фильтры и выгрузка."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pairs: tuple[tuple[int, int], ...] = ()

        self.model = PacketTableModel(controller.wall_offset, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

        self.direction_box = QComboBox()
        self.direction_box.addItem(texts.FILTER_ANY, None)
        for direction in models.direction_choices():
            self.direction_box.addItem(direction.value, direction.value)

        self.pair_box = QComboBox()
        self.pair_box.addItem(texts.FILTER_ANY_PAIR, None)

        self.pause_box = QCheckBox(texts.LOG_PAUSED)
        self.pause_box.setToolTip(texts.LOG_PAUSED_HINT)
        self.export_button = QPushButton(texts.BUTTON_EXPORT)
        self.export_button.clicked.connect(self._on_export)
        self.status_label = QLabel()

        controls = QHBoxLayout()
        controls.addWidget(QLabel(texts.FILTER_DIRECTION))
        controls.addWidget(self.direction_box)
        controls.addWidget(QLabel(texts.FILTER_ID_FC))
        controls.addWidget(self.pair_box)
        controls.addWidget(self.pause_box)
        controls.addStretch(1)
        controls.addWidget(self.status_label)
        controls.addWidget(self.export_button)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.table, 1)

        self.refresh(controller.snapshot())

    # --- Фильтры -----------------------------------------------------------------------

    def selected_direction(self) -> Direction | None:
        """Выбранное направление или None — «любое».

        Данные элемента приходят из `QVariant`, а он типов Python не хранит:
        `StrEnum` возвращается обычной строкой. Поэтому в списке лежит строка,
        а перечисление собирается здесь.
        """
        data = self.direction_box.currentData()
        return None if data is None else Direction(str(data))

    def selected_pair(self) -> tuple[int, int] | None:
        """Выбранная пара (ID, FC) или None — «любая». Причина та же."""
        data = self.pair_box.currentData()
        return None if data is None else models.parse_id_fc_pair(str(data))

    def _sync_pairs(self, records: tuple[PacketRecord, ...]) -> None:
        """Обновляет список пар, сохраняя выбор пользователя.

        Список строится по тому, что реально встретилось: в журнале интереснее
        всего пары, которых в списке известных команд нет. Перезаполняется
        только при появлении новой — иначе выбор сбрасывался бы десять раз
        в секунду.
        """
        pairs = models.id_fc_choices(records)
        if pairs == self._pairs:
            return
        self._pairs = pairs
        current = self.pair_box.currentData()
        self.pair_box.blockSignals(True)
        self.pair_box.clear()
        self.pair_box.addItem(texts.FILTER_ANY_PAIR, None)
        for pair in pairs:
            label = models.format_id_fc_pair(pair)
            self.pair_box.addItem(label, label)
        if current is not None:
            index = self.pair_box.findData(current)
            self.pair_box.setCurrentIndex(max(index, 0))
        self.pair_box.blockSignals(False)

    # --- Обновление --------------------------------------------------------------------

    def refresh(self, snapshot: AppSnapshot) -> None:
        """Читает снимок кольца и обновляет таблицу. Таймер окна, 10 Гц."""
        if self.pause_box.isChecked():
            return
        everything = self._controller.packet_records()
        self._sync_pairs(everything)
        records = self._controller.packet_records(
            direction=self.selected_direction(), id_fc=self.selected_pair()
        )
        at_bottom = self._at_bottom()
        self.model.set_records(records)
        if at_bottom:
            self.table.scrollToBottom()
        if snapshot.log is not None:
            self.status_label.setText(
                f"{len(records)} из {len(everything)} · принято "
                f"{snapshot.log.records_in} · потеряно {snapshot.log.lost_records}"
            )

    def _at_bottom(self) -> bool:
        """Прокручена ли таблица вниз. Если да — держим её внизу."""
        scrollbar = self.table.verticalScrollBar()
        return scrollbar.value() >= scrollbar.maximum() - 1

    # --- Экспорт -----------------------------------------------------------------------

    def _on_export(self) -> None:
        """Выгружает кольцо с применённым фильтром."""
        suggested = models.export_suggested_name()
        chosen, _ = QFileDialog.getSaveFileName(
            self, texts.EXPORT_DIALOG_TITLE, suggested, texts.EXPORT_FILTER
        )
        if not chosen:
            return
        self.export_to(Path(chosen))

    def export_to(self, path: Path) -> int:
        """Экспорт в заданный файл. Отдельно от диалога — ради тестов."""
        try:
            return self._controller.export_packets(
                path, direction=self.selected_direction(), id_fc=self.selected_pair()
            )
        except OSError as exc:
            self._controller.note(f"журнал не выгружен: {exc}")
            return 0
