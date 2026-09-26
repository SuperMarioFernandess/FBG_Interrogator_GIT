"""Сжатая история графиков вне Qt и вне приёмного тракта.

Самописец читает уже принятые кадры последовательным ``FrameCursor`` из
``AppController.snapshot()``. Поток приёма ничего о нём не знает. Сырая
частота преобразуется в длину волны и складывается в неподвижные интервалы
по 100 мс: среднее, минимум, максимум, σ и число валидных кадров. Арифметику
окон задаёт ``core.averaging.fixed_window_average`` — второй реализации
временной сетки здесь нет.

Поверх базовых интервалов строится **пирамида разрешений** (Р88): уровни
1 с, 10 с, 100 с, 1000 с. Свёртка точная: ``min`` — минимум минимумов, ``max``
— максимум максимумов, ``mean`` — среднее, взвешенное по ``n``, σ — через
объединённую дисперсию. Сегмент самописца входит в ключ группы, поэтому
интервал, накрывающий «Стоп → Запуск» или ``# GAP``, режется, а не склеивает
данные по обе стороны. Для экрана выбирается уровень, дающий не больше
``MAX_VIEW_POINTS`` точек на видимую ширину, и копируется только видимая
область с полями: стоимость такта не зависит от длины опыта (№44).

Номер интервала сетки хранится **целым числом** для каждой строки. Восстанавливать
его из времени начала нельзя: ``floor(4.3 / 0.1) == 42``, и именно так чат №20
дважды записывал один интервал и терял огибающую соседнего.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fbg.core.averaging import fixed_window_average
from fbg.core.pipeline import FrameBatch
from fbg.core.profile import C_NM_GHZ

HISTORY_INTERVAL_S = 0.1
"""Разрешение самописца: 200 raw-кадров при штатных 2 кГц."""

DEFAULT_HISTORY_DEPTH_S = 86_400.0
MIN_HISTORY_DEPTH_S = 1.0
MAX_HISTORY_DEPTH_S = 7 * 86_400.0

PYRAMID_FACTOR = 10
"""Во сколько раз каждый уровень пирамиды грубее предыдущего."""

PYRAMID_LEVELS = 5
"""0.1 · 1 · 10 · 100 · 1000 с. Неделя на верхнем уровне — 605 точек."""

MAX_VIEW_POINTS = 1000
"""Предел точек на **видимую** ширину графика: порядка ширины в пикселях."""

VIEW_MARGIN_FRACTION = 0.5
"""Поле с каждой стороны видимой области, в долях её ширины.

Нужно, чтобы панорама остановленного графика не показывала пустоту до
следующего такта и не перестраивала модель на каждом сдвиге мыши.
"""

VIEW_TILE_ROWS = 100
"""Границы копируемой области квантуются плитками этого числа строк уровня."""

VIEW_ROW_LIMIT = int(
    MAX_VIEW_POINTS * (1.0 + 2.0 * VIEW_MARGIN_FRACTION)
    + 2 * (VIEW_TILE_ROWS + round(5.0 / HISTORY_INTERVAL_S))
    + 2
)
"""Структурный предел строк одного снимка на кривую при непрерывном сегменте.

Видимая область даёт не больше ``MAX_VIEW_POINTS``, поля — ещё столько же,
квантование плитками — по плитке и одному окну усреднения (до 5 с = 50
базовых интервалов) с каждой стороны, плюс незакрытый интервал в конце.
Каждая граница сегмента внутри области добавляет не больше двух строк:
разрезанный интервал и разрыв линии. От длины истории предел не зависит.
"""

_BLOCK_ROWS = 1024
_ROW_COMPLETE_TOLERANCE_S = 1e-6

Slot = tuple[int, int]

VIEW_FOLLOW = "follow"
VIEW_ALL = "all"
VIEW_MANUAL = "manual"


def level_bins(level: int) -> int:
    """Сколько базовых интервалов 100 мс в одном интервале уровня."""
    return int(PYRAMID_FACTOR**level)


def level_interval_s(level: int) -> float:
    """Длительность интервала уровня пирамиды, с."""
    return HISTORY_INTERVAL_S * level_bins(level)


@dataclass(frozen=True)
class HistoryViewRequest:
    """Какую область истории показывает график: режим оси X оператора (Р87).

    ``follow`` — последние ``span_s`` секунд, ``all`` — вся история, ``manual``
    — диапазон ``low_s … high_s`` в координатах графика (секунды от начала
    истории). Запрос не содержит данных и ничего не меняет в самописце.
    """

    kind: str = VIEW_ALL
    span_s: float = 60.0
    low_s: float = 0.0
    high_s: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {VIEW_FOLLOW, VIEW_ALL, VIEW_MANUAL}:
            raise ValueError(f"неизвестный режим области графика: {self.kind!r}")
        if self.kind == VIEW_FOLLOW and not (math.isfinite(self.span_s) and self.span_s > 0.0):
            raise ValueError("span_s должен быть положительным")
        if self.kind == VIEW_MANUAL and not (
            math.isfinite(self.low_s) and math.isfinite(self.high_s) and self.high_s > self.low_s
        ):
            raise ValueError("ручной диапазон должен быть конечным и возрастающим")


@dataclass(frozen=True)
class CompressedHistorySnapshot:
    """Самодостаточная копия выбранных линий сжатой истории.

    Строки относятся к одному уровню пирамиды: ``interval_s`` — его интервал,
    ``base_bin`` — номер базового 100-мс интервала, с которого начинается
    интервал строки на её уровне. Поля после ``origin_mono`` описывают, какая
    область истории скопирована; у снимков, собранных вручную в тестах, они
    имеют умолчания прежнего, полного снимка.
    """

    positions: tuple[Slot, ...]
    start_mono: np.ndarray
    stop_mono: np.ndarray
    segment: np.ndarray
    mean_nm: np.ndarray
    min_nm: np.ndarray
    max_nm: np.ndarray
    sigma_nm: np.ndarray
    n: np.ndarray
    first_valid_nm: np.ndarray
    running: bool
    interval_s: float
    depth_s: float
    version: int
    origin_mono: float | None
    level: int = 0
    base_bin: np.ndarray | None = None
    first_mono: float | None = None
    last_mono: float | None = None
    current_segment: int | None = None
    includes_tail: bool = True
    view_key: tuple[object, ...] = ()

    @property
    def windows(self) -> int:
        return int(self.start_mono.size)

    @property
    def span_s(self) -> float:
        if self.windows == 0:
            return 0.0
        return float(self.stop_mono[-1] - self.start_mono[0])

    @property
    def t_mono(self) -> np.ndarray:
        """Центры интервалов для отрисовки."""
        return (self.start_mono + self.stop_mono) / 2.0


@dataclass(frozen=True)
class HistoryAggregate:
    """Та же история после точного объединения интервалов по ``n``."""

    start_mono: np.ndarray
    stop_mono: np.ndarray
    segment: np.ndarray
    mean_nm: np.ndarray
    min_nm: np.ndarray
    max_nm: np.ndarray
    sigma_nm: np.ndarray
    n: np.ndarray

    @property
    def windows(self) -> int:
        return int(self.start_mono.size)


# --------------------------------------------------------------------------------------
# Точная свёртка интервалов
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pooled:
    n: np.ndarray
    mean: np.ndarray
    sigma: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def pool_intervals(
    group_first: np.ndarray,
    n: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> _Pooled:
    """Объединяет подряд идущие группы строк ``(rows, columns)`` без потери точности.

    ``group_first`` — индексы первых строк групп, возрастающие, первый равен 0.
    Строка с ``n = 0`` вклада не даёт ни во что; группа из одних пустых строк
    даёт ``mean = σ = min = max = NaN`` и ``n = 0`` (№7, №41). σ — population,
    как и у исходных интервалов (Р82).
    """
    counts = np.asarray(n, dtype=np.int64)
    valid = counts > 0
    groups = int(group_first.size)
    columns = counts.shape[1]
    if groups == 0:
        empty = np.empty((0, columns), dtype=np.float64)
        return _Pooled(np.empty((0, columns), np.int64), empty, empty, empty, empty)
    total = np.add.reduceat(counts, group_first, axis=0)
    safe_mean = np.where(valid, mean, 0.0)
    weighted = np.add.reduceat(safe_mean * counts, group_first, axis=0)
    has = total > 0
    pooled_mean = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(weighted, total, out=pooled_mean, where=has)

    sizes = np.diff(np.r_[group_first, counts.shape[0]])
    group_mean = np.repeat(np.where(has, pooled_mean, 0.0), sizes, axis=0)
    safe_sigma = np.where(valid & np.isfinite(sigma), sigma, 0.0)
    deviation = np.where(valid, mean - group_mean, 0.0)
    terms = counts * (safe_sigma * safe_sigma + deviation * deviation)
    second = np.add.reduceat(terms, group_first, axis=0)
    pooled_sigma = np.full(total.shape, np.nan, dtype=np.float64)
    variance = np.zeros(total.shape, dtype=np.float64)
    np.divide(second, total, out=variance, where=has)
    pooled_sigma[has] = np.sqrt(np.maximum(variance[has], 0.0))

    low = np.minimum.reduceat(np.where(valid, minimum, np.inf), group_first, axis=0)
    high = np.maximum.reduceat(np.where(valid, maximum, -np.inf), group_first, axis=0)
    low[~has] = np.nan
    high[~has] = np.nan
    return _Pooled(total, pooled_mean, pooled_sigma, low, high)


def _group_starts(*keys: np.ndarray) -> np.ndarray:
    """Индексы первых строк групп подряд идущих одинаковых ключей."""
    rows = int(keys[0].size)
    if rows == 0:
        return np.empty(0, dtype=np.intp)
    change = np.zeros(rows, dtype=bool)
    change[0] = True
    for key in keys:
        change[1:] |= key[1:] != key[:-1]
    return np.flatnonzero(change)


# --------------------------------------------------------------------------------------
# Хранение уровня
# --------------------------------------------------------------------------------------


@dataclass
class _Rows:
    """Плотная копия строк уровня для выбранных позиций."""

    start: np.ndarray
    stop: np.ndarray
    segment: np.ndarray
    base_bin: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    sigma: np.ndarray
    n: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.start.size)

    @staticmethod
    def empty(columns: int) -> _Rows:
        values = np.empty((0, columns), dtype=np.float64)
        return _Rows(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            values,
            values.copy(),
            values.copy(),
            values.copy(),
            np.empty((0, columns), dtype=np.int64),
        )

    def take(self, selection: np.ndarray | slice) -> _Rows:
        return _Rows(
            self.start[selection],
            self.stop[selection],
            self.segment[selection],
            self.base_bin[selection],
            self.mean[selection],
            self.minimum[selection],
            self.maximum[selection],
            self.sigma[selection],
            self.n[selection],
        )

    @staticmethod
    def concat(parts: Sequence[_Rows], columns: int) -> _Rows:
        present = [part for part in parts if part.rows]
        if not present:
            return _Rows.empty(columns)
        if len(present) == 1:
            return present[0]
        return _Rows(
            np.concatenate([part.start for part in present]),
            np.concatenate([part.stop for part in present]),
            np.concatenate([part.segment for part in present]),
            np.concatenate([part.base_bin for part in present]),
            np.concatenate([part.mean for part in present], axis=0),
            np.concatenate([part.minimum for part in present], axis=0),
            np.concatenate([part.maximum for part in present], axis=0),
            np.concatenate([part.sigma for part in present], axis=0),
            np.concatenate([part.n for part in present], axis=0),
        )


class _Block:
    """Фиксированный кусок уровня; данные позиции выделяются только при наличии."""

    __slots__ = ("base_bin", "first_row", "rows", "segment", "start", "stop", "values")

    def __init__(self, first_row: int) -> None:
        self.first_row = first_row
        self.rows = 0
        self.start = np.empty(_BLOCK_ROWS, dtype=np.float64)
        self.stop = np.empty(_BLOCK_ROWS, dtype=np.float64)
        self.segment = np.empty(_BLOCK_ROWS, dtype=np.int64)
        self.base_bin = np.empty(_BLOCK_ROWS, dtype=np.int64)
        self.values: dict[
            Slot, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    @property
    def free(self) -> int:
        return _BLOCK_ROWS - self.rows

    def extend(self, rows: _Rows, positions: tuple[Slot, ...], offset: int, count: int) -> None:
        begin = self.rows
        target = slice(begin, begin + count)
        source = slice(offset, offset + count)
        self.start[target] = rows.start[source]
        self.stop[target] = rows.stop[source]
        self.segment[target] = rows.segment[source]
        self.base_bin[target] = rows.base_bin[source]
        present = np.flatnonzero(np.any(rows.n[source] > 0, axis=0))
        for column_raw in present:
            column = int(column_raw)
            slot = positions[column]
            arrays = self.values.get(slot)
            if arrays is None:
                arrays = (
                    np.full(_BLOCK_ROWS, np.nan, dtype=np.float64),
                    np.full(_BLOCK_ROWS, np.nan, dtype=np.float64),
                    np.full(_BLOCK_ROWS, np.nan, dtype=np.float64),
                    np.full(_BLOCK_ROWS, np.nan, dtype=np.float64),
                    np.zeros(_BLOCK_ROWS, dtype=np.int64),
                )
                self.values[slot] = arrays
            arrays[0][target] = rows.mean[source, column]
            arrays[1][target] = rows.minimum[source, column]
            arrays[2][target] = rows.maximum[source, column]
            arrays[3][target] = rows.sigma[source, column]
            arrays[4][target] = rows.n[source, column]
        self.rows += count


class _Level:
    """Один уровень пирамиды: блоки строк с глобальной нумерацией."""

    def __init__(self, level: int) -> None:
        self.level = level
        self.bins = level_bins(level)
        self.blocks: list[_Block] = []
        self.block_first_bin: list[int] = []
        self.end_row = 0
        self.open_start = 0
        """Глобальный номер первой строки, ещё не свёрнутой в следующий уровень."""

    @property
    def first_row(self) -> int:
        return self.blocks[0].first_row if self.blocks else self.end_row

    @property
    def rows(self) -> int:
        return self.end_row - self.first_row

    def last(self) -> tuple[float, int] | None:
        """Конец последней строки и её номер интервала."""
        if not self.blocks:
            return None
        block = self.blocks[-1]
        return float(block.stop[block.rows - 1]), int(block.base_bin[block.rows - 1])

    def extend(self, rows: _Rows, positions: tuple[Slot, ...]) -> None:
        offset = 0
        total = rows.rows
        while offset < total:
            if not self.blocks or self.blocks[-1].free == 0:
                self.blocks.append(_Block(self.end_row))
                self.block_first_bin.append(int(rows.base_bin[offset]))
            block = self.blocks[-1]
            count = min(block.free, total - offset)
            block.extend(rows, positions, offset, count)
            offset += count
            self.end_row += count

    def gather(self, first: int, end: int, positions: Sequence[Slot]) -> _Rows:
        """Плотная копия глобальных строк ``[first, end)`` для позиций."""
        first = max(first, self.first_row)
        end = min(end, self.end_row)
        columns = len(positions)
        if end <= first:
            return _Rows.empty(columns)
        size = end - first
        out = _Rows(
            np.empty(size, dtype=np.float64),
            np.empty(size, dtype=np.float64),
            np.empty(size, dtype=np.int64),
            np.empty(size, dtype=np.int64),
            np.full((size, columns), np.nan, dtype=np.float64),
            np.full((size, columns), np.nan, dtype=np.float64),
            np.full((size, columns), np.nan, dtype=np.float64),
            np.full((size, columns), np.nan, dtype=np.float64),
            np.zeros((size, columns), dtype=np.int64),
        )
        index = self._block_index(first)
        cursor = 0
        while cursor < size:
            block = self.blocks[index]
            local_begin = first + cursor - block.first_row
            count = min(block.rows - local_begin, size - cursor)
            src = slice(local_begin, local_begin + count)
            dst = slice(cursor, cursor + count)
            out.start[dst] = block.start[src]
            out.stop[dst] = block.stop[src]
            out.segment[dst] = block.segment[src]
            out.base_bin[dst] = block.base_bin[src]
            for column, slot in enumerate(positions):
                arrays = block.values.get(slot)
                if arrays is None:
                    continue
                out.mean[dst, column] = arrays[0][src]
                out.minimum[dst, column] = arrays[1][src]
                out.maximum[dst, column] = arrays[2][src]
                out.sigma[dst, column] = arrays[3][src]
                out.n[dst, column] = arrays[4][src]
            cursor += count
            index += 1
        return out

    def _block_index(self, row: int) -> int:
        starts = [block.first_row for block in self.blocks]
        return max(0, bisect.bisect_right(starts, row) - 1)

    def row_for_bin(self, base_bin: int) -> int:
        """Глобальный номер первой строки с номером интервала ``>= base_bin``."""
        if not self.blocks:
            return self.end_row
        index = max(0, bisect.bisect_right(self.block_first_bin, base_bin) - 1)
        while index < len(self.blocks):
            block = self.blocks[index]
            local = int(np.searchsorted(block.base_bin[: block.rows], base_bin, side="left"))
            if local < block.rows:
                return block.first_row + local
            index += 1
        return self.end_row

    def first_row_with(self, values_name: str, cutoff: float) -> int:
        """Первая строка, у которой ``start``/``stop`` не раньше ``cutoff``."""
        for block in self.blocks:
            values = getattr(block, values_name)[: block.rows]
            if values[-1] >= cutoff:
                local = int(np.searchsorted(values, cutoff, side="left"))
                return block.first_row + local
        return self.end_row

    def trim(self, values_name: str, cutoff: float) -> None:
        """Отбрасывает целые блоки, у которых последняя строка раньше ``cutoff``."""
        while len(self.blocks) > 1:
            block = self.blocks[0]
            if float(getattr(block, values_name)[block.rows - 1]) >= cutoff:
                break
            self.blocks.pop(0)
            self.block_first_bin.pop(0)
        self.open_start = max(self.open_start, self.first_row)


# --------------------------------------------------------------------------------------
# Самописец
# --------------------------------------------------------------------------------------


def choose_view_level(
    width_s: float,
    averaging_window_s: float | None,
    max_points: int = MAX_VIEW_POINTS,
) -> tuple[int, bool]:
    """Уровень пирамиды для видимой ширины и признак свёртки окнами усреднения.

    Без усреднения — самый подробный уровень, дающий не больше ``max_points``
    точек. С усреднением окно ``W`` задаёт наименьший интервал на экране: если
    окна W помещаются, строки берутся с самого грубого уровня, кратного W,
    и модель сворачивает их в окна W точно. Если не помещаются — берётся
    уровень грубее W: мельче чем W показывать нельзя, а мельче чем пиксель
    незачем.
    """
    width = max(float(width_s), HISTORY_INTERVAL_S)
    top = PYRAMID_LEVELS - 1
    if averaging_window_s is not None and can_aggregate_exactly(averaging_window_s):
        window_bins = round(averaging_window_s / HISTORY_INTERVAL_S)
        exact_level = 0
        while exact_level < top and window_bins % level_bins(exact_level + 1) == 0:
            exact_level += 1
        if width / level_interval_s(exact_level) <= max_points:
            return exact_level, True
        for level in range(top + 1):
            interval = level_interval_s(level)
            if interval > averaging_window_s and width / interval <= max_points:
                return level, False
        return top, False
    for level in range(top + 1):
        if width / level_interval_s(level) <= max_points:
            return level, False
    return top, False


class CompressedHistoryRecorder:
    """Самописец фиксированной сетки для всех слотов прибора.

    Объект Qt-свободен. ``AppController`` создаёт отдельный экземпляр для
    графика измерения и графика датчиков, потому что их Start/Stop независимы.
    """

    def __init__(
        self,
        channels: int,
        positions_per_channel: int,
        *,
        interval_s: float = HISTORY_INTERVAL_S,
        depth_s: float = DEFAULT_HISTORY_DEPTH_S,
    ) -> None:
        if channels < 1 or positions_per_channel < 1:
            raise ValueError("геометрия истории должна быть положительной")
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("interval_s должен быть положительным")
        if abs(interval_s - HISTORY_INTERVAL_S) > 1e-12:
            # Пирамида и окна усреднения считают номера интервалов в единицах
            # 100 мс. Иная сетка потребовала бы второго набора констант.
            raise ValueError(f"interval_s самописца фиксирован: {HISTORY_INTERVAL_S} с (Р85)")
        self._positions = tuple(
            (channel, position)
            for channel in range(channels)
            for position in range(positions_per_channel)
        )
        self._columns = len(self._positions)
        self._interval_s = float(interval_s)
        self._depth_s = self._validated_depth(depth_s)
        self._levels = [_Level(level) for level in range(PYRAMID_LEVELS)]
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)
        self._first_valid: dict[Slot, float] = {}
        self._segment_first_valid: dict[tuple[int, Slot], float] = {}
        self._segment_first_valid_bin: dict[tuple[int, Slot], int] = {}
        self._active: set[Slot] = set()
        self._running = False
        self._started_once = False
        self._segment = 0
        self._origin_mono: float | None = None
        self._version = 0

    @staticmethod
    def _validated_depth(depth_s: float) -> float:
        value = float(depth_s)
        if not math.isfinite(value) or not MIN_HISTORY_DEPTH_S <= value <= MAX_HISTORY_DEPTH_S:
            raise ValueError(
                f"depth_s должен быть в диапазоне {MIN_HISTORY_DEPTH_S}…{MAX_HISTORY_DEPTH_S}"
            )
        return value

    @property
    def running(self) -> bool:
        return self._running

    @property
    def depth_s(self) -> float:
        return self._depth_s

    @property
    def interval_s(self) -> float:
        return self._interval_s

    @property
    def version(self) -> int:
        return self._version

    @property
    def active_positions(self) -> tuple[Slot, ...]:
        return tuple(slot for slot in self._positions if slot in self._active)

    @property
    def segment(self) -> int:
        return self._segment

    def level_rows(self, level: int) -> int:
        """Сколько закрытых строк хранит уровень; нужно тестам и оценкам."""
        return self._levels[level].rows

    def first_valid_nm(self, slot: Slot, *, segment: int | None = None) -> float | None:
        if segment is None:
            return self._first_valid.get(slot)
        return self._segment_first_valid.get((segment, slot))

    def start(self) -> None:
        if self._running:
            return
        if self._started_once:
            self._segment += 1
        self._started_once = True
        self._running = True
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)
        self._version += 1

    def stop(self) -> None:
        if not self._running:
            return
        self._finish_segment()
        self._running = False
        self._version += 1

    def clear(self) -> None:
        self._levels = [_Level(level) for level in range(PYRAMID_LEVELS)]
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)
        self._first_valid.clear()
        self._segment_first_valid.clear()
        self._segment_first_valid_bin.clear()
        self._active.clear()
        self._origin_mono = None
        if self._running:
            self._segment += 1
        else:
            self._segment = 0
            self._started_once = False
        self._version += 1

    def set_depth(self, depth_s: float) -> None:
        value = self._validated_depth(depth_s)
        if value == self._depth_s:
            return
        self._depth_s = value
        self._trim()
        self._version += 1

    # --- Приём ---------------------------------------------------------------------

    def ingest_batch(self, batch: FrameBatch) -> None:
        if not self._running or len(batch) == 0:
            return
        if batch.gap:
            self._finish_segment()
            self._segment += 1
        freq = np.asarray(batch.freq_ghz, dtype=np.float64)
        matrix = freq.reshape(freq.shape[0], self._columns)
        nm = np.full(matrix.shape, np.nan, dtype=np.float64)
        np.divide(C_NM_GHZ, matrix, out=nm, where=np.isfinite(matrix) & (matrix > 0.0))
        self.ingest(np.asarray(batch.t_mono, dtype=np.float64), nm)

    def ingest(self, t_mono: np.ndarray, wavelength_nm: np.ndarray) -> None:
        """Добавляет raw-копию; предназначено также для Qt-free тестов."""
        if not self._running:
            return
        times = np.asarray(t_mono, dtype=np.float64)
        matrix = np.asarray(wavelength_nm, dtype=np.float64)
        if times.ndim != 1 or matrix.shape != (times.size, self._columns):
            raise ValueError("wavelength_nm должен иметь форму (кадры, все позиции)")
        if times.size == 0:
            return
        if np.any(np.diff(times) < 0.0):
            raise ValueError("t_mono должен быть отсортирован")
        self._remember_first_valid(matrix)
        if self._pending_t.size:
            times = np.concatenate((self._pending_t, times))
            matrix = np.concatenate((self._pending_nm, matrix), axis=0)
        self._consume(times, matrix, final=False)

    def _remember_first_valid(self, matrix: np.ndarray) -> None:
        finite = np.isfinite(matrix)
        columns = np.flatnonzero(np.any(finite, axis=0))
        for column_raw in columns:
            column = int(column_raw)
            slot = self._positions[column]
            if slot in self._first_valid and (self._segment, slot) in self._segment_first_valid:
                continue
            value = float(matrix[int(np.argmax(finite[:, column])), column])
            self._first_valid.setdefault(slot, value)
            self._segment_first_valid.setdefault((self._segment, slot), value)

    def _finish_segment(self) -> None:
        if self._pending_t.size:
            self._consume(self._pending_t, self._pending_nm, final=True)
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)

    def _consume(self, times: np.ndarray, matrix: np.ndarray, *, final: bool) -> None:
        averaged = fixed_window_average(times, matrix, self._interval_s)
        complete = np.flatnonzero(averaged.complete)
        sample_bins = np.floor(times / self._interval_s).astype(np.int64)
        if complete.size:
            # Номер интервала восстанавливается округлением, а не floor: начало
            # окна равно ``bin * 0.1`` в двоичной арифметике, и floor от него
            # даёт соседний интервал примерно в каждом десятом случае. Отсчёты
            # относятся к интервалам той же функцией, что в core.averaging.
            bins = np.rint(averaged.start_mono[complete] / self._interval_s).astype(np.int64)
            lower = np.searchsorted(sample_bins, bins, side="left")
            upper = np.searchsorted(sample_bins, bins, side="right")
            minimum = np.full((complete.size, self._columns), np.nan, dtype=np.float64)
            maximum = np.full((complete.size, self._columns), np.nan, dtype=np.float64)
            finite = np.isfinite(matrix)
            for row, (begin, end) in enumerate(zip(lower, upper, strict=True)):
                if end <= begin:
                    continue
                block = matrix[begin:end]
                mask = finite[begin:end]
                has = np.any(mask, axis=0)
                if np.any(has):
                    minimum[row, has] = np.min(np.where(mask, block, np.inf), axis=0)[has]
                    maximum[row, has] = np.max(np.where(mask, block, -np.inf), axis=0)[has]
            self._append_rows(
                _Rows(
                    averaged.start_mono[complete].copy(),
                    averaged.stop_mono[complete].copy(),
                    np.full(complete.size, self._segment, dtype=np.int64),
                    bins,
                    averaged.mean[complete],
                    minimum,
                    maximum,
                    averaged.sigma[complete],
                    averaged.n[complete].astype(np.int64, copy=False),
                )
            )
            keep = sample_bins > int(bins[-1])
            self._pending_t = times[keep].copy()
            self._pending_nm = matrix[keep].copy()
        else:
            self._pending_t = times.copy()
            self._pending_nm = matrix.copy()
        if final:
            self._pending_t = np.empty(0, dtype=np.float64)
            self._pending_nm = np.empty((0, self._columns), dtype=np.float64)

    def append_intervals(
        self,
        start_mono: np.ndarray,
        stop_mono: np.ndarray,
        mean_nm: np.ndarray,
        min_nm: np.ndarray,
        max_nm: np.ndarray,
        sigma_nm: np.ndarray,
        n: np.ndarray,
    ) -> None:
        """Добавляет готовые базовые интервалы текущего сегмента.

        Это тот же путь, которым идёт ``ingest``, без свёртки raw-кадров.
        Нужен тестам, которым требуется история в сутки: собирать её из
        172 млн кадров незачем, а арифметика пирамиды та же.
        """
        if not self._running:
            return
        starts = np.asarray(start_mono, dtype=np.float64)
        count = starts.size
        shape = (count, self._columns)
        arrays = [np.asarray(item) for item in (mean_nm, min_nm, max_nm, sigma_nm, n)]
        if any(item.shape != shape for item in arrays):
            raise ValueError("интервалы должны иметь форму (строки, все позиции)")
        if count == 0:
            return
        bins = np.rint(starts / self._interval_s).astype(np.int64)
        if np.any(np.diff(bins) <= 0):
            raise ValueError("интервалы должны идти по возрастанию без повторов")
        last = self._levels[0].last()
        if last is not None and int(bins[0]) <= last[1]:
            raise ValueError("интервал уже есть в истории")
        self._append_rows(
            _Rows(
                starts.copy(),
                np.asarray(stop_mono, dtype=np.float64).copy(),
                np.full(count, self._segment, dtype=np.int64),
                bins,
                arrays[0].astype(np.float64),
                arrays[1].astype(np.float64),
                arrays[2].astype(np.float64),
                arrays[3].astype(np.float64),
                arrays[4].astype(np.int64),
            )
        )

    def _append_rows(self, rows: _Rows) -> None:
        if self._origin_mono is None:
            self._origin_mono = float(rows.start[0])
        valid = rows.n > 0
        for column_raw in np.flatnonzero(np.any(valid, axis=0)):
            column = int(column_raw)
            slot = self._positions[column]
            self._active.add(slot)
            key = (self._segment, slot)
            if key not in self._segment_first_valid_bin:
                first = int(np.argmax(valid[:, column]))
                self._segment_first_valid_bin[key] = int(rows.base_bin[first])
        self._levels[0].extend(rows, self._positions)
        self._cascade(0)
        self._version += 1
        self._trim()

    def _cascade(self, index: int) -> None:
        """Закрывает интервалы следующего уровня, в которые пришла строка из нового."""
        if index + 1 >= len(self._levels):
            return
        child = self._levels[index]
        parent = self._levels[index + 1]
        first = max(child.open_start, child.first_row)
        if child.end_row - first <= 1:
            return
        keys = child.gather(first, child.end_row, ())
        parent_bin = keys.base_bin // parent.bins
        starts = _group_starts(keys.segment, parent_bin)
        if starts.size <= 1:
            return
        closed_end = first + int(starts[-1])
        rows = child.gather(first, closed_end, self._positions)
        pooled = self._pool(rows, starts[:-1], parent.bins)
        parent.extend(pooled, self._positions)
        child.open_start = closed_end
        self._cascade(index + 1)

    @staticmethod
    def _pool(rows: _Rows, group_first: np.ndarray, bins: int) -> _Rows:
        stats = pool_intervals(
            group_first, rows.n, rows.mean, rows.sigma, rows.minimum, rows.maximum
        )
        group_last = np.r_[group_first[1:], rows.rows] - 1
        return _Rows(
            rows.start[group_first].copy(),
            rows.stop[group_last].copy(),
            rows.segment[group_first].copy(),
            (rows.base_bin[group_first] // bins) * bins,
            stats.mean,
            stats.minimum,
            stats.maximum,
            stats.sigma,
            stats.n,
        )

    def _tail(self, level: int, positions: Sequence[Slot]) -> _Rows:
        """Незакрытые интервалы уровня, досчитанные из строк уровнем ниже.

        Их не больше двух на уровень: интервал, закрытию которого ещё не пришла
        строка нового, и уже начавшийся следующий. Досчёт стоит порядка десяти
        строк нижнего уровня на уровень, от длины истории не зависит.
        """
        columns = len(positions)
        if level == 0:
            return _Rows.empty(columns)
        child = self._levels[level - 1]
        rows = _Rows.concat(
            (
                child.gather(max(child.open_start, child.first_row), child.end_row, positions),
                self._tail(level - 1, positions),
            ),
            columns,
        )
        if rows.rows == 0:
            return rows
        bins = level_bins(level)
        starts = _group_starts(rows.segment, rows.base_bin // bins)
        return self._pool(rows, starts, bins)

    def _trim(self) -> None:
        last = self._levels[0].last()
        if last is None:
            return
        cutoff = last[0] - self._depth_s
        for index, level in enumerate(self._levels):
            level.trim("stop" if index == 0 else "start", cutoff)

    # --- Снимки -----------------------------------------------------------------------

    def _level_rows(
        self,
        level: int,
        positions: Sequence[Slot],
        first_bin: int,
        end_bin: int,
    ) -> _Rows:
        """Строки уровня с номером интервала в ``[first_bin, end_bin)`` в пределах глубины."""
        store = self._levels[level]
        first = store.row_for_bin(first_bin)
        end = store.row_for_bin(end_bin)
        last = self._levels[0].last()
        cutoff = -math.inf if last is None else last[0] - self._depth_s
        # База держит интервал, пока не кончился; грубый уровень — только
        # целиком внутри глубины, иначе в нём были бы уже вытесненные кадры.
        live = store.first_row_with("stop" if level == 0 else "start", cutoff)
        rows = store.gather(max(first, live), end, positions)
        tail = self._tail(level, positions)
        if tail.rows:
            inside = (tail.base_bin >= first_bin) & (tail.base_bin < end_bin)
            inside &= tail.start >= cutoff
            rows = _Rows.concat((rows, tail.take(inside)), len(positions))
        return rows

    def _snapshot_from_rows(
        self,
        rows: _Rows,
        selected: tuple[Slot, ...],
        *,
        level: int,
        includes_tail: bool,
        first_mono: float | None,
        last_mono: float | None,
        view_key: tuple[object, ...],
    ) -> CompressedHistorySnapshot:
        first_valid = np.asarray(
            [self._first_valid.get(slot, np.nan) for slot in selected], dtype=np.float64
        )
        return CompressedHistorySnapshot(
            positions=selected,
            start_mono=rows.start,
            stop_mono=rows.stop,
            segment=rows.segment,
            mean_nm=rows.mean,
            min_nm=rows.minimum,
            max_nm=rows.maximum,
            sigma_nm=rows.sigma,
            n=rows.n,
            first_valid_nm=first_valid,
            running=self._running,
            interval_s=level_interval_s(level),
            depth_s=self._depth_s,
            version=self._version,
            origin_mono=self._origin_mono,
            level=level,
            base_bin=rows.base_bin,
            first_mono=first_mono,
            last_mono=last_mono,
            current_segment=self._segment,
            includes_tail=includes_tail,
            view_key=view_key,
        )

    def _bounds(self) -> tuple[float, float, int] | None:
        """Начало живых данных, конец истории и номер последнего интервала."""
        base = self._levels[0]
        last = base.last()
        if last is None:
            return None
        live = base.first_row_with("stop", last[0] - self._depth_s)
        first = base.gather(live, live + 1, ())
        if first.rows == 0:
            return None
        return float(first.start[0]), last[0], last[1]

    def _select(self, positions: Sequence[Slot] | None) -> tuple[Slot, ...]:
        requested = self.active_positions if positions is None else tuple(positions)
        return tuple(slot for slot in requested if slot in self._active)

    def snapshot(self, positions: Sequence[Slot] | None = None) -> CompressedHistorySnapshot:
        """Полная копия базового уровня. **Не для такта интерфейса** — стоит O(истории)."""
        selected = self._select(positions)
        bounds = self._bounds()
        if bounds is None:
            return self._snapshot_from_rows(
                _Rows.empty(len(selected)),
                selected,
                level=0,
                includes_tail=True,
                first_mono=None,
                last_mono=None,
                view_key=("full",),
            )
        rows = self._level_rows(0, selected, -(2**62), 2**62)
        return self._snapshot_from_rows(
            rows,
            selected,
            level=0,
            includes_tail=True,
            first_mono=bounds[0],
            last_mono=bounds[1],
            view_key=("full",),
        )

    def view(
        self,
        positions: Sequence[Slot] | None,
        request: HistoryViewRequest | None = None,
        *,
        averaging_window_s: float | None = None,
        max_points: int = MAX_VIEW_POINTS,
    ) -> CompressedHistorySnapshot:
        """Копия видимой области на уровне, помещающемся в пиксели (Р88).

        Объём копии ограничен ``VIEW_ROW_LIMIT`` строк на позицию и от длины
        истории не зависит. Уровень выбирается по той части видимой области,
        где данные есть: отдалённый мышью график с часом данных посередине
        недели получает час данных в полном допустимом разрешении.
        """
        selected = self._select(positions)
        request = request or HistoryViewRequest()
        bounds = self._bounds()
        origin = self._origin_mono
        if bounds is None or origin is None:
            return self._snapshot_from_rows(
                _Rows.empty(len(selected)),
                selected,
                level=0,
                includes_tail=True,
                first_mono=None,
                last_mono=None,
                view_key=("empty",),
            )
        data_low, data_high, last_bin = bounds
        if request.kind == VIEW_FOLLOW:
            low, high = data_high - request.span_s, data_high
        elif request.kind == VIEW_MANUAL:
            low, high = origin + request.low_s, origin + request.high_s
        else:
            low, high = data_low, data_high
        low, high = max(low, data_low), min(high, data_high)
        window = (
            averaging_window_s
            if averaging_window_s is not None and can_aggregate_exactly(averaging_window_s)
            else None
        )
        if high <= low:
            level, _aggregate = choose_view_level(HISTORY_INTERVAL_S, window, max_points)
            return self._snapshot_from_rows(
                _Rows.empty(len(selected)),
                selected,
                level=level,
                includes_tail=False,
                first_mono=data_low,
                last_mono=data_high,
                view_key=("outside", level),
            )
        width = high - low
        level, aggregate = choose_view_level(width, window, max_points)
        bins = level_bins(level)
        tile = bins * VIEW_TILE_ROWS
        if aggregate and window is not None:
            window_bins = round(window / HISTORY_INTERVAL_S)
            tile = window_bins * max(1, round(tile / window_bins))
        margin = VIEW_MARGIN_FRACTION * width
        first_bin = math.floor((low - margin) / HISTORY_INTERVAL_S / tile) * tile
        end_bin = math.ceil((high + margin) / HISTORY_INTERVAL_S / tile) * tile
        rows = self._level_rows(level, selected, first_bin, end_bin)
        return self._snapshot_from_rows(
            rows,
            selected,
            level=level,
            includes_tail=end_bin > last_bin,
            first_mono=data_low,
            last_mono=data_high,
            view_key=(level, first_bin, end_bin, aggregate),
        )

    def first_window_mean(self, slot: Slot, segment: int, window_s: float) -> float | None:
        """Среднее первого **завершённого** окна W сегмента, где у слота есть данные.

        Контракт однократного автозаполнения λ₀ (Р86): прежде он сворачивал
        всю историю на каждом такте. Окно находится по номеру первого валидного
        интервала сегмента, и копируется не больше W / 100 мс строк.
        """
        if not can_aggregate_exactly(window_s):
            return None
        first_bin = self._segment_first_valid_bin.get((segment, slot))
        if first_bin is None:
            return None
        window_bins = round(window_s / HISTORY_INTERVAL_S)
        group = first_bin // window_bins
        begin, end = group * window_bins, (group + 1) * window_bins
        rows = self._level_rows(0, (slot,), begin, end)
        rows = rows.take(rows.segment == segment)
        last = self._levels[0].last()
        includes_tail = last is None or end > last[1]
        history = self._snapshot_from_rows(
            rows,
            (slot,),
            level=0,
            includes_tail=includes_tail,
            first_mono=None,
            last_mono=None,
            view_key=("first_window", segment),
        )
        averaged = aggregate_history(history, window_s)
        valid = np.flatnonzero(averaged.n[:, 0] > 0)
        if valid.size == 0:
            return None
        return float(averaged.mean_nm[int(valid[0]), 0])


def history_memory_estimate_bytes(
    depth_s: float,
    active_positions: int,
    *,
    interval_s: float = HISTORY_INTERVAL_S,
) -> int:
    """Оценка верхнего объёма массивов самописца при данном числе линий.

    Это именно оценка для подписи UI: блоковая организация и Python-объекты
    дают небольшой служебный расход сверх формулы. Пирамида добавляет к базе
    ``1/10 + 1/100 + …`` — около 11 %.
    """
    if depth_s <= 0.0 or interval_s <= 0.0 or active_positions < 0:
        raise ValueError("параметры оценки истории должны быть неотрицательными")
    windows = math.ceil(depth_s / interval_s)
    pyramid = sum(1.0 / PYRAMID_FACTOR**level for level in range(PYRAMID_LEVELS))
    global_bytes = 8 + 8 + 8 + 8  # start, stop, segment, base_bin
    per_position_bytes = 8 + 8 + 8 + 8 + 8  # mean, min, max, sigma, n
    return math.ceil(windows * pyramid) * (global_bytes + active_positions * per_position_bytes)


def can_aggregate_exactly(window_s: float, interval_s: float = HISTORY_INTERVAL_S) -> bool:
    """Можно ли собрать окно без дробления уже сжатого интервала."""
    if not math.isfinite(window_s) or window_s < interval_s * (1.0 - 1e-9):
        return False
    ratio = window_s / interval_s
    return abs(ratio - round(ratio)) <= 1e-9


def aggregate_history(
    history: CompressedHistorySnapshot,
    window_s: float,
) -> HistoryAggregate:
    """Объединяет интервалы с весом по ``n`` без потери среднего.

    Окно обязано быть целым числом интервалов строк. Иначе точное среднее
    старых данных математически невосстановимо: один интервал пришлось бы
    делить между двумя окнами, а распределение raw-кадров уже сжато.

    Окна группируются по целому номеру интервала, если он есть в снимке.
    Незавершённым считается только последнее окно текущего идущего сегмента,
    и только если снимок доходит до конца истории: окно, обрезанное краем
    скопированной области, отбрасывается как неполное, а не выдаётся за
    полное.
    """
    if not can_aggregate_exactly(window_s, history.interval_s):
        raise ValueError(
            f"окно усреднения должно быть не меньше {history.interval_s * 1000:.0f} мс "
            f"и кратно {history.interval_s * 1000:.0f} мс"
        )
    rows = history.windows
    columns = len(history.positions)
    if rows == 0:
        return _empty_aggregate(columns)

    if history.base_bin is not None:
        base_bins = np.asarray(history.base_bin, dtype=np.int64)
    else:
        base_bins = np.rint(history.start_mono / HISTORY_INTERVAL_S).astype(np.int64)
    window_bins = round(window_s / HISTORY_INTERVAL_S)
    groups_key = base_bins // window_bins
    segment = np.asarray(history.segment, dtype=np.int64)
    first = _group_starts(segment, groups_key)
    last = np.r_[first[1:], rows] - 1

    keep = np.ones(first.size, dtype=bool)
    current = (
        int(history.segment[-1]) if history.current_segment is None else history.current_segment
    )
    tail = first.size - 1
    if history.running and history.includes_tail and int(segment[first[tail]]) == current:
        # Только живой хвост может быть неполным. Копия для экрана выровнена
        # по сетке окна (``view``), поэтому окно внутри копии краем не режется.
        nominal_stop = (int(groups_key[first[tail]]) + 1) * window_s
        last_stop = float(history.stop_mono[last[tail]])
        keep[tail] = last_stop >= nominal_stop - _ROW_COMPLETE_TOLERANCE_S
    stats = pool_intervals(
        first,
        history.n,
        history.mean_nm,
        history.sigma_nm,
        history.min_nm,
        history.max_nm,
    )
    selection = np.flatnonzero(keep)
    if selection.size == 0:
        return _empty_aggregate(columns)
    return HistoryAggregate(
        start_mono=np.asarray(history.start_mono, dtype=np.float64)[first[selection]],
        stop_mono=np.asarray(history.stop_mono, dtype=np.float64)[last[selection]],
        segment=segment[first[selection]].astype(np.int32),
        mean_nm=stats.mean[selection],
        min_nm=stats.minimum[selection],
        max_nm=stats.maximum[selection],
        sigma_nm=stats.sigma[selection],
        n=stats.n[selection],
    )


def _empty_aggregate(columns: int) -> HistoryAggregate:
    empty2 = np.empty((0, columns), dtype=np.float64)
    return HistoryAggregate(
        np.empty(0),
        np.empty(0),
        np.empty(0, dtype=np.int32),
        empty2.copy(),
        empty2.copy(),
        empty2.copy(),
        empty2.copy(),
        np.empty((0, columns), dtype=np.int64),
    )
