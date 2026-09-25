"""Сжатая история графиков вне Qt и вне приёмного тракта.

Самописец читает уже принятые кадры последовательным ``FrameCursor`` из
``AppController.snapshot()``. Поток приёма ничего о нём не знает. Сырая
частота преобразуется в длину волны и складывается в неподвижные интервалы
по 100 мс: среднее, минимум, максимум и число валидных кадров. Арифметику
окон задаёт ``core.averaging.fixed_window_average`` — второй реализации
временной сетки здесь нет.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fbg.core.averaging import WindowAverages, fixed_window_average
from fbg.core.pipeline import FrameBatch
from fbg.core.profile import C_NM_GHZ

HISTORY_INTERVAL_S = 0.1
"""Разрешение самописца: 200 raw-кадров при штатных 2 кГц."""

DEFAULT_HISTORY_DEPTH_S = 86_400.0
MIN_HISTORY_DEPTH_S = 1.0
MAX_HISTORY_DEPTH_S = 7 * 86_400.0
_HISTORY_BLOCK_WINDOWS = 1024

Slot = tuple[int, int]


@dataclass(frozen=True)
class CompressedHistorySnapshot:
    """Самодостаточная копия выбранных линий сжатой истории."""

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
    """Та же история после точного объединения базовых интервалов по ``n``."""

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


class _Block:
    """Фиксированный кусок истории; данные позиции выделяются только при успехе."""

    __slots__ = ("rows", "segment", "start", "stop", "values")

    def __init__(self) -> None:
        self.rows = 0
        self.start = np.empty(_HISTORY_BLOCK_WINDOWS, dtype=np.float64)
        self.stop = np.empty(_HISTORY_BLOCK_WINDOWS, dtype=np.float64)
        self.segment = np.empty(_HISTORY_BLOCK_WINDOWS, dtype=np.int32)
        self.values: dict[
            Slot, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def append(
        self,
        start: float,
        stop: float,
        segment: int,
        positions: tuple[Slot, ...],
        mean: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        sigma: np.ndarray,
        count: np.ndarray,
    ) -> None:
        row = self.rows
        self.start[row] = start
        self.stop[row] = stop
        self.segment[row] = segment
        valid_columns = np.flatnonzero(count > 0)
        for column in valid_columns:
            slot = positions[int(column)]
            arrays = self.values.get(slot)
            if arrays is None:
                means = np.full(_HISTORY_BLOCK_WINDOWS, np.nan, dtype=np.float64)
                minima = np.full(_HISTORY_BLOCK_WINDOWS, np.nan, dtype=np.float64)
                maxima = np.full(_HISTORY_BLOCK_WINDOWS, np.nan, dtype=np.float64)
                sigmas = np.full(_HISTORY_BLOCK_WINDOWS, np.nan, dtype=np.float64)
                counts = np.zeros(_HISTORY_BLOCK_WINDOWS, dtype=np.int32)
                arrays = (means, minima, maxima, sigmas, counts)
                self.values[slot] = arrays
            arrays[0][row] = float(mean[column])
            arrays[1][row] = float(minimum[column])
            arrays[2][row] = float(maximum[column])
            arrays[3][row] = float(sigma[column])
            arrays[4][row] = int(count[column])
        self.rows += 1

    @property
    def full(self) -> bool:
        return self.rows >= _HISTORY_BLOCK_WINDOWS


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
        self._positions = tuple(
            (channel, position)
            for channel in range(channels)
            for position in range(positions_per_channel)
        )
        self._columns = len(self._positions)
        self._interval_s = float(interval_s)
        self._depth_s = self._validated_depth(depth_s)
        self._blocks: deque[_Block] = deque()
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)
        self._first_valid: dict[Slot, float] = {}
        self._segment_first_valid: dict[tuple[int, Slot], float] = {}
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
        self._blocks.clear()
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)
        self._first_valid.clear()
        self._segment_first_valid.clear()
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
        for column, slot in enumerate(self._positions):
            valid = np.flatnonzero(np.isfinite(matrix[:, column]))
            if not valid.size:
                continue
            value = float(matrix[int(valid[0]), column])
            self._first_valid.setdefault(slot, value)
            self._segment_first_valid.setdefault((self._segment, slot), value)

    def _finish_segment(self) -> None:
        if self._pending_t.size:
            self._consume(self._pending_t, self._pending_nm, final=True)
        self._pending_t = np.empty(0, dtype=np.float64)
        self._pending_nm = np.empty((0, self._columns), dtype=np.float64)

    def _consume(self, times: np.ndarray, matrix: np.ndarray, *, final: bool) -> None:
        averaged = fixed_window_average(times, matrix, self._interval_s)
        complete_indices = np.flatnonzero(averaged.complete)
        if complete_indices.size:
            for output_index in complete_indices:
                index = int(output_index)
                minimum, maximum = self._extrema_for_window(times, matrix, averaged, index)
                self._append_window(
                    float(averaged.start_mono[index]),
                    float(averaged.stop_mono[index]),
                    averaged.mean[index],
                    minimum,
                    maximum,
                    averaged.sigma[index],
                    averaged.n[index],
                )
            last_index = int(complete_indices[-1])
            last_bin = math.floor(float(averaged.start_mono[last_index]) / self._interval_s)
            sample_bins = np.floor(times / self._interval_s).astype(np.int64)
            keep = sample_bins > last_bin
            self._pending_t = times[keep].copy()
            self._pending_nm = matrix[keep].copy()
        else:
            self._pending_t = times.copy()
            self._pending_nm = matrix.copy()
        if final:
            self._pending_t = np.empty(0, dtype=np.float64)
            self._pending_nm = np.empty((0, self._columns), dtype=np.float64)

    def _extrema_for_window(
        self,
        times: np.ndarray,
        matrix: np.ndarray,
        averaged: WindowAverages,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Принадлежность raw-строки повторяет неподвижную сетку
        # ``core.averaging``. Использовать ``stop-start`` нельзя: окно,
        # обрезанное GAP, короче базового интервала и сместило бы bins.
        bin_index = math.floor(float(averaged.start_mono[index]) / self._interval_s)
        sample_bins = np.floor(times / self._interval_s).astype(np.int64)
        block = matrix[sample_bins == bin_index]
        columns = matrix.shape[1]
        minimum = np.full(columns, np.nan, dtype=np.float64)
        maximum = np.full(columns, np.nan, dtype=np.float64)
        if block.size == 0:
            return minimum, maximum
        finite = np.isfinite(block)
        has = np.any(finite, axis=0)
        if np.any(has):
            minimum[has] = np.min(np.where(finite[:, has], block[:, has], np.inf), axis=0)
            maximum[has] = np.max(np.where(finite[:, has], block[:, has], -np.inf), axis=0)
        return minimum, maximum

    def _append_window(
        self,
        start: float,
        stop: float,
        mean: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        sigma: np.ndarray,
        count: np.ndarray,
    ) -> None:
        if self._origin_mono is None:
            self._origin_mono = start
        valid = np.flatnonzero(count > 0)
        for column in valid:
            self._active.add(self._positions[int(column)])
        if not self._blocks or self._blocks[-1].full:
            self._blocks.append(_Block())
        self._blocks[-1].append(
            start,
            stop,
            self._segment,
            self._positions,
            mean,
            minimum,
            maximum,
            sigma,
            count,
        )
        self._version += 1
        self._trim()

    def _trim(self) -> None:
        if not self._blocks:
            return
        last = self._blocks[-1]
        if last.rows == 0:
            return
        cutoff = float(last.stop[last.rows - 1]) - self._depth_s
        while len(self._blocks) > 1:
            first = self._blocks[0]
            if first.rows == 0 or float(first.stop[first.rows - 1]) >= cutoff:
                break
            self._blocks.popleft()

    def snapshot(self, positions: Sequence[Slot] | None = None) -> CompressedHistorySnapshot:
        requested = self.active_positions if positions is None else tuple(positions)
        selected = tuple(slot for slot in requested if slot in self._active)
        row_count = sum(block.rows for block in self._blocks)
        start = np.empty(row_count, dtype=np.float64)
        stop = np.empty(row_count, dtype=np.float64)
        segment = np.empty(row_count, dtype=np.int32)
        shape = (row_count, len(selected))
        mean = np.full(shape, np.nan, dtype=np.float64)
        minimum = np.full(shape, np.nan, dtype=np.float64)
        maximum = np.full(shape, np.nan, dtype=np.float64)
        sigma = np.full(shape, np.nan, dtype=np.float64)
        count = np.zeros(shape, dtype=np.int32)
        cursor = 0
        for block in self._blocks:
            rows = block.rows
            target = slice(cursor, cursor + rows)
            start[target] = block.start[:rows]
            stop[target] = block.stop[:rows]
            segment[target] = block.segment[:rows]
            for column, slot in enumerate(selected):
                arrays = block.values.get(slot)
                if arrays is None:
                    continue
                mean[target, column] = arrays[0][:rows]
                minimum[target, column] = arrays[1][:rows]
                maximum[target, column] = arrays[2][:rows]
                sigma[target, column] = arrays[3][:rows]
                count[target, column] = arrays[4][:rows]
            cursor += rows
        if row_count:
            cutoff = float(stop[-1]) - self._depth_s
            first = int(np.searchsorted(stop, cutoff, side="left"))
            start = start[first:]
            stop = stop[first:]
            segment = segment[first:]
            mean = mean[first:]
            minimum = minimum[first:]
            maximum = maximum[first:]
            sigma = sigma[first:]
            count = count[first:]
        first_valid = np.asarray(
            [self._first_valid.get(slot, np.nan) for slot in selected], dtype=np.float64
        )
        return CompressedHistorySnapshot(
            positions=selected,
            start_mono=start,
            stop_mono=stop,
            segment=segment,
            mean_nm=mean,
            min_nm=minimum,
            max_nm=maximum,
            sigma_nm=sigma,
            n=count,
            first_valid_nm=first_valid,
            running=self._running,
            interval_s=self._interval_s,
            depth_s=self._depth_s,
            version=self._version,
            origin_mono=self._origin_mono,
        )


def history_memory_estimate_bytes(
    depth_s: float,
    active_positions: int,
    *,
    interval_s: float = HISTORY_INTERVAL_S,
) -> int:
    """Оценка верхнего объёма массивов самописца при данном числе линий.

    Это именно оценка для подписи UI: блоковая организация и Python-объекты
    дают небольшой служебный расход сверх формулы.
    """
    if depth_s <= 0.0 or interval_s <= 0.0 or active_positions < 0:
        raise ValueError("параметры оценки истории должны быть неотрицательными")
    windows = math.ceil(depth_s / interval_s)
    global_bytes = 8 + 8 + 4  # start, stop, segment
    per_position_bytes = 8 + 8 + 8 + 8 + 4  # mean, min, max, sigma, n
    return windows * (global_bytes + active_positions * per_position_bytes)


def can_aggregate_exactly(window_s: float, interval_s: float = HISTORY_INTERVAL_S) -> bool:
    """Можно ли собрать окно без дробления уже сжатого базового интервала."""
    if not math.isfinite(window_s) or window_s < interval_s:
        return False
    ratio = window_s / interval_s
    return abs(ratio - round(ratio)) <= 1e-9


def aggregate_history(
    history: CompressedHistorySnapshot,
    window_s: float,
) -> HistoryAggregate:
    """Объединяет базовые интервалы с весом по ``n`` без потери среднего.

    Окно обязано быть целым числом базовых интервалов. Иначе точное среднее
    старых данных математически невосстановимо: один 100-мс интервал пришлось
    бы делить между двумя окнами, а распределение raw-кадров уже сжато.
    """
    if not can_aggregate_exactly(window_s, history.interval_s):
        raise ValueError(
            f"окно усреднения должно быть не меньше {history.interval_s * 1000:.0f} мс "
            f"и кратно {history.interval_s * 1000:.0f} мс"
        )
    rows = history.windows
    columns = len(history.positions)
    if rows == 0:
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

    bins = np.floor(history.start_mono / window_s).astype(np.int64)
    starts = np.empty(rows, dtype=bool)
    starts[0] = True
    if rows > 1:
        starts[1:] = (history.segment[1:] != history.segment[:-1]) | (bins[1:] != bins[:-1])
    first = np.flatnonzero(starts)
    last = np.r_[first[1:], rows]

    out_start: list[float] = []
    out_stop: list[float] = []
    out_segment: list[int] = []
    means: list[np.ndarray] = []
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    current_segment = int(history.segment[-1])

    for _group_index, (begin_raw, end_raw) in enumerate(zip(first, last, strict=True)):
        begin = int(begin_raw)
        end = int(end_raw)
        segment = int(history.segment[begin])
        bin_index = int(bins[begin])
        nominal_stop = (bin_index + 1) * window_s
        followed_by_new_segment = end < rows and int(history.segment[end]) != segment
        is_last_running_segment = history.running and segment == current_segment and end == rows
        complete = followed_by_new_segment or not is_last_running_segment
        if is_last_running_segment:
            complete = float(history.stop_mono[end - 1]) >= nominal_stop - 1e-12
        if not complete:
            continue

        block_n = history.n[begin:end].astype(np.int64, copy=False)
        total_n = np.sum(block_n, axis=0, dtype=np.int64)
        block_mean_all = history.mean_nm[begin:end]
        weighted = np.where(np.isfinite(block_mean_all), block_mean_all, 0.0)
        weighted_sum = np.sum(weighted * block_n, axis=0, dtype=np.float64)
        mean = np.full(columns, np.nan, dtype=np.float64)
        valid = total_n > 0
        mean[valid] = weighted_sum[valid] / total_n[valid]
        sigma = np.full(columns, np.nan, dtype=np.float64)
        if np.any(valid):
            block_mean = history.mean_nm[begin:end, valid]
            block_sigma = history.sigma_nm[begin:end, valid]
            block_counts = block_n[:, valid]
            combined_mean = mean[valid]
            variance_terms = np.where(
                block_counts > 0,
                block_counts
                * (
                    np.where(np.isfinite(block_sigma), block_sigma * block_sigma, 0.0)
                    + (block_mean - combined_mean) ** 2
                ),
                0.0,
            )
            variance = np.sum(variance_terms, axis=0, dtype=np.float64) / total_n[valid]
            sigma[valid] = np.sqrt(np.maximum(variance, 0.0))

        block_min = history.min_nm[begin:end]
        block_max = history.max_nm[begin:end]
        minimum = np.full(columns, np.nan, dtype=np.float64)
        maximum = np.full(columns, np.nan, dtype=np.float64)
        if np.any(valid):
            minimum[valid] = np.min(
                np.where(np.isfinite(block_min[:, valid]), block_min[:, valid], np.inf), axis=0
            )
            maximum[valid] = np.max(
                np.where(np.isfinite(block_max[:, valid]), block_max[:, valid], -np.inf), axis=0
            )

        out_start.append(float(history.start_mono[begin]))
        out_stop.append(float(history.stop_mono[end - 1]))
        out_segment.append(segment)
        means.append(mean)
        minima.append(minimum)
        maxima.append(maximum)
        sigmas.append(sigma)
        counts.append(total_n)

    if not means:
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
    return HistoryAggregate(
        start_mono=np.asarray(out_start, dtype=np.float64),
        stop_mono=np.asarray(out_stop, dtype=np.float64),
        segment=np.asarray(out_segment, dtype=np.int32),
        mean_nm=np.vstack(means),
        min_nm=np.vstack(minima),
        max_nm=np.vstack(maxima),
        sigma_nm=np.vstack(sigmas),
        n=np.vstack(counts).astype(np.int64, copy=False),
    )
