"""Усреднение по неподвижной временной сетке.

Модуль ничего не знает про Qt, pipeline и файловый формат. Он содержит только
арифметику окон, которую используют два требуемых пути: модель интерфейса и
офлайн-пересчёт записи. Общая реализация нужна не как абстракция «на будущее»,
а чтобы live-график и производный CSV не разошлись в семантике NaN, ``n`` и σ.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowAverages:
    """Среднее, σ и число валидных отсчётов для последовательности окон."""

    start_mono: np.ndarray
    stop_mono: np.ndarray
    mean: np.ndarray
    sigma: np.ndarray
    n: np.ndarray
    segment: np.ndarray
    complete: np.ndarray

    @property
    def windows(self) -> int:
        return int(self.start_mono.size)

    @property
    def columns(self) -> int:
        if self.mean.ndim != 2:
            return 0
        return int(self.mean.shape[1])


def empty_window_averages(columns: int) -> WindowAverages:
    """Пустой результат с известным числом колонок."""
    if columns < 0:
        raise ValueError("columns не может быть отрицательным")
    return WindowAverages(
        start_mono=np.empty(0, dtype=np.float64),
        stop_mono=np.empty(0, dtype=np.float64),
        mean=np.empty((0, columns), dtype=np.float64),
        sigma=np.empty((0, columns), dtype=np.float64),
        n=np.empty((0, columns), dtype=np.int64),
        segment=np.empty(0, dtype=np.int64),
        complete=np.empty(0, dtype=bool),
    )


def _normalise_gaps(gaps: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = tuple((float(start), float(stop)) for start, stop in gaps)
    previous_stop = -np.inf
    for start, stop in ordered:
        if not np.isfinite(start) or not np.isfinite(stop):
            raise ValueError("границы разрыва должны быть конечными")
        if stop < start:
            raise ValueError("конец разрыва не может быть раньше начала")
        if start < previous_stop:
            raise ValueError("разрывы должны быть отсортированы и не пересекаться")
        previous_stop = stop
    return ordered


def valid_mean_sigma_n(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Считает mean/σ/``n`` независимо по колонкам, игнорируя только NaN/inf.

    Дисперсия считается вторым проходом относительно уже найденного среднего,
    а не как ``E[x²] - E[x]²``. Для длин волн около 1550 нм последняя формула
    теряет точность именно на пикометровом шуме, ради которого нужна полоса σ.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise ValueError("values должен быть двумерным")

    columns = matrix.shape[1]
    means = np.full(columns, np.nan, dtype=np.float64)
    sigmas = np.full(columns, np.nan, dtype=np.float64)
    finite = np.isfinite(matrix)
    counts = np.count_nonzero(finite, axis=0).astype(np.int64)
    nonzero = counts > 0
    if not np.any(nonzero):
        return means, sigmas, counts

    safe = np.where(finite, matrix, 0.0)
    sums = np.sum(safe, axis=0, dtype=np.float64)
    means[nonzero] = sums[nonzero] / counts[nonzero]

    centered = np.where(finite, matrix - means, 0.0)
    sums_sq = np.sum(centered * centered, axis=0, dtype=np.float64)
    variance = np.zeros(columns, dtype=np.float64)
    variance[nonzero] = sums_sq[nonzero] / counts[nonzero]
    sigmas[nonzero] = np.sqrt(variance[nonzero])
    return means, sigmas, counts


def _sample_segments_and_bins(
    times: np.ndarray,
    window_s: float,
    gap_starts: np.ndarray,
    gap_stops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Назначает каждому отсчёту непрерывный сегмент и bin общей сетки."""
    if gap_stops.size:
        segments = np.searchsorted(gap_stops, times, side="right").astype(np.int64)
    else:
        segments = np.zeros(times.size, dtype=np.int64)
    bins = np.floor(times / window_s).astype(np.int64)

    # t_mono_from — последний полученный кадр до тишины (Р42). Если он ровно
    # совпал с границей глобального bin, он относится к окну *до* GAP, а не к
    # нулевой по длительности части окна после этой границы.
    for gap_index, start in enumerate(gap_starts):
        at_left_boundary = (segments == gap_index) & (times == start)
        if np.any(at_left_boundary):
            before = np.nextafter(start, -np.inf)
            bins[at_left_boundary] = int(np.floor(before / window_s))
    return segments, bins


def fixed_window_average(
    t_mono: np.ndarray,
    values: np.ndarray,
    window_s: float,
    *,
    gaps: Sequence[tuple[float, float]] = (),
) -> WindowAverages:
    """Усредняет строки по фиксированным временным окнам.

    Сетка имеет якорь ``t=0`` и потому не зависит от момента вызова функции.
    Разрыв разрезает окно точно по своим границам: предыдущее окно заканчивается
    в ``gap_from``, следующее начинается в ``gap_to``. Значения внутри разрыва
    считаются противоречием входных данных.

    Внутри известного непрерывного сегмента присутствуют и bins без строк:
    они дают ``mean=NaN``, ``sigma=NaN``, ``n=0``, а не исчезают с графика.
    Это важно для офлайн-записи с децимацией, где окно может оказаться между
    двумя сохранёнными строками. До первого наблюдаемого кадра и после
    последнего никакие пустые окна не додумываются.

    NaN/inf исключаются независимо по каждой колонке. ``n`` хранит число
    валидных отсчётов. Для одного валидного отсчёта σ равна нулю (population
    standard deviation).
    """
    if not np.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("window_s должен быть положительным конечным числом")

    times = np.asarray(t_mono, dtype=np.float64)
    matrix = np.asarray(values, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("t_mono должен быть одномерным")
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2 or matrix.shape[0] != times.size:
        raise ValueError("values должен иметь форму (кадры, колонки)")
    if times.size == 0:
        return empty_window_averages(matrix.shape[1])
    if not np.all(np.isfinite(times)):
        raise ValueError("t_mono содержит нечисловую метку")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("t_mono должен быть отсортирован")

    normalized_gaps = _normalise_gaps(gaps)
    if normalized_gaps:
        gap_starts = np.asarray([item[0] for item in normalized_gaps], dtype=np.float64)
        gap_stops = np.asarray([item[1] for item in normalized_gaps], dtype=np.float64)
        for start, stop in normalized_gaps:
            if np.any((times > start) & (times < stop)):
                raise ValueError("t_mono содержит отсчёт внутри объявленного разрыва")
    else:
        gap_starts = np.empty(0, dtype=np.float64)
        gap_stops = np.empty(0, dtype=np.float64)

    segments, bins = _sample_segments_and_bins(times, window_s, gap_starts, gap_stops)

    # Строки одинакового (segment, bin) идут подряд, поэтому достаточно карты
    # на срез матрицы. Пустые bins в эту карту не попадают и будут созданы ниже.
    starts = np.empty(times.size, dtype=bool)
    starts[0] = True
    if times.size > 1:
        starts[1:] = (segments[1:] != segments[:-1]) | (bins[1:] != bins[:-1])
    first = np.flatnonzero(starts)
    last = np.r_[first[1:], times.size]
    groups = {
        (int(segments[begin]), int(bins[begin])): (int(begin), int(end))
        for begin, end in zip(first, last, strict=True)
    }

    descriptors: list[tuple[int, int, float, float]] = []
    gap_count = len(normalized_gaps)
    for segment in range(gap_count + 1):
        observed = np.flatnonzero(segments == segment)
        if segment == 0:
            if observed.size == 0:
                continue
            first_bin = int(bins[int(observed[0])])
            lower = -np.inf
        else:
            lower = float(gap_stops[segment - 1])
            first_bin = int(np.floor(lower / window_s))

        if segment < gap_count:
            upper = float(gap_starts[segment])
            if upper < lower:
                raise ValueError("непрерывный сегмент имеет обратные границы")
            if upper == lower and observed.size == 0:
                continue
            before_upper = np.nextafter(upper, -np.inf)
            last_bin = int(np.floor(before_upper / window_s))
        else:
            if observed.size == 0:
                continue
            upper = np.inf
            last_bin = int(bins[int(observed[-1])])

        if observed.size:
            first_bin = min(first_bin, int(np.min(bins[observed])))
            last_bin = max(last_bin, int(np.max(bins[observed])))
        if last_bin < first_bin:
            continue

        for bin_index in range(first_bin, last_bin + 1):
            nominal_start = float(bin_index) * window_s
            nominal_stop = nominal_start + window_s
            start = max(nominal_start, lower) if np.isfinite(lower) else nominal_start
            stop = min(nominal_stop, upper) if np.isfinite(upper) else nominal_stop
            if stop < start:
                continue
            descriptors.append((segment, bin_index, start, stop))

    rows = len(descriptors)
    columns = matrix.shape[1]
    means = np.full((rows, columns), np.nan, dtype=np.float64)
    sigmas = np.full((rows, columns), np.nan, dtype=np.float64)
    counts = np.zeros((rows, columns), dtype=np.int64)
    window_starts = np.empty(rows, dtype=np.float64)
    window_stops = np.empty(rows, dtype=np.float64)
    window_segments = np.empty(rows, dtype=np.int64)
    complete = np.empty(rows, dtype=bool)

    last_seen = float(times[-1])
    for output_index, (segment, bin_index, start, stop) in enumerate(descriptors):
        window_starts[output_index] = start
        window_stops[output_index] = stop
        window_segments[output_index] = segment
        complete[output_index] = segment < gap_count or stop <= last_seen

        bounds = groups.get((segment, bin_index))
        if bounds is None:
            continue
        begin, end = bounds
        mean, sigma, n = valid_mean_sigma_n(matrix[begin:end])
        means[output_index] = mean
        sigmas[output_index] = sigma
        counts[output_index] = n

    return WindowAverages(
        start_mono=window_starts,
        stop_mono=window_stops,
        mean=means,
        sigma=sigmas,
        n=counts,
        segment=window_segments,
        complete=complete,
    )


def concatenate_window_averages(
    prefix: WindowAverages,
    tail: WindowAverages,
) -> WindowAverages:
    """Склеивает два результата одинаковой ширины без изменения данных."""
    if prefix.columns != tail.columns:
        raise ValueError("число колонок усреднений не совпадает")
    if prefix.windows == 0:
        return tail
    if tail.windows == 0:
        return prefix
    return WindowAverages(
        start_mono=np.concatenate((prefix.start_mono, tail.start_mono)),
        stop_mono=np.concatenate((prefix.stop_mono, tail.stop_mono)),
        mean=np.concatenate((prefix.mean, tail.mean), axis=0),
        sigma=np.concatenate((prefix.sigma, tail.sigma), axis=0),
        n=np.concatenate((prefix.n, tail.n), axis=0),
        segment=np.concatenate((prefix.segment, tail.segment)),
        complete=np.concatenate((prefix.complete, tail.complete)),
    )


def slice_window_averages(windows: WindowAverages, selection: slice | np.ndarray) -> WindowAverages:
    """Возвращает копию выбранных окон; helper нужен инкрементальной UI-модели."""
    return WindowAverages(
        start_mono=windows.start_mono[selection].copy(),
        stop_mono=windows.stop_mono[selection].copy(),
        mean=windows.mean[selection].copy(),
        sigma=windows.sigma[selection].copy(),
        n=windows.n[selection].copy(),
        segment=windows.segment[selection].copy(),
        complete=windows.complete[selection].copy(),
    )
