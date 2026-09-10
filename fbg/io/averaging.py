"""Офлайн-усреднение готовой записи Recorder по временному окну.

Исходные CSV не меняются. Все части ротации читаются как одна временная
последовательность, поэтому граница файла сама по себе окно не разрывает.
Разрывает его только явный ``# GAP``; маркер копируется в производный файл.

Модуль отдельный от :mod:`fbg.io.recalibrate`: калибровка добавляет физические
величины строка-в-строку, а здесь меняется временная дискретизация и схема
колонок. Общий код между ними ограничивается поиском частей одной записи.

Пересчёт потоковый: в памяти живёт статистика только одного текущего окна.
Это важно для многочасовых записей при 2 кГц — офлайн-операция не должна
требовать загрузки исходного CSV целиком.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from fbg.io.recalibrate import recording_parts
from fbg.io.recorder import SEPARATOR

_GAP_BOUNDS_RE = re.compile(
    r"^# GAP .*?t_mono_from=(?P<start>[-+0-9.eE]+|nan) "
    r"t_mono_to=(?P<stop>[-+0-9.eE]+|nan)"
)


@dataclass(frozen=True, slots=True)
class AveragingExportResult:
    """Итог усреднения одной записи через все части ротации."""

    inputs: tuple[Path, ...]
    output: Path
    windows: int
    gaps: int


@dataclass(frozen=True, slots=True)
class _GapBounds:
    start: float | None
    stop: float | None


class _OnlineWindow:
    """Welford-статистика одного окна без хранения исходных строк."""

    def __init__(self, bin_index: int, start: float, stop: float, columns: int) -> None:
        self.bin_index = bin_index
        self.start = float(start)
        self.stop = float(stop)
        self.n = np.zeros(columns, dtype=np.int64)
        self.mean = np.zeros(columns, dtype=np.float64)
        self.m2 = np.zeros(columns, dtype=np.float64)
        self.samples = 0

    def add(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        if not np.any(finite):
            self.samples += 1
            return
        old_n = self.n[finite].copy()
        new_n = old_n + 1
        delta = values[finite] - self.mean[finite]
        self.mean[finite] += delta / new_n
        delta2 = values[finite] - self.mean[finite]
        self.m2[finite] += delta * delta2
        self.n[finite] = new_n
        self.samples += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.full(self.mean.shape, np.nan, dtype=np.float64)
        sigma = np.full(self.mean.shape, np.nan, dtype=np.float64)
        valid = self.n > 0
        mean[valid] = self.mean[valid]
        sigma[valid] = np.sqrt(self.m2[valid] / self.n[valid])
        return mean, sigma, self.n.copy()


def averaged_path(path: Path, window_s: float) -> Path:
    """Имя производного файла рядом с первой частью записи."""
    window_ms = window_s * 1000.0
    token = f"{window_ms:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return path.with_name(f"{path.stem}_averaged_{token}ms.csv")


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.9g}"


def _parse_gap_bounds(line: str) -> _GapBounds | None:
    match = _GAP_BOUNDS_RE.match(line.rstrip("\r\n"))
    if match is None:
        return None
    raw_start = float(match.group("start"))
    raw_stop = float(match.group("stop"))
    start = raw_start if math.isfinite(raw_start) else None
    stop = raw_stop if math.isfinite(raw_stop) else None
    if start is not None and stop is not None and stop < start:
        raise ValueError("# GAP содержит обратные границы времени")
    return _GapBounds(start, stop)


def _output_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    if len(columns) < 4 or columns[:3] != ("frame_no", "t_mono", "t_wall"):
        raise ValueError("неожиданная схема CSV Recorder")
    result = ["window_start_mono", "window_stop_mono", "t_wall_mean"]
    for name in columns[3:]:
        result.extend((f"{name}_mean", f"{name}_n", f"{name}_sigma"))
    return tuple(result)


def _write_header(output: TextIO, columns: tuple[str, ...], window_s: float, parts: int) -> None:
    output.write(SEPARATOR.join(_output_columns(columns)) + "\n")
    output.write("# fbg-interrogator averaged export\n")
    output.write(f"# averaging_window_ms={window_s * 1000.0:.9g}\n")
    output.write(f"# source_parts={parts}\n")
    output.write("# time_grid=t_mono_zero_fixed windows_do_not_cross_GAP\n")
    output.write("# statistics=valid_samples_only sigma=population n=valid_sample_count\n")
    output.write("# empty_window_value=nan\n")


class _WindowWriter:
    """Потоково превращает строки Recorder в окна фиксированной сетки."""

    def __init__(self, output: TextIO, window_s: float, value_columns: int) -> None:
        self.output = output
        self.window_s = window_s
        # Нулевая колонка внутренней статистики — t_wall; остальные — данные.
        self.columns = value_columns + 1
        self.current: _OnlineWindow | None = None
        self.segment_lower: float | None = None
        self.last_input_t: float | None = None
        self.deferred_boundary: tuple[float, float, np.ndarray] | None = None
        self.windows = 0

    def _bin(self, t_mono: float, *, before_boundary: bool = False) -> int:
        value = np.nextafter(t_mono, -np.inf) if before_boundary else t_mono
        return math.floor(value / self.window_s)

    def _new_window(self, bin_index: int) -> _OnlineWindow:
        nominal_start = bin_index * self.window_s
        start = nominal_start
        if self.segment_lower is not None:
            start = max(start, self.segment_lower)
        return _OnlineWindow(bin_index, start, nominal_start + self.window_s, self.columns)

    def _write_current(self, *, stop: float | None = None) -> None:
        window = self.current
        if window is None:
            return
        actual_stop = window.stop if stop is None else min(window.stop, stop)
        # Нулевой интервал без отсчётов не является окном. С отсчётом такой
        # случай возможен только при неизвестной/битой границе и сохраняется.
        if actual_stop <= window.start and window.samples == 0:
            self.current = None
            return
        mean, sigma, n = window.result()
        fields = [
            f"{window.start:.6f}",
            f"{actual_stop:.6f}",
            _format_float(float(mean[0])),
        ]
        for column in range(1, self.columns):
            fields.extend(
                (
                    _format_float(float(mean[column])),
                    str(int(n[column])),
                    _format_float(float(sigma[column])),
                )
            )
        self.output.write(SEPARATOR.join(fields) + "\n")
        self.windows += 1
        self.current = None

    def _advance_to(self, target_bin: int) -> None:
        if self.current is None:
            if self.segment_lower is None:
                self.current = self._new_window(target_bin)
                return
            first_bin = self._bin(self.segment_lower)
            self.current = self._new_window(first_bin)
        if target_bin < self.current.bin_index:
            raise ValueError("t_mono вернулся в уже завершённое окно")
        while self.current.bin_index < target_bin:
            next_bin = self.current.bin_index + 1
            self._write_current()
            self.current = self._new_window(next_bin)

    def _commit_row(
        self,
        t_mono: float,
        t_wall: float,
        values: np.ndarray,
        *,
        before_boundary: bool = False,
    ) -> None:
        target = self._bin(t_mono, before_boundary=before_boundary)
        self._advance_to(target)
        assert self.current is not None
        vector = np.empty(self.columns, dtype=np.float64)
        vector[0] = t_wall
        vector[1:] = values
        self.current.add(vector)

    def _commit_deferred(self, *, before_boundary: bool = False) -> None:
        deferred = self.deferred_boundary
        if deferred is None:
            return
        self.deferred_boundary = None
        self._commit_row(*deferred, before_boundary=before_boundary)

    def add_row(self, t_mono: float, t_wall: float, values: np.ndarray) -> None:
        if self.last_input_t is not None and t_mono < self.last_input_t:
            raise ValueError("t_mono в записи должен возрастать через все части ротации")
        self.last_input_t = t_mono

        # Последняя строка всегда задерживается на один входной элемент. Если
        # следом идёт GAP с той же левой границей, это последний полученный
        # кадр *до* тишины (Р42), и на точной границе глобального bin его надо
        # отнести к предыдущему окну. Один отложенный ряд дешевле и надёжнее,
        # чем распознавать кратность больших perf_counter-меток по float.
        self._commit_deferred()
        self.deferred_boundary = (t_mono, t_wall, values.copy())

    def add_gap(self, line: str, bounds: _GapBounds | None) -> None:
        if bounds is not None and bounds.start is not None:
            start = bounds.start
            if self.deferred_boundary is not None and self.deferred_boundary[0] == start:
                self._commit_deferred(before_boundary=True)
            else:
                self._commit_deferred()

            pre_gap_bin = self._bin(start, before_boundary=True)
            if self.current is None:
                if self.segment_lower is not None and self.segment_lower < start:
                    self._advance_to(pre_gap_bin)
            elif self.current.bin_index < pre_gap_bin:
                self._advance_to(pre_gap_bin)
            elif self.current.bin_index > pre_gap_bin:
                raise ValueError("# GAP начинается раньше уже прочитанных строк")
            if self.current is not None:
                self._write_current(stop=start)
        else:
            # Левая граница неизвестна: склеивать значения через маркер нельзя,
            # но и придумывать точный момент окончания окна нельзя.
            self._commit_deferred()
            self._write_current()

        self.output.write(line if line.endswith("\n") else line + "\n")
        self.current = None
        self.segment_lower = None if bounds is None else bounds.stop

    def finish(self) -> None:
        self._commit_deferred()
        self._write_current()


def average_recording(path: Path, window_s: float) -> AveragingExportResult:
    """Усредняет выбранную запись и все её части в один производный CSV.

    Окно задаётся временем, поэтому исходная `decimation` только уменьшает
    фактическое ``n`` и не меняет границы сетки. Все raw-колонки после
    ``frame_no/t_mono/t_wall`` получают рядом ``mean``, ``n`` и ``sigma``.
    Пустой временной bin внутри известного непрерывного участка не исчезает:
    он записывается как ``nan, n=0, sigma=nan``.
    """
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("window_s должен быть положительным конечным числом")
    parts = recording_parts(path)
    output_path = averaged_path(parts[0], window_s)
    temporary = output_path.with_name(output_path.name + ".tmp")
    gaps_seen = 0
    columns: tuple[str, ...] | None = None
    writer: _WindowWriter | None = None

    try:
        with temporary.open("w", encoding="ascii", newline="\n") as output:
            for part in parts:
                with part.open("r", encoding="ascii", errors="strict") as source:
                    raw_header = source.readline()
                    if not raw_header:
                        raise ValueError(f"{part}: пустой файл")
                    part_columns = tuple(raw_header.rstrip("\r\n").split(SEPARATOR))
                    if columns is None:
                        columns = part_columns
                        _write_header(output, columns, window_s, len(parts))
                        writer = _WindowWriter(output, window_s, len(columns) - 3)
                    elif part_columns != columns:
                        raise ValueError(f"{part}: схема колонок отличается от первой части")

                    for line in source:
                        if line.startswith("#"):
                            if line.startswith("# GAP"):
                                assert writer is not None
                                writer.add_gap(line, _parse_gap_bounds(line))
                                gaps_seen += 1
                            else:
                                metadata = line.rstrip("\r\n")
                                metadata = (
                                    metadata[2:] if metadata.startswith("# ") else metadata[1:]
                                )
                                output.write(f"# source_metadata {metadata}\n")
                            continue
                        text = line.strip()
                        if not text:
                            continue
                        assert columns is not None and writer is not None
                        parsed = np.fromstring(text, sep=SEPARATOR, dtype=np.float64)
                        if parsed.size != len(columns):
                            raise ValueError(
                                f"{part}: строка содержит {parsed.size} полей вместо {len(columns)}"
                            )
                        t_mono = float(parsed[1])
                        if not math.isfinite(t_mono):
                            raise ValueError(f"{part}: t_mono должен быть конечным")
                        writer.add_row(t_mono, float(parsed[2]), parsed[3:])

            if columns is None or writer is None:
                raise ValueError(f"{path}: запись не содержит частей")
            writer.finish()
        os.replace(temporary, output_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise

    return AveragingExportResult(parts, output_path, writer.windows, gaps_seen)
