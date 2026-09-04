"""Постобработка записанного CSV: сырые нанометры → физические величины.

Исходные файлы Recorder не меняются. Для каждой части ротации рядом создаётся
`*_calibrated.csv` с теми же исходными колонками, комментариями и `# GAP`, плюс
по одной колонке величины на датчик. Это намеренно **не** второй писатель в
тракте приёма: калибровку можно исправить и повторить задним числом, а сырые
нанометры по KB_05 №4 остаются первичным материалом.

Модуль не импортирует Qt. Долгую функцию вызывает обычный `threading.Thread`
панели, поэтому GUI не блокируется (KB_05 №34).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from fbg.core.calibration import Sensor, validate_sensors
from fbg.io.recorder import SEPARATOR

_BATCH_ROWS = 4096
_PART_RE = re.compile(r"\bfile_part=(\d+)\b")
_START_RE = re.compile(r"^# t_wall_start=(.+)$")
_WAVELENGTH_COLUMN_RE = re.compile(r"^ch(\d+)_fbg(\d+)_nm$")


@dataclass(frozen=True, slots=True)
class RecalibrationResult:
    """Итог обработки одной записи, включая все найденные части ротации."""

    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    rows: int
    gaps: int


def _read_recording_identity(path: Path) -> tuple[str, int]:
    """Возвращает `(t_wall_start, file_part)` из ASCII-шапки Recorder."""
    started: str | None = None
    part: int | None = None
    with path.open("r", encoding="ascii", errors="strict") as handle:
        # Первая строка — имена колонок. Идентификаторы записи идут сразу далее.
        handle.readline()
        for _ in range(32):
            line = handle.readline()
            if not line:
                break
            text = line.rstrip("\r\n")
            match = _START_RE.match(text)
            if match:
                started = match.group(1)
            part_match = _PART_RE.search(text)
            if part_match:
                part = int(part_match.group(1))
            if started is not None and part is not None:
                return started, part
    raise ValueError(f"{path}: в шапке нет t_wall_start/file_part")


def recording_parts(path: Path) -> tuple[Path, ...]:
    """Находит все части ротации той же записи и сортирует по `file_part`."""
    target = path.resolve()
    started, _part = _read_recording_identity(target)
    found: list[tuple[int, Path]] = []
    for candidate in target.parent.glob("*.csv"):
        if candidate.stem.endswith("_calibrated"):
            continue
        try:
            candidate_started, candidate_part = _read_recording_identity(candidate)
        except (OSError, UnicodeError, ValueError):
            continue
        if candidate_started == started:
            found.append((candidate_part, candidate))
    if not found:
        return (target,)
    found.sort(key=lambda item: item[0])
    parts = tuple(candidate for _part_no, candidate in found)
    if target not in tuple(candidate.resolve() for candidate in parts):
        raise ValueError(f"{path}: выбранный файл не найден среди частей своей записи")
    return parts


def calibrated_path(path: Path) -> Path:
    """Путь производного файла рядом с исходным."""
    return path.with_name(path.stem + "_calibrated.csv")


def _safe_sensor_column(sensor: Sensor, index: int) -> str:
    """ASCII-имя колонки независимо от человеческого имени/ID датчика."""
    ascii_id = "".join(
        char if char.isascii() and (char.isalnum() or char == "_") else "_"
        for char in sensor.id
    ).strip("_")
    suffix = ascii_id or "sensor"
    return f"sensor{index + 1:03d}_{suffix}_value"


def _channel_columns(columns: tuple[str, ...]) -> dict[int, np.ndarray]:
    """Индексы raw-nm колонок по 0-based каналу."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for index, name in enumerate(columns):
        match = _WAVELENGTH_COLUMN_RE.match(name)
        if not match:
            continue
        channel = int(match.group(1)) - 1
        position = int(match.group(2)) - 1
        grouped.setdefault(channel, []).append((position, index))
    return {
        channel: np.asarray([index for _position, index in sorted(items)], dtype=np.intp)
        for channel, items in grouped.items()
    }


def _match_batch(matrix: np.ndarray, columns: np.ndarray, sensor: Sensor) -> np.ndarray:
    """Векторно находит единственный пик датчика для каждой строки пачки."""
    rows = matrix.shape[0]
    if columns.size == 0:
        return np.full(rows, np.nan, dtype=np.float64)
    wavelengths = matrix[:, columns]
    inside = np.isfinite(wavelengths) & (
        np.abs(wavelengths - sensor.expected_nm) <= sensor.window_nm
    )
    count = np.count_nonzero(inside, axis=1)
    positions = np.argmax(inside, axis=1)
    found = wavelengths[np.arange(rows), positions].astype(np.float64, copy=True)
    found[count != 1] = np.nan
    return found


def _calibrate_batch(
    matrix: np.ndarray,
    sensors: tuple[Sensor, ...],
    channel_columns: dict[int, np.ndarray],
) -> np.ndarray:
    """Считает все датчики пачки; NaN и компенсация совпадают с core.calibration."""
    rows = matrix.shape[0]
    result = np.full((rows, len(sensors)), np.nan, dtype=np.float64)
    by_id = {sensor.id: index for index, sensor in enumerate(sensors)}

    def calculate(index: int, sensor: Sensor) -> None:
        columns = channel_columns.get(sensor.channel, np.empty(0, dtype=np.intp))
        found = _match_batch(matrix, columns, sensor)
        delta = found - sensor.expected_nm
        value = sensor.value0 + sensor.k1 * delta + sensor.k2 * delta * delta
        if sensor.compensation is not None:
            reference_index = by_id[sensor.compensation.reference]
            reference = result[:, reference_index]
            value = value + sensor.compensation.coeff * (reference - sensor.compensation.base)
            value[~np.isfinite(reference)] = np.nan
        result[:, index] = value

    deferred: list[tuple[int, Sensor]] = []
    for index, sensor in enumerate(sensors):
        if sensor.compensation is None:
            calculate(index, sensor)
        else:
            deferred.append((index, sensor))
    for index, sensor in deferred:
        calculate(index, sensor)
    return result


def _format_value(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.9g}"


def _flush_batch(
    output: TextIO,
    raw_lines: list[str],
    rows: list[np.ndarray],
    sensors: tuple[Sensor, ...],
    channel_columns: dict[int, np.ndarray],
) -> int:
    if not rows:
        return 0
    matrix = np.vstack(rows)
    calibrated = _calibrate_batch(matrix, sensors, channel_columns)
    for raw, values in zip(raw_lines, calibrated, strict=True):
        suffix = SEPARATOR.join(_format_value(float(value)) for value in values)
        output.write(raw.rstrip("\r\n") + SEPARATOR + suffix + "\n")
    count = len(rows)
    raw_lines.clear()
    rows.clear()
    return count


def _recalibrate_part(
    path: Path, output_path: Path, sensors: tuple[Sensor, ...]
) -> tuple[int, int]:
    """Обрабатывает одну часть ротации атомарно."""
    temporary = output_path.with_name(output_path.name + ".tmp")
    rows_written = 0
    gaps = 0
    try:
        with path.open("r", encoding="ascii", errors="strict") as source, temporary.open(
            "w", encoding="ascii", newline="\n"
        ) as output:
            header = source.readline()
            if not header:
                raise ValueError(f"{path}: пустой файл")
            columns = tuple(header.rstrip("\r\n").split(SEPARATOR))
            channel_columns = _channel_columns(columns)
            if not channel_columns:
                raise ValueError(f"{path}: не найдены колонки chN_fbgM_nm")
            extra = tuple(
                _safe_sensor_column(sensor, index) for index, sensor in enumerate(sensors)
            )
            output.write(SEPARATOR.join(columns + extra) + "\n")

            raw_lines: list[str] = []
            parsed_rows: list[np.ndarray] = []
            for line in source:
                if line.startswith("#"):
                    rows_written += _flush_batch(
                        output, raw_lines, parsed_rows, sensors, channel_columns
                    )
                    output.write(line)
                    if line.startswith("# GAP"):
                        gaps += 1
                    continue
                text = line.strip()
                if not text:
                    continue
                row = np.fromstring(text, sep=SEPARATOR, dtype=np.float64)
                if row.size != len(columns):
                    raise ValueError(
                        f"{path}: строка данных содержит {row.size} полей вместо {len(columns)}"
                    )
                raw_lines.append(line)
                parsed_rows.append(row)
                if len(parsed_rows) >= _BATCH_ROWS:
                    rows_written += _flush_batch(
                        output, raw_lines, parsed_rows, sensors, channel_columns
                    )
            rows_written += _flush_batch(output, raw_lines, parsed_rows, sensors, channel_columns)
        os.replace(temporary, output_path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return rows_written, gaps


def recalibrate_recording(path: Path, sensors: tuple[Sensor, ...]) -> RecalibrationResult:
    """Пересчитывает выбранную запись и все её части ротации.

    `# GAP` обоих видов копируются буквально, децимация сохраняется в исходных
    `frame_no`, а `nan` остаётся `nan`. Исходные файлы не открываются на запись.
    """
    if not sensors:
        raise ValueError("нет датчиков для пересчёта")
    problems = validate_sensors(sensors)
    if problems:
        raise ValueError("; ".join(problems))
    parts = recording_parts(path)
    outputs: list[Path] = []
    rows = 0
    gaps = 0
    for part in parts:
        output = calibrated_path(part)
        part_rows, part_gaps = _recalibrate_part(part, output, sensors)
        outputs.append(output)
        rows += part_rows
        gaps += part_gaps
    return RecalibrationResult(parts, tuple(outputs), rows, gaps)
