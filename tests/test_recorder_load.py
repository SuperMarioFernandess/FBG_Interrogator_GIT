"""Нагрузочные тесты записи: сквозной тракт до файла и предельный темп формата.

Чат №6 довёл поток до курсора писателя. Здесь замыкается последнее звено —
диск: симулятор → транспорт → сессия → pipeline → recorder → файл, 2000 Гц,
60 секунд, с проверкой полноты **по номерам кадров в самом файле**, а не
по счётчикам в памяти. Счётчик может ошибаться заодно с кодом, который он
считает; ряд номеров в файле — независимый свидетель.

Отдельно меряется собственный потолок записи: сколько строк в секунду
recorder способен отформатировать и положить на диск. Форматирование
ожидалось узким местом — замер показывает, что это не так, и число
печатается всегда, а не только при падении (KB_05).

Маркер `slow`: прогон длится больше минуты, отдельная job в CI.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from fbg.core.profile import DeviceProfile
from fbg.io.recorder import Recorder, RecorderConfig, column_names, format_rows, row_format
from tests.test_pipeline_load import LOAD_SECONDS, MAX_LOSS_FRACTION, TARGET_RATE_HZ, Stand
from tests.test_recorder import files_of, frame_numbers, gap_lines
from tests.test_session import wait_until

pytestmark = pytest.mark.slow

#: Ротация под нагрузкой: 32 МБ дают несколько файлов за минуту потока
#: и проверяют, что смена файла на ходу кадров не роняет.
ROTATE_BYTES = 32 << 20


def _run_to_disk(
    directory: Path,
    seconds: float,
    *,
    decimation: int = 1,
    rotate_bytes: int | None = ROTATE_BYTES,
) -> tuple[Stand, Recorder, float]:
    """Гоняет поток заданное время через полный тракт с записью в файл."""
    stand = Stand(TARGET_RATE_HZ, history_frames=20_000)
    recorder = Recorder(
        stand.pipeline,
        RecorderConfig(
            directory=directory,
            serial=94401220,
            firmware="4.10",
            decimation=decimation,
            rotate_bytes=rotate_bytes,
            rotate_seconds=None,
        ),
    )
    try:
        assert stand.session.connect().ok, "подключение к симулятору не удалось"
        stand.pipeline.start()
        recorder.start()
        started = time.perf_counter()
        assert stand.session.start_stream(TARGET_RATE_HZ).ok
        time.sleep(seconds)
        elapsed = time.perf_counter() - started
        assert stand.session.stop_stream().ok
        assert wait_until(lambda: not stand.sim.streaming, timeout=2.0)
        # Долёт последних датаграмм и добор кольца: иначе потери мнимые.
        time.sleep(1.0)
    finally:
        recorder.stop()
        stand.close()
    return stand, recorder, elapsed


def _report(name: str, stand: Stand, recorder: Recorder, directory: Path, seconds: float) -> None:
    """Печатает результат прогона. Это результат теста, а не диагностика."""
    sent = stand.sim.stats.frames_sent
    metrics = stand.pipeline.metrics()
    stats = recorder.stats
    paths = files_of(directory)
    volume = sum(path.stat().st_size for path in paths)
    print(f"\n--- {name} ---")
    print(
        f"отправлено {sent}, разобрано {metrics.frames}, "
        f"строк в файлах {stats.rows}, потеряно записью {stats.lost_frames}"
    )
    print(
        f"файлов {len(paths)}, объём {volume / (1 << 20):.1f} МБ "
        f"({volume / seconds / (1 << 20):.2f} МБ/с), "
        f"строка в среднем {volume // max(stats.rows, 1)} Б"
    )
    print(f"маркеров разрыва {stats.gaps}, ошибка записи: {stats.error or 'нет'}")
    print(
        f"темп факт {metrics.frame_rate_hz:.1f} Гц, записано "
        f"{stats.rows / seconds:.1f} строк/с при децимации {recorder.config.decimation}"
    )


def test_запись_2000_кадров_в_секунду_60_секунд(tmp_path: Path) -> None:
    """Сквозной прогон до файла: в файлах ровно столько строк, сколько кадров дошло.

    Проверка полноты — по номерам в самом файле: ряд `frame_no` обязан быть
    непрерывным от нуля без единого пропуска, а маркеров разрыва быть
    не должно. Ротация при этом происходит на ходу: смена файла кадров
    не роняет, потому что recorder тянет кадры сам, а кольцо держит 10 секунд.
    """
    stand, recorder, elapsed = _run_to_disk(tmp_path, LOAD_SECONDS)
    _report("запись на диск, 2000 Гц", stand, recorder, tmp_path, elapsed)

    sent = stand.sim.stats.frames_sent
    metrics = stand.pipeline.metrics()
    paths = files_of(tmp_path)
    numbers = frame_numbers(paths)

    assert recorder.stats.error is None, f"запись отвалилась: {recorder.stats.error}"
    assert stand.sim.pace.rate_hz == pytest.approx(TARGET_RATE_HZ, rel=0.02), (
        "отправитель не выдержал 2000 Гц — сравнивать нечего"
    )
    assert len(paths) >= 2, "ротация под нагрузкой не сработала — стык не проверен"
    assert metrics.parse_errors == 0
    assert recorder.stats.gaps == 0, "на паспортном темпе разрывов быть не должно"
    assert len(numbers) == metrics.frames, (
        "каждый разобранный кадр обязан оказаться строкой в файле"
    )
    assert numbers == list(range(len(numbers))), "ряд номеров кадров обязан быть непрерывным"
    loss = 1.0 - len(numbers) / sent
    assert loss < MAX_LOSS_FRACTION, f"сквозные потери до файла {loss * 100:.4f} %"


def test_децимация_снижает_объём_и_не_даёт_разрывов(tmp_path: Path) -> None:
    """Децимация 10 при 2 кГц: строк вдесятеро меньше, шаг номеров ровный.

    Именно так поток 2 кГц пишется в рабочем режиме: 200 строк в секунду
    вместо 2000. Пропуск по настройке разрывом не считается и маркером
    не отмечается.
    """
    stand, recorder, elapsed = _run_to_disk(tmp_path, 20.0, decimation=10)
    _report("децимация 10, 2000 Гц", stand, recorder, tmp_path, elapsed)

    numbers = frame_numbers(files_of(tmp_path))
    metrics = stand.pipeline.metrics()

    assert recorder.stats.error is None
    assert gap_lines(files_of(tmp_path)[0]) == []
    assert recorder.stats.gaps == 0
    assert numbers == [index * 10 for index in range(len(numbers))]
    assert len(numbers) == pytest.approx(metrics.frames / 10, rel=0.01)


def test_потолок_форматирования_и_записи(tmp_path: Path) -> None:
    """Сколько строк в секунду recorder способен положить на диск.

    Худший случай: все 30 позиций каждого канала заполнены, то есть строка
    вдвое длиннее рабочей. Меряется ровно то, чем занят поток записи, —
    форматирование пачки и `write`, — без сети и без разбора кадров.
    """
    profile = DeviceProfile()
    rows_per_chunk = 512
    chunks = 240
    total = rows_per_chunk * chunks

    template = row_format(profile.channels, profile.fbg_per_channel)
    generator = np.random.default_rng(0)
    frame_no = np.arange(rows_per_chunk, dtype=np.int64)
    t_mono = np.linspace(0.0, rows_per_chunk / TARGET_RATE_HZ, rows_per_chunk)
    t_wall = t_mono + 1.7e9
    nm = 1544.8 + generator.normal(
        0.0, 0.01, (rows_per_chunk, profile.channels, profile.fbg_per_channel)
    )
    temp = np.full((rows_per_chunk, profile.channels), 16.85)

    path = tmp_path / "throughput.csv"
    format_seconds = 0.0
    started = time.perf_counter()
    with path.open("wb", buffering=1 << 20) as handle:
        for _ in range(chunks):
            mark = time.perf_counter()
            text = format_rows(template, frame_no, t_mono, t_wall, nm, temp)
            format_seconds += time.perf_counter() - mark
            handle.write(text.encode("ascii"))
        handle.flush()
    elapsed = time.perf_counter() - started

    volume = path.stat().st_size
    print("\n--- потолок записи, худший случай ---")
    print(
        f"{total} строк по {volume // total} Б за {elapsed:.2f} с: "
        f"{total / elapsed:,.0f} строк/с, {volume / elapsed / (1 << 20):.0f} МБ/с"
    )
    print(
        f"из них форматирование {format_seconds:.2f} с "
        f"({format_seconds / elapsed * 100:.0f} %), запись {elapsed - format_seconds:.2f} с"
    )
    print(f"запас над паспортными {TARGET_RATE_HZ} Гц: ×{total / elapsed / TARGET_RATE_HZ:.1f}")

    lines = path.read_text(encoding="ascii").splitlines()
    width = len(column_names(profile.channels, profile.fbg_per_channel))
    assert len(lines) == total, "строки обязаны быть все на месте"
    assert len(lines[-1].split(";")) == width, "строка обязана быть целой"
    assert total / elapsed > 5 * TARGET_RATE_HZ, (
        f"темп записи {total / elapsed:.0f} строк/с — запаса над 2000 Гц почти нет"
    )
