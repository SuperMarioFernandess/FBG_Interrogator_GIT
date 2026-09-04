"""Тесты записи измерений в CSV.

Кадры собираются `fbg.sim.encode` — тем же независимым от кодека кодом, что
и у симулятора (KB_05 №11), и подаются в `Pipeline.on_telemetry` напрямую:
сеть здесь не проверяется, она закрыта тестами транспорта и сессии.

Метки времени задаются явно. Поток записи в модульных тестах не запускается:
`pump()` вызывается вручную, иначе проверялся бы планировщик ОС, а не логика.
Отдельные тесты потока и живого темпа помечены и вынесены.

Главное, что здесь проверяется, — различимость трёх событий (KB_05 №24):
кадр записан, кадр был но пиков нет, кадров не было. Формат выбран ровно
ради этого различия, и тесты бьют в него прямо.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fbg.core.pipeline import Pipeline, PipelineConfig
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.io.recorder import (
    NM_DECIMALS,
    Recorder,
    RecorderConfig,
    column_names,
    format_gap,
    row_format,
)
from fbg.sim.encode import encode_measurement, nm_to_raw

#: Решётки стенда, ✅ скрининг: прибор распознаёт две из четырёх.
STAND_NM = (1544.80, 1551.51)

#: Тесное кольцо: разрыв у писателя устраивается за десяток кадров.
SMALL_RING = PipelineConfig(history_frames=16, ui_period_s=0.01)

#: Обычное кольцо для тестов, где разрыв не нужен.
ROOMY_RING = PipelineConfig(history_frames=4096, ui_period_s=0.01)


# --------------------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------------------


def make_frame(
    profile: DeviceProfile,
    wavelengths: dict[tuple[int, int], float] | None = None,
    *,
    temp_c: float = 16.85,
) -> bytes:
    """Кадр телеметрии: позиция → длина волны, остальное «пик не найден»."""
    divisor = profile.freq_divisor or 10
    freq = np.zeros((profile.channels, profile.fbg_per_channel), dtype=np.uint32)
    for (channel, position), nm in (wavelengths or {}).items():
        freq[channel, position] = nm_to_raw(nm, divisor)
    temp = np.full(profile.channels, round(temp_c / profile.case_temp_scale), dtype=np.int32)
    return encode_measurement(profile, freq, temp)


def feed(pipeline: Pipeline, frames: int, *, start_s: float, rate_hz: float = 2000.0) -> float:
    """Подаёт кадры с двумя распознанными решётками. Возвращает метку следующего."""
    period = 1.0 / rate_hz
    for index in range(frames):
        data = make_frame(pipeline.profile, {(0, 0): STAND_NM[0], (0, 1): STAND_NM[1]})
        pipeline.on_telemetry(data, start_s + index * period)
    return start_s + frames * period


def data_lines(path: Path) -> list[str]:
    """Строки данных файла: без имён колонок и без комментариев."""
    lines = path.read_text(encoding="ascii").splitlines()
    return [line for line in lines[1:] if line and not line.startswith("#")]


def comment_lines(path: Path) -> list[str]:
    """Строки-комментарии файла."""
    return [line for line in path.read_text(encoding="ascii").splitlines() if line.startswith("#")]


def gap_lines(path: Path) -> list[str]:
    """Маркеры разрыва файла."""
    return [line for line in comment_lines(path) if line.startswith("# GAP")]


def frame_numbers(paths: list[Path]) -> list[int]:
    """Номера кадров из всех файлов подряд, в порядке файлов."""
    numbers: list[int] = []
    for path in paths:
        numbers += [int(line.split(";", 1)[0]) for line in data_lines(path)]
    return numbers


def files_of(directory: Path, prefix: str = "data") -> list[Path]:
    """Файлы записи в порядке появления — по именам, а не по времени создания.

    Сортировка по `st_ctime_ns` выглядела очевиднее и была неверна: на Windows
    разрешение времени создания грубее интервала между ротациями, и девять
    файлов одной секунды выстраивались произвольно. Порядок обязан читаться
    из имён — ровно так его будет восстанавливать постобработка.
    """
    return sorted(directory.glob(f"{prefix}_*.csv"))


@pytest.fixture
def profile() -> DeviceProfile:
    """Профиль прибора со стенда."""
    return DeviceProfile()


@pytest.fixture
def pipeline(profile: DeviceProfile) -> Pipeline:
    """Тракт с просторным кольцом, поток публикации не запущен."""
    return Pipeline(profile, ROOMY_RING)


def recorder_for(pipeline: Pipeline, directory: Path, **kwargs: object) -> Recorder:
    """Recorder с параметрами по умолчанию, кроме заданных."""
    config = RecorderConfig(directory=directory, **kwargs)  # type: ignore[arg-type]
    return Recorder(pipeline, config)


# --------------------------------------------------------------------------------------
# Конфигурация и раскладка колонок
# --------------------------------------------------------------------------------------


def test_config_rejects_nonsense(tmp_path: Path) -> None:
    """Некорректные параметры — баг вызывающего, значит ValueError (KB_05)."""
    with pytest.raises(ValueError, match="decimation"):
        RecorderConfig(directory=tmp_path, decimation=0)
    with pytest.raises(ValueError, match="fbg_limit"):
        RecorderConfig(directory=tmp_path, fbg_limit=0)
    with pytest.raises(ValueError, match="rotate_seconds"):
        RecorderConfig(directory=tmp_path, rotate_seconds=0.0)
    with pytest.raises(ValueError, match="rotate_bytes"):
        RecorderConfig(directory=tmp_path, rotate_bytes=0)
    with pytest.raises(ValueError, match="poll_period_s"):
        RecorderConfig(directory=tmp_path, poll_period_s=0.0)
    with pytest.raises(ValueError, match="flush_period_s"):
        RecorderConfig(directory=tmp_path, flush_period_s=-1.0)
    with pytest.raises(ValueError, match="batch_limit"):
        RecorderConfig(directory=tmp_path, batch_limit=0)
    with pytest.raises(ValueError, match="chunk_frames"):
        RecorderConfig(directory=tmp_path, chunk_frames=0)
    with pytest.raises(ValueError, match="prefix"):
        RecorderConfig(directory=tmp_path, prefix="")


def test_колонки_в_порядке_из_kb03() -> None:
    """Порядок колонок задан KB_03: номер, два времени, все λ, затем температуры."""
    names = column_names(4, 30)
    assert len(names) == 3 + 4 * 30 + 4
    assert names[:3] == ("frame_no", "t_mono", "t_wall")
    assert names[3] == "ch1_fbg1_nm"
    assert names[3 + 29] == "ch1_fbg30_nm"
    assert names[3 + 30] == "ch2_fbg1_nm"
    assert names[-4:] == ("ch1_temp", "ch2_temp", "ch3_temp", "ch4_temp")


def test_шаблон_строки_совпадает_по_числу_полей() -> None:
    """Число `%`-полей шаблона обязано совпасть с числом колонок."""
    assert row_format(4, 30).count("%") == len(column_names(4, 30))
    assert row_format(2, 5).count("%") == len(column_names(2, 5))


# --------------------------------------------------------------------------------------
# Шапка
# --------------------------------------------------------------------------------------


def test_файл_целиком_ascii(pipeline: Pipeline, tmp_path: Path) -> None:
    """Ни одного не-ASCII байта, включая шапку.

    Охранник, а не придирка: `numpy.genfromtxt` открывает файл в текстовом
    режиме с локальной кодировкой ОС, и на Windows — целевой платформе
    приложения — это cp1252. Русские пояснения в шапке роняли бы наивную
    постобработку с `UnicodeDecodeError`, то есть ровно то, ради чего
    комментарии и выбраны форматом отметки разрыва. Тест нашёл это в CI
    чата №7 и оставлен, чтобы следующая правка шапки не вернула проблему.
    """
    recorder = recorder_for(pipeline, tmp_path, serial=94401220, firmware="4.10")
    recorder.open()
    feed(pipeline, 5, start_s=0.0)
    recorder.pump()
    recorder.close()

    raw = (recorder.path or tmp_path).read_bytes()
    assert raw.isascii(), "файл данных обязан быть чисто ASCII"
    # Та же проверка снаружи: файл читается любой однобайтовой кодировкой.
    assert raw.decode("cp1252").startswith("frame_no;")


def test_нельзя_положить_в_шапку_не_ascii(tmp_path: Path) -> None:
    """Поля конфигурации, попадающие в файл, проверяются на ASCII при создании."""
    for field, value in (
        ("device_model", "прибор"),
        ("firmware", "версия"),
        ("app_version", "нулевая"),
        ("prefix", "данные"),
    ):
        with pytest.raises(ValueError, match="ASCII"):
            RecorderConfig(directory=tmp_path, **{field: value})  # type: ignore[arg-type]


def test_шапка_содержит_все_поля(pipeline: Pipeline, tmp_path: Path) -> None:
    """Шапка делает файл самодостаточным: прибор, развёртка, единицы, децимация."""
    recorder = recorder_for(
        pipeline, tmp_path, serial=94401220, firmware="4.10", decimation=5, app_version="0.1.0"
    )
    recorder.open()
    feed(pipeline, 10, start_s=100.0)
    recorder.pump()
    recorder.close()

    text = "\n".join(comment_lines(recorder.path or tmp_path))
    for token in (
        "fbg-interrogator 0.1.0",
        "device=GC-97001C-03-01-A-F",
        "sn=94401220",
        "fw=4.10",
        "sweep_start_ghz=196249",
        "sweep_stop_ghz=191149",
        "step_ghz=2",
        "adc_step_ghz=2",
        "sweep_speed_hz=2000",
        "channels=4",
        "fbg_per_channel=30",
        "fbg_written=30",
        "freq_divisor=10",
        "decimation=5",
        "t_wall_start=",
        "t_wall_file=",
        "file_part=1",
        "frame_no=received_frame_index_since_recording_start step=5",
        "t_wall=unix_epoch_seconds_utc",
        "nan=peak_not_found",
        "# GAP frames=N",
        "# GAP frames=unknown",
    ):
        assert token in text, f"в шапке нет «{token}»"


def test_неизвестный_прибор_не_выдумывается(pipeline: Pipeline, tmp_path: Path) -> None:
    """Серийник и прошивка, которых приложение не знает, — `unknown` (правило №10)."""
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 2, start_s=0.0)
    recorder.pump()
    recorder.close()
    text = "\n".join(comment_lines(recorder.path or tmp_path))
    assert "sn=unknown" in text
    assert "fw=unknown" in text


def test_шапка_повторяется_после_ротации(pipeline: Pipeline, tmp_path: Path) -> None:
    """Каждый файл после ротации самодостаточен: шапка полная, номер части растёт."""
    recorder = recorder_for(pipeline, tmp_path, rotate_bytes=2000, chunk_frames=1)
    recorder.open()
    feed(pipeline, 40, start_s=0.0)
    recorder.pump()
    recorder.close()

    paths = files_of(tmp_path)
    assert len(paths) >= 2, "ротация по размеру не сработала — сравнивать нечего"
    for index, path in enumerate(paths, start=1):
        text = "\n".join(comment_lines(path))
        assert "fbg-interrogator" in text
        assert "sweep_speed_hz=2000" in text
        assert f"file_part={index}" in text
        assert path.read_text(encoding="ascii").splitlines()[0].startswith("frame_no;t_mono;")


# --------------------------------------------------------------------------------------
# Номера кадров
# --------------------------------------------------------------------------------------


def test_frame_no_сквозной_через_ротацию(pipeline: Pipeline, tmp_path: Path) -> None:
    """Нумерация не сбрасывается при смене файла: ряд непрерывен через стык."""
    recorder = recorder_for(pipeline, tmp_path, rotate_bytes=3000, chunk_frames=1)
    recorder.open()
    feed(pipeline, 60, start_s=0.0)
    recorder.pump()
    recorder.close()

    paths = files_of(tmp_path)
    assert len(paths) >= 2
    numbers = frame_numbers(paths)
    assert numbers == list(range(60))
    # Стык проверяется тем же способом, что и середина (Р41).
    first_of_second = int(data_lines(paths[1])[0].split(";", 1)[0])
    last_of_first = int(data_lines(paths[0])[-1].split(";", 1)[0])
    assert first_of_second == last_of_first + 1


def test_frame_no_начинается_с_нуля_независимо_от_возраста_тракта(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    """Нумерация сквозная **от старта записи**, а не от старта приёма."""
    feed(pipeline, 500, start_s=0.0)
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 10, start_s=1.0)
    recorder.pump()
    recorder.close()
    assert frame_numbers([recorder.path]) == list(range(10))  # type: ignore[list-item]


# --------------------------------------------------------------------------------------
# Три различимых события: строка, строка NaN, разрыв
# --------------------------------------------------------------------------------------


def test_кадр_без_пиков_это_строка_из_nan(pipeline: Pipeline, tmp_path: Path) -> None:
    """Прибор не нашёл ни одного пика — рядовая ситуация, а не пропуск (KB_05 №24)."""
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    pipeline.on_telemetry(make_frame(pipeline.profile, {(0, 0): STAND_NM[0]}), 1.0)
    pipeline.on_telemetry(make_frame(pipeline.profile, {}), 1.0005)
    pipeline.on_telemetry(make_frame(pipeline.profile, {(0, 0): STAND_NM[0]}), 1.001)
    recorder.pump()
    recorder.close()

    rows = data_lines(recorder.path)  # type: ignore[arg-type]
    assert len(rows) == 3, "кадр без пиков обязан быть строкой, а не пропуском"
    empty = rows[1].split(";")
    assert empty[0] == "1"
    assert all(value == "nan" for value in empty[3:-4]), "все λ обязаны быть nan"
    # Температура корпуса при этом измерена: пиков нет, а прибор жив.
    assert all(float(value) == pytest.approx(16.85) for value in empty[-4:])
    assert gap_lines(recorder.path) == [], "кадр без пиков разрывом не является"  # type: ignore[arg-type]


def test_разрыв_это_комментарий_а_не_строка_nan(profile: DeviceProfile, tmp_path: Path) -> None:
    """Потеря кадров отмечается `# GAP`, и ни одной строки при этом не выдумывается.

    Стимул: тесное кольцо и писатель, который между двумя `pump` пропускает
    больше кадров, чем кольцо вмещает.
    """
    pipeline = Pipeline(profile, SMALL_RING)
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 4, start_s=10.0)
    recorder.pump()
    feed(pipeline, 40, start_s=20.0)
    recorder.pump()
    recorder.close()

    rows = data_lines(recorder.path)  # type: ignore[arg-type]
    numbers = [int(row.split(";", 1)[0]) for row in rows]
    marks = gap_lines(recorder.path)  # type: ignore[arg-type]

    assert len(marks) == 1, "потеря обязана быть отмечена ровно одним маркером"
    lost = int(marks[0].split("frames=")[1].split()[0])
    assert lost > 0
    # Номера подтверждают ту же потерю арифметикой (Р41).
    assert numbers[4] - numbers[3] == lost + 1
    assert numbers[-1] + 1 - len(numbers) == lost
    # Ни одной выдуманной строки: NaN-строк в файле нет вовсе.
    for row in rows:
        assert not all(value == "nan" for value in row.split(";")[3:-4])


def test_сетевой_разрыв_пишется_один_раз_с_точными_границами(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    """Обрыв источника виден, хотя pipeline не может посчитать потерянные кадры.

    У телеметрии нет sequence number, поэтому число кадров во время сетевой
    тишины не угадывается по 2000 Гц. Session сообщает Recorder только две
    реально наблюдённые границы, а тот ставит `frames=unknown` ровно между
    последним кадром до паузы и первым после неё.
    """
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 4, start_s=10.0)
    recorder.mark_gap(10.0015, 20.0)
    feed(pipeline, 3, start_s=20.0)
    recorder.pump()
    recorder.close()

    marks = gap_lines(recorder.path)  # type: ignore[arg-type]
    assert marks == ["# GAP frames=unknown t_mono_from=10.001500 t_mono_to=20.000000"]
    assert frame_numbers([recorder.path]) == list(range(7))  # type: ignore[list-item]
    assert recorder.stats.gaps == 1
    assert recorder.stats.lost_frames == 0, "неизвестное число кадров нельзя выдавать за оценку"


def test_маркер_разрыва_несёт_границы_по_времени(profile: DeviceProfile, tmp_path: Path) -> None:
    """`t_mono_from` — последняя записанная строка, `t_mono_to` — первая после разрыва."""
    pipeline = Pipeline(profile, SMALL_RING)
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 4, start_s=10.0)
    recorder.pump()
    feed(pipeline, 40, start_s=20.0)
    recorder.pump()
    recorder.close()

    rows = data_lines(recorder.path)  # type: ignore[arg-type]
    mark = gap_lines(recorder.path)[0]  # type: ignore[arg-type]
    since = float(mark.split("t_mono_from=")[1].split()[0])
    until = float(mark.split("t_mono_to=")[1].split()[0])
    assert since == pytest.approx(float(rows[3].split(";")[1]))
    assert until == pytest.approx(float(rows[4].split(";")[1]))


def test_разрыв_до_первой_строки_даёт_nan_левую_границу(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Разрыв в самом начале записи: слева границы нет, и она честно `nan`.

    Подставлять вместо неё момент открытия файла значило бы выдать за метку
    кадра время, которого в данных нет.
    """
    pipeline = Pipeline(profile, SMALL_RING)
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    # Ни одной строки ещё не записано, а кольцо уже провернулось.
    feed(pipeline, 40, start_s=10.0)
    recorder.pump()
    recorder.close()

    marks = gap_lines(recorder.path)  # type: ignore[arg-type]
    assert len(marks) == 1
    assert "t_mono_from=nan" in marks[0]
    assert "t_mono_to=" in marks[0] and "t_mono_to=nan" not in marks[0]
    assert data_lines(recorder.path)[0].split(";")[0] == "0"  # type: ignore[arg-type]


def test_разрыв_без_последующей_строки_даёт_nan_правую_границу(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """При децимации разрыв может оказаться последним событием файла.

    Стимул точный, а не случайный: кольцо на 8 кадров, децимация 10. После
    провала курсор оказывается на кадрах 12…19 — ни одного кратного десяти,
    писать нечего, и отставание при этом уже нулевое. Маркер обязан попасть
    в файл всё равно: потеря не должна исчезать оттого, что после неё нечего
    записать. Правая граница остаётся `nan`.
    """
    pipeline = Pipeline(profile, PipelineConfig(history_frames=8, ui_period_s=0.01))
    recorder = recorder_for(pipeline, tmp_path, decimation=10)
    recorder.open()
    feed(pipeline, 1, start_s=10.0)
    recorder.pump()
    feed(pipeline, 19, start_s=20.0)
    recorder.pump()
    recorder.close()

    marks = gap_lines(recorder.path)  # type: ignore[arg-type]
    assert len(marks) == 1, "разрыв обязан быть отмечен даже без строки после него"
    assert "frames=11" in marks[0]
    assert "t_mono_to=nan" in marks[0]
    assert frame_numbers([recorder.path]) == [0]  # type: ignore[list-item]
    assert recorder.stats.lost_frames == 11


def test_format_gap_печатает_обе_границы() -> None:
    """Маркер собирается ровно в формате Р42."""
    assert format_gap(1247, 123.456789, 124.080123) == (
        "# GAP frames=1247 t_mono_from=123.456789 t_mono_to=124.080123\n"
    )
    assert "t_mono_from=nan" in format_gap(3, float("nan"), 1.0)
    assert format_gap(None, 1.0, 2.0) == (
        "# GAP frames=unknown t_mono_from=1.000000 t_mono_to=2.000000\n"
    )


# --------------------------------------------------------------------------------------
# Децимация
# --------------------------------------------------------------------------------------


def test_децимация_даёт_ровный_шаг_и_не_считается_разрывом(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    """Пропуск по настройке — не потеря: маркера нет, шаг номеров равен N."""
    recorder = recorder_for(pipeline, tmp_path, decimation=10)
    recorder.open()
    feed(pipeline, 100, start_s=0.0)
    recorder.pump()
    recorder.close()

    numbers = frame_numbers([recorder.path])  # type: ignore[list-item]
    assert len(numbers) == 10
    assert numbers == [index * 10 for index in range(10)]
    assert gap_lines(recorder.path) == []  # type: ignore[arg-type]
    assert recorder.stats.lost_frames == 0


def test_децимация_переживает_пустые_пачки(pipeline: Pipeline, tmp_path: Path) -> None:
    """Пачка, целиком выпавшая по децимации, разрывом не становится.

    Тонкое место: курсор в этом случае сдвигает позицию и возвращает None,
    и наивный учёт «позиция уехала — значит потеря» дал бы ложный `# GAP`.
    """
    recorder = recorder_for(pipeline, tmp_path, decimation=10)
    recorder.open()
    for index in range(50):
        pipeline.on_telemetry(make_frame(pipeline.profile, {(0, 0): STAND_NM[0]}), index * 0.0005)
        recorder.pump()  # по одному кадру за раз: 9 пачек из 10 пустые
    recorder.close()

    assert gap_lines(recorder.path) == []  # type: ignore[arg-type]
    assert frame_numbers([recorder.path]) == [0, 10, 20, 30, 40]  # type: ignore[list-item]


def test_ограничение_числа_датчиков_отмечено_в_шапке(pipeline: Pipeline, tmp_path: Path) -> None:
    """Пишем только первые N позиций канала — и говорим об этом в шапке."""
    recorder = recorder_for(pipeline, tmp_path, fbg_limit=4)
    recorder.open()
    feed(pipeline, 5, start_s=0.0)
    recorder.pump()
    recorder.close()

    names = (recorder.path or tmp_path).read_text(encoding="ascii").splitlines()[0].split(";")
    assert len(names) == 3 + 4 * 4 + 4
    assert "ch1_fbg4_nm" in names
    assert "ch1_fbg5_nm" not in names
    assert "fbg_written=4" in "\n".join(comment_lines(recorder.path))  # type: ignore[arg-type]
    assert "fbg_per_channel=30" in "\n".join(comment_lines(recorder.path))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Значения
# --------------------------------------------------------------------------------------


def test_пишутся_сырые_нанометры(pipeline: Pipeline, tmp_path: Path) -> None:
    """В файл идёт λ = c / f того же кадра — без калибровки (правило №4)."""
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    pipeline.on_telemetry(make_frame(pipeline.profile, {(0, 0): STAND_NM[0]}), 5.0)
    recorder.pump()
    recorder.close()

    row = data_lines(recorder.path)[0].split(";")  # type: ignore[arg-type]
    raw = nm_to_raw(STAND_NM[0], 10)
    expected = C_NM_GHZ / (raw / 10)
    assert float(row[3]) == pytest.approx(expected, abs=10.0**-NM_DECIMALS)
    # Квантование прибора — 0.8 пм; четыре знака мельче его втрое с лишним.
    assert abs(float(row[3]) - STAND_NM[0]) < 0.001


def test_время_записано_обеими_шкалами(pipeline: Pipeline, tmp_path: Path) -> None:
    """`t_mono` — та же метка, что у кадра; `t_wall` — Unix epoch, сдвиг постоянный."""
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 3, start_s=42.0)
    recorder.pump()
    recorder.close()

    rows = [line.split(";") for line in data_lines(recorder.path)]  # type: ignore[arg-type]
    monos = [float(row[1]) for row in rows]
    walls = [float(row[2]) for row in rows]
    assert monos == pytest.approx([42.0, 42.0005, 42.001])
    offsets = [wall - mono for wall, mono in zip(walls, monos, strict=True)]
    assert offsets[0] == pytest.approx(offsets[-1], abs=1e-6), "сдвиг обязан быть постоянным"
    assert walls[0] > 1_700_000_000.0, "t_wall обязано быть настенным временем, а не нулём"


# --------------------------------------------------------------------------------------
# Ротация
# --------------------------------------------------------------------------------------


def test_ротация_по_размеру(pipeline: Pipeline, tmp_path: Path) -> None:
    """Файл закрывается, когда перерос лимит; данные не теряются."""
    recorder = recorder_for(pipeline, tmp_path, rotate_bytes=4000, chunk_frames=1)
    recorder.open()
    feed(pipeline, 50, start_s=0.0)
    recorder.pump()
    recorder.close()

    paths = files_of(tmp_path)
    assert len(paths) >= 2
    assert frame_numbers(paths) == list(range(50))
    for path in paths[:-1]:
        assert path.stat().st_size >= 4000


def test_ротация_по_времени(pipeline: Pipeline, tmp_path: Path) -> None:
    """Второй лимит независим от первого: по времени файл тоже сменяется."""
    recorder = recorder_for(
        pipeline, tmp_path, rotate_seconds=0.05, rotate_bytes=None, chunk_frames=1
    )
    recorder.open()
    feed(pipeline, 5, start_s=0.0)
    recorder.pump()
    time.sleep(0.06)
    feed(pipeline, 5, start_s=1.0)
    recorder.pump()
    recorder.close()

    paths = files_of(tmp_path)
    assert len(paths) == 2
    assert frame_numbers(paths) == list(range(10))


def test_имена_файлов_не_сталкиваются(pipeline: Pipeline, tmp_path: Path) -> None:
    """Разрешение имени — секунда; при частой ротации добавляется номер.

    Дописывать в уже существующий файл нельзя: его шапка описывает другую запись.
    """
    recorder = recorder_for(pipeline, tmp_path, rotate_bytes=2000, chunk_frames=1)
    recorder.open()
    feed(pipeline, 40, start_s=0.0)
    recorder.pump()
    recorder.close()

    paths = files_of(tmp_path)
    assert len(paths) >= 2
    assert len({path.name for path in paths}) == len(paths)
    assert all(path.stat().st_size > 0 for path in paths)
    # Номер дополнен нулями, иначе `_10` встало бы перед `_2`.
    assert paths[1].name.endswith("_002.csv")


def test_порядок_имён_совпадает_с_порядком_создания(pipeline: Pipeline, tmp_path: Path) -> None:
    """Алфавитная сортировка имён обязана давать порядок частей записи.

    Свойство не косметическое: постобработка склеивает файлы по
    `sorted(glob(...))`, и перепутанный порядок разорвал бы непрерывный ряд
    кадров. Опираться на время создания файла нельзя — на Windows его
    разрешение грубее интервала между ротациями, что и уронило CI чата №7.

    Стимул точный: лимит в 2 КБ при строке около 550 байт даёт десяток
    файлов за одну секунду, то есть двузначные номера частей, — именно там
    ломается сортировка без дополнения нулями.
    """
    recorder = recorder_for(pipeline, tmp_path, rotate_bytes=2000, chunk_frames=1)
    recorder.open()
    feed(pipeline, 60, start_s=0.0)
    recorder.pump()
    recorder.close()

    paths = sorted(tmp_path.glob("data_*.csv"))
    assert len(paths) >= 10, "стимул не сработал: двузначных номеров частей нет"
    parts = [
        int(
            next(line for line in comment_lines(path) if "file_part=" in line).split("file_part=")[
                1
            ]
        )
        for path in paths
    ]
    assert parts == list(range(1, len(paths) + 1))
    assert frame_numbers(paths) == list(range(60))


# --------------------------------------------------------------------------------------
# Читаемость сторонними инструментами
# --------------------------------------------------------------------------------------


def test_файл_читается_pandas(pipeline: Pipeline, tmp_path: Path) -> None:
    """`pandas.read_csv(sep=';', comment='#')` даёт непрерывный ряд без маркеров.

    pandas разрешён только здесь — в постобработке готового файла (KB_05 №2).
    В тракте приёма его нет и быть не может.
    """
    pandas = pytest.importorskip("pandas")
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    pipeline.on_telemetry(make_frame(pipeline.profile, {}), 0.0)
    feed(pipeline, 20, start_s=0.001)
    recorder.pump()
    recorder.close()

    table = pandas.read_csv(recorder.path, sep=";", comment="#")
    assert list(table.columns) == list(column_names(4, 30))
    assert len(table) == 21
    assert table["frame_no"].tolist() == list(range(21))
    assert bool(np.isnan(table["ch1_fbg1_nm"].iloc[0])), "кадр без пиков обязан читаться как NaN"
    assert table["ch1_fbg1_nm"].iloc[1] == pytest.approx(STAND_NM[0], abs=0.001)


def test_файл_читается_numpy_genfromtxt(pipeline: Pipeline, tmp_path: Path) -> None:
    """`numpy.genfromtxt(delimiter=';', names=True)` читает файл без лишних аргументов.

    Ради этого строка имён колонок стоит первой: genfromtxt берёт имена
    из первой строки файла, и блок комментариев сверху сломал бы разбор.
    """
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 15, start_s=0.0)
    recorder.pump()
    recorder.close()

    table = np.genfromtxt(recorder.path, delimiter=";", names=True)
    assert table.dtype.names is not None
    assert list(table.dtype.names) == list(column_names(4, 30))
    assert table.shape == (15,)
    assert table["ch1_fbg2_nm"][0] == pytest.approx(STAND_NM[1], abs=0.001)
    assert np.isnan(table["ch1_fbg3_nm"]).all()


def test_маркер_разрыва_невидим_наивной_постобработке(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Комментарий пропускается обоими читателями: наивный разбор не ломается."""
    pipeline = Pipeline(profile, SMALL_RING)
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 4, start_s=10.0)
    recorder.pump()
    feed(pipeline, 40, start_s=20.0)
    recorder.pump()
    recorder.close()

    assert gap_lines(recorder.path), "стимул не сработал: разрыва нет"  # type: ignore[arg-type]
    table = np.genfromtxt(recorder.path, delimiter=";", names=True)
    assert table.shape == (len(data_lines(recorder.path)),)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Надёжность
# --------------------------------------------------------------------------------------


def test_файл_на_диске_состоит_из_целых_строк(pipeline: Pipeline, tmp_path: Path) -> None:
    """Процесс убит без закрытия файла: на диске только целые строки.

    Сброс буфера ограничивает потерю хвостом длиной `flush_period_s`.
    Всё, что до него доехало, разбирается без ухищрений.
    """
    recorder = recorder_for(pipeline, tmp_path, flush_period_s=0.0001, chunk_frames=8)
    recorder.open()
    feed(pipeline, 64, start_s=0.0)
    recorder.pump()
    time.sleep(0.001)
    feed(pipeline, 64, start_s=1.0)
    recorder.pump()
    # `close` намеренно не вызывается: имитируем падение процесса.

    path = recorder.path
    assert path is not None
    rows = data_lines(path)
    assert len(rows) >= 64
    assert all(len(row.split(";")) == len(column_names(4, 30)) for row in rows)
    assert frame_numbers([path]) == list(range(len(rows)))


def test_обрыв_на_середине_строки_портит_только_её(pipeline: Pipeline, tmp_path: Path) -> None:
    """Недописанная последняя строка допустима, повреждённая середина — нет."""
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    feed(pipeline, 40, start_s=0.0)
    recorder.pump()
    recorder.close()

    raw = (recorder.path or tmp_path).read_bytes()
    truncated = tmp_path / "truncated.csv"
    truncated.write_bytes(raw[: len(raw) - 500])

    lines = truncated.read_text(encoding="ascii").splitlines()
    rows = [line for line in lines[1:] if line and not line.startswith("#")]
    width = len(column_names(4, 30))
    assert all(len(row.split(";")) == width for row in rows[:-1]), "середина обязана быть цела"
    numbers = [int(row.split(";", 1)[0]) for row in rows[:-1]]
    assert numbers == list(range(len(numbers)))


class ExplodingFile:
    """Файл, который отказывается писать. Стимул для «диск заполнен»."""

    def __init__(self) -> None:
        self.closed = False

    def write(self, payload: bytes) -> int:
        """Всегда отказ: ENOSPC — самая обычная причина остановки записи."""
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        """Ничего не делает: до сброса дело не доходит."""

    def close(self) -> None:
        """Отмечает, что recorder закрыл файл после отказа."""
        self.closed = True


def test_ошибка_записи_не_роняет_приём(pipeline: Pipeline, tmp_path: Path) -> None:
    """Отказ диска останавливает только запись; приём кадров продолжается."""
    seen: list[str] = []
    config = RecorderConfig(directory=tmp_path)
    recorder = Recorder(pipeline, config, on_error=seen.append)
    recorder.open()
    feed(pipeline, 5, start_s=0.0)
    recorder.pump()

    # Подмена файла — стимул: реального заполненного диска в тестах нет.
    real = recorder._file
    assert real is not None
    real.close()
    broken = ExplodingFile()
    recorder._file = broken  # type: ignore[assignment]

    feed(pipeline, 5, start_s=1.0)
    assert recorder.pump() == 0, "запись обязана остановиться, а не бросить исключение"
    assert recorder.stats.error is not None
    assert "No space left" in recorder.stats.error
    assert seen and "No space left" in seen[0]
    assert not recorder.is_open
    assert broken.closed, "после отказа файл обязан быть закрыт"

    # Приём данных отказа записи не заметил.
    before = pipeline.sequence
    feed(pipeline, 20, start_s=2.0)
    assert pipeline.sequence == before + 20
    assert pipeline.metrics().parse_errors == 0
    # Повторный `pump` не оживает и не бросает.
    assert recorder.pump() == 0
    recorder.stop()


def test_отсутствующая_папка_создаётся(pipeline: Pipeline, tmp_path: Path) -> None:
    """Папку записи создаёт сам recorder: отдельного шага настройки не нужно."""
    target = tmp_path / "nested" / "run01"
    recorder = recorder_for(pipeline, target)
    recorder.open()
    feed(pipeline, 3, start_s=0.0)
    recorder.pump()
    recorder.close()
    assert target.is_dir()
    assert len(files_of(target)) == 1


# --------------------------------------------------------------------------------------
# Поток записи
# --------------------------------------------------------------------------------------


def test_поток_забирает_кадры_и_не_остаётся(pipeline: Pipeline, tmp_path: Path) -> None:
    """`start`/`stop` работают, поток добирает остаток кольца и завершается."""
    recorder = recorder_for(pipeline, tmp_path, poll_period_s=0.002)
    recorder.start()
    assert recorder.is_running
    for index in range(200):
        pipeline.on_telemetry(make_frame(pipeline.profile, {(0, 0): STAND_NM[0]}), index * 0.0005)
        if index % 50 == 0:
            time.sleep(0.003)
    recorder.stop()

    assert not recorder.is_running
    assert "fbg-recorder" not in {thread.name for thread in threading.enumerate()}
    assert frame_numbers([recorder.path]) == list(range(200))  # type: ignore[list-item]
    assert recorder.stats.rows == 200
    assert recorder.stats.error is None


def test_контекстный_менеджер_закрывает_файл(pipeline: Pipeline, tmp_path: Path) -> None:
    """`with` открывает запись и гарантированно закрывает её."""
    with recorder_for(pipeline, tmp_path, poll_period_s=0.002) as recorder:
        feed(pipeline, 20, start_s=0.0)
        time.sleep(0.02)
    assert not recorder.is_open
    assert not recorder.is_running
    assert recorder.stats.rows == 20


def test_повторные_open_и_stop_безвредны(pipeline: Pipeline, tmp_path: Path) -> None:
    """Идемпотентность: второй `open` не заводит второй файл, второй `stop` не падает."""
    recorder = recorder_for(pipeline, tmp_path)
    recorder.open()
    recorder.open()
    feed(pipeline, 5, start_s=0.0)
    recorder.pump()
    recorder.stop()
    recorder.stop()
    assert len(files_of(tmp_path)) == 1
    assert recorder.stats.files == 1
