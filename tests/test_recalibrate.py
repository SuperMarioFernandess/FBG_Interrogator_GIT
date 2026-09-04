"""Постобработка Recorder: сырые nm остаются первичными, калибровка — производная."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from fbg.core.calibration import Sensor, SensorType
from fbg.core.profile import DeviceProfile
from fbg.io.recalibrate import calibrated_path, recalibrate_recording, recording_parts
from fbg.io.recorder import RecorderConfig, build_header, column_names

PROFILE = DeviceProfile()
START = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def sensor(**overrides: object) -> Sensor:
    params: dict[str, object] = {
        "id": "T1",
        "name": "Температура",
        "channel": 0,
        "type": SensorType.TEMPERATURE,
        "expected_nm": 1544.80,
        "window_nm": 0.20,
        "value0": 25.0,
        "k1": 100.0,
    }
    params.update(overrides)
    return Sensor(**params)  # type: ignore[arg-type]


def make_part(path: Path, part: int, lines: list[str], *, decimation: int = 1) -> Path:
    config = RecorderConfig(directory=path.parent, decimation=decimation)
    columns = column_names(PROFILE.channels, 2)
    header = build_header(
        PROFILE,
        config,
        columns=columns,
        fbg_written=2,
        t_wall_start=START,
        t_wall_file=START,
        part=part,
    )
    path.write_text(header + "".join(lines), encoding="ascii")
    return path


def row(frame_no: int, first_nm: str, second_nm: str = "nan") -> str:
    # 3 служебных поля + 4 канала × 2 nm + 4 температуры.
    fields = [str(frame_no), f"{frame_no / 10:.6f}", f"{1700000000 + frame_no:.6f}"]
    fields += [first_nm, second_nm, "nan", "nan", "nan", "nan", "nan", "nan"]
    fields += ["20.00", "20.00", "20.00", "20.00"]
    return ";".join(fields) + "\n"


def data_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]


def test_пересчёт_добавляет_величину_не_меняя_raw_nm(tmp_path: Path) -> None:
    source = make_part(tmp_path / "data_0001.csv", 1, [row(0, "1544.8000"), row(1, "1544.8100")])
    before = source.read_text(encoding="ascii")

    result = recalibrate_recording(source, (sensor(),))

    assert result.inputs == (source,)
    assert result.outputs == (calibrated_path(source),)
    assert result.rows == 2 and result.gaps == 0
    assert source.read_text(encoding="ascii") == before
    rows = data_lines(result.outputs[0])
    header = rows[0].split(";")
    assert header[-1] == "sensor001_T1_value"
    assert rows[1].split(";")[:-1] == row(0, "1544.8000").strip().split(";")
    assert float(rows[1].split(";")[-1]) == pytest.approx(25.0)
    assert float(rows[2].split(";")[-1]) == pytest.approx(26.0)


def test_nan_и_неоднозначность_остаются_nan(tmp_path: Path) -> None:
    source = make_part(
        tmp_path / "data_0001.csv",
        1,
        [
            row(0, "nan"),
            row(1, "1544.8000", "1544.8500"),  # два кандидата в одном окне
        ],
    )
    output = recalibrate_recording(source, (sensor(),)).outputs[0]
    rows = data_lines(output)
    assert rows[1].split(";")[-1] == "nan"
    assert rows[2].split(";")[-1] == "nan"


def test_ротация_сортируется_по_file_part_а_не_имени(tmp_path: Path) -> None:
    second = make_part(tmp_path / "zzz.csv", 2, [row(20, "1544.8200")], decimation=10)
    first = make_part(tmp_path / "aaa.csv", 1, [row(10, "1544.8100")], decimation=10)

    assert recording_parts(second) == (first, second)
    result = recalibrate_recording(second, (sensor(),))
    assert result.inputs == (first, second)
    assert result.outputs == (calibrated_path(first), calibrated_path(second))
    assert result.rows == 2
    # frame_no сохраняет исходную децимацию, ничего не перенумеровывается.
    assert data_lines(result.outputs[0])[1].startswith("10;")
    assert data_lines(result.outputs[1])[1].startswith("20;")


def test_gap_обоих_видов_копируется_буквально(tmp_path: Path) -> None:
    known = "# GAP frames=17\n"
    unknown = "# GAP frames=unknown t_mono_from=1.000000 t_mono_to=2.000000\n"
    source = make_part(
        tmp_path / "data_0001.csv",
        1,
        [row(0, "1544.8000"), known, row(1, "1544.8100"), unknown, row(2, "nan")],
    )

    result = recalibrate_recording(source, (sensor(),))

    text = result.outputs[0].read_text(encoding="ascii")
    assert known in text and unknown in text
    assert result.gaps == 2 and result.rows == 3


def test_выходной_csv_остаётся_ascii_даже_при_русском_имени(tmp_path: Path) -> None:
    source = make_part(tmp_path / "data_0001.csv", 1, [row(0, "1544.8000")])
    output = recalibrate_recording(source, (sensor(id="температура", name="Свая №3"),)).outputs[0]
    payload = output.read_bytes()
    payload.decode("ascii")
    assert b"sensor001_sensor_value" in payload.splitlines()[0]


def test_пустой_набор_датчиков_не_создаёт_бессмысленный_файл(tmp_path: Path) -> None:
    source = make_part(tmp_path / "data_0001.csv", 1, [row(0, "1544.8000")])
    with pytest.raises(ValueError, match="нет датчиков"):
        recalibrate_recording(source, ())
    assert not calibrated_path(source).exists()


def test_вход_читается_numpy_после_пересчёта(tmp_path: Path) -> None:
    """Производный файл остаётся совместимым с простым numpy-чтением."""
    source = make_part(tmp_path / "data_0001.csv", 1, [row(0, "1544.8000"), row(1, "nan")])
    output = recalibrate_recording(source, (sensor(),)).outputs[0]
    data = np.genfromtxt(output, delimiter=";", names=True, comments="#")
    assert data.dtype.names is not None
    assert data.dtype.names[-1] == "sensor001_T1_value"
    assert data["sensor001_T1_value"][0] == pytest.approx(25.0)
    assert np.isnan(data["sensor001_T1_value"][1])
