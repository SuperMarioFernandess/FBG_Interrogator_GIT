from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from fbg.core.averaging import fixed_window_average
from fbg.core.profile import DeviceProfile
from fbg.io.averaging import average_recording
from fbg.io.recorder import RecorderConfig, build_header, column_names, format_gap

PROFILE = DeviceProfile()
START = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


def _row(frame_no: int, t_mono: float, first_nm: str) -> str:
    fields = [str(frame_no), f"{t_mono:.6f}", f"{1700000000 + t_mono:.6f}"]
    fields += [first_nm, "nan", "nan", "nan"]
    fields += ["20.0", "20.0", "20.0", "20.0"]
    return ";".join(fields) + "\n"


def _part(path: Path, part: int, lines: list[str]) -> Path:
    config = RecorderConfig(directory=path.parent, decimation=10)
    columns = column_names(PROFILE.channels, 1)
    header = build_header(
        PROFILE,
        config,
        columns=columns,
        fbg_written=1,
        t_wall_start=START,
        t_wall_file=START,
        part=part,
    )
    path.write_text(header + "".join(lines), encoding="ascii")
    return path


def _data_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]


def test_ротированная_запись_с_gap_decimation_nan_даёт_mean_n_sigma(tmp_path: Path) -> None:
    gap1 = format_gap(17, 0.5, 0.8)
    gap2 = format_gap(None, 1.4, 1.7)
    first = _part(
        tmp_path / "zzz.csv",
        1,
        [
            _row(0, 0.0, "1544.0"),
            _row(10, 0.5, "1546.0"),
            gap1,
            _row(20, 1.0, "1548.0"),
        ],
    )
    second = _part(
        tmp_path / "aaa.csv",
        2,
        [
            _row(30, 1.4, "1550.0"),
            gap2,
            _row(40, 1.8, "nan"),
            _row(50, 2.2, "1552.0"),
        ],
    )
    before_first = first.read_bytes()
    before_second = second.read_bytes()

    result = average_recording(second, 1.0)

    assert result.inputs == (first, second)
    assert result.windows == 5
    assert result.gaps == 2
    assert first.read_bytes() == before_first
    assert second.read_bytes() == before_second
    payload = result.output.read_bytes()
    payload.decode("ascii")
    text = payload.decode("ascii")
    assert gap1.strip() in text
    assert gap2.strip() in text
    assert "# source_metadata freq_divisor=" in text
    assert "decimation=10" in text

    lines = _data_lines(result.output)
    header = lines[0].split(";")
    mean_index = header.index("ch1_fbg1_nm_mean")
    n_index = header.index("ch1_fbg1_nm_n")
    sigma_index = header.index("ch1_fbg1_nm_sigma")

    first_window = lines[1].split(";")
    assert float(first_window[0]) == pytest.approx(0.0)
    assert float(first_window[1]) == pytest.approx(0.5)
    assert float(first_window[mean_index]) == pytest.approx(1545.0)
    assert int(first_window[n_index]) == 2
    assert float(first_window[sigma_index]) == pytest.approx(1.0)

    first_empty_window = lines[2].split(";")
    assert float(first_empty_window[0]) == pytest.approx(0.8)
    assert float(first_empty_window[1]) == pytest.approx(1.0)
    assert first_empty_window[mean_index] == "nan"
    assert int(first_empty_window[n_index]) == 0
    assert first_empty_window[sigma_index] == "nan"

    second_empty_window = lines[4].split(";")
    assert float(second_empty_window[0]) == pytest.approx(1.7)
    assert float(second_empty_window[1]) == pytest.approx(2.0)
    assert second_empty_window[mean_index] == "nan"
    assert int(second_empty_window[n_index]) == 0
    assert second_empty_window[sigma_index] == "nan"

    data = np.genfromtxt(result.output, delimiter=";", names=True, comments="#")
    assert data.dtype.names is not None
    assert "ch1_fbg1_nm_mean" in data.dtype.names
    assert "ch1_fbg1_nm_n" in data.dtype.names
    assert "ch1_fbg1_nm_sigma" in data.dtype.names


def test_gap_ровно_на_границе_сетки_не_создаёт_нулевого_окна(tmp_path: Path) -> None:
    gap = format_gap(None, 0.05, 0.075)
    source = _part(
        tmp_path / "data.csv",
        1,
        [
            _row(0, 0.049, "1544.0"),
            _row(10, 0.05, "1546.0"),
            gap,
            _row(20, 0.08, "1548.0"),
        ],
    )

    result = average_recording(source, 0.05)

    assert result.windows == 2
    lines = _data_lines(result.output)
    header = lines[0].split(";")
    mean_index = header.index("ch1_fbg1_nm_mean")
    n_index = header.index("ch1_fbg1_nm_n")
    first = lines[1].split(";")
    second = lines[2].split(";")
    assert (float(first[0]), float(first[1])) == pytest.approx((0.0, 0.05))
    assert float(first[mean_index]) == pytest.approx(1545.0)
    assert int(first[n_index]) == 2
    assert (float(second[0]), float(second[1])) == pytest.approx((0.075, 0.10))
    assert float(second[mean_index]) == pytest.approx(1548.0)
    assert int(second[n_index]) == 1


def test_потоковый_экспорт_совпадает_с_общей_оконной_математикой(tmp_path: Path) -> None:
    gap1 = format_gap(17, 0.5, 0.8)
    gap2 = format_gap(None, 1.4, 1.7)
    first = _part(
        tmp_path / "first.csv",
        1,
        [
            _row(0, 0.0, "1544.0"),
            _row(10, 0.5, "1546.0"),
            gap1,
            _row(20, 1.0, "1548.0"),
        ],
    )
    _part(
        tmp_path / "second.csv",
        2,
        [
            _row(30, 1.4, "1550.0"),
            gap2,
            _row(40, 1.8, "nan"),
            _row(50, 2.2, "1552.0"),
        ],
    )

    exported = average_recording(first, 1.0)
    reference = fixed_window_average(
        np.asarray([0.0, 0.5, 1.0, 1.4, 1.8, 2.2]),
        np.asarray([[1544.0], [1546.0], [1548.0], [1550.0], [np.nan], [1552.0]]),
        1.0,
        gaps=((0.5, 0.8), (1.4, 1.7)),
    )

    lines = _data_lines(exported.output)
    header = lines[0].split(";")
    mean_index = header.index("ch1_fbg1_nm_mean")
    n_index = header.index("ch1_fbg1_nm_n")
    sigma_index = header.index("ch1_fbg1_nm_sigma")
    rows = [line.split(";") for line in lines[1:]]

    np.testing.assert_allclose(
        [float(row[0]) for row in rows], reference.start_mono, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        [float(row[1]) for row in rows], reference.stop_mono, rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        [float(row[mean_index]) for row in rows],
        reference.mean[:, 0],
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
    np.testing.assert_array_equal([int(row[n_index]) for row in rows], reference.n[:, 0])
    np.testing.assert_allclose(
        [float(row[sigma_index]) for row in rows],
        reference.sigma[:, 0],
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
