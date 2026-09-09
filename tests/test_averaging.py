import numpy as np
import pytest

from fbg.core import codec
from fbg.core.averaging import fixed_window_average
from fbg.core.profile import DeviceProfile
from tests.synthetic import load_vectors


def test_усреднение_работает_на_реальном_векторе_телеметрии() -> None:
    profile = DeviceProfile()
    raw = load_vectors("measurement_real.hex")["measurement_real"]
    first = codec.parse_measurement(raw, profile, t_mono=0.001).unwrap().wavelength_nm()[0, 0]
    second = codec.parse_measurement(raw, profile, t_mono=0.002).unwrap().wavelength_nm()[0, 0]

    result = fixed_window_average(
        np.asarray([0.001, 0.002]),
        np.asarray([[first], [second]]),
        0.05,
    )

    assert result.windows == 1
    assert result.mean[0, 0] == pytest.approx(1544.785, abs=0.01)
    assert result.n[0, 0] == 2
    assert result.sigma[0, 0] == 0.0


def test_пустое_по_валидным_данным_окно_даёт_nan_и_n_0() -> None:
    result = fixed_window_average(
        np.asarray([0.001, 0.010, 0.020]),
        np.asarray([[np.nan], [np.nan], [np.nan]]),
        0.05,
    )

    assert result.windows == 1
    assert result.n[:, 0].tolist() == [0]
    assert np.isnan(result.mean[0, 0])
    assert np.isnan(result.sigma[0, 0])


def test_один_валидный_из_ста_сохраняет_значение_n_1_и_sigma_0() -> None:
    times = np.arange(100, dtype=np.float64) * 0.0005
    values = np.full((100, 1), np.nan)
    values[37, 0] = 1549.1234

    result = fixed_window_average(times, values, 0.05)

    assert result.windows == 1
    assert result.mean[0, 0] == pytest.approx(1549.1234)
    assert result.n[0, 0] == 1
    assert result.sigma[0, 0] == 0.0


def test_gap_режет_одно_глобальное_окно_на_два_по_точным_границам() -> None:
    times = np.asarray([0.010, 0.020, 0.080, 0.090])
    values = np.asarray([[1.0], [3.0], [5.0], [7.0]])

    result = fixed_window_average(times, values, 0.1, gaps=((0.025, 0.075),))

    assert result.windows == 2
    assert result.start_mono.tolist() == pytest.approx([0.0, 0.075])
    assert result.stop_mono.tolist() == pytest.approx([0.025, 0.1])
    assert result.mean[:, 0].tolist() == pytest.approx([2.0, 6.0])
    assert result.n[:, 0].tolist() == [2, 2]


def test_границы_окон_не_зависят_от_повторного_такта() -> None:
    times = np.asarray([12.011, 12.049, 12.051, 12.099])
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]])

    first = fixed_window_average(times, values, 0.05)
    second = fixed_window_average(times.copy(), values.copy(), 0.05)

    np.testing.assert_array_equal(first.start_mono, second.start_mono)
    np.testing.assert_array_equal(first.stop_mono, second.stop_mono)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.n, second.n)


def test_sigma_известного_набора_и_константы() -> None:
    times = np.asarray([0.001, 0.002, 0.003, 0.004])
    values = np.asarray(
        [
            [1.0, 7.0],
            [2.0, 7.0],
            [3.0, 7.0],
            [4.0, 7.0],
        ]
    )

    result = fixed_window_average(times, values, 0.05)

    assert result.mean[0, 0] == pytest.approx(2.5)
    assert result.sigma[0, 0] == pytest.approx(np.sqrt(1.25))
    assert result.sigma[0, 1] == 0.0
    assert result.n[0].tolist() == [4, 4]


def test_отсчёт_внутри_объявленного_gap_отвергается() -> None:
    with pytest.raises(ValueError, match="внутри объявленного разрыва"):
        fixed_window_average(
            np.asarray([1.0, 1.5, 2.0]),
            np.asarray([[1.0], [2.0], [3.0]]),
            1.0,
            gaps=((1.2, 1.8),),
        )


def test_sigma_на_синтетическом_шуме_сходится_к_заданной() -> None:
    rng = np.random.default_rng(20260910)
    samples = rng.normal(loc=12.0, scale=2.0, size=50_000)
    times = np.linspace(0.0, 0.999999, samples.size, dtype=np.float64)

    result = fixed_window_average(times, samples[:, np.newaxis], 1.0)

    assert result.windows == 1
    assert result.n[0, 0] == samples.size
    assert result.mean[0, 0] == pytest.approx(12.0, abs=0.03)
    assert result.sigma[0, 0] == pytest.approx(2.0, abs=0.03)


def test_кадр_ровно_на_левой_границе_gap_остаётся_до_разрыва() -> None:
    result = fixed_window_average(
        np.asarray([0.050, 0.080]),
        np.asarray([[1.0], [2.0]]),
        0.05,
        gaps=((0.050, 0.075),),
    )

    assert result.start_mono.tolist() == pytest.approx([0.0, 0.075])
    assert result.stop_mono.tolist() == pytest.approx([0.05, 0.10])
    assert result.mean[:, 0].tolist() == pytest.approx([1.0, 2.0])
    assert result.n[:, 0].tolist() == [1, 1]


def test_n_считается_независимо_для_каждой_колонки() -> None:
    result = fixed_window_average(
        np.asarray([0.001, 0.002, 0.003]),
        np.asarray([[1.0, np.nan], [np.nan, 10.0], [3.0, 12.0]]),
        0.05,
    )

    np.testing.assert_allclose(result.mean[0], [2.0, 11.0])
    np.testing.assert_array_equal(result.n[0], [2, 2])
    np.testing.assert_allclose(result.sigma[0], [1.0, 1.0])
