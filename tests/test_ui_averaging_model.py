"""Qt-свободная проверка усреднения на обоих графиках UI."""

import numpy as np
import pytest

from fbg.core.calibration import Sensor, SensorType
from fbg.core.endpoint import Endpoint
from fbg.core.pipeline import TraceHistorySnapshot
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.ui import models

PROFILE = DeviceProfile()


def _snapshot(**kwargs: object) -> models.AppSnapshot:
    base: dict[str, object] = {
        "endpoint": Endpoint(),
        "profile": PROFILE,
        "state": SessionState.STREAMING,
    }
    base.update(kwargs)
    return models.AppSnapshot(**base)  # type: ignore[arg-type]


def test_усреднённый_график_измерения_incremental_равен_полному() -> None:
    slot = (models.SlotRef(0, 0),)
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=3,
        t_mono=np.asarray([0.001, 0.020, 0.051]),
        wavelength_nm=np.asarray([[1550.000], [1550.002], [1550.004]]),
    )
    first = models.measurement_graph_model(
        _snapshot(trace_history=first_history),
        slot,
        averaging_window_s=0.05,
    )
    full_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=5,
        t_mono=np.asarray([0.001, 0.020, 0.051, 0.080, 0.101]),
        wavelength_nm=np.asarray(
            [[1550.000], [1550.002], [1550.004], [1550.006], [1550.008]]
        ),
    )
    incremental = models.measurement_graph_model(
        _snapshot(trace_history=full_history),
        slot,
        previous=first,
        averaging_window_s=0.05,
    )
    full = models.measurement_graph_model(
        _snapshot(trace_history=full_history),
        slot,
        averaging_window_s=0.05,
    )

    assert incremental.averaged is not None and full.averaged is not None
    np.testing.assert_allclose(incremental.averaged.start_mono, full.averaged.start_mono)
    np.testing.assert_allclose(incremental.averaged.stop_mono, full.averaged.stop_mono)
    np.testing.assert_allclose(incremental.averaged.mean, full.averaged.mean)
    np.testing.assert_allclose(incremental.averaged.sigma, full.averaged.sigma)
    np.testing.assert_array_equal(incremental.averaged.n, full.averaged.n)
    np.testing.assert_allclose(incremental.traces[0].delta_nm, full.traces[0].delta_nm)


def test_усреднённый_график_датчика_считает_каждый_raw_кадр_а_не_ui_историю() -> None:
    sensor = Sensor(
        id="T1",
        name="T1",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=1550.0,
        window_nm=0.1,
        value0=20.0,
        k1=100.0,
    )
    raw = TraceHistorySnapshot(
        positions=((0, 0), (0, 1)),
        seq_start=100,
        seq_stop=104,
        t_mono=np.asarray([1.001, 1.010, 1.020, 1.030]),
        wavelength_nm=np.asarray(
            [
                [1550.000, np.nan],
                [1550.010, np.nan],
                [np.nan, np.nan],
                [1550.030, np.nan],
            ]
        ),
    )
    # UI-history намеренно содержит другое число: при включённом усреднении
    # модель обязана брать raw TraceHistorySnapshot.
    ui_history = models.SensorHistorySnapshot(
        t_mono=np.asarray([1.03]),
        sensor_ids=("T1",),
        values=np.asarray([[999.0]]),
    )

    graph = models.sensor_graph_model(
        _snapshot(
            sensors=(sensor,),
            sensor_history=ui_history,
            sensor_trace_history=raw,
        ),
        ("T1",),
        averaging_window_s=0.05,
    )

    assert len(graph.traces) == 1
    trace = graph.traces[0]
    assert trace.n is not None and trace.sigma is not None
    assert trace.n.tolist() == [3]
    assert trace.values[0] == pytest.approx((20.0 + 21.0 + 23.0) / 3.0)
    assert trace.values[0] != 999.0
    assert trace.sigma[0] == pytest.approx(np.std([20.0, 21.0, 23.0]))



def test_график_измерения_вмещает_полосу_sigma_в_y_диапазон() -> None:
    slot = (models.SlotRef(0, 0),)
    history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=4,
        t_mono=np.asarray([0.001, 0.010, 0.020, 0.030]),
        wavelength_nm=np.asarray([[1549.99], [1550.01], [1549.99], [1550.01]]),
    )

    graph = models.measurement_graph_model(
        _snapshot(trace_history=history),
        slot,
        averaging_window_s=0.05,
    )

    trace = graph.traces[0]
    assert trace.sigma_nm is not None
    finite = np.isfinite(trace.delta_nm) & np.isfinite(trace.sigma_nm)
    lower = np.min(trace.delta_nm[finite] - trace.sigma_nm[finite])
    upper = np.max(trace.delta_nm[finite] + trace.sigma_nm[finite])
    assert graph.y_min_nm <= lower
    assert graph.y_max_nm >= upper

def test_число_кадров_в_окне_следует_текущей_скорости() -> None:
    snapshot = _snapshot()
    assert models.expected_averaging_frames(snapshot, 50.0) == 100


def test_incremental_усреднение_с_уже_известным_gap_не_дублирует_окно() -> None:
    slot = (models.SlotRef(0, 0),)
    gaps = ((0.025, 0.075),)
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=3,
        t_mono=np.asarray([0.080, 0.090, 0.110]),
        wavelength_nm=np.asarray([[1550.0], [1550.2], [1550.4]]),
    )
    first = models.measurement_graph_model(
        _snapshot(trace_history=first_history, stream_gaps=gaps),
        slot,
        averaging_window_s=0.05,
    )
    full_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=4,
        t_mono=np.asarray([0.080, 0.090, 0.110, 0.160]),
        wavelength_nm=np.asarray([[1550.0], [1550.2], [1550.4], [1550.6]]),
    )
    incremental = models.measurement_graph_model(
        _snapshot(trace_history=full_history, stream_gaps=gaps),
        slot,
        previous=first,
        averaging_window_s=0.05,
    )
    full = models.measurement_graph_model(
        _snapshot(trace_history=full_history, stream_gaps=gaps),
        slot,
        averaging_window_s=0.05,
    )

    assert incremental.averaged is not None and full.averaged is not None
    np.testing.assert_allclose(incremental.averaged.start_mono, full.averaged.start_mono)
    np.testing.assert_allclose(incremental.averaged.stop_mono, full.averaged.stop_mono)
    np.testing.assert_allclose(incremental.averaged.mean, full.averaged.mean)
    np.testing.assert_array_equal(incremental.averaged.n, full.averaged.n)



def test_incremental_усреднение_при_сдвиге_не_пересчитывает_готовое_окно() -> None:
    slot = (models.SlotRef(0, 0),)
    first_times = np.arange(1.54, 5.50, 0.011, dtype=np.float64)
    current_times = np.arange(1.76, 5.72, 0.011, dtype=np.float64)
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=100,
        seq_stop=100 + first_times.size,
        t_mono=first_times,
        wavelength_nm=(1550.0 + 0.01 * np.sin(first_times))[:, np.newaxis],
    )
    first = models.measurement_graph_model(
        _snapshot(trace_history=first_history),
        slot,
        averaging_window_s=0.5,
    )
    current_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=120,
        seq_stop=120 + current_times.size,
        t_mono=current_times,
        wavelength_nm=(1550.0 + 0.01 * np.sin(current_times))[:, np.newaxis],
    )

    incremental = models.measurement_graph_model(
        _snapshot(trace_history=current_history),
        slot,
        previous=first,
        averaging_window_s=0.5,
    )
    full_from_truncated_raw = models.measurement_graph_model(
        _snapshot(trace_history=current_history),
        slot,
        averaging_window_s=0.5,
    )

    assert first.averaged is not None
    assert incremental.averaged is not None
    assert full_from_truncated_raw.averaged is not None
    # Окно 1.5…2.0 уже завершилось в первом такте. После вытеснения части
    # его raw-строк из кольца оно остаётся прежним, а не пересчитывается по
    # усечённому хвосту. Это и есть требование Р76 для фиксированной сетки.
    first_index = int(np.flatnonzero(first.averaged.start_mono == 1.5)[0])
    current_index = int(np.flatnonzero(incremental.averaged.start_mono == 1.5)[0])
    full_index = int(np.flatnonzero(full_from_truncated_raw.averaged.start_mono == 1.5)[0])
    assert incremental.averaged.mean[current_index, 0] == first.averaged.mean[first_index, 0]
    assert incremental.averaged.n[current_index, 0] == first.averaged.n[first_index, 0]
    assert incremental.averaged.mean[current_index, 0] != pytest.approx(
        full_from_truncated_raw.averaged.mean[full_index, 0], abs=1e-8
    )


def test_incremental_усреднение_при_сдвиге_сохраняет_границы_gap() -> None:
    slot = (models.SlotRef(0, 0),)
    gaps = ((2.5, 2.9),)
    first_times = np.asarray([2.244, 2.414, 2.499, 2.924, 3.10, 3.30, 3.70, 4.10])
    current_times = np.asarray([2.414, 2.499, 2.924, 3.10, 3.30, 3.70, 4.10, 4.30])
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=100,
        seq_stop=108,
        t_mono=first_times,
        wavelength_nm=(1550.0 + first_times * 0.001)[:, np.newaxis],
    )
    first = models.measurement_graph_model(
        _snapshot(trace_history=first_history, stream_gaps=gaps),
        slot,
        averaging_window_s=0.4,
    )
    current_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=101,
        seq_stop=109,
        t_mono=current_times,
        wavelength_nm=(1550.0 + current_times * 0.001)[:, np.newaxis],
    )

    incremental = models.measurement_graph_model(
        _snapshot(trace_history=current_history, stream_gaps=gaps),
        slot,
        previous=first,
        averaging_window_s=0.4,
    )

    assert incremental.averaged is not None
    pairs = list(
        zip(
            incremental.averaged.start_mono.tolist(),
            incremental.averaged.stop_mono.tolist(),
            strict=True,
        )
    )
    assert len(pairs) == len(set(pairs))
    assert any(stop == pytest.approx(2.5) for _start, stop in pairs)
    assert any(start == pytest.approx(2.9) for start, _stop in pairs)
    assert all(not (start < 2.5 and stop > 2.9) for start, stop in pairs)


def test_датчики_сохраняют_завершённые_окна_старше_raw_хвоста() -> None:
    sensor = Sensor(
        id="T1",
        name="T1",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=1550.0,
        window_nm=0.2,
        value0=0.0,
        k1=100.0,
    )
    first_raw = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=6,
        t_mono=np.asarray([0.01, 0.04, 0.06, 0.09, 0.11, 0.14]),
        wavelength_nm=np.asarray(
            [[1550.00], [1550.01], [1550.02], [1550.03], [1550.04], [1550.05]]
        ),
    )
    first_ui_history = models.SensorHistorySnapshot(
        t_mono=np.asarray([0.01, 0.14]),
        sensor_ids=("T1",),
        values=np.asarray([[0.0], [5.0]]),
    )
    first = models.sensor_graph_model(
        _snapshot(
            sensors=(sensor,),
            sensor_history=first_ui_history,
            sensor_trace_history=first_raw,
        ),
        ("T1",),
        averaging_window_s=0.05,
    )
    current_raw = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=4,
        seq_stop=8,
        t_mono=np.asarray([0.11, 0.14, 0.16, 0.19]),
        wavelength_nm=np.asarray([[1550.04], [1550.05], [1550.06], [1550.07]]),
    )
    current_ui_history = models.SensorHistorySnapshot(
        t_mono=np.asarray([0.01, 0.19]),
        sensor_ids=("T1",),
        values=np.asarray([[0.0], [7.0]]),
    )

    current = models.sensor_graph_model(
        _snapshot(
            sensors=(sensor,),
            sensor_history=current_ui_history,
            sensor_trace_history=current_raw,
        ),
        ("T1",),
        previous=first,
        averaging_window_s=0.05,
    )

    assert current.averaged is not None
    assert current.averaged.start_mono[0] == pytest.approx(0.0)
    assert current.averaged.stop_mono[0] == pytest.approx(0.05)
    assert current.traces[0].n is not None
    assert current.traces[0].n[0] == 2

def test_incremental_путь_пересчитывает_только_последний_хвост(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot = (models.SlotRef(0, 0),)
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=3,
        t_mono=np.asarray([0.001, 0.020, 0.051]),
        wavelength_nm=np.asarray([[1550.000], [1550.002], [1550.004]]),
    )
    first = models.measurement_graph_model(
        _snapshot(trace_history=first_history),
        slot,
        averaging_window_s=0.05,
    )
    full_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=5,
        t_mono=np.asarray([0.001, 0.020, 0.051, 0.080, 0.101]),
        wavelength_nm=np.asarray(
            [[1550.000], [1550.002], [1550.004], [1550.006], [1550.008]]
        ),
    )

    original = models.fixed_window_average
    rows_seen: list[int] = []

    def counted(t_mono: np.ndarray, *args: object, **kwargs: object):
        rows_seen.append(int(t_mono.size))
        return original(t_mono, *args, **kwargs)

    monkeypatch.setattr(models, "fixed_window_average", counted)
    models.measurement_graph_model(
        _snapshot(trace_history=full_history),
        slot,
        previous=first,
        averaging_window_s=0.05,
    )

    assert rows_seen == [3]
    assert rows_seen[0] < full_history.frames
