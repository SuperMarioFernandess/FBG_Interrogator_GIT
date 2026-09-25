"""Qt-free тесты самописца графиков чата №20."""

import numpy as np
import pytest

from fbg.ui.history import (
    HISTORY_INTERVAL_S,
    CompressedHistoryRecorder,
    aggregate_history,
    can_aggregate_exactly,
    history_memory_estimate_bytes,
)


def _matrix(times: np.ndarray, columns: int = 2) -> np.ndarray:
    values = np.full((times.size, columns), np.nan, dtype=np.float64)
    values[:, 0] = 1550.0 + np.arange(times.size) * 0.001
    if columns > 1:
        values[:, 1] = 1560.0
    return values


def test_самописец_хранит_mean_min_max_n_на_неподвижной_сетке() -> None:
    recorder = CompressedHistoryRecorder(1, 2)
    recorder.start()
    times = 0.005 + np.arange(31) * 0.01
    recorder.ingest(times, _matrix(times))
    snapshot = recorder.snapshot()

    assert snapshot.positions == ((0, 0), (0, 1))
    assert snapshot.windows == 3
    assert snapshot.start_mono == pytest.approx([0.0, 0.1, 0.2])
    assert snapshot.stop_mono == pytest.approx([0.1, 0.2, 0.3])
    assert snapshot.n[:, 0].tolist() == [10, 10, 10]
    assert snapshot.min_nm[0, 0] == pytest.approx(1550.0)
    assert snapshot.max_nm[0, 0] == pytest.approx(1550.009)


def test_stop_замораживает_start_продолжает_с_новым_сегментом_clear_сбрасывает() -> None:
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    first_t = np.arange(0.01, 0.25, 0.01)
    recorder.ingest(first_t, _matrix(first_t, 1))
    recorder.stop()
    stopped = recorder.snapshot()
    assert not stopped.running
    stopped_version = stopped.version

    ignored_t = np.arange(1.0, 1.25, 0.01)
    recorder.ingest(ignored_t, _matrix(ignored_t, 1))
    assert recorder.snapshot().version == stopped_version

    recorder.start()
    second_t = np.arange(2.01, 2.25, 0.01)
    recorder.ingest(second_t, _matrix(second_t, 1))
    recorder.stop()
    resumed = recorder.snapshot()
    assert set(resumed.segment.tolist()) == {0, 1}
    assert resumed.windows > stopped.windows

    recorder.clear()
    cleared = recorder.snapshot()
    assert cleared.windows == 0
    assert cleared.positions == ()
    assert cleared.origin_mono is None


def test_разрешение_истории_не_зависит_от_частоты_вызова_ingest() -> None:
    times = np.arange(0.001, 1.001, 0.001)
    values = _matrix(times, 1)

    fast = CompressedHistoryRecorder(1, 1)
    fast.start()
    for begin in range(0, times.size, 50):
        fast.ingest(times[begin : begin + 50], values[begin : begin + 50])
    fast.stop()

    slow = CompressedHistoryRecorder(1, 1)
    slow.start()
    for begin in range(0, times.size, 500):
        slow.ingest(times[begin : begin + 500], values[begin : begin + 500])
    slow.stop()

    a = fast.snapshot()
    b = slow.snapshot()
    assert a.interval_s == b.interval_s == HISTORY_INTERVAL_S
    assert a.start_mono == pytest.approx(b.start_mono)
    assert a.mean_nm == pytest.approx(b.mean_nm)
    assert a.n.tolist() == b.n.tolist()


def test_усреднение_сжатых_интервалов_взвешивает_mean_по_n() -> None:
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    first_t = np.arange(0.001, 0.101, 0.001)
    second_t = np.arange(0.101, 0.201, 0.002)
    recorder.ingest(first_t, np.full((first_t.size, 1), 10.0))
    recorder.ingest(second_t, np.full((second_t.size, 1), 20.0))
    recorder.ingest(np.asarray([0.201]), np.asarray([[20.0]]))
    history = recorder.snapshot()

    averaged = aggregate_history(history, 0.2)
    assert averaged.windows >= 1
    assert averaged.n[0, 0] == 150
    assert averaged.mean_nm[0, 0] == pytest.approx((1000.0 + 1000.0) / 150.0)
    assert averaged.min_nm[0, 0] == 10.0
    assert averaged.max_nm[0, 0] == 20.0


def test_окно_меньше_интервала_или_некратное_честно_недоступно() -> None:
    assert not can_aggregate_exactly(0.05)
    assert not can_aggregate_exactly(0.15)
    assert can_aggregate_exactly(0.1)
    assert can_aggregate_exactly(0.5)


def test_оценка_памяти_растёт_с_числом_активных_позиций() -> None:
    one = history_memory_estimate_bytes(86_400.0, 1)
    five = history_memory_estimate_bytes(86_400.0, 5)
    assert five > one > 0
