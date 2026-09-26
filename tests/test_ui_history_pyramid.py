"""Пирамида разрешений самописца (Р88) и структурный критерий такта (№44).

Критерий производительности — не секундомер (чат №15): проверяется число
строк, которые снимок отдаёт на кривую, и что оно не зависит от длины истории.
Точность пирамиды сверяется с прямым расчётом по базовым интервалам.
"""

from pathlib import Path

import numpy as np
import pytest

from fbg.core.calibration import Sensor, SensorType
from fbg.core.endpoint import Endpoint
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.ui import models
from fbg.ui.app import AppController
from fbg.ui.history import (
    HISTORY_INTERVAL_S,
    MAX_VIEW_POINTS,
    PYRAMID_LEVELS,
    VIEW_ROW_LIMIT,
    CompressedHistoryRecorder,
    HistoryViewRequest,
    aggregate_history,
    choose_view_level,
    level_bins,
    level_interval_s,
)

LINE_NM = np.asarray([1538.22, 1544.78, 1549.68, 1551.35, 1559.77])
SLOTS = tuple((0, index) for index in range(LINE_NM.size))


def synthetic_rows(first_bin: int, rows: int, *, seed: int = 7) -> dict[str, np.ndarray]:
    """Базовые интервалы, значения которых — функция **абсолютного** номера.

    Поэтому последняя минута суточной и минутной истории, кончающихся в один
    момент, совпадает побитно, и сравнивать их можно на равенство.
    """
    bins = np.arange(first_bin, first_bin + rows, dtype=np.int64)
    start = bins * HISTORY_INTERVAL_S
    phase = bins[:, np.newaxis].astype(np.float64)
    mean = LINE_NM + 0.01 * np.sin(phase / 997.0) + 0.001 * np.sin(phase * 1.3)
    sigma = np.broadcast_to(0.002 + 0.0005 * np.cos(phase * 0.7), mean.shape).copy()
    n = np.full((rows, LINE_NM.size), 200, dtype=np.int64)
    empty = (bins % 101 == 0)[:, np.newaxis] & (np.arange(LINE_NM.size) == 2)
    n[empty] = 0
    mean[n == 0] = np.nan
    sigma[n == 0] = np.nan
    del seed
    return {
        "start_mono": start,
        "stop_mono": start + HISTORY_INTERVAL_S,
        "mean_nm": mean,
        "min_nm": mean - 3.0 * sigma,
        "max_nm": mean + 3.0 * sigma,
        "sigma_nm": sigma,
        "n": n,
    }


def recorder_with(seconds: float, *, end_bin: int = 10_000_000) -> CompressedHistoryRecorder:
    rows = round(seconds / HISTORY_INTERVAL_S)
    recorder = CompressedHistoryRecorder(1, LINE_NM.size)
    recorder.start()
    recorder.append_intervals(**synthetic_rows(end_bin - rows, rows))
    return recorder


@pytest.fixture(scope="module")
def day() -> CompressedHistoryRecorder:
    return recorder_with(86_400.0)


@pytest.fixture(scope="module")
def minute() -> CompressedHistoryRecorder:
    return recorder_with(60.0)


def _aligned(view, reference, level: int):
    """Строки уровня и прямого расчёта с одинаковым номером интервала."""
    bins = level_bins(level)
    reference_bins = (
        np.rint(reference.start_mono / HISTORY_INTERVAL_S).astype(np.int64) // bins
    ) * bins
    common, left, right = np.intersect1d(view.base_bin, reference_bins, return_indices=True)
    return common, left, right


# --- сетка базового уровня ------------------------------------------------------------


def test_сетка_не_дублирует_интервалы_и_не_теряет_огибающую() -> None:
    """Регресс чата №20: ``floor(4.3 / 0.1) == 42`` писал интервал дважды."""
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    times = np.arange(1, 20_001) / 2000.0
    values = (1550.0 + 0.01 * np.sin(times))[:, np.newaxis]
    for begin in range(0, times.size, 200):
        recorder.ingest(times[begin : begin + 200], values[begin : begin + 200])
    history = recorder.snapshot()

    assert history.windows == 100
    assert np.unique(history.base_bin).size == 100
    assert int(history.n.sum()) == 20_000 - 1  # последний кадр ждёт конца интервала
    sample_bins = np.floor(times / HISTORY_INTERVAL_S).astype(np.int64)
    for row, base_bin in enumerate(history.base_bin):
        raw = values[sample_bins == base_bin, 0]
        assert history.n[row, 0] == raw.size
        assert history.min_nm[row, 0] == raw.min()
        assert history.max_nm[row, 0] == raw.max()


# --- точность -----------------------------------------------------------------------


@pytest.mark.parametrize("level", range(1, PYRAMID_LEVELS))
def test_каждый_уровень_совпадает_с_прямым_расчётом(
    day: CompressedHistoryRecorder, level: int
) -> None:
    full = day.snapshot()
    reference = aggregate_history(full, level_interval_s(level))
    points = int(86_400.0 / level_interval_s(level)) + 2
    view = day.view(None, HistoryViewRequest(), max_points=points)
    assert view.level == level
    common, left, right = _aligned(view, reference, level)
    assert common.size >= reference.windows - 1

    assert np.array_equal(view.n[left], reference.n[right])
    assert np.array_equal(view.min_nm[left], reference.min_nm[right], equal_nan=True)
    assert np.array_equal(view.max_nm[left], reference.max_nm[right], equal_nan=True)
    np.testing.assert_allclose(view.mean_nm[left], reference.mean_nm[right], rtol=0, atol=1e-10)
    np.testing.assert_allclose(view.sigma_nm[left], reference.sigma_nm[right], rtol=0, atol=1e-10)


def test_прямой_расчёт_независим_от_пирамиды() -> None:
    """Эталон сверяется не с тем же кодом: mean и σ группы — вторым проходом по кадрам."""
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    rng = np.random.default_rng(3)
    times = np.sort(rng.uniform(0.0, 30.0, 40_000))
    values = (1550.0 + rng.normal(0.0, 0.003, times.size))[:, np.newaxis]
    values[(times > 12.0) & (times < 14.0)] = np.nan
    recorder.ingest(times, values)
    recorder.stop()

    view = recorder.view(None, HistoryViewRequest(), max_points=40)
    assert view.level == 1
    # «Стоп» отбрасывает незаконченный последний 100-мс интервал (чат №20);
    # эталон строится из кадров тех интервалов, которые в базе есть.
    base_bins = np.floor(times / HISTORY_INTERVAL_S).astype(np.int64)
    kept = np.isin(base_bins, recorder.snapshot().base_bin)
    sample_bins = base_bins // 10 * 10
    for row, base_bin in enumerate(view.base_bin):
        raw = values[(sample_bins == base_bin) & kept, 0]
        raw = raw[np.isfinite(raw)]
        assert view.n[row, 0] == raw.size
        if raw.size == 0:
            assert np.isnan(view.mean_nm[row, 0])
            continue
        assert view.mean_nm[row, 0] == pytest.approx(raw.mean(), abs=1e-11)
        assert view.sigma_nm[row, 0] == pytest.approx(raw.std(), abs=1e-11)
        assert view.min_nm[row, 0] == raw.min()
        assert view.max_nm[row, 0] == raw.max()


def test_пустые_интервалы_остаются_разрывом_на_всех_уровнях() -> None:
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    rows = synthetic_rows(1000, 3000)
    rows = {key: value[:, :1] if value.ndim == 2 else value for key, value in rows.items()}
    silent = slice(1000, 1200)  # 20 секунд без пиков: две 10-секундные клетки целиком
    for key in ("mean_nm", "min_nm", "max_nm", "sigma_nm"):
        rows[key][silent] = np.nan
    rows["n"][silent] = 0
    recorder.append_intervals(**rows)

    view = recorder.view(None, HistoryViewRequest(), max_points=40)
    assert view.level == 2
    empty = view.n[:, 0] == 0
    assert np.count_nonzero(empty) == 2
    assert np.all(np.isnan(view.mean_nm[empty, 0]))
    assert np.all(np.isnan(view.min_nm[empty, 0]))


def test_разрыв_сегмента_режет_грубый_интервал() -> None:
    recorder = CompressedHistoryRecorder(1, LINE_NM.size)
    recorder.start()
    recorder.append_intervals(**synthetic_rows(1000, 55))  # 100.0 … 105.5 с
    recorder.stop()
    recorder.start()
    recorder.append_intervals(**synthetic_rows(1057, 63))  # 105.7 … 112.0 с
    recorder.stop()

    view = recorder.view(None, HistoryViewRequest(), max_points=15)
    assert view.level == 1
    # Секунда [105, 106) накрыта разрывом: в ней две строки, по одной на сегмент.
    assert view.segment.tolist() == [0] * 6 + [1] * 7
    assert view.base_bin[5] == view.base_bin[6] == 1050
    assert view.stop_mono[5] == pytest.approx(105.5)
    assert view.start_mono[6] == pytest.approx(105.7)
    base = recorder.snapshot()
    for row in (5, 6):
        members = (base.base_bin // 10 * 10 == 1050) & (base.segment == view.segment[row])
        assert np.array_equal(view.n[row], base.n[members].sum(axis=0))


def test_хвост_уровня_досчитывается_и_виден() -> None:
    recorder = recorder_with(125.0)
    view = recorder.view(None, HistoryViewRequest(), max_points=15)
    assert view.level == 2
    full = recorder.snapshot()
    assert int(view.n.sum()) == int(full.n.sum())
    assert view.stop_mono[-1] == pytest.approx(full.stop_mono[-1])


# --- структурный критерий №44 --------------------------------------------------------


def test_последняя_минута_одинакова_на_минутной_и_суточной_истории(
    day: CompressedHistoryRecorder, minute: CompressedHistoryRecorder
) -> None:
    follow = HistoryViewRequest("follow", span_s=60.0)
    short = minute.view(SLOTS, follow)
    long = day.view(SLOTS, follow)
    assert short.level == long.level == 0
    tail = short.windows
    assert long.windows <= VIEW_ROW_LIMIT
    np.testing.assert_array_equal(long.mean_nm[-tail:], short.mean_nm)
    np.testing.assert_array_equal(long.n[-tail:], short.n)


@pytest.mark.parametrize(
    "request_",
    [
        HistoryViewRequest("follow", span_s=60.0),
        HistoryViewRequest("follow", span_s=3600.0),
        HistoryViewRequest(),
        HistoryViewRequest("manual", low_s=10_000.0, high_s=50_000.0),
    ],
)
@pytest.mark.parametrize("window_s", [None, 0.5, 2.0, 4.9])
def test_объём_снимка_ограничен_при_любой_длине_истории(
    day: CompressedHistoryRecorder,
    minute: CompressedHistoryRecorder,
    request_: HistoryViewRequest,
    window_s: float | None,
) -> None:
    for recorder in (minute, day):
        view = recorder.view(SLOTS, request_, averaging_window_s=window_s)
        assert view.windows <= VIEW_ROW_LIMIT
        assert view.mean_nm.shape == (view.windows, len(SLOTS))


def test_сутки_во_весь_экран_дают_не_больше_тысячи_видимых_точек(
    day: CompressedHistoryRecorder,
) -> None:
    view = day.view(SLOTS, HistoryViewRequest())
    assert view.level == 3
    assert view.windows <= MAX_VIEW_POINTS


def test_выбор_уровня() -> None:
    assert choose_view_level(60.0, None) == (0, False)
    assert choose_view_level(600.0, None) == (1, False)
    assert choose_view_level(3600.0, None) == (2, False)
    assert choose_view_level(86_400.0, None) == (3, False)
    assert choose_view_level(7 * 86_400.0, None) == (4, False)
    # Окна W помещаются — строки с грубейшего уровня, кратного W.
    assert choose_view_level(60.0, 0.5) == (0, True)
    assert choose_view_level(600.0, 2.0) == (1, True)
    # Не помещаются — уровень грубее W, мельче окна не показываем.
    assert choose_view_level(600.0, 0.5) == (1, False)
    assert choose_view_level(3600.0, 0.5) == (2, False)
    assert choose_view_level(3600.0, 4.9) == (2, False)
    assert choose_view_level(30.0, 4.9) == (0, True)


# --- окна W и Δλ на выбранном уровне ---------------------------------------------------


@pytest.mark.parametrize(("window_s", "span_s"), [(0.5, 60.0), (2.0, 600.0), (5.0, 600.0)])
def test_окна_w_из_уровня_совпадают_с_окнами_из_базы(
    day: CompressedHistoryRecorder, window_s: float, span_s: float
) -> None:
    view = day.view(SLOTS, HistoryViewRequest("follow", span_s=span_s), averaging_window_s=window_s)
    assert view.interval_s <= window_s
    from_view = aggregate_history(view, window_s)
    reference = aggregate_history(day.snapshot(SLOTS), window_s)
    common, left, right = np.intersect1d(
        np.round(from_view.start_mono, 6), np.round(reference.start_mono, 6), return_indices=True
    )
    assert common.size == from_view.windows
    assert np.array_equal(from_view.n[left], reference.n[right])
    np.testing.assert_allclose(from_view.mean_nm[left], reference.mean_nm[right], atol=1e-10)
    np.testing.assert_allclose(from_view.sigma_nm[left], reference.sigma_nm[right], atol=1e-10)


def test_дельта_на_грубом_уровне_равна_расчёту_по_базе(day: CompressedHistoryRecorder) -> None:
    lambda0 = ((0, 1, 1544.70),)
    view = day.view(SLOTS, HistoryViewRequest())
    snap = models.AppSnapshot(
        endpoint=Endpoint(),
        profile=DeviceProfile(),
        state=SessionState.STREAMING,
        measurement_history=view,
        measurement_lambda0_nm=lambda0,
    )
    graph = models.measurement_history_graph_model(snap, (models.SlotRef(0, 1),), mode="delta")
    assert graph.resolution_s == pytest.approx(level_interval_s(view.level))
    reference = aggregate_history(day.snapshot(SLOTS), level_interval_s(view.level))
    _common, left, right = _aligned(view, reference, view.level)
    values = graph.traces[0].values_nm[left]
    np.testing.assert_allclose(values, reference.mean_nm[right, 1] - 1544.70, atol=1e-10)
    np.testing.assert_array_equal(
        graph.traces[0].min_nm[left], reference.min_nm[right, 1] - 1544.70
    )


# --- датчик на грубом уровне ----------------------------------------------------------


def _sensor_snapshot(history, sensor: Sensor) -> models.AppSnapshot:
    return models.AppSnapshot(
        endpoint=Endpoint(),
        profile=DeviceProfile(),
        state=SessionState.STREAMING,
        sensors=(sensor,),
        sensor_version=1,
        sensor_wavelength_history=history,
    )


def test_датчик_на_грубом_уровне_не_смешивает_решётки_при_перестановке_слотов() -> None:
    """Р30: слабая решётка появилась ниже — выбранная переехала в соседний слот."""
    recorder = CompressedHistoryRecorder(1, 2)
    recorder.start()
    rows = 100  # 10 секунд = одна 10-секундная клетка
    starts = (1000 + np.arange(rows)) * HISTORY_INTERVAL_S
    mean = np.full((rows, 2), np.nan)
    mean[:50, 0] = 1549.68  # сначала решётка в слоте 0
    mean[50:, 0] = 1546.50  # затем в слот 0 встала слабая решётка
    mean[50:, 1] = 1549.68  # а наша ушла в слот 1
    n = np.where(np.isfinite(mean), 200, 0)
    sigma = np.where(np.isfinite(mean), 0.002, np.nan)
    recorder.append_intervals(starts, starts + 0.1, mean, mean, mean, sigma, n)
    recorder.append_intervals(
        np.asarray([1100 * 0.1]),
        np.asarray([1101 * 0.1]),
        mean[-1:],
        mean[-1:],
        mean[-1:],
        sigma[-1:],
        n[-1:],
    )
    sensor = Sensor("T", "T", 0, SensorType.TEMPERATURE, 1549.68, 0.35, 25.0, 100.0)

    base = recorder.view(None, HistoryViewRequest(), max_points=10**6)
    base_graph = models.sensor_history_graph_model(_sensor_snapshot(base, sensor), ("T",))
    assert np.all(np.isfinite(base_graph.traces[0].values[:rows]))

    coarse = recorder.view(None, HistoryViewRequest(), max_points=2)
    assert coarse.level == 2
    graph = models.sensor_history_graph_model(_sensor_snapshot(coarse, sensor), ("T",))
    # Смесь 1549.68 и 1546.50 не выдаётся за измерение.
    assert np.isnan(graph.traces[0].values[0])
    assert graph.traces[0].n[0] == 0


def test_датчик_на_грубом_уровне_без_перестановки_точно_равен_свёртке() -> None:
    recorder = recorder_with(3600.0)
    sensor = Sensor("T", "T", 0, SensorType.TEMPERATURE, 1551.35, 0.35, 25.0, 100.0, k2=7.0)
    coarse = recorder.view(None, HistoryViewRequest(), max_points=4000)
    assert coarse.level == 1
    graph = models.sensor_history_graph_model(_sensor_snapshot(coarse, sensor), ("T",))
    base = recorder.view(None, HistoryViewRequest(), max_points=10**6)
    reference = aggregate_history(base, 1.0)
    _common, left, right = _aligned(coarse, reference, 1)
    delta = reference.mean_nm[right, 3] - 1551.35
    sigma = reference.sigma_nm[right, 3]
    expected = 25.0 + 100.0 * delta + 7.0 * (delta * delta + sigma * sigma)
    np.testing.assert_allclose(graph.traces[0].values[left], expected, atol=1e-9)


# --- автозаполнение λ₀ и контроллер ------------------------------------------------------


def test_первое_окно_w_совпадает_с_прежним_расчётом_по_всей_истории() -> None:
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    times = np.arange(1, 12_001) / 2000.0
    values = np.full((times.size, 1), np.nan)
    values[times > 1.23, 0] = 1550.0 + 0.001 * np.sin(times[times > 1.23])
    recorder.ingest(times, values)
    for window_s in (0.2, 0.5, 1.0):
        expected_all = aggregate_history(recorder.snapshot(), window_s)
        first = int(np.flatnonzero(expected_all.n[:, 0] > 0)[0])
        value = recorder.first_window_mean((0, 0), 0, window_s)
        assert value == pytest.approx(float(expected_all.mean_nm[first, 0]), abs=1e-12)


def test_первое_окно_ждёт_завершения() -> None:
    recorder = CompressedHistoryRecorder(1, 1)
    recorder.start()
    times = np.arange(1, 401) / 2000.0  # 0.2 с: окно 1 с ещё идёт
    recorder.ingest(times, np.full((times.size, 1), 1550.0))
    assert recorder.first_window_mean((0, 0), 0, 1.0) is None


def test_снимок_контроллера_не_копирует_всю_историю(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = AppController(
        AppConfig(
            calibration_path=tmp_path / "sensors.json",
            packet_log=PacketLogConfig(directory=None),
        )
    )
    controller.start()
    try:

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("полный snapshot() на такте интерфейса")

        monkeypatch.setattr(controller._measurement_history, "snapshot", forbidden)
        monkeypatch.setattr(controller._sensor_graph_history, "snapshot", forbidden)
        controller.start_measurement_history(averaging_window_s=0.5)
        controller.snapshot()
        controller.snapshot(include_trace_history=True, include_sensor_data=True)
    finally:
        controller.shutdown()
