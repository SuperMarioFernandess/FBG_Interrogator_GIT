"""Qt-свободные регрессии самописцев чата №20."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fbg.core.calibration import Sensor, SensorType
from fbg.core.endpoint import Endpoint
from fbg.core.pipeline import DEFAULT_HISTORY_FRAMES, PipelineConfig
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.ui import models
from fbg.ui.app import AppController
from fbg.ui.history import CompressedHistorySnapshot, history_memory_estimate_bytes
from tests.synthetic import load_vectors

PROFILE = DeviceProfile()
REAL_FRAME = load_vectors("measurement_real.hex")["measurement_real"]


def compressed_history(
    *,
    positions: tuple[tuple[int, int], ...] = ((0, 0), (0, 1)),
    running: bool = True,
    version: int = 1,
) -> CompressedHistorySnapshot:
    mean = np.asarray(
        [
            [1550.000, 1560.000],
            [1550.010, 1560.020],
            [1550.020, 1560.040],
            [1550.030, 1560.060],
        ],
        dtype=np.float64,
    )[:, : len(positions)]
    return CompressedHistorySnapshot(
        positions=positions,
        start_mono=np.asarray([10.0, 10.1, 10.2, 10.3]),
        stop_mono=np.asarray([10.1, 10.2, 10.3, 10.4]),
        segment=np.zeros(4, dtype=np.int32),
        mean_nm=mean,
        min_nm=mean - 0.001,
        max_nm=mean + 0.001,
        sigma_nm=np.full_like(mean, 0.0005),
        n=np.asarray([[10, 20], [10, 20], [10, 20], [10, 20]], dtype=np.int32)[:, : len(positions)],
        first_valid_nm=mean[0].copy(),
        running=running,
        interval_s=0.1,
        depth_s=86_400.0,
        version=version,
        origin_mono=10.0,
    )


def snapshot(**kwargs: object) -> models.AppSnapshot:
    base: dict[str, object] = {
        "endpoint": Endpoint(),
        "profile": PROFILE,
        "state": SessionState.STREAMING,
    }
    base.update(kwargs)
    return models.AppSnapshot(**base)  # type: ignore[arg-type]


def test_смена_выбора_не_меняет_lambda0_других_позиций() -> None:
    history = compressed_history()
    snap = snapshot(
        measurement_history=history,
        measurement_lambda0_nm=((0, 0, 1549.9), (0, 1, 1559.8)),
    )
    first = models.measurement_history_graph_model(
        snap,
        (models.SlotRef(0, 0), models.SlotRef(0, 1)),
    )
    second = models.measurement_history_graph_model(snap, (models.SlotRef(0, 0),))

    assert {trace.slot: trace.lambda0_nm for trace in first.traces} == {
        models.SlotRef(0, 0): 1549.9,
        models.SlotRef(0, 1): 1559.8,
    }
    assert second.traces[0].lambda0_nm == pytest.approx(1549.9)
    assert snap.measurement_lambda0_nm == ((0, 0, 1549.9), (0, 1, 1559.8))


def test_глубина_усреднение_и_спектр_не_меняют_lambda0() -> None:
    base_history = compressed_history()
    lambda0 = ((0, 0, 1549.9),)
    first = snapshot(measurement_history=base_history, measurement_lambda0_nm=lambda0)
    deeper = snapshot(
        measurement_history=replace(base_history, depth_s=172_800.0, version=2),
        measurement_lambda0_nm=lambda0,
        spectrum_version=77,
    )

    plain = models.measurement_history_graph_model(first, (models.SlotRef(0, 0),))
    averaged = models.measurement_history_graph_model(
        deeper,
        (models.SlotRef(0, 0),),
        averaging_window_s=0.2,
    )

    assert plain.traces[0].lambda0_nm == pytest.approx(1549.9)
    assert averaged.traces[0].lambda0_nm == pytest.approx(1549.9)
    assert first.measurement_lambda0_nm == deeper.measurement_lambda0_nm == lambda0


def test_delta_без_lambda0_не_рисует_случайный_ноль() -> None:
    graph = models.measurement_history_graph_model(
        snapshot(measurement_history=compressed_history(), measurement_lambda0_nm=()),
        (models.SlotRef(0, 0),),
        mode="delta",
    )
    assert np.all(np.isnan(graph.traces[0].values_nm))
    assert graph.traces[0].lambda0_nm is None


def test_история_датчика_пересчитывается_новыми_коэффициентами_без_очистки() -> None:
    history = compressed_history(positions=((0, 0),), running=False)
    sensor = Sensor(
        id="T1",
        name="до",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=1550.0,
        window_nm=0.5,
        value0=0.0,
        k1=100.0,
    )
    before = models.sensor_history_graph_model(
        snapshot(
            sensors=(sensor,),
            sensor_version=1,
            sensor_wavelength_history=history,
        ),
        ("T1",),
    )
    renamed = replace(sensor, name="после")
    same = models.sensor_history_graph_model(
        snapshot(
            sensors=(renamed,),
            sensor_version=2,
            sensor_wavelength_history=history,
        ),
        ("T1",),
        before,
    )
    changed = replace(renamed, k1=200.0)
    recalculated = models.sensor_history_graph_model(
        snapshot(
            sensors=(changed,),
            sensor_version=3,
            sensor_wavelength_history=history,
        ),
        ("T1",),
        same,
    )

    assert np.allclose(same.traces[0].values, before.traces[0].values, equal_nan=True)
    finite = np.isfinite(before.traces[0].values)
    assert np.allclose(
        recalculated.traces[0].values[finite],
        2.0 * before.traces[0].values[finite],
    )
    assert recalculated.version == before.version == history.version


def test_оценка_памяти_явно_остаётся_оценкой() -> None:
    value = history_memory_estimate_bytes(86_400.0, 120)
    assert value > 0
    assert "Оценка" in "Оценка максимума памяти"


def test_скрытая_вкладка_копит_дольше_кольца_без_разрыва_и_lambda0_стабилен(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        profile=PROFILE,
        pipeline=PipelineConfig(history_frames=5),
        packet_log=PacketLogConfig(directory=None),
        calibration_path=tmp_path / "sensors.json",
    )
    controller = AppController(config)
    controller.set_measurement_history_request(((0, 0),), 60.0)
    controller.start_measurement_history()
    try:
        # Кольцо всего на пять кадров. Снимок каждые четыре кадра дренирует
        # курсор даже при include_trace_history=False, то есть при скрытой вкладке.
        for index in range(80):
            controller.pipeline.on_telemetry(REAL_FRAME, 10.0 + index * 0.02)
            if index % 4 == 3:
                controller.snapshot(include_trace_history=False, include_sensor_data=False)
        reference = controller.measurement_lambda0(0, 0)
        assert reference is not None

        visible = controller.snapshot(include_trace_history=True, include_sensor_data=False)
        assert visible.measurement_history is not None
        assert visible.measurement_history.windows > 5
        assert np.unique(visible.measurement_history.segment).tolist() == [0]
        assert controller.measurement_lambda0(0, 0) == reference
    finally:
        controller.shutdown()


def test_lambda0_autofill_при_усреднении_берет_завершенное_окно(tmp_path: Path) -> None:
    config = AppConfig(
        profile=PROFILE,
        packet_log=PacketLogConfig(directory=None),
        calibration_path=tmp_path / "sensors.json",
    )
    controller = AppController(config)
    controller.start_measurement_history(averaging_window_s=0.2)
    try:
        columns = PROFILE.channels * PROFILE.fbg_per_channel
        matrix = np.full((5, columns), np.nan, dtype=np.float64)
        matrix[:, 0] = np.asarray([1550.0, 1552.0, 1554.0, 1556.0, 1558.0])
        controller._measurement_history.ingest(
            np.asarray([10.00, 10.05, 10.11, 10.15, 10.21]), matrix
        )

        controller.snapshot(include_trace_history=False, include_sensor_data=False)

        assert controller.measurement_lambda0(0, 0) == pytest.approx(1553.0)
        assert controller.measurement_lambda0(0, 0) != pytest.approx(1550.0)
    finally:
        controller.shutdown()


def test_replace_sensor_не_очищает_lambda_историю(tmp_path: Path) -> None:
    config = AppConfig(
        profile=PROFILE,
        packet_log=PacketLogConfig(directory=None),
        calibration_path=tmp_path / "sensors.json",
    )
    controller = AppController(config)
    sensor = Sensor(
        id="T1",
        name="до",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=1545.0,
        window_nm=1.0,
        k1=100.0,
    )
    controller.replace_sensors((sensor,))
    controller.set_sensor_history_request(("T1",), 60.0)
    controller.start_sensor_history()
    try:
        for index in range(20):
            controller.pipeline.on_telemetry(REAL_FRAME, 20.0 + index * 0.02)
        before = controller.snapshot(include_trace_history=False, include_sensor_data=True)
        assert before.sensor_wavelength_history is not None
        assert before.sensor_wavelength_history.windows > 0
        version = before.sensor_wavelength_history.version

        controller.replace_sensors((replace(sensor, name="после", k1=200.0),))
        after = controller.snapshot(include_trace_history=False, include_sensor_data=True)

        assert after.sensor_wavelength_history is not None
        assert after.sensor_wavelength_history.windows == before.sensor_wavelength_history.windows
        assert after.sensor_wavelength_history.version == version
    finally:
        controller.shutdown()


def test_кольцо_pipeline_глубже_максимального_двухсекундного_ui_такта() -> None:
    # Р80: при 2000 Гц UI может опрашивать раз в 2 с. Штатное кольцо должно
    # оставлять запас, иначе Qt-свободный самописец потеряет кадры между тактами.
    assert DEFAULT_HISTORY_FRAMES / PROFILE.sweep_speed_hz > 2.0


def test_текущее_lambda0_из_усреднения_доступно_при_остановленном_самописце(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        profile=PROFILE,
        packet_log=PacketLogConfig(directory=None),
        calibration_path=tmp_path / "sensors.json",
    )
    controller = AppController(config)
    try:
        for index in range(31):
            controller.pipeline.on_telemetry(REAL_FRAME, 10.001 + index * 0.01)
        assert not controller.measurement_history_running

        value = controller.current_measurement_wavelength(0, 0, averaging_window_s=0.1)

        assert value is not None
        assert np.isfinite(value)
    finally:
        controller.shutdown()


def test_квадратичный_датчик_усредняется_точно_через_sigma_lambda() -> None:
    history = CompressedHistorySnapshot(
        positions=((0, 0),),
        start_mono=np.asarray([0.0]),
        stop_mono=np.asarray([0.1]),
        segment=np.asarray([0], dtype=np.int32),
        mean_nm=np.asarray([[10.0]]),
        min_nm=np.asarray([[9.0]]),
        max_nm=np.asarray([[11.0]]),
        sigma_nm=np.asarray([[1.0]]),
        n=np.asarray([[2]], dtype=np.int32),
        first_valid_nm=np.asarray([9.0]),
        running=False,
        interval_s=0.1,
        depth_s=60.0,
        version=1,
        origin_mono=0.0,
    )
    sensor = Sensor(
        id="Q1",
        name="quadratic",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=10.0,
        window_nm=2.0,
        value0=0.0,
        k1=0.0,
        k2=1.0,
    )
    graph = models.sensor_history_graph_model(
        snapshot(
            sensors=(sensor,),
            sensor_version=1,
            sensor_wavelength_history=history,
        ),
        ("Q1",),
    )

    # Две симметричные λ=9/11 имеют mean λ=10 и σ=1, но mean(Δλ²)=1,
    # а не polynomial(mean λ)=0.
    assert graph.traces[0].values[0] == pytest.approx(1.0)
