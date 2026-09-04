"""Qt-свободная модель панели измерения и её контракт с pipeline/recorder.

Реальный вектор `measurement_real.hex` используется там, где проверяется
геометрия и заполненность последнего кадра. Искусственные истории отдельно
помечены: они нужны для NaN, дрейфа и пачечного времени, которых один кадр
захвата сам по себе не содержит.
"""

import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest

from fbg.core.endpoint import Endpoint
from fbg.core.pipeline import Pipeline, PipelineConfig, TraceHistorySnapshot
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.io.recorder import RecorderConfig, RecorderStats
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene
from fbg.ui import models
from fbg.ui.app import AppController
from tests.synthetic import load_vectors
from tests.test_recorder import files_of, frame_numbers, gap_lines
from tests.test_session import QUIET, TEST_ENDPOINT_KWARGS, wait_until

PROFILE = DeviceProfile()
REAL_FRAME = load_vectors("measurement_real.hex")["measurement_real"]


def app_snapshot(
    *,
    ui=None,
    trace_history: TraceHistorySnapshot | None = None,
    metrics=None,
    recorder_config: RecorderConfig | None = None,
    recorder: RecorderStats | None = None,
    recording: bool = False,
    recording_elapsed_s: float = 0.0,
) -> models.AppSnapshot:
    """Минимальный снимок для чистых функций модели."""
    return models.AppSnapshot(
        endpoint=Endpoint(),
        profile=PROFILE,
        state=SessionState.STREAMING,
        ui=ui,
        trace_history=trace_history,
        metrics=metrics,
        recorder_config=recorder_config,
        recorder=recorder,
        recording=recording,
        recording_elapsed_s=recording_elapsed_s,
    )


def real_pipeline_snapshot() -> tuple[Pipeline, models.AppSnapshot]:
    """Последний кадр и метрики на реальном векторе из KB_02/KB_06."""
    pipeline = Pipeline(PROFILE, PipelineConfig(history_frames=20_000))
    pipeline.set_expected_rate(2000.0)
    pipeline.on_telemetry(REAL_FRAME, 1.0)
    ui = pipeline.publish_now()
    assert ui is not None
    return pipeline, app_snapshot(ui=ui, metrics=pipeline.metrics())


# --------------------------------------------------------------------------------------
# График
# --------------------------------------------------------------------------------------


def test_выбор_позиций_фильтрует_график() -> None:
    """Не выбранная линия не должна влиять ни на кривые, ни на масштаб."""
    history = TraceHistorySnapshot(
        positions=((0, 0), (0, 1), (1, 0)),
        seq_start=10,
        seq_stop=14,
        t_mono=np.asarray([0.0, 1.0, 2.0, 3.0]),
        wavelength_nm=np.asarray(
            [
                [1550.000, 1560.000, 1540.000],
                [1550.001, 1560.050, 1540.001],
                [1550.002, 1560.100, 1540.002],
                [1550.003, 1560.150, 1540.003],
            ]
        ),
    )
    selected = (models.SlotRef(0, 0), models.SlotRef(1, 0))
    graph = models.measurement_graph_model(app_snapshot(trace_history=history), selected)

    assert [trace.slot for trace in graph.traces] == list(selected)
    assert models.SlotRef(0, 1) not in {trace.slot for trace in graph.traces}
    assert graph.t_s.tolist() == [-3.0, -2.0, -1.0, 0.0]
    # Невыбранный дрейф 0.150 нм не раздувает видимый диапазон выбранных 0.003 нм.
    assert graph.y_max_nm < 0.01


def test_nan_остаётся_разрывом_и_не_интерполируется() -> None:
    """Правило №7: пропавший пик не превращается в последнее известное значение."""
    history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=5,
        t_mono=np.arange(5, dtype=np.float64),
        wavelength_nm=np.asarray([[1550.0], [1550.001], [np.nan], [1550.003], [1550.004]]),
    )
    graph = models.measurement_graph_model(
        app_snapshot(trace_history=history), (models.SlotRef(0, 0),)
    )
    delta = graph.traces[0].delta_nm

    assert delta[0] == pytest.approx(0.0)
    assert delta[1] == pytest.approx(0.001)
    assert np.isnan(delta[2])
    assert delta[3] == pytest.approx(0.003)
    assert graph.traces[0].valid_points == 4


def test_масштаб_графика_строится_по_delta_видимого_диапазона() -> None:
    """Абсолютные 1550 нм не превращают пикометровый дрейф в прямую линию."""
    history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=3,
        t_mono=np.asarray([0.0, 1.0, 2.0]),
        wavelength_nm=np.asarray([[1550.0000], [1550.0005], [1550.0010]]),
    )
    graph = models.measurement_graph_model(
        app_snapshot(trace_history=history), (models.SlotRef(0, 0),)
    )

    assert graph.traces[0].baseline_nm == pytest.approx(1550.0)
    assert graph.y_max_nm - graph.y_min_nm == pytest.approx(models.GRAPH_MIN_SPAN_NM)
    assert abs(graph.y_min_nm) < 0.01
    assert abs(graph.y_max_nm) < 0.01


def test_модель_графика_достраивается_только_новым_хвостом() -> None:
    """Р76: 200 новых кадров не заставляют заново считать старые 20 000."""
    slot = (models.SlotRef(0, 0),)
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=100,
        seq_stop=104,
        t_mono=np.asarray([1.0, 2.0, 3.0, 4.0]),
        wavelength_nm=np.asarray([[1550.000], [1550.001], [np.nan], [1550.003]]),
    )
    first = models.measurement_graph_model(app_snapshot(trace_history=first_history), slot)
    second_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=102,
        seq_stop=106,
        t_mono=np.asarray([3.0, 4.0, 5.0, 6.0]),
        wavelength_nm=np.asarray([[np.nan], [1550.003], [1550.004], [1550.005]]),
    )
    second = models.measurement_graph_model(
        app_snapshot(trace_history=second_history), slot, previous=first
    )

    assert second.seq_start == 102 and second.seq_stop == 106
    assert second.t_s.tolist() == [-3.0, -2.0, -1.0, 0.0]
    assert np.isnan(second.traces[0].delta_nm[0])
    assert second.traces[0].delta_nm[1:].tolist() == pytest.approx([0.003, 0.004, 0.005])
    assert second.traces[0].baseline_nm == pytest.approx(1550.0)
    assert second.traces[0].valid_points == 3


def test_ось_y_расширяется_сразу_а_сужается_только_по_редкому_пересчёту() -> None:
    """Краткий выброс виден сразу, но исчезновение выброса не дёргает ось каждый тик."""
    slot = (models.SlotRef(0, 0),)
    first_history = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=3,
        t_mono=np.arange(3, dtype=np.float64),
        wavelength_nm=np.asarray([[1550.0], [1550.001], [1550.002]]),
    )
    first = models.measurement_graph_model(app_snapshot(trace_history=first_history), slot)

    with_spike = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=0,
        seq_stop=4,
        t_mono=np.arange(4, dtype=np.float64),
        wavelength_nm=np.asarray([[1550.0], [1550.001], [1550.002], [1550.050]]),
    )
    expanded = models.measurement_graph_model(
        app_snapshot(trace_history=with_spike), slot, previous=first
    )
    assert expanded.y_max_nm > 0.05

    after_spike = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=3,
        seq_stop=6,
        t_mono=np.arange(3, 6, dtype=np.float64),
        wavelength_nm=np.asarray([[1550.050], [1550.003], [1550.004]]),
    )
    sticky = models.measurement_graph_model(
        app_snapshot(trace_history=after_spike), slot, previous=expanded
    )
    # Пока выброс ещё в окне, диапазон точно не сужается.
    assert sticky.y_max_nm == expanded.y_max_nm

    no_spike = TraceHistorySnapshot(
        positions=((0, 0),),
        seq_start=4,
        seq_stop=7,
        t_mono=np.arange(4, 7, dtype=np.float64),
        wavelength_nm=np.asarray([[1550.003], [1550.004], [1550.005]]),
    )
    still_sticky = models.measurement_graph_model(
        app_snapshot(trace_history=no_spike), slot, previous=sticky
    )
    assert still_sticky.y_max_nm == expanded.y_max_nm
    recalculated = models.measurement_graph_model(
        app_snapshot(trace_history=no_spike), slot, previous=sticky, recalculate_y=True
    )
    assert recalculated.y_max_nm < expanded.y_max_nm


def test_pipeline_отдаёт_копию_истории_только_выбранных_слотов() -> None:
    """Р36: график получает snapshot, а не `RingHistory`; real frame задаёт байты."""
    pipeline = Pipeline(PROFILE, PipelineConfig(history_frames=32))
    for index in range(6):
        pipeline.on_telemetry(REAL_FRAME, index * 0.0005)
    history = pipeline.trace_history(((0, 0), (0, 1)), 10.0)

    assert history.positions == ((0, 0), (0, 1))
    assert history.wavelength_nm.shape == (6, 2)
    assert history.frames == 6
    assert history.wavelength_nm.base is not pipeline.history.freq_ghz


# --------------------------------------------------------------------------------------
# Таблица
# --------------------------------------------------------------------------------------


def test_таблица_реального_кадра_имеет_4x30_и_валидность() -> None:
    """Реальный канал 1 содержит две длины волны; позиции 3…30 — NaN."""
    _, snapshot = real_pipeline_snapshot()
    table = models.measurement_table_model(snapshot)

    assert table.wavelength_nm.shape == (4, 30)
    assert table.valid.shape == (4, 30)
    assert table.valid[0, :2].tolist() == [True, True]
    assert not bool(table.valid[0, 2])
    assert int(table.valid.sum()) == 2
    assert table.case_temp_c.shape == (4,)
    assert np.all(np.isfinite(table.case_temp_c))


def test_переменное_число_заполненных_позиций_не_считается_ошибкой() -> None:
    """Два соседних кадра могут содержать 3 и 2 пика — это штатно по скринингу."""
    from tests.test_pipeline import make_frame

    pipeline = Pipeline(PROFILE, PipelineConfig(history_frames=32))
    pipeline.on_telemetry(
        make_frame(PROFILE, {(0, 0): 1544.8, (0, 1): 1546.5, (0, 2): 1551.5}), 1.0
    )
    first_ui = pipeline.publish_now()
    pipeline.on_telemetry(make_frame(PROFILE, {(0, 0): 1544.8, (0, 1): 1551.5}), 2.0)
    second_ui = pipeline.publish_now()
    assert first_ui is not None and second_ui is not None

    first = models.measurement_table_model(app_snapshot(ui=first_ui))
    second = models.measurement_table_model(app_snapshot(ui=second_ui))
    assert int(first.valid[0].sum()) == 3
    assert int(second.valid[0].sum()) == 2
    assert np.isnan(second.wavelength_nm[0, 2])


# --------------------------------------------------------------------------------------
# Расчёт объёма и состояние записи
# --------------------------------------------------------------------------------------


def test_оценка_объёма_учитывает_децимацию_и_число_позиций(tmp_path: Path) -> None:
    """До старта видно цену 2 кГц; децимация делит её, лимит колонок уменьшает."""
    _, base = real_pipeline_snapshot()
    full = RecorderConfig(directory=tmp_path, decimation=1, fbg_limit=None)
    half = RecorderConfig(directory=tmp_path, decimation=2, fbg_limit=None)
    narrow = RecorderConfig(directory=tmp_path, decimation=1, fbg_limit=4)

    full_bytes = models.estimate_recording_bytes(base, full)
    half_bytes = models.estimate_recording_bytes(base, half)
    narrow_bytes = models.estimate_recording_bytes(base, narrow)

    assert 600_000_000 < full_bytes < 750_000_000
    assert half_bytes == pytest.approx(full_bytes / 2, rel=0.001)
    assert narrow_bytes < full_bytes


def test_старт_и_остановка_записи_попадают_в_снимок(tmp_path: Path) -> None:
    controller = AppController(
        AppConfig(
            recorder=RecorderConfig(directory=tmp_path),
            packet_log=PacketLogConfig(directory=None),
        )
    )
    controller.start()
    try:
        recorder = controller.start_recording()
        active = controller.snapshot()
        assert active.recording
        assert active.recorder is not None
        assert active.recorder.path == recorder.path
        assert active.recording_elapsed_s >= 0.0

        controller.stop_recording()
        stopped = controller.snapshot()
        assert not stopped.recording
        assert stopped.recorder is not None
        assert stopped.recorder.path == recorder.path
    finally:
        controller.shutdown()


class ExplodingFile:
    """Файл-стимул ENOSPC для проверки состояния панели."""

    def write(self, _payload: bytes) -> int:
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_ошибка_записи_видна_и_приём_продолжается(tmp_path: Path) -> None:
    """Recorder падает отдельно; AppSnapshot сохраняет ошибку, pipeline жив."""
    controller = AppController(
        AppConfig(
            pipeline=PipelineConfig(history_frames=256),
            recorder=RecorderConfig(directory=tmp_path, poll_period_s=0.001),
            packet_log=PacketLogConfig(directory=None),
        )
    )
    controller.start()
    try:
        recorder = controller.start_recording()
        real_file = recorder._file
        assert real_file is not None
        real_file.close()
        recorder._file = ExplodingFile()  # type: ignore[assignment]

        controller.pipeline.on_telemetry(REAL_FRAME, time.perf_counter())
        deadline = time.perf_counter() + 2.0
        while recorder.stats.error is None and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert recorder.stats.error is not None

        before = controller.pipeline.sequence
        for index in range(20):
            controller.pipeline.on_telemetry(REAL_FRAME, time.perf_counter() + index * 0.0005)
        assert controller.pipeline.sequence == before + 20

        snapshot = controller.snapshot()  # заодно reaps завершившийся writer
        state = models.recording_panel_model(snapshot)
        assert not snapshot.recording
        assert state.error is not None and "No space left" in state.error
        assert controller.pipeline.metrics().parse_errors == 0
    finally:
        controller.shutdown()


def test_разрыв_записи_остаётся_предупреждением_после_остановки(tmp_path: Path) -> None:
    config = RecorderConfig(directory=tmp_path)
    stats = RecorderStats(
        files=1,
        rows=100,
        frames_span=150,
        gaps=2,
        lost_frames=50,
        pending_gap=0,
        bytes_written=123_456,
        path=tmp_path / "data.csv",
        last_frame_no=149,
        error=None,
    )
    state = models.recording_panel_model(
        app_snapshot(recorder_config=config, recorder=stats, recording=False)
    )
    assert state.has_gaps
    assert state.gaps == 2
    assert state.lost_frames == 50


# --------------------------------------------------------------------------------------
# Р65 — долгосрочный темп на пачечном входе
# --------------------------------------------------------------------------------------


def test_длинное_окно_темпа_ровное_на_пачечном_входе() -> None:
    """Пачки и паузы не должны превращаться в тревогу при среднем ровно 2 кГц."""
    pipeline = Pipeline(PROFILE, PipelineConfig(history_frames=20_000))
    pipeline.set_expected_rate(2000.0)

    # Один цикл — 200 межкадровых интервалов и ровно 0.1 с. В нём есть
    # 2 паузы по 5.5 мс и быстрые приходы внутри пачек; средний темп 2000 Гц.
    intervals = [0.000498] * 100 + [0.000100] * 40 + [0.005500] * 2
    remainder = 0.1 - sum(intervals)
    intervals += [remainder / 58.0] * 58
    assert len(intervals) == 200
    assert sum(intervals) == pytest.approx(0.1)

    t_mono = 0.0
    pipeline.on_telemetry(REAL_FRAME, t_mono)
    for index in range(19_999):
        t_mono += intervals[index % len(intervals)]
        pipeline.on_telemetry(REAL_FRAME, t_mono)

    metrics = pipeline.metrics()
    assert metrics.frame_rate_hz == pytest.approx(2000.0, rel=0.002)
    assert metrics.loss_estimate == pytest.approx(0.0, abs=0.002)


def test_оценка_потерь_не_бывает_отрицательной_при_небольшом_переразгоне() -> None:
    """Частота выше ожидаемой — это не «минус потери», а нулевая оценка."""
    pipeline = Pipeline(PROFILE, PipelineConfig(history_frames=20_000))
    pipeline.set_expected_rate(2000.0)
    for index in range(2000):
        pipeline.on_telemetry(REAL_FRAME, index * 0.00049)

    assert pipeline.frame_rate_hz() > 2000.0
    assert pipeline.metrics().loss_estimate == 0.0


@pytest.mark.slow
def test_120_линий_инкрементальный_такт_укладывается_в_ui_budget() -> None:
    """Чат 15: 120 линий × 20 000 точек, новый хвост 200 кадров < 100 мс.

    Тест намеренно использует чистую Qt-free модель: он измеряет ту часть такта,
    которая до правки каждый раз пересчитывала всю историю. Отрисовочное
    прореживание pyqtgraph проверяется отдельно UI-тестом. На базовом e13bad4
    тест не проходит: `measurement_graph_model` не имеет инкрементального пути
    `previous=` и модель каждого тика строится заново.
    """
    lines = PROFILE.channels * PROFILE.fbg_per_channel
    frames = 20_000
    tail = 200
    selected = tuple(
        models.SlotRef(ch, pos)
        for ch in range(PROFILE.channels)
        for pos in range(PROFILE.fbg_per_channel)
    )
    positions = tuple((slot.channel, slot.position) for slot in selected)
    t0 = np.arange(frames, dtype=np.float64) / 2000.0
    base = 1540.0 + np.arange(lines, dtype=np.float64) * 0.01
    wave0 = base[np.newaxis, :] + np.arange(frames, dtype=np.float64)[:, np.newaxis] * 1e-7
    first_history = TraceHistorySnapshot(
        positions=positions,
        seq_start=0,
        seq_stop=frames,
        t_mono=t0,
        wavelength_nm=wave0,
    )
    previous = models.measurement_graph_model(app_snapshot(trace_history=first_history), selected)

    t1 = np.arange(tail, frames + tail, dtype=np.float64) / 2000.0
    wave1 = (
        base[np.newaxis, :] + np.arange(tail, frames + tail, dtype=np.float64)[:, np.newaxis] * 1e-7
    )
    second_history = TraceHistorySnapshot(
        positions=positions,
        seq_start=tail,
        seq_stop=frames + tail,
        t_mono=t1,
        wavelength_nm=wave1,
    )

    started = time.perf_counter()
    current = models.measurement_graph_model(
        app_snapshot(trace_history=second_history),
        selected,
        previous=previous,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(f"\nчат 15: 120 линий, инкрементальный UI-такт {elapsed_ms:.2f} мс")
    assert len(current.traces) == 120
    assert current.seq_start == tail and current.seq_stop == frames + tail
    assert elapsed_ms < 100.0


# --------------------------------------------------------------------------------------
# Совместная нагрузка: поток + recorder + штатный журнал + модель графика
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_одновременная_запись_журнал_и_график_не_создают_обратного_давления(
    tmp_path: Path,
) -> None:
    """2 кГц проходят в CSV при штатном журнале и опросе графика 10 Гц.

    Это первый тест, где четыре потребителя работают вместе. Журнал оставлен
    в штатном режиме `telemetry_stride=0`: отдельный нагрузочный тест уже
    проверяет его полный режим. График строится чистой моделью без Qt: здесь
    измеряется копирование истории и расчёт диапазона, а не оконная система CI.
    """
    rate_hz = 2000.0
    seconds = 10.0
    profile = DeviceProfile()
    sim = DeviceSimulator(
        profile=profile,
        scene=Scene(
            profile,
            [
                Grating(0, 0, 1544.80),
                Grating(0, 1, 1549.46),
                Grating(0, 2, 1551.50),
            ],
        ),
        reply_to=("127.0.0.1", 1),
        frame_rate_hz=rate_hz,
    )
    sim.start()
    host, port = sim.address
    endpoint = Endpoint(
        device_ip=host,
        device_port=port,
        **TEST_ENDPOINT_KWARGS,  # type: ignore[arg-type]
    )
    controller = AppController(
        AppConfig(
            endpoint=endpoint,
            profile=profile,
            session=QUIET,
            pipeline=PipelineConfig(history_frames=20_000, expected_rate_hz=rate_hz),
            recorder=RecorderConfig(
                directory=tmp_path / "data",
                rotate_bytes=None,
                rotate_seconds=None,
            ),
            packet_log=PacketLogConfig(
                directory=tmp_path / "log",
                telemetry_stride=0,
                rotate_bytes=None,
                keep_files=None,
            ),
        )
    )
    selected = tuple(models.SlotRef(0, position) for position in range(4))
    snapshots = 0
    graph_points = 0
    try:
        controller.start()
        # Как у реального прибора: адрес назначения ответов известен заранее,
        # поэтому объявляем открытый порт до первого Stop из connect().
        controller.session._transport.open()
        sim.reply_to = controller.session.local_address
        assert controller.connect().ok
        controller.set_measurement_trace_request(
            [(slot.channel, slot.position) for slot in selected], 10.0
        )
        assert controller.start_stream().ok
        assert wait_until(lambda: controller.pipeline.sequence >= 100, timeout=2.0)
        controller.start_recording()

        started = time.perf_counter()
        deadline = started + seconds
        while time.perf_counter() < deadline:
            snapshot = controller.snapshot()
            graph = models.measurement_graph_model(snapshot, selected)
            snapshots += 1
            graph_points += sum(trace.delta_nm.size for trace in graph.traces)
            time.sleep(0.1)
        elapsed = time.perf_counter() - started

        assert controller.stop_stream().ok
        assert wait_until(lambda: not sim.streaming, timeout=2.0)
        time.sleep(0.5)  # долёт сокета, recorder и packet_log добирают свои очереди
        controller.stop_recording()
        # KB_05, «Тесты»: сравнивать счётчик потребителя с принятым можно
        # только после опустошения очереди транспорта. Иначе последняя
        # датаграмма уже принята RX-потоком, но ещё не дошла до session tap.
        assert wait_until(lambda: controller.session._transport.queue_depth == 0, timeout=5.0)
        assert wait_until(lambda: controller.packet_log.stats.queue_depth == 0, timeout=5.0)

        final = controller.snapshot()
        sent = sim.stats.frames_sent
        metrics = controller.pipeline.metrics()
        transport = controller.session.transport_stats
        log_stats = controller.packet_log.stats
        recorder_stats = final.recorder
        assert recorder_stats is not None

        print("\n--- чат 12: поток + запись + журнал + график ---")
        print(
            f"{elapsed:.2f} с · отправлено {sent} · разобрано {metrics.frames} · "
            f"записано {recorder_stats.rows} · снимков графика {snapshots}"
        )
        print(
            f"потери транспорта {transport.dropped_queue_full} · "
            f"разрывы recorder {recorder_stats.gaps}/{recorder_stats.lost_frames} · "
            f"потери журнала {log_stats.dropped_queue_full}/{log_stats.lost_records}"
        )
        print(
            f"темп длинного окна {metrics.frame_rate_hz:.2f} Гц · "
            f"оценка потерь {(metrics.loss_estimate or 0.0) * 100.0:.4f} % · "
            f"обработано точек графика {graph_points}"
        )

        assert metrics.parse_errors == 0
        assert sent > rate_hz * seconds * 0.95, "симулятор не выдержал паспортный темп"
        assert metrics.frames == sent, "между симулятором и pipeline потерялся кадр"
        assert transport.dropped_queue_full == 0
        assert recorder_stats.error is None
        assert recorder_stats.gaps == 0
        assert recorder_stats.lost_frames == 0
        assert recorder_stats.rows == recorder_stats.frames_span
        assert log_stats.error is None
        assert log_stats.telemetry_seen == metrics.frames
        assert log_stats.telemetry_admitted == 0
        assert log_stats.telemetry_skipped == metrics.frames
        assert log_stats.dropped_queue_full == 0
        assert log_stats.lost_records == 0
        assert snapshots >= int(elapsed * 8.0), "модель графика не выдержала обновление около 10 Гц"
        assert graph_points > 0
    finally:
        controller.shutdown()
        sim.stop()


@pytest.mark.slow
def test_запись_2кгц_переживает_сетевой_обрыв_с_одним_gap(tmp_path: Path) -> None:
    """2 кГц + Recorder + N11: запись продолжается через двухсекундную тишину.

    `go_silent` оставляет флаг Streaming прибора включённым — это именно
    наблюдаемое поведение N11, а не гипотеза reboot. Пассивное окно здесь
    длиннее тишины, поэтому ни Stop, ни Start восстановлению не нужны.
    """
    rate_hz = 2000.0
    profile = DeviceProfile()
    sim = DeviceSimulator(
        profile=profile,
        scene=Scene(profile, [Grating(0, 0, 1544.80), Grating(0, 1, 1551.50)]),
        reply_to=("127.0.0.1", 1),
        frame_rate_hz=rate_hz,
    )
    sim.start()
    host, port = sim.address
    endpoint = Endpoint(
        device_ip=host,
        device_port=port,
        **TEST_ENDPOINT_KWARGS,  # type: ignore[arg-type]
    )
    session_config = dataclasses.replace(
        QUIET,
        stream_stall_floor_s=0.3,
        stream_resume_wait_s=3.0,
        backoff_schedule=(0.05, 0.1),
    )
    controller = AppController(
        AppConfig(
            endpoint=endpoint,
            profile=profile,
            session=session_config,
            pipeline=PipelineConfig(history_frames=20_000, expected_rate_hz=rate_hz),
            recorder=RecorderConfig(
                directory=tmp_path / "data",
                rotate_bytes=None,
                rotate_seconds=None,
            ),
            packet_log=PacketLogConfig(directory=None),
        )
    )
    try:
        controller.start()
        controller.session._transport.open()
        sim.reply_to = controller.session.local_address
        assert controller.connect().ok
        assert controller.start_stream().ok
        assert wait_until(lambda: controller.pipeline.sequence >= 100, timeout=2.0)
        recorder = controller.start_recording()
        assert wait_until(lambda: recorder.stats.rows >= 100, timeout=2.0)
        commands_before = controller.session.stats().commands

        sim.go_silent(2.0)
        assert wait_until(lambda: controller.session.state is SessionState.DEGRADED, timeout=2.0)
        assert controller.is_recording
        assert wait_until(lambda: controller.session.state is SessionState.STREAMING, timeout=4.0)
        assert controller.is_recording
        assert wait_until(lambda: recorder.stats.gaps == 1, timeout=2.0)
        assert wait_until(lambda: recorder.stats.rows >= 300, timeout=2.0)

        assert controller.stop_stream().ok
        assert wait_until(lambda: not sim.streaming, timeout=2.0)
        assert wait_until(lambda: controller.session._transport.queue_depth == 0, timeout=2.0)
        controller.stop_recording()

        paths = files_of(tmp_path / "data")
        numbers = frame_numbers(paths)
        marks = [mark for path in paths for mark in gap_lines(path)]
        stats = recorder.stats
        transport = controller.session.transport_stats

        print("\n--- чат 14: 2 кГц + recorder + N11 ---")
        print(
            f"строк {stats.rows}, GAP {stats.gaps}, потеряно recorder {stats.lost_frames}, "
            f"потери транспорта {transport.dropped_queue_full}"
        )
        print(f"маркер: {marks[0] if marks else 'нет'}")

        assert stats.error is None
        assert stats.gaps == 1
        assert stats.lost_frames == 0
        assert len(marks) == 1
        assert "frames=unknown" in marks[0]
        assert transport.dropped_queue_full == 0
        assert numbers == list(range(len(numbers))), (
            "локальные номера полученных кадров непрерывны; сетевой провал отмечает # GAP"
        )
        # После N11 добавились только пять чтений конфигурации 0x10 и штатный
        # Stop от явного завершения теста. Восстановление не посылало Start.
        assert controller.session.stats().commands >= commands_before + 5
    finally:
        controller.shutdown()
        sim.stop()
