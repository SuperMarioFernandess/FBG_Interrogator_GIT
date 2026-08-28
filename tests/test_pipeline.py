"""Тесты приёмного тракта: кольцо, курсор писателя, децимация UI, агрегаты, метрики.

Кадры собираются `fbg.sim.encode` — тем же независимым от кодека кодом, что
и у симулятора (KB_05 №11), и подаются в `Pipeline.on_telemetry` напрямую:
сеть здесь не проверяется, она закрыта тестами транспорта и сессии.

Метки времени задаются тестом явно. Это не удобство, а требование: децимация
для UI работает по времени кадра, и проверить «одинаковый темп UI при 3 Гц
и при 2000 Гц» на реальных часах значило бы проверять планировщик ОС.

⚠️ Тесты фиксируют поведение **pipeline**, а не факты о приборе. Раскладка
кадра (N4), единицы частоты (D1) и код «пик не найден» (N3) закрыты скринингом
и приходят сюда из профиля.
"""

import threading
import time

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.pipeline import Aggregates, Pipeline, PipelineConfig, RingHistory
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.sim.encode import encode_measurement, nm_to_raw

#: Решётки стенда, ✅ скрининг: прибор распознаёт две из четырёх.
STAND_NM = (1544.80, 1551.51)

#: Быстрые параметры: маленькое кольцо и частая публикация, чтобы тесты
#: не занимали 20 МБ и не ждали.
FAST = PipelineConfig(history_frames=256, ui_period_s=0.01, aggregate_window_s=1.0)


def make_frame(
    profile: DeviceProfile,
    wavelengths: dict[tuple[int, int], float] | None = None,
    *,
    temp_c: float = 16.85,
    raw_overrides: dict[tuple[int, int], int] | None = None,
) -> bytes:
    """Собирает кадр телеметрии: позиция → длина волны, остальное «пик не найден».

    Позиции задаются явно и произвольно: pipeline не привязывает датчик
    к позиции (решение Р30), и тест не должен закреплять такую привязку.
    """
    divisor = profile.freq_divisor or 10
    freq = np.zeros((profile.channels, profile.fbg_per_channel), dtype=np.uint32)
    for (channel, position), nm in (wavelengths or {}).items():
        freq[channel, position] = nm_to_raw(nm, divisor)
    for (channel, position), raw in (raw_overrides or {}).items():
        freq[channel, position] = raw
    temp = np.full(profile.channels, round(temp_c / profile.case_temp_scale), dtype=np.int32)
    return encode_measurement(profile, freq, temp)


def feed(
    pipeline: Pipeline,
    frames: int,
    *,
    rate_hz: float,
    start_s: float = 0.0,
    builder=None,
) -> float:
    """Подаёт кадры с равномерными метками времени. Возвращает метку следующего кадра."""
    profile = pipeline.profile
    period = 1.0 / rate_hz
    for index in range(frames):
        data = builder(index) if builder is not None else make_frame(profile, {(0, 0): 1544.80})
        pipeline.on_telemetry(data, start_s + index * period)
    return start_s + frames * period


@pytest.fixture
def profile() -> DeviceProfile:
    """Профиль прибора со стенда."""
    return DeviceProfile()


@pytest.fixture
def pipeline(profile: DeviceProfile) -> Pipeline:
    """Тракт с маленьким кольцом, поток публикации не запущен."""
    return Pipeline(profile, FAST)


# --------------------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------------------


def test_config_rejects_nonsense() -> None:
    """Некорректные параметры — баг вызывающего, значит ValueError (KB_05)."""
    with pytest.raises(ValueError, match="history_frames"):
        PipelineConfig(history_frames=0)
    with pytest.raises(ValueError, match="ui_period_s"):
        PipelineConfig(ui_period_s=0.0)
    with pytest.raises(ValueError, match="aggregate_window_s"):
        PipelineConfig(aggregate_window_s=-1.0)
    with pytest.raises(ValueError, match="rate_window_frames"):
        PipelineConfig(rate_window_frames=1)
    with pytest.raises(ValueError, match="expected_rate_hz"):
        PipelineConfig(expected_rate_hz=0.0)


def test_expected_rate_settable_after_probe(pipeline: Pipeline) -> None:
    """Ожидаемый темп приходит из прочитанной 10 04, а не из конструктора."""
    assert pipeline.metrics().expected_rate_hz is None
    pipeline.set_expected_rate(2000.0)
    assert pipeline.metrics().expected_rate_hz == 2000.0
    pipeline.set_expected_rate(None)
    assert pipeline.metrics().expected_rate_hz is None
    with pytest.raises(ValueError, match="rate_hz"):
        pipeline.set_expected_rate(0.0)


# --------------------------------------------------------------------------------------
# Кольцо
# --------------------------------------------------------------------------------------


def test_ring_segments_wrap() -> None:
    """Логический диапазон разбивается на один или два физических отрезка."""
    ring = RingHistory(8, 1, 1)
    assert ring.segments(0, 4) == ((0, 4),)
    assert ring.segments(6, 10) == ((6, 8), (0, 2))
    assert ring.segments(8, 12) == ((0, 4),)
    assert ring.segments(3, 3) == ()


def test_ring_size_is_honest() -> None:
    """Объём кольца считается, а не берётся «с запасом».

    Умолчание — 20 000 кадров: 10 секунд при 2000 Гц и около 20 МБ. Пять минут
    той же истории стоили бы 610 МБ, поэтому кольцо и не является архивом.
    """
    ring = RingHistory(20_000, 4, 30)
    assert 20e6 < ring.nbytes < 21e6
    per_frame = ring.nbytes / ring.capacity
    assert 1000 < per_frame < 1040
    assert RingHistory(600_000, 4, 30).nbytes > 600e6


def test_ring_does_not_grow_and_evicts_oldest(pipeline: Pipeline) -> None:
    """Кольцо фиксированного размера: вытесняется старейший, массивы не растут."""
    ring = pipeline.history
    before = ring.nbytes
    capacity = ring.capacity

    feed(pipeline, capacity + 40, rate_hz=2000.0)

    assert ring.nbytes == before, "кольцо не должно перевыделять массивы"
    assert ring.used == capacity
    assert ring.written == capacity + 40
    assert ring.oldest_seq == 40
    assert not ring.holds(39)
    assert ring.holds(40)


# --------------------------------------------------------------------------------------
# Приём и порядок
# --------------------------------------------------------------------------------------


def test_frames_are_accepted_in_order(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Порядок кадров сохраняется: курсор отдаёт их той же чередой, что и приём."""
    cursor = pipeline.cursor()
    marks = [1530.0 + index * 0.5 for index in range(20)]

    for index, nm in enumerate(marks):
        pipeline.on_telemetry(make_frame(profile, {(0, 0): nm}), index * 0.001)

    batch = cursor.take()
    assert batch is not None
    assert len(batch) == 20
    assert batch.seq_start == 0
    assert batch.gap == 0
    assert list(batch.seq) == list(range(20))
    got = batch.wavelength_nm()[:, 0, 0]
    # Допуск — шаг квантования прибора 8.01 пм (KB_01).
    assert np.allclose(got, marks, atol=0.009)
    assert np.all(np.diff(batch.t_mono) > 0)


def test_cursor_starts_from_now(pipeline: Pipeline) -> None:
    """Курсор заводится «с этого момента»: прошлое писателю не отдаётся."""
    feed(pipeline, 10, rate_hz=1000.0)
    cursor = pipeline.cursor()
    assert cursor.take() is None
    feed(pipeline, 5, rate_hz=1000.0, start_s=1.0)
    batch = cursor.take()
    assert batch is not None
    assert batch.seq_start == 10
    assert len(batch) == 5


def test_parse_error_is_counted_not_stored(pipeline: Pipeline) -> None:
    """Неразобравшаяся датаграмма в историю не попадает и считается отдельно."""
    pipeline.on_telemetry(b"\x30\x02\x00\x00", 1.0)
    pipeline.on_telemetry(bytes(494), 1.1)

    metrics = pipeline.metrics()
    assert metrics.frames == 0
    assert metrics.parse_errors == 2
    assert sum(metrics.errors.values()) == 2
    assert pipeline.sequence == 0


def test_take_limit_splits_batches(pipeline: Pipeline) -> None:
    """Ограничение размера пачки не теряет кадров и не путает порядок."""
    cursor = pipeline.cursor()
    feed(pipeline, 50, rate_hz=1000.0)

    seen: list[int] = []
    while (batch := cursor.take(limit=7)) is not None:
        seen.extend(int(value) for value in batch.seq)
    assert seen == list(range(50))
    assert cursor.delivered == 50
    assert cursor.lost == 0


# --------------------------------------------------------------------------------------
# Потеря у писателя отмечается
# --------------------------------------------------------------------------------------


def test_slow_writer_gets_marked_gap(pipeline: Pipeline) -> None:
    """Отставший писатель получает разрыв числом, а не молча теряет кадры."""
    cursor = pipeline.cursor()
    capacity = pipeline.history.capacity

    feed(pipeline, capacity + 30, rate_hz=2000.0)
    batch = cursor.take()

    assert batch is not None
    assert batch.gap == 30, "разрыв обязан быть отмечен, а не проглочен"
    assert batch.seq_start == 30
    assert len(batch) == capacity
    assert cursor.lost == 30
    assert cursor.gaps == 1
    assert pipeline.metrics().evicted == 30


def test_eviction_is_not_loss(pipeline: Pipeline) -> None:
    """Вытеснение кадра из кольца — не потеря, если читатель успел его забрать.

    Кольцо конечно, и при долгом потоке оно оборачивается непрерывно: счётчик
    `evicted` растёт всегда. Потерей это становится только для отставшего
    читателя, и считает её курсор. Смешивать эти два числа нельзя — на этом
    ошибся нагрузочный тест, пока счётчик назывался «вытеснено».
    """
    cursor = pipeline.cursor()
    capacity = pipeline.history.capacity
    stamp = 0.0
    for _ in range(6):
        feed(pipeline, capacity // 2, rate_hz=2000.0, start_s=stamp)
        stamp += capacity / 2 / 2000.0
        cursor.take()

    metrics = pipeline.metrics()
    assert metrics.evicted > capacity, "кольцо обязано было обернуться"
    assert cursor.lost == 0, "читатель не отставал — потери быть не должно"
    assert cursor.gaps == 0
    assert cursor.delivered == metrics.frames


def test_gap_registered_even_when_batch_is_empty(pipeline: Pipeline) -> None:
    """Если кольцо обогнало читателя, разрыв учитывается и при пустой выдаче.

    `limit=0` — вырожденный опрос: он доводит до ветки «читать нечего,
    но разрыв уже случился», которую иначе не достать.
    """
    cursor = pipeline.cursor()
    feed(pipeline, pipeline.history.capacity * 2, rate_hz=2000.0)
    cursor.take(limit=0)
    assert cursor.lost == pipeline.history.capacity
    assert cursor.position == pipeline.history.oldest_seq


def test_cursor_stride_records_every_nth(pipeline: Pipeline) -> None:
    """Децимация записи идёт по сквозному номеру, поэтому переживает разрыв."""
    cursor = pipeline.cursor(stride=5)
    feed(pipeline, 20, rate_hz=1000.0)

    batch = cursor.take()
    assert batch is not None
    assert list(batch.seq) == [0, 5, 10, 15]
    assert batch.seq_start == 0
    assert batch.seq_stop == 20

    feed(pipeline, 3, rate_hz=1000.0, start_s=1.0)
    tail = cursor.take()
    assert tail is not None
    assert list(tail.seq) == [20]

    with pytest.raises(ValueError, match="stride"):
        pipeline.cursor(stride=0)


def test_cursor_lag_visible(pipeline: Pipeline) -> None:
    """Отставание читателя видно числом до того, как оно превратится в потерю."""
    cursor = pipeline.cursor()
    feed(pipeline, 40, rate_hz=1000.0)
    assert cursor.lag == 40
    cursor.take()
    assert cursor.lag == 0


# --------------------------------------------------------------------------------------
# Децимация для UI
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate_hz", [3.0, 2000.0])
def test_ui_decimation_is_time_based(profile: DeviceProfile, rate_hz: float) -> None:
    """Темп UI задаётся временем, а не числом кадров.

    Десять секунд потока при 3 Гц и при 2000 Гц отличаются числом кадров
    в 666 раз, а числом обновлений UI — ничем: децимация сравнивает метки
    времени кадров, а не считает их.
    """
    pipeline = Pipeline(profile, PipelineConfig(history_frames=64, ui_period_s=1.0))
    feed(pipeline, int(rate_hz * 10), rate_hz=rate_hz)

    metrics = pipeline.metrics()
    assert metrics.frames == int(rate_hz * 10)
    assert metrics.ui_gates == 10, "при периоде 1 с за 10 секунд ровно 10 тактов UI"


def test_ui_snapshot_published_by_thread(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Поток-публикатор собирает снимок сам, приёмный поток агрегаты не считает."""
    with pipeline:
        assert pipeline.snapshot() is None
        base = time.perf_counter()
        for index in range(10):
            pipeline.on_telemetry(make_frame(profile, {(0, 0): 1544.80}), base + index * 0.05)
        deadline = time.perf_counter() + 5.0
        while pipeline.snapshot() is None and time.perf_counter() < deadline:
            time.sleep(0.005)

    snapshot = pipeline.snapshot()
    assert snapshot is not None
    assert snapshot.seq >= 0
    assert snapshot.filled_total == 1
    assert snapshot.freq_divisor == profile.freq_divisor
    assert pipeline.metrics().ui_updates >= 1


def test_snapshot_carries_last_frame_and_positions(
    pipeline: Pipeline, profile: DeviceProfile
) -> None:
    """Снимок отдаёт то, что пришло: позицию, длину волны, признак валидности."""
    pipeline.on_telemetry(make_frame(profile, {(0, 0): 1544.80}), 1.0)
    pipeline.on_telemetry(make_frame(profile, {(0, 1): 1551.51, (2, 0): 1530.00}), 1.1)

    snapshot = pipeline.publish_now()
    assert snapshot is not None
    assert snapshot.seq == 1
    assert snapshot.t_mono == 1.1
    nm = snapshot.wavelength_nm
    assert nm[0, 1] == pytest.approx(1551.51, abs=0.009)
    assert nm[2, 0] == pytest.approx(1530.00, abs=0.009)
    assert np.isnan(nm[0, 0]), "позиция без пика обязана остаться NaN"
    assert list(snapshot.filled) == [1, 0, 1, 0]
    assert snapshot.case_temp_c[0] == pytest.approx(16.85, abs=0.005)
    assert snapshot.latency_s > 0.0


def test_slow_ui_does_not_slow_the_writer(profile: DeviceProfile) -> None:
    """Медленный потребитель UI не мешает ни приёму, ни записи.

    Публикатор здесь искусственно заторможен: каждый снимок собирается
    50 мс. Приёмный путь этого не замечает — он только выставляет событие, —
    и писатель получает все кадры до одного.
    """

    class SluggishPipeline(Pipeline):
        """Тракт с намеренно медленной публикацией снимка."""

        def publish_now(self):
            time.sleep(0.05)
            return super().publish_now()

    pipeline = SluggishPipeline(profile, PipelineConfig(history_frames=4096, ui_period_s=0.001))
    cursor = pipeline.cursor()
    frame = make_frame(profile, {(0, 0): 1544.80})

    with pipeline:
        started = time.perf_counter()
        for index in range(2000):
            pipeline.on_telemetry(frame, started + index * 0.0005)
        ingest_s = time.perf_counter() - started

    collected = 0
    while (batch := cursor.take()) is not None:
        collected += len(batch)

    assert collected == 2000, "запись обязана получить все кадры, как бы ни тормозил UI"
    assert cursor.lost == 0
    assert ingest_s < 1.0, f"приём занял {ingest_s:.3f} с — публикатор его подпёр"


# --------------------------------------------------------------------------------------
# NaN в агрегатах
# --------------------------------------------------------------------------------------


def test_nan_survives_aggregates(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Позиция без пика остаётся NaN во всех агрегатах и не становится нулём."""
    for index in range(10):
        pipeline.on_telemetry(make_frame(profile, {(0, 0): 1544.80}), index * 0.01)

    snapshot = pipeline.publish_now()
    assert snapshot is not None
    agg = snapshot.aggregates
    assert agg.frames == 10

    assert np.isnan(agg.mean_ghz[0, 1])
    assert np.isnan(agg.min_ghz[0, 1])
    assert np.isnan(agg.max_ghz[0, 1])
    assert np.isnan(agg.peak_to_peak_ghz[0, 1])
    assert np.isnan(agg.mean_nm[0, 1])
    assert np.isnan(agg.peak_to_peak_nm[0, 1])
    assert agg.valid[0, 1] == 0

    assert agg.valid[0, 0] == 10
    assert agg.mean_nm[0, 0] == pytest.approx(1544.80, abs=0.009)
    assert agg.peak_to_peak_ghz[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_aggregates_ignore_missing_frames(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Среднее считается по кадрам с пиком, а не по всем: NaN не тянет вниз."""
    stamps = 0.0
    for index in range(10):
        found = index % 2 == 0
        data = make_frame(profile, {(0, 0): 1544.80 + index * 0.01} if found else {})
        pipeline.on_telemetry(data, stamps)
        stamps += 0.01

    agg = pipeline.publish_now().aggregates  # type: ignore[union-attr]
    assert agg.valid[0, 0] == 5
    # Пики стояли на 1544.80, 1544.82, 1544.84, 1544.86, 1544.88 — среднее 1544.84.
    assert agg.mean_nm[0, 0] == pytest.approx(1544.84, abs=0.009)
    assert agg.peak_to_peak_nm[0, 0] == pytest.approx(0.08, abs=0.009)
    assert 0.0 not in set(np.ravel(agg.mean_ghz[~np.isnan(agg.mean_ghz)]))


def test_aggregates_span_and_extremes(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Размах считается по окну, а минимум и максимум переводятся в нм без путаницы."""
    values = [1544.70, 1544.90, 1544.80]
    for index, nm in enumerate(values):
        pipeline.on_telemetry(make_frame(profile, {(1, 3): nm}), index * 0.01)

    agg = pipeline.publish_now().aggregates  # type: ignore[union-attr]
    assert agg.min_nm[1, 3] == pytest.approx(1544.70, abs=0.009)
    assert agg.max_nm[1, 3] == pytest.approx(1544.90, abs=0.009)
    assert agg.peak_to_peak_nm[1, 3] == pytest.approx(0.20, abs=0.009)
    # Меньшей длине волны соответствует бо́льшая частота: перевод инвертирует.
    assert agg.max_ghz[1, 3] == pytest.approx(C_NM_GHZ / 1544.70, rel=1e-6)
    assert agg.span_s == pytest.approx(0.02, abs=1e-9)


def test_aggregate_window_is_time_bounded(profile: DeviceProfile) -> None:
    """В окно попадает только последняя секунда, а не всё кольцо."""
    pipeline = Pipeline(profile, PipelineConfig(history_frames=4096, aggregate_window_s=0.5))
    feed(pipeline, 2000, rate_hz=1000.0)

    agg = pipeline.publish_now().aggregates  # type: ignore[union-attr]
    assert agg.frames == pytest.approx(500, abs=2)
    assert agg.span_s == pytest.approx(0.5, abs=0.01)


def test_empty_aggregates_are_nan(pipeline: Pipeline) -> None:
    """Без кадров агрегатов нет, а не «нули»."""
    assert pipeline.publish_now() is None
    empty = Aggregates(
        0,
        0.0,
        np.zeros((1, 1), dtype=np.int64),
        np.full((1, 1), np.nan),
        np.full((1, 1), np.nan),
        np.full((1, 1), np.nan),
    )
    assert np.isnan(empty.peak_to_peak_nm[0, 0])


# --------------------------------------------------------------------------------------
# Переменное число валидных пиков
# --------------------------------------------------------------------------------------


def test_varying_peak_count_is_not_an_error(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Число заполненных позиций меняется от кадра к кадру — это норма, не сбой.

    ✅ Скрининг: прибор распознаёт две решётки из четырёх, и число найденных
    пиков плавает. Метрика «сколько позиций заполнено» полезна, ошибкой быть
    не должна (KB_04, N15).
    """
    cursor = pipeline.cursor()
    plans: list[dict[tuple[int, int], float]] = [
        {(0, 0): STAND_NM[0], (0, 1): STAND_NM[1]},
        {(0, 0): STAND_NM[0]},
        {},
        {(0, 0): STAND_NM[0], (0, 1): STAND_NM[1]},
    ]
    for index, plan in enumerate(plans):
        pipeline.on_telemetry(make_frame(profile, plan), index * 0.01)

    metrics = pipeline.metrics()
    assert metrics.parse_errors == 0
    assert metrics.frames == 4
    assert metrics.errors == {}
    assert metrics.filled_by_channel == (2, 0, 0, 0)

    batch = cursor.take()
    assert batch is not None
    assert list(batch.filled[:, 0]) == [2, 1, 0, 2]


def test_position_is_a_slot_not_a_sensor(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """Pipeline не привязывает датчик к позиции: он отдаёт то, что пришло.

    ✅ Скрининг: 1551.5 нм лежит в позиции 1, будучи четвёртой решёткой линии
    (решение Р30). Тот же датчик в другом кадре может оказаться в позиции 0,
    и pipeline обязан отдать это как есть, не «исправляя» порядок. Привязка
    по λ с допуском — работа калибровки, чат №10.
    """
    pipeline.on_telemetry(make_frame(profile, {(0, 0): STAND_NM[0], (0, 1): STAND_NM[1]}), 0.0)
    first = pipeline.publish_now()
    pipeline.on_telemetry(make_frame(profile, {(0, 0): STAND_NM[1]}), 0.1)
    second = pipeline.publish_now()

    assert first is not None and second is not None
    assert first.wavelength_nm[0, 1] == pytest.approx(STAND_NM[1], abs=0.009)
    assert second.wavelength_nm[0, 0] == pytest.approx(STAND_NM[1], abs=0.009)
    assert np.isnan(second.wavelength_nm[0, 1])


def test_missing_code_and_out_of_range_both_give_nan(
    pipeline: Pipeline, profile: DeviceProfile
) -> None:
    """Ноль «пик не найден» и значение вне диапазона одинаково дают NaN (KB_05 №9)."""
    data = make_frame(
        profile,
        {(0, 0): 1544.80},
        raw_overrides={(0, 1): 0x000000, (0, 2): 0xFFFFFF},
    )
    pipeline.on_telemetry(data, 0.0)

    snapshot = pipeline.publish_now()
    assert snapshot is not None
    assert not np.isnan(snapshot.wavelength_nm[0, 0])
    assert np.isnan(snapshot.wavelength_nm[0, 1])
    assert np.isnan(snapshot.wavelength_nm[0, 2])
    assert snapshot.filled[0] == 1


# --------------------------------------------------------------------------------------
# Метрики темпа и потерь
# --------------------------------------------------------------------------------------


def test_frame_rate_measured_from_stamps(pipeline: Pipeline) -> None:
    """Фактический темп берётся из меток времени, а не из настройки прибора."""
    assert pipeline.frame_rate_hz() == 0.0
    feed(pipeline, 200, rate_hz=2000.0)
    assert pipeline.frame_rate_hz() == pytest.approx(2000.0, rel=0.02)


def test_loss_estimate_on_known_input(pipeline: Pipeline) -> None:
    """Оценка потерь на заведомом входе: половина кадров — половина оценки."""
    pipeline.set_expected_rate(2000.0)
    feed(pipeline, 200, rate_hz=2000.0)
    assert pipeline.metrics().loss_estimate == pytest.approx(0.0, abs=0.02)

    half = Pipeline(pipeline.profile, FAST)
    half.set_expected_rate(2000.0)
    feed(half, 200, rate_hz=1000.0)
    assert half.metrics().loss_estimate == pytest.approx(0.5, abs=0.02)


def test_loss_estimate_unknown_without_expected_rate(pipeline: Pipeline) -> None:
    """Пока `10 04` не прочитана, оценивать потери не с чем — None, а не ноль.

    Протокол не даёт ни счётчиков кадров, ни последовательных номеров, поэтому
    другого способа оценить потерю не существует вовсе.
    """
    feed(pipeline, 50, rate_hz=2000.0)
    assert pipeline.metrics().loss_estimate is None


def test_rate_window_follows_slow_stream(profile: DeviceProfile) -> None:
    """При 3 Гц темп измерим: окно ограничено и временем, и числом кадров."""
    pipeline = Pipeline(profile, PipelineConfig(history_frames=64, rate_window_s=1.0))
    feed(pipeline, 12, rate_hz=3.0)
    assert pipeline.frame_rate_hz() == pytest.approx(3.0, rel=0.1)


def test_metrics_report_history_and_lag(pipeline: Pipeline) -> None:
    """Метрики показывают заполнение кольца и задержку приёмного потока."""
    feed(pipeline, 100, rate_hz=2000.0)
    metrics = pipeline.metrics()
    assert metrics.history_frames == FAST.history_frames
    assert metrics.history_used == 100
    assert metrics.history_bytes == pipeline.history.nbytes
    assert metrics.evicted == 0
    assert metrics.ingest_lag_s != 0.0


# --------------------------------------------------------------------------------------
# Старт, остановка, потоки
# --------------------------------------------------------------------------------------


def _pipeline_threads() -> list[str]:
    """Живые потоки тракта по имени."""
    return [thread.name for thread in threading.enumerate() if thread.name == "fbg-pipeline"]


def test_start_stop_joins_threads(pipeline: Pipeline) -> None:
    """После остановки не остаётся ни висящих потоков, ни работы в фоне."""
    assert not pipeline.is_running
    pipeline.start()
    assert pipeline.is_running
    assert _pipeline_threads() == ["fbg-pipeline"]

    pipeline.start()  # повторный старт безвреден
    assert _pipeline_threads() == ["fbg-pipeline"]

    pipeline.stop()
    assert not pipeline.is_running
    assert _pipeline_threads() == []
    pipeline.stop()  # повторная остановка безвредна


def test_context_manager_stops_thread(profile: DeviceProfile) -> None:
    """Контекстный менеджер закрывает поток даже при исключении."""
    pipeline = Pipeline(profile, FAST)
    with pytest.raises(RuntimeError, match="стенд"), pipeline:
        assert pipeline.is_running
        raise RuntimeError("стенд отвалился")
    assert not pipeline.is_running
    assert _pipeline_threads() == []


def test_ingest_works_without_started_thread(pipeline: Pipeline) -> None:
    """Кадры принимаются и без публикатора: он нужен только снимкам UI."""
    cursor = pipeline.cursor()
    feed(pipeline, 30, rate_hz=1000.0)
    assert not pipeline.is_running
    batch = cursor.take()
    assert batch is not None and len(batch) == 30
    assert pipeline.metrics().ui_updates == 0


# --------------------------------------------------------------------------------------
# Стыковка с сессией
# --------------------------------------------------------------------------------------


def test_callback_signature_matches_session(pipeline: Pipeline, profile: DeviceProfile) -> None:
    """`on_telemetry` подходит сессии по сигнатуре: она зовёт его как (bytes, float).

    Тест держит контракт: сессия отдаёт **сырые байты** и метку времени приёма,
    разбор — работа pipeline (KB_03, таблица ответственности).
    """
    telemetry = make_frame(profile, {(0, 0): 1544.80})
    assert codec.classify(telemetry) == (codec.ID_MODE, codec.FC_STREAM)
    pipeline.on_telemetry(telemetry, 1.25)
    assert pipeline.sequence == 1
    assert pipeline.history.t_mono[0] == 1.25
