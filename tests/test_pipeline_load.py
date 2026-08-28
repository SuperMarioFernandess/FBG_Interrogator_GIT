"""Нагрузочный тест приёмного тракта целиком: симулятор → транспорт → сессия → pipeline.

Чат №2 снял риск R2 со стороны отправки, чат №3 — со стороны приёма датаграмм.
Здесь замыкается последнее звено: разбор кадров, кольцевая история и выдача
писателю на паспортных 2000 Гц.

Критерий тот же, что и у транспорта (KB_05): потери менее 0.1 % за 60 секунд,
и считаются они **на выходе курсора**, то есть по числу кадров, реально
доехавших до того, кто будет писать файл. Всё, что потерялось раньше —
в сети, в очереди транспорта, в кольце, — в это число уже входит.

Задержка меряется двумя величинами, и путать их нельзя:
  * задержка приёмного потока — от `recvfrom` до конца разбора кадра; это
    вклад самого pipeline;
  * задержка доставки — от `recvfrom` до момента, когда кадр забрал писатель;
    в неё входит период опроса писателя, поэтому она измеряется вместе с ним.

Маркер `slow`: прогон длится больше минуты, отдельная job в CI.
"""

import threading
import time

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.endpoint import Endpoint
from fbg.core.pipeline import Pipeline, PipelineConfig
from fbg.core.profile import DeviceProfile
from fbg.core.session import Session, SessionState
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene
from tests.test_session import QUIET, TEST_ENDPOINT_KWARGS, open_port_and_announce, wait_until

pytestmark = pytest.mark.slow

#: Паспортный темп прибора: 2000 Гц, код 0x00CA, ✅ прочитан командой 10 04.
TARGET_RATE_HZ = 2000

#: Длительность основного прогона (KB_05: критерий измеряется за 60 секунд).
LOAD_SECONDS = 60.0

#: Допустимая доля потерь, KB_05.
MAX_LOSS_FRACTION = 0.001

#: Как часто писатель забирает кадры. 10 мс — реалистичная пауза для того,
#: кто пишет CSV пачками, и она входит в измеренную задержку доставки.
DRAIN_PERIOD_S = 0.01


class Writer:
    """Читатель-писатель: тянет кадры курсором в своём потоке и копит статистику.

    Файлов не пишет — recorder это следующий чат. Здесь он изображает
    потребителя, которому нужны **все** кадры: считает полученное, разрывы
    и задержку доставки.
    """

    def __init__(
        self, pipeline: Pipeline, capacity: int, drain_period_s: float = DRAIN_PERIOD_S
    ) -> None:
        self.cursor = pipeline.cursor()
        self.drain_period_s = drain_period_s
        self.latency_us = np.empty(capacity, dtype=np.float64)
        self.count = 0
        self.frames = 0
        self.gaps = 0
        self.lost = 0
        self.peak_lag = 0
        self.batches = 0
        self.max_batch = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="test-writer", daemon=True)

    def start(self) -> None:
        """Запускает поток чтения."""
        self._thread.start()

    def stop(self) -> None:
        """Останавливает поток, добрав всё, что осталось в кольце."""
        self._stop.set()
        self._thread.join(timeout=10.0)
        while self._drain():
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._drain():
                time.sleep(self.drain_period_s)
            elif self.drain_period_s > DRAIN_PERIOD_S:
                # Намеренно медленный читатель: пауза даже когда кадры есть.
                time.sleep(self.drain_period_s)

    def _drain(self) -> bool:
        self.peak_lag = max(self.peak_lag, self.cursor.lag)
        batch = self.cursor.take()
        if batch is None:
            return False
        size = len(batch)
        self.frames += size
        self.batches += 1
        self.max_batch = max(self.max_batch, size)
        if batch.gap:
            self.gaps += 1
            self.lost += batch.gap
        room = min(size, self.latency_us.size - self.count)
        if room > 0:
            self.latency_us[self.count : self.count + room] = batch.latency_s()[:room] * 1e6
            self.count += room
        return True

    def percentiles(self) -> tuple[float, float, float, float]:
        """p50, p99, p99.9 и максимум задержки доставки, мкс."""
        sample = self.latency_us[: self.count]
        if sample.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.percentile(sample, 50)),
            float(np.percentile(sample, 99)),
            float(np.percentile(sample, 99.9)),
            float(sample.max()),
        )


class Stand:
    """Симулятор, сессия и pipeline, соединённые как прибор и приложение.

    Порядок соединения повторяет реальный: прибор поднимается первым, сессия
    открывает приёмный порт и сообщает его прибору — тот отвечает на
    прописанный адрес назначения, а не на source-порт (KB_01).
    """

    def __init__(self, rate_hz: int, history_frames: int) -> None:
        self.profile = DeviceProfile()
        self.pipeline = Pipeline(
            self.profile,
            PipelineConfig(
                history_frames=history_frames,
                ui_period_s=0.05,
                aggregate_window_s=1.0,
                expected_rate_hz=float(rate_hz),
            ),
        )
        self.sim = DeviceSimulator(
            profile=self.profile,
            scene=Scene(self.profile, [Grating(0, 0, 1544.80), Grating(0, 1, 1551.51)]),
            reply_to=("127.0.0.1", 1),
            frame_rate_hz=float(rate_hz),
        )
        self.sim.start()
        host, port = self.sim.address
        self.endpoint = Endpoint(device_ip=host, device_port=port, **TEST_ENDPOINT_KWARGS)  # type: ignore[arg-type]
        self.session = Session(
            self.endpoint,
            self.profile,
            QUIET,
            on_telemetry=self.pipeline.on_telemetry,
        )
        self.state_after_stop = SessionState.DISCONNECTED
        """Состояние автомата на момент остановки потока, до сворачивания стенда."""

        open_port_and_announce(self.session, self._announce)

    def _announce(self, address: tuple[str, int]) -> None:
        self.sim.reply_to = address

    def close(self) -> None:
        """Сначала замолкает прибор, потом сворачиваются сессия и тракт."""
        self.sim.stop()
        self.session.disconnect()
        self.pipeline.stop()


def _run(
    seconds: float,
    rate_hz: int,
    history_frames: int,
    drain_period_s: float = DRAIN_PERIOD_S,
) -> tuple[Stand, Writer, float]:
    """Гоняет поток заданное время через полный тракт и возвращает стенд и писателя."""
    stand = Stand(rate_hz, history_frames)
    capacity = int(rate_hz * seconds * 1.2)
    writer = Writer(stand.pipeline, capacity, drain_period_s)
    try:
        assert stand.session.connect().ok, "подключение к симулятору не удалось"
        stand.pipeline.start()
        writer.start()
        started = time.perf_counter()
        assert stand.session.start_stream(rate_hz).ok
        time.sleep(seconds)
        elapsed = time.perf_counter() - started
        assert stand.session.stop_stream().ok
        stand.state_after_stop = stand.session.state
        assert wait_until(lambda: not stand.sim.streaming, timeout=2.0)
        # Долёт последних датаграмм и добор кольца: иначе потери мнимые.
        time.sleep(1.0)
    finally:
        writer.stop()
        stand.close()
    return stand, writer, elapsed


def _report(name: str, stand: Stand, writer: Writer, seconds: float) -> float:
    """Печатает метрики прогона и возвращает сквозную долю потерь."""
    sent = stand.sim.stats.frames_sent
    metrics = stand.pipeline.metrics()
    transport = stand.session.transport_stats
    loss = 1.0 - writer.frames / sent if sent else 1.0
    p50, p99, p999, worst = writer.percentiles()
    lag = stand.pipeline.history.lag_s[: stand.pipeline.history.used] * 1e6

    print(f"\n--- {name} ---")
    print(f"темп отправки симулятора: {stand.sim.pace.describe()}")
    print(
        f"отправлено {sent}, вынуто из сокета {transport.datagrams_received}, "
        f"разобрано {metrics.frames}, доехало до писателя {writer.frames}"
    )
    print(
        f"потери сквозные {loss * 100:.4f} % · отказов разбора {metrics.parse_errors} · "
        f"разрывов у писателя {writer.gaps} (кадров {writer.lost})"
    )
    print(
        f"очередь транспорта: пик {transport.queue_peak} из "
        f"{stand.endpoint.rx_queue_capacity}, вытеснено {transport.dropped_queue_full}"
    )
    print(
        f"кольцо: {metrics.history_used} из {metrics.history_frames} кадров, "
        f"{metrics.history_bytes / (1 << 20):.1f} МБ, вытеснено {metrics.evicted}; "
        f"пик отставания писателя {writer.peak_lag} кадров, "
        f"крупнейшая пачка {writer.max_batch}"
    )
    print(
        f"задержка доставки писателю, мкс: p50 {p50:.0f} · p99 {p99:.0f} · "
        f"p99.9 {p999:.0f} · макс {worst:.0f} (в неё входит опрос "
        f"{writer.drain_period_s * 1e3:.0f} мс)"
    )
    if lag.size:
        print(
            f"задержка приёмного потока, мкс: p50 {np.percentile(lag, 50):.1f} · "
            f"p99 {np.percentile(lag, 99):.1f} · p99.9 {np.percentile(lag, 99.9):.1f}"
        )
    print(
        f"темп факт {metrics.frame_rate_hz:.1f} Гц при ожидании "
        f"{metrics.expected_rate_hz:.0f} Гц, оценка потерь по темпу "
        f"{(metrics.loss_estimate or 0.0) * 100:.3f} %"
    )
    print(
        f"снимков UI {metrics.ui_updates} за {seconds:.0f} с "
        f"({metrics.ui_updates / seconds:.1f} Гц), тактов децимации {metrics.ui_gates}"
    )
    print(f"ошибки транспорта: {dict(transport.errors) or 'нет'}")
    return loss


def test_сквозной_приём_2000_кадров_в_секунду_60_секунд() -> None:
    """Полный тракт на паспортных 2000 Гц: потери менее 0.1 % за 60 секунд.

    Метрики печатаются всегда, а не только при падении: это результат теста,
    а не диагностика. Отдельно проверяется, что писатель получил кадры **без
    разрывов** — потеря у него допустима только с отметкой, а её появление
    на паспортном темпе означало бы, что кольцо мало.
    """
    stand, writer, elapsed = _run(LOAD_SECONDS, TARGET_RATE_HZ, history_frames=20_000)
    loss = _report("полный тракт, 2000 Гц", stand, writer, elapsed)
    metrics = stand.pipeline.metrics()

    assert stand.sim.pace.rate_hz == pytest.approx(TARGET_RATE_HZ, rel=0.02), (
        "отправитель не выдержал 2000 Гц — сравнивать потери не с чем"
    )
    assert metrics.parse_errors == 0, "кадры симулятора обязаны разбираться без отказов"
    assert writer.gaps == 0, f"писатель отстал: {writer.lost} кадров вытеснено из кольца"
    assert loss < MAX_LOSS_FRACTION, f"сквозные потери {loss * 100:.4f} % при допустимых 0.1 %"
    assert stand.state_after_stop is SessionState.IDLE, (
        "после Stop сессия обязана вернуться в Idle, а не остаться в Streaming"
    )


def test_ui_не_ускоряется_вслед_за_потоком() -> None:
    """При 2000 кадрах/с UI обновляется те же 20 раз в секунду, а не 2000.

    Проверяется на живом потоке, а не на подставленных метках времени:
    децимация обязана держать темп UI и тогда, когда кадры идут лавиной.
    """
    seconds = 10.0
    stand, writer, elapsed = _run(seconds, TARGET_RATE_HZ, history_frames=20_000)
    _report("темп UI под нагрузкой", stand, writer, elapsed)
    metrics = stand.pipeline.metrics()

    expected_updates = elapsed / stand.pipeline.config.ui_period_s
    assert metrics.frames > 15_000, "поток не разогнался, проверять нечего"
    assert metrics.ui_updates == pytest.approx(expected_updates, rel=0.25)
    assert metrics.ui_updates < metrics.frames / 50


def test_отставший_писатель_получает_разрыв_числом() -> None:
    """Писатель, который заведомо не успевает, теряет кадры **со счётом**.

    Стимул, а не наблюдение: кольцо урезано до 512 кадров, а пауза писателя
    поднята до полусекунды — при 2000 Гц это 1000 кадров на паузу, вдвое
    больше кольца. Проверяется не сам факт потери, а её учёт: каждый принятый
    кадр обязан быть либо отдан писателю, либо посчитан в разрыве. Молчаливой
    потери в тракте быть не должно нигде.

    Вытеснение из кольца при этом потерей **не является** и с разрывом
    не совпадает: кадр, который читатель успел забрать, уходит из истории
    штатно. Поэтому `evicted` заметно больше, чем `lost`.
    """
    seconds = 5.0
    stand, writer, elapsed = _run(seconds, TARGET_RATE_HZ, history_frames=512, drain_period_s=0.5)
    _report("отставший писатель, 2000 Гц", stand, writer, elapsed)
    metrics = stand.pipeline.metrics()

    assert metrics.frames > 5_000
    assert writer.gaps > 0, "стимул не сработал: писатель успел за потоком"
    assert writer.frames + writer.lost == metrics.frames, (
        "каждый принятый кадр обязан быть либо отдан писателю, либо посчитан потерянным"
    )
    assert metrics.evicted >= writer.lost
    assert metrics.parse_errors == 0


def test_запас_по_темпу_сквозного_тракта() -> None:
    """Показывает, до какого темпа держит весь тракт. Утверждается паспортный.

    Выше 2000 Гц прибор не умеет; остальные точки печатаются, чтобы запас
    был виден числом, а не словом.
    """
    for rate in (1000, 2000, 4000):
        stand, writer, elapsed = _run(5.0, rate, history_frames=20_000)
        loss = _report(f"{rate} Гц", stand, writer, elapsed)
        sent = stand.sim.stats.frames_sent
        assert sent > 0
        if rate <= TARGET_RATE_HZ:
            assert loss < MAX_LOSS_FRACTION, f"на {rate} Гц потери {loss * 100:.4f} %"
            assert writer.gaps == 0


def test_pipeline_потоки_не_остаются() -> None:
    """После прогона не остаётся ни потока публикатора, ни потока писателя."""
    stand, writer, _ = _run(2.0, TARGET_RATE_HZ, history_frames=20_000)
    assert not stand.pipeline.is_running
    names = {thread.name for thread in threading.enumerate()}
    assert "fbg-pipeline" not in names
    assert "test-writer" not in names
    assert writer.frames > 0
    assert codec.classify(codec.build_stop()) == (codec.ID_MODE, codec.FC_STOP)
