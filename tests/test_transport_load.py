"""Нагрузочный тест приёмного тракта: 2000 датаграмм/с по 494 байта, 60 секунд.

Чат №2 снял риск R2 со стороны отправки: симулятор выдерживает 2000 кадров/с.
Здесь закрывается вторая половина — приём. Критерий из KB_05: потери менее
0.1 % за 60 секунд, и считаются они **на выходе `tap`**, то есть по числу
датаграмм, реально дошедших до потребителя, а не по числу, которые приёмный
поток успел вынуть из сокета. Так виден весь тракт целиком.

Потери на уровне ядра здесь не спрашиваются у сокета: `SO_RXQ_OVFL` есть
в Linux, в Windows переносимого способа нет, а прибор целевой под Windows.
Поэтому потери меряются сравнением со счётчиком отправителя — симулятора,
который знает, сколько датаграмм он отдал в сеть.

Маркер `slow`: прогон длится больше минуты, отдельная job в CI.
"""

import time

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.frames import MeasurementFrame
from fbg.core.profile import DeviceProfile
from tests.test_transport import Rig, wait_until

pytestmark = pytest.mark.slow

#: Паспортный темп прибора: 2000 Гц, код 0x00CA, ✅ прочитан командой 10 04.
TARGET_RATE_HZ = 2000

#: Длительность основного прогона (KB_05: критерий измеряется за 60 секунд).
LOAD_SECONDS = 60.0

#: Допустимая доля потерь, KB_05.
MAX_LOSS_FRACTION = 0.001


class LoadSink:
    """Потребитель `tap` для замера: считает датаграммы и пишет метки времени.

    Аллокаций на датаграмму нет: массив меток предвыделен, счётчики целые.
    Разбора нет намеренно — он проверяется отдельным тестом ниже, чтобы
    стоимость приёма и стоимость разбора были видны по отдельности.
    """

    def __init__(self, capacity: int, frame_size: int) -> None:
        self.stamps = np.empty(capacity, dtype=np.float64)
        self.frame_size = frame_size
        self.count = 0
        self.frames = 0
        self.other = 0

    def __call__(self, data: bytes, t_mono: float) -> None:
        if len(data) == self.frame_size:
            self.frames += 1
        else:
            self.other += 1
        count = self.count
        if count < self.stamps.size:
            self.stamps[count] = t_mono
            self.count = count + 1

    def intervals_us(self) -> np.ndarray:
        """Межкадровые интервалы, мкс."""
        return np.diff(self.stamps[: self.count]) * 1e6


class ParsingSink(LoadSink):
    """То же, плюс полный разбор кадра телеметрии в переиспользуемый буфер.

    Показывает стоимость настоящего тракта: транспорт плюс `codec`. Буфер
    принадлежит вызывающему (решение Р13 в KB_03), поэтому разбор не аллоцирует
    массивы на кадр.
    """

    def __init__(self, capacity: int, profile: DeviceProfile) -> None:
        super().__init__(capacity, profile.frame_size)
        self.profile = profile
        self.buffer = MeasurementFrame(profile.channels, profile.fbg_per_channel)
        self.parsed = 0
        self.parse_errors = 0

    def __call__(self, data: bytes, t_mono: float) -> None:
        super().__call__(data, t_mono)
        if len(data) != self.frame_size:
            return
        if codec.parse_measurement(data, self.profile, t_mono, out=self.buffer).ok:
            self.parsed += 1
        else:
            self.parse_errors += 1


def _report(name: str, sink: LoadSink, rig: Rig, seconds: float) -> float:
    """Печатает метрики прогона и возвращает долю потерь."""
    stats = rig.transport.stats()
    sent = rig.sim.stats.frames_sent
    loss = 1.0 - sink.frames / sent if sent else 1.0
    intervals = sink.intervals_us()

    print(f"\n--- {name} ---")
    print(f"приёмный буфер сокета: {rig.transport.rcvbuf_actual / (1 << 20):.2f} МБ")
    print(f"темп отправки: {rig.sim.pace.describe()}")
    print(
        f"отправлено {sent}, вынуто из сокета {stats.datagrams_received}, "
        f"дошло до потребителя {sink.frames} (прочих датаграмм {sink.other})"
    )
    print(
        f"потери сквозные {loss * 100:.4f} % · "
        f"вытеснено очередью {stats.dropped_queue_full} · "
        f"пик очереди {stats.queue_peak} из {rig.endpoint.rx_queue_capacity}"
    )
    if intervals.size:
        print(
            f"межкадровый интервал, мкс: p50 {np.percentile(intervals, 50):.1f} · "
            f"p99 {np.percentile(intervals, 99):.1f} · "
            f"p99.9 {np.percentile(intervals, 99.9):.1f} · "
            f"макс {intervals.max():.0f}"
        )
    print(f"фактический темп приёма: {sink.frames / seconds:.1f} Гц")
    print(f"ошибки транспорта: {dict(stats.errors) or 'нет'}")
    return loss


def _run(sink: LoadSink, seconds: float, rate_hz: int) -> Rig:
    """Гоняет поток заданное время через транспорт и возвращает стенд."""
    rig = Rig(tap=sink, rate_hz=float(rate_hz))  # type: ignore[arg-type]
    try:
        rig.transport.send(codec.build_start_stream(rate_hz))
        time.sleep(seconds)
        rig.transport.send(codec.build_stop())
        assert wait_until(lambda: not rig.sim.streaming, timeout=2.0)
        # Долёт последних датаграмм и разбор очереди: иначе потери мнимые.
        time.sleep(1.0)
    finally:
        rig.transport.close()
        rig.sim.stop()
    return rig


def test_приём_2000_датаграмм_в_секунду_60_секунд() -> None:
    """Основной прогон: потери менее 0.1 % за 60 секунд (KB_05).

    Метрики печатаются всегда, а не только при падении: фактический темп,
    перцентили межкадрового интервала и пик очереди — это результат теста.
    От него зависит, остаётся приём на Python или переносится на C, поэтому
    подгонять здесь нечего.
    """
    profile = DeviceProfile()
    capacity = int(TARGET_RATE_HZ * LOAD_SECONDS * 1.2)
    sink = LoadSink(capacity, profile.frame_size)
    rig = _run(sink, LOAD_SECONDS, TARGET_RATE_HZ)

    loss = _report("приём без разбора", sink, rig, LOAD_SECONDS)
    stats = rig.transport.stats()

    assert rig.sim.pace.rate_hz == pytest.approx(TARGET_RATE_HZ, rel=0.02), (
        "отправитель не выдержал 2000 Гц — сравнивать потери не с чем"
    )
    assert stats.errors == {}, f"ошибки транспорта: {dict(stats.errors)}"
    assert loss < MAX_LOSS_FRACTION, f"сквозные потери {loss * 100:.4f} % при допустимых 0.1 %"


def test_приём_с_полным_разбором_кадра() -> None:
    """Тот же темп, но потребитель ещё и разбирает каждый кадр кодеком.

    Это уже настоящий тракт приёма: транспорт довозит байты, `codec` их
    разбирает в переиспользуемый буфер. Прогон короче основного — задача
    показать, что разбор (21 мкс на кадр, замер чата №1) укладывается
    в бюджет 500 мкс и не съедает запас по потерям.
    """
    profile = DeviceProfile()
    seconds = 20.0
    sink = ParsingSink(int(TARGET_RATE_HZ * seconds * 1.2), profile)
    rig = _run(sink, seconds, TARGET_RATE_HZ)

    loss = _report("приём с разбором", sink, rig, seconds)
    print(f"разобрано {sink.parsed}, отказов разбора {sink.parse_errors}")

    assert sink.parse_errors == 0, "кадры симулятора обязаны разбираться кодеком без отказов"
    assert sink.parsed == sink.frames
    assert loss < MAX_LOSS_FRACTION, f"сквозные потери {loss * 100:.4f} % при допустимых 0.1 %"


def test_запас_по_темпу_приёма() -> None:
    """Показывает, до какого темпа тракт держит поток без потерь.

    Утверждается только паспортный темп: выше 2000 Гц прибор не умеет.
    Остальные точки печатаются, чтобы запас был виден числом.
    """
    profile = DeviceProfile()
    for rate in (1000, 2000, 4000):
        sink = LoadSink(int(rate * 5.0 * 1.2), profile.frame_size)
        rig = _run(sink, 5.0, rate)
        loss = _report(f"{rate} Гц", sink, rig, 5.0)
        if rate <= TARGET_RATE_HZ:
            assert loss < MAX_LOSS_FRACTION, f"на {rate} Гц потери {loss * 100:.4f} %"
