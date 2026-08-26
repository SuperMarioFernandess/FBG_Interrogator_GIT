"""Нагрузочные тесты: симулятор гонит 2000 кадров/с по 494 байта на loopback.

Критерий из KB_05: потери менее 0.1 % за 60 секунд. Приёмник здесь намеренно
примитивный — голый `recvfrom` в потоке со счётчиком, без разбора и без очереди.
Это нижняя граница: настоящий тракт приёма (`transport` + `pipeline`) появится
позже и обязан уложиться в тот же бюджет с разбором кадров в придачу.

Маркер `slow`: тесты идут больше минуты и запускаются отдельной job в CI.
"""

import socket
import threading
import time

import pytest

from fbg.core import codec
from fbg.core.profile import DeviceProfile
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene

pytestmark = pytest.mark.slow

#: Целевой темп прибора: 2000 Гц, код 0x00CA, ✅ прочитан командой 10 04.
TARGET_RATE_HZ = 2000

#: Длительность основного прогона. KB_05: критерий потерь измеряется за 60 секунд.
LOAD_SECONDS = 60.0

#: Допустимая доля потерь, KB_05.
MAX_LOSS_FRACTION = 0.001

#: Приёмный буфер сокета. При 2000 кадрах/с поток равен 0.99 МБ/с (KB_01),
#: и буфера по умолчанию (208 КБ в Linux) хватило бы лишь на 0.2 секунды затыка.
RECEIVER_BUFFER_BYTES = 16 << 20


class CountingReceiver:
    """Минимальный приёмник: считает датаграммы, ничего не разбирая.

    Кадры телеметрии отличаются от ответов на команды по длине: разбор здесь
    был бы лишней работой, а задача — измерить, сколько датаграмм вообще
    доходит до пользовательского процесса.
    """

    def __init__(self, frame_size: int) -> None:
        self.frame_size = frame_size
        self.frames = 0
        self.other = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECEIVER_BUFFER_BYTES)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.settimeout(0.2)
        # Размер фиксируется сразу: после `stop` сокет закрыт и опросить его нельзя.
        self.buffer_bytes = int(self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="load-rx", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.socket.getsockname()[:2]
        return str(host), int(port)

    def _loop(self) -> None:
        recv = self.socket.recv
        while not self._stop.is_set():
            try:
                datagram = recv(2048)
            except TimeoutError:
                continue
            except OSError:
                return
            if len(datagram) == self.frame_size:
                self.frames += 1
            else:
                self.other += 1

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.socket.close()


def _run_load(profile: DeviceProfile, seconds: float, rate_hz: int) -> tuple[DeviceSimulator, int]:
    """Гоняет поток заданное время и возвращает симулятор и число принятых кадров."""
    scene = Scene(
        profile,
        [
            Grating(channel=0, position=0, wavelength_nm=1545.0),
            Grating(channel=0, position=1, wavelength_nm=1550.0),
            Grating(channel=1, position=0, wavelength_nm=1560.0),
        ],
    )
    receiver = CountingReceiver(profile.frame_size)
    receiver.start()

    simulator = DeviceSimulator(
        profile=profile,
        scene=scene,
        reply_to=receiver.address,
        deviation_capacity=int(rate_hz * seconds) + 1000,
    )
    simulator.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(codec.build_start_stream(rate_hz), simulator.address)
        time.sleep(seconds)
        sender.sendto(codec.build_stop(), simulator.address)
        # Пауза на долёт последних датаграмм: иначе потери окажутся мнимыми.
        time.sleep(0.5)
    finally:
        sender.close()
        simulator.stop()
        receiver.stop()

    print(f"\nприёмный буфер сокета: {receiver.buffer_bytes / (1 << 20):.1f} МБ")
    print(f"датаграмм не 494 байта: {receiver.other}")
    return simulator, receiver.frames


def test_поток_2000_кадров_в_секунду_60_секунд() -> None:
    """Основной нагрузочный прогон: 2000 Гц × 60 с, потери менее 0.1 % (KB_05).

    Отчёт печатается всегда, а не только при падении: фактический темп
    и джиттер — это результат теста, а не диагностика. Распределение
    отклонений имеет тяжёлый хвост от пауз планировщика, поэтому вместе
    с σ выводятся перцентили: 99 % кадров могут уходить с точностью
    в единицы микросекунд при отдельных выбросах в десятки миллисекунд,
    и одна σ этого не показывает.
    """
    profile = DeviceProfile()
    simulator, received = _run_load(profile, LOAD_SECONDS, TARGET_RATE_HZ)

    report = simulator.pace
    sent = simulator.stats.frames_sent
    loss = 1.0 - received / sent if sent else 1.0

    print(f"темп: {report.describe()}")
    print(f"отправлено {sent}, принято {received}, потери {loss * 100:.4f} %")
    print(f"пропускная способность: {sent * profile.frame_size / LOAD_SECONDS / 1e6:.2f} МБ/с")

    assert sent > 0
    assert report.rate_hz == pytest.approx(TARGET_RATE_HZ, rel=0.02), (
        "темп 2000 Гц не достигнут — сравнивать потери не с чем"
    )
    assert loss < MAX_LOSS_FRACTION, f"потери {loss * 100:.4f} % при допустимых 0.1 %"


def test_потолок_темпа_на_этой_машине() -> None:
    """Показывает, до какого темпа отправитель ещё выдерживает расписание.

    Утверждается только достижение паспортных 2000 Гц: выше прибор не умеет.
    Остальные точки печатаются, чтобы запас был виден числом, а не на словах.
    """
    profile = DeviceProfile()
    achieved: list[tuple[int, float, float]] = []

    for rate in (500, 1000, 2000, 4000):
        simulator, received = _run_load(profile, 3.0, rate)
        report = simulator.pace
        achieved.append((rate, report.rate_hz, report.jitter_us))
        print(
            f"цель {rate:5d} Гц → факт {report.rate_hz:8.1f} Гц "
            f"(σ {report.jitter_us:6.1f} мкс, p99 {report.percentile_us(99):6.1f} мкс), "
            f"принято {received}/{simulator.stats.frames_sent}"
        )

    for rate, actual, _ in achieved:
        if rate <= TARGET_RATE_HZ:
            assert actual == pytest.approx(rate, rel=0.05), f"паспортный темп {rate} Гц не выдержан"


def test_потеря_кадров_видна_приёмнику() -> None:
    """Внесённая потеря 10 % кадров действительно доходит до приёмника как потеря.

    Нужен, чтобы основной тест не оказался ложноположительным: если бы
    приёмник считал что-то не то, эта проверка бы это вскрыла.
    """
    profile = DeviceProfile()
    scene = Scene(profile, [Grating(0, 0, 1550.0)])
    receiver = CountingReceiver(profile.frame_size)
    receiver.start()

    simulator = DeviceSimulator(profile=profile, scene=scene, reply_to=receiver.address)
    simulator.faults.frame_drop_probability = 0.1
    simulator.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(codec.build_start_stream(TARGET_RATE_HZ), simulator.address)
        time.sleep(5.0)
        sender.sendto(codec.build_stop(), simulator.address)
        time.sleep(0.5)
    finally:
        sender.close()
        simulator.stop()
        receiver.stop()

    stats = simulator.stats
    scheduled = stats.frames_sent + stats.frames_dropped
    dropped_share = stats.frames_dropped / scheduled
    print(f"\nзапланировано {scheduled}, отброшено {stats.frames_dropped} ({dropped_share:.3f})")
    print(f"принято {receiver.frames} из отправленных {stats.frames_sent}")

    assert dropped_share == pytest.approx(0.1, abs=0.02)
    assert receiver.frames == pytest.approx(stats.frames_sent, rel=0.001)
