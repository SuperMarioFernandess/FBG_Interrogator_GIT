"""Тесты транспорта: сокет, приёмный поток, контракт tap, счётчики.

Проверяется против симулятора `fbg.sim.device_sim`, а не против железа.
Транспорт байты не разбирает, поэтому кодек здесь используется только тестом —
чтобы убедиться, что довезённые байты остались теми же, какими их собрал
симулятор.

⚠️ Тесты фиксируют поведение **транспорта и симулятора**, а не факты о приборе.
Раскладка кадра телеметрии (N4), единицы частоты (D1) и число датаграмм
в длинных ответах (D5) остаются открытыми вопросами KB_04: захватов нет.
"""

import dataclasses
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from fbg.core import codec
from fbg.core.endpoint import Endpoint
from fbg.core.profile import DeviceProfile
from fbg.core.transport import UdpTransport, disable_udp_conn_reset
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene

#: Щедрые таймауты: тесты не должны мигать на загруженной машине.
WAIT_TIMEOUT_S = 5.0

#: Темп телеметрии в функциональных тестах. Нагрузочные 2 кГц — в test_transport_load.py.
TEST_FRAME_RATE_HZ = 200.0

#: Период опроса сокета в тестах: делает close быстрым и проверяемым.
TEST_POLL_S = 0.02


def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_TIMEOUT_S) -> bool:
    """Ждёт выполнения условия. False — не дождались."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class Collector:
    """Потребитель `tap`: копит датаграммы, при желании медленно.

    `delay_s` имитирует потребителя, который не успевает за приёмом: именно
    так проверяется, что переполнение приводит к контролируемой потере,
    а не к остановке приёма.
    """

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.raise_on_call = False
        self._items: list[tuple[bytes, float]] = []
        self._lock = threading.Lock()

    def __call__(self, data: bytes, t_mono: float) -> None:
        if self.raise_on_call:
            raise RuntimeError("потребитель упал")
        if self.delay_s:
            time.sleep(self.delay_s)
        with self._lock:
            self._items.append((data, t_mono))

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def frames(self) -> list[bytes]:
        with self._lock:
            return [data for data, _ in self._items]

    def stamps(self) -> list[float]:
        with self._lock:
            return [t for _, t in self._items]


class Rig:
    """Симулятор плюс транспорт, связанные так же, как прибор и приложение.

    Порядок соединения повторяет реальный: симулятор поднимается первым и
    сообщает свой адрес, транспорт биндится на эфемерный порт 127.0.0.1
    и сообщает свой, после чего симулятору проставляется `reply_to` —
    прибор отвечает на прописанный адрес назначения, а не на source-порт (KB_01).
    """

    def __init__(
        self,
        *,
        tap: Collector | None = None,
        profile: DeviceProfile | None = None,
        scene: Scene | None = None,
        rate_hz: float = TEST_FRAME_RATE_HZ,
        queue_capacity: int = 4096,
        strict_source_port: bool = False,
    ) -> None:
        self.profile = profile or DeviceProfile()
        self.collector = tap or Collector()
        self.sim = DeviceSimulator(
            profile=self.profile,
            scene=scene or Scene(self.profile, [Grating(0, 0, 1545.0), Grating(1, 0, 1560.0)]),
            reply_to=("127.0.0.1", 1),
            frame_rate_hz=rate_hz,
        )
        self.sim.start()
        host, port = self.sim.address
        self.endpoint = Endpoint(
            device_ip=host,
            device_port=port,
            local_ip="127.0.0.1",
            local_port=0,
            rx_poll_timeout_s=TEST_POLL_S,
            rx_queue_capacity=queue_capacity,
            strict_source_port=strict_source_port,
        )
        self.transport = UdpTransport(self.endpoint, self.collector)
        self.transport.open()
        self.sim.reply_to = self.transport.local_address

    def close(self) -> None:
        """Сначала замолкает прибор, потом закрывается приём.

        Обратный порядок оставил бы симулятор отправляющим на закрытый порт,
        а Windows превращает ответный ICMP в ошибку на его собственном сокете.
        """
        self.sim.stop()
        self.transport.close()


@pytest.fixture
def rig() -> Iterator[Rig]:
    """Стенд по умолчанию: быстрый потребитель, телеметрия 200 Гц."""
    stand = Rig()
    try:
        yield stand
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------------------


def test_endpoint_значения_по_умолчанию_заводские() -> None:
    """Умолчания совпадают с сетевыми настройками прибора из KB_01."""
    endpoint = Endpoint()
    assert endpoint.device_ip == "192.168.0.19"
    assert endpoint.device_port == 4567
    assert endpoint.local_ip == "0.0.0.0"
    assert endpoint.local_port == 8001
    assert endpoint.rcvbuf_bytes == 8 * 1024 * 1024
    assert endpoint.read_timeout_s == 0.5
    assert endpoint.write_timeout_s == 1.0
    assert endpoint.retries == 3


def test_endpoint_неизменяем() -> None:
    """Настройки — данные: правка на месте запрещена."""
    endpoint = Endpoint()
    with pytest.raises(dataclasses.FrozenInstanceError):
        endpoint.device_ip = "10.0.0.1"  # type: ignore[misc]


def test_endpoint_адреса_собираются_парами() -> None:
    """`device_address` и `bind_address` отдают готовые кортежи для сокета."""
    endpoint = Endpoint(device_ip="10.0.0.5", device_port=1234, local_ip="10.0.0.9", local_port=99)
    assert endpoint.device_address == ("10.0.0.5", 1234)
    assert endpoint.bind_address == ("10.0.0.9", 99)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_ip": ""},
        {"local_ip": ""},
        {"device_port": 0},
        {"device_port": 70000},
        {"local_port": -1},
        {"rcvbuf_bytes": 0},
        {"read_timeout_s": 0.0},
        {"write_timeout_s": -1.0},
        {"rx_poll_timeout_s": 0.0},
        {"retries": -1},
        {"rx_queue_capacity": 0},
    ],
)
def test_endpoint_отвергает_несогласованные_настройки(kwargs: dict[str, object]) -> None:
    """Некорректный Endpoint собирает программа, а не сеть: это ValueError (KB_05)."""
    with pytest.raises(ValueError):
        Endpoint(**kwargs)  # type: ignore[arg-type]


def test_endpoint_допускает_эфемерный_локальный_порт() -> None:
    """Порт 0 нужен тестам, чтобы не занимать 8001."""
    assert Endpoint(local_port=0).local_port == 0


# --------------------------------------------------------------------------------------
# Жизненный цикл
# --------------------------------------------------------------------------------------


def test_open_close_идемпотентны() -> None:
    """Повторные open и close не создают потоков, не бросают и не меняют адрес."""
    transport = UdpTransport(
        Endpoint(local_ip="127.0.0.1", local_port=0, rx_poll_timeout_s=TEST_POLL_S), Collector()
    )
    try:
        transport.open()
        address = transport.local_address
        transport.open()
        assert transport.local_address == address
        assert transport.is_open
    finally:
        transport.close()
        transport.close()
    assert not transport.is_open


def test_close_освобождает_порт() -> None:
    """После close тот же порт можно занять снова."""
    first = UdpTransport(
        Endpoint(local_ip="127.0.0.1", local_port=0, rx_poll_timeout_s=TEST_POLL_S), Collector()
    )
    first.open()
    port = first.local_address[1]
    first.close()

    second = UdpTransport(
        Endpoint(local_ip="127.0.0.1", local_port=port, rx_poll_timeout_s=TEST_POLL_S), Collector()
    )
    try:
        second.open()
        assert second.local_address[1] == port
    finally:
        second.close()


def test_close_не_оставляет_висящих_потоков() -> None:
    """Оба потока транспорта присоединены к моменту возврата из close."""
    transport = UdpTransport(
        Endpoint(local_ip="127.0.0.1", local_port=0, rx_poll_timeout_s=TEST_POLL_S), Collector()
    )
    transport.open()
    assert {"fbg-rx", "fbg-tap"} <= {thread.name for thread in threading.enumerate()}
    transport.close()
    assert not {"fbg-rx", "fbg-tap"} & {thread.name for thread in threading.enumerate()}


def test_таймаут_чтения_не_блокирует_close() -> None:
    """Тишина в сети не должна удлинять остановку.

    Приёмный поток выходит по таймауту `recvfrom`, поэтому close укладывается
    примерно в `rx_poll_timeout_s`, а не ждёт данных.
    """
    endpoint = Endpoint(local_ip="127.0.0.1", local_port=0, rx_poll_timeout_s=0.05)
    transport = UdpTransport(endpoint, Collector())
    transport.open()
    time.sleep(0.1)
    started = time.perf_counter()
    transport.close()
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"close занял {elapsed:.3f} с при периоде опроса 0.05 с"


def test_send_на_закрытом_транспорте_это_баг_вызывающего() -> None:
    """Отправка без open — программная ошибка, а не штатная ситуация (KB_05)."""
    transport = UdpTransport(Endpoint(local_ip="127.0.0.1", local_port=0), Collector())
    with pytest.raises(RuntimeError, match="закрыт"):
        transport.send(codec.build_stop())


def test_контекстный_менеджер_закрывает_сокет() -> None:
    """`with` открывает и гарантированно закрывает транспорт."""
    endpoint = Endpoint(local_ip="127.0.0.1", local_port=0, rx_poll_timeout_s=TEST_POLL_S)
    with UdpTransport(endpoint, Collector()) as transport:
        assert transport.is_open
    assert not transport.is_open


def test_адрес_приёма_недоступен_до_open() -> None:
    """Пока сокет не открыт, порта нет — спрашивать его бессмысленно."""
    transport = UdpTransport(Endpoint(local_ip="127.0.0.1", local_port=0), Collector())
    with pytest.raises(RuntimeError):
        _ = transport.local_address


def test_приёмный_буфер_запрашивается_у_ядра() -> None:
    """Фактический SO_RCVBUF читается и доступен: ядро вправе выдать не то, что просили."""
    endpoint = Endpoint(
        local_ip="127.0.0.1", local_port=0, rcvbuf_bytes=1 << 20, rx_poll_timeout_s=TEST_POLL_S
    )
    with UdpTransport(endpoint, Collector()) as transport:
        assert transport.rcvbuf_actual > 0


# --------------------------------------------------------------------------------------
# Платформенная ветка
# --------------------------------------------------------------------------------------


def test_отключение_wsaeconnreset_безопасно_на_любой_платформе() -> None:
    """Вызов не падает нигде и всегда возвращает bool.

    Ни `ValueError`, ни `OSError`, ни `AttributeError` не должны выходить
    наружу: `open` вызывает эту функцию до `bind`, и любое исключение здесь
    уронило бы транспорт целиком. Именно так CI и упал в первый раз —
    `socket.ioctl` бросает `ValueError` на любом коде, кроме трёх известных
    ему (CPython, `sock_ioctl`, ветка `default`), а ловился только `OSError`.

    Успешность применения на Windows здесь **не утверждается**: проверить это
    можно лишь на Windows-машине, а закреплять непроверенное утверждение
    тестом запрещено (KB_05 №12). Открытый вопрос N16 в KB_04.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        applied = disable_udp_conn_reset(sock)
    finally:
        sock.close()
    assert isinstance(applied, bool)
    if sys.platform != "win32":
        assert applied is False


def test_открытие_транспорта_проходит_платформенную_ветку(rig: Rig) -> None:
    """`open` проходит платформенную ветку и открывает сокет на любой ОС."""
    assert rig.transport.is_open
    if sys.platform != "win32":
        assert rig.transport.conn_reset_disabled is False


# --------------------------------------------------------------------------------------
# Обмен с симулятором
# --------------------------------------------------------------------------------------


def test_команда_уходит_и_ответ_приходит(rig: Rig) -> None:
    """Базовый обмен: один сокет и на отправку, и на приём (KB_01)."""
    assert rig.transport.send(codec.build_read_version()) is True
    assert wait_until(lambda: rig.collector.count >= 1)

    frame = rig.collector.frames()[0]
    assert codec.parse_version(frame).unwrap() == 410

    stats = rig.transport.stats()
    assert stats.commands_sent == 1
    assert stats.bytes_sent == 4
    assert stats.datagrams_received == 1
    assert stats.bytes_received == len(frame)
    assert stats.last_rx_mono > 0.0
    assert stats.errors == {}


def test_ответы_приходят_на_тот_же_сокет_с_которого_ушла_команда(rig: Rig) -> None:
    """Симулятор шлёт на прописанный reply_to — совпадающий с адресом приёма транспорта."""
    assert rig.sim.reply_to == rig.transport.local_address
    rig.transport.send(codec.build_read_serial())
    assert wait_until(lambda: rig.collector.count >= 1)
    assert codec.parse_serial(rig.collector.frames()[0]).unwrap() == 94_401_220


def test_длинный_ответ_не_усекается(rig: Rig) -> None:
    """Ответ 30 07 длиннее кадра телеметрии и обязан дойти целиком.

    Приёмный буфер меньше датаграммы усёк бы её молча. 🟡 Симулятор шлёт
    ответ одной датаграммой; сколько их шлёт прибор — открытый вопрос D5.
    """
    rig.transport.send(codec.build_read_raw_adc(0, rig.profile))
    assert wait_until(lambda: rig.collector.count >= 1)

    frame = rig.collector.frames()[0]
    expected = 2 + rig.profile.mode_len_width + 4 + rig.profile.adc_points * 2
    assert len(frame) == expected
    block = codec.parse_raw_adc(frame, rig.profile).unwrap()
    assert block.adc.size == rig.profile.adc_points


def test_поток_телеметрии_доходит_целиком(rig: Rig) -> None:
    """Кадры телеметрии доезжают побайтово целыми и разбираются кодеком."""
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 20)
    rig.transport.send(codec.build_stop())

    frames = rig.collector.frames()
    telemetry = [frame for frame in frames if len(frame) == rig.profile.frame_size]
    assert len(telemetry) >= 20

    for frame in telemetry[:20]:
        assert codec.classify(frame) == (codec.ID_MODE, codec.FC_STREAM)
        assert int.from_bytes(frame[2:6], "big") == rig.profile.frame_size
        assert codec.parse_measurement(frame, rig.profile).ok


def test_метка_времени_монотонна_и_растёт(rig: Rig) -> None:
    """`t_mono` снимается в приёмном потоке сразу после recvfrom и не убывает."""
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 10)
    rig.transport.send(codec.build_stop())

    stamps = rig.collector.stamps()
    assert stamps == sorted(stamps)
    assert stamps[-1] > stamps[0]


def test_счётчики_приёма_совпадают_с_принятым(rig: Rig) -> None:
    """Число и объём принятых датаграмм считаются точно."""
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 15)
    rig.transport.send(codec.build_stop())
    assert wait_until(lambda: not rig.sim.streaming)
    time.sleep(0.2)
    # Датаграмма может быть уже принята, но ещё не отдана в tap: сравнивать
    # счётчик приёма с собранным потребителем можно только на пустой очереди.
    assert wait_until(lambda: rig.transport.queue_depth == 0)

    stats = rig.transport.stats()
    frames = rig.collector.frames()
    assert stats.datagrams_received == len(frames)
    assert stats.bytes_received == sum(len(frame) for frame in frames)
    assert stats.dropped_queue_full == 0


# --------------------------------------------------------------------------------------
# Посторонний источник
# --------------------------------------------------------------------------------------


def test_датаграмма_с_чужого_порта_не_проходит() -> None:
    """При строгой сверке порта чужая датаграмма считается и отбрасывается."""
    stand = Rig(strict_source_port=True)
    try:
        intruder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            intruder.sendto(b"\x30\x02\x00\x00\x01\xee", stand.transport.local_address)
            assert wait_until(lambda: stand.transport.stats().foreign_datagrams >= 1)
        finally:
            intruder.close()

        time.sleep(0.1)
        stats = stand.transport.stats()
        assert stats.foreign_datagrams == 1
        assert stats.datagrams_received == 0
        assert stand.collector.count == 0
    finally:
        stand.close()


@pytest.mark.skipif(sys.platform == "win32", reason="127.0.0.2 не биндится на Windows")
def test_датаграмма_с_чужого_ip_не_проходит(rig: Rig) -> None:
    """Фильтрация по IP работает всегда, независимо от строгости по порту."""
    intruder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        intruder.bind(("127.0.0.2", 0))
        intruder.sendto(b"\x10\x01\x00\x08\x00\x00\x01\x9a", rig.transport.local_address)
        assert wait_until(lambda: rig.transport.stats().foreign_datagrams >= 1)
    finally:
        intruder.close()

    time.sleep(0.1)
    assert rig.transport.stats().datagrams_received == 0
    assert rig.collector.count == 0


def test_строгая_сверка_порта_выключена_по_умолчанию() -> None:
    """Source-порт прибора захватом не подтверждён (KB_06 пуст) — фильтр по нему опасен."""
    assert Endpoint().strict_source_port is False


# --------------------------------------------------------------------------------------
# Режимы сбоев симулятора
# --------------------------------------------------------------------------------------


def test_молчание_прибора_не_ломает_транспорт(rig: Rig) -> None:
    """Сценарий G1: прибор замолчал. Транспорт жив, счётчики не растут, close работает."""
    rig.sim.go_silent(2.0)
    rig.transport.send(codec.build_read_version())
    time.sleep(0.3)

    stats = rig.transport.stats()
    assert stats.commands_sent == 1
    assert stats.datagrams_received == 0
    assert stats.errors == {}
    assert rig.transport.is_open


def test_мусор_довозится_как_есть(rig: Rig) -> None:
    """Сценарий G5: разбор не дело транспорта, мусор обязан дойти до потребителя."""
    rig.sim.faults.garbage = True
    rig.transport.send(codec.build_read_version())
    assert wait_until(lambda: rig.collector.count >= 1)

    frame = rig.collector.frames()[0]
    assert len(frame) == 16
    assert rig.transport.stats().datagrams_received == 1


def test_неверный_len_довозится_как_есть(rig: Rig) -> None:
    """Сценарий G6: LEN не совпадает с длиной — это увидит кодек, не транспорт."""
    rig.sim.faults.bad_len = True
    rig.transport.send(codec.build_read_version())
    assert wait_until(lambda: rig.collector.count >= 1)

    frame = rig.collector.frames()[0]
    assert len(frame) == 8
    assert int.from_bytes(frame[2:4], "big") != len(frame)
    assert not codec.parse_version(frame).ok


def test_потеря_кадров_на_стороне_прибора_видна_как_недосчёт(rig: Rig) -> None:
    """Риск R3: транспорт принимает ровно то, что ушло в сеть, и не выдумывает пропуски."""
    rig.sim.faults.frame_drop_probability = 0.5
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 20)
    rig.transport.send(codec.build_stop())
    assert wait_until(lambda: not rig.sim.streaming)
    time.sleep(0.3)

    # Ответ на Stop — тоже датаграмма, поэтому телеметрия отбирается по длине.
    telemetry = [frame for frame in rig.collector.frames() if len(frame) == rig.profile.frame_size]
    assert rig.sim.stats.frames_dropped > 0
    assert len(telemetry) == rig.sim.stats.frames_sent
    assert rig.transport.stats().dropped_queue_full == 0


def test_перезагрузка_прибора_посреди_потока_не_ломает_приём(rig: Rig) -> None:
    """Сценарий G2: поток обрывается, транспорт остаётся открытым и готовым."""
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 10)
    rig.sim.reboot()
    time.sleep(0.2)
    before = rig.transport.stats().datagrams_received

    rig.transport.send(codec.build_read_version())
    assert wait_until(lambda: rig.transport.stats().datagrams_received > before)
    assert rig.transport.stats().errors == {}


# --------------------------------------------------------------------------------------
# Контракт tap
# --------------------------------------------------------------------------------------


def test_медленный_потребитель_даёт_контролируемую_потерю_а_не_остановку_приёма() -> None:
    """Ключевое требование контракта tap.

    Потребитель заведомо не успевает: 20 мс на датаграмму при потоке 200 Гц.
    Ожидается ровно это — очередь упирается в ёмкость, лишнее вытесняется
    со счётчиком, но приём продолжается, и число принятых датаграмм совпадает
    с числом отправленных симулятором.
    """
    stand = Rig(tap=Collector(delay_s=0.02), queue_capacity=16)
    try:
        stand.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
        assert wait_until(lambda: stand.transport.stats().dropped_queue_full > 0)
        time.sleep(1.0)
        stand.transport.send(codec.build_stop())
        assert wait_until(lambda: not stand.sim.streaming)
        time.sleep(0.3)

        stats = stand.transport.stats()
        sent = stand.sim.stats.frames_sent
        assert stats.dropped_queue_full > 0, "медленный потребитель обязан приводить к потере"
        assert stats.queue_peak == 16, "очередь обязана упереться в ёмкость, а не расти дальше"
        assert stats.datagrams_received >= sent - 1, "приём не должен вставать из-за потребителя"
        assert stand.collector.count < stats.datagrams_received
        assert (
            stats.dropped_queue_full + stand.collector.count + stand.transport.queue_depth
            <= stats.datagrams_received
        )
    finally:
        stand.close()


def test_исключение_в_tap_не_роняет_приём(rig: Rig) -> None:
    """Падение потребителя учитывается в errors и не останавливает поток датаграмм."""
    rig.collector.raise_on_call = True
    rig.transport.send(codec.build_read_version())
    assert wait_until(lambda: rig.transport.stats().errors.get("RuntimeError", 0) >= 1)

    rig.collector.raise_on_call = False
    rig.transport.send(codec.build_read_serial())
    assert wait_until(lambda: rig.collector.count >= 1)
    assert rig.transport.stats().datagrams_received == 2


def test_данные_принадлежат_потребителю(rig: Rig) -> None:
    """`tap` получает копию: приёмный буфер переиспользуется, данные — нет."""
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 5)
    rig.transport.send(codec.build_stop())

    frames = rig.collector.frames()[:5]
    assert len({id(frame) for frame in frames}) == len(frames)
    assert all(isinstance(frame, bytes) for frame in frames)


def test_пиковая_заполненность_очереди_считается(rig: Rig) -> None:
    """Пик очереди — метрика запаса приёмного тракта, и она не может быть нулевой."""
    rig.transport.send(codec.build_start_stream(int(TEST_FRAME_RATE_HZ)))
    assert wait_until(lambda: rig.collector.count >= 10)
    rig.transport.send(codec.build_stop())
    assert rig.transport.stats().queue_peak >= 1


def test_счётчики_отправки_считают_байты(rig: Rig) -> None:
    """Отправленные команды и байты учитываются отдельно от принятых."""
    commands = [codec.build_stop(), codec.build_read_version(), codec.build_read_sweep()]
    for command in commands:
        assert rig.transport.send(command) is True

    stats = rig.transport.stats()
    assert stats.commands_sent == 3
    assert stats.bytes_sent == sum(len(command) for command in commands)
