"""UDP-транспорт: один сокет, приёмный поток, развязывающая очередь, счётчики.

Транспорт возит байты и ничего о них не знает: `codec` он не импортирует.
Его выход — пара «байты + метка `perf_counter`, снятая сразу после `recvfrom`».
Классификация по (ID, FC), корреляция ответов, `Stop` при подключении и в
`finally` — ответственность `session` (KB_03, таблица ответственности).

Один сокет, а не два
--------------------
Прибор отвечает на жёстко прописанный Destination IP:Port, а не на source-порт
запроса (KB_01). Поэтому команды уходят из того же сокета, который слушает
`local_port`: два отдельных сокета — типичная ошибка, при которой ответы
не приходят вовсе.

Два потока
----------
`rx` только принимает и кладёт в очередь; `dispatch` вызывает `tap`. Второй
поток здесь не украшение: при синхронном вызове `tap` прямо из `rx` медленный
потребитель останавливает приём, датаграммы копятся в буфере ядра и теряются
там молча — узнать о потере неоткуда, `SO_RXQ_OVFL` есть только в Linux.
С развязкой переполняется наша очередь, а не ядро, и каждая потерянная
датаграмма попадает в счётчик `dropped_queue_full`.

Что не измеряется
-----------------
Потери на уровне ядра. В Linux их отдаёт `SO_RXQ_OVFL`, в Windows переносимого
способа нет, а приложение целевое под Windows. Поэтому потери оцениваются
сравнением принятого со счётчиком отправителя (в тестах — со счётчиком
симулятора), а не запросом у сокета. Придумывать здесь нечего: см. KB_05 №12.
"""

import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import TracebackType

from fbg.core.endpoint import Endpoint

#: Код ioctl Windows, отключающий превращение ICMP «port unreachable»
#: в `ConnectionResetError` на следующем `recvfrom` (SIO_UDP_CONNRESET).
SIO_UDP_CONNRESET = 0x9800000C

#: Размер приёмного буфера под одну датаграмму. Кадр телеметрии — 494 байта,
#: но ответ `30 07` — 5112 байт, а `30 03` — около 21 КБ (KB_01), и приходят ли
#: они одной датаграммой, неизвестно (вопрос D5). Симулятор шлёт одной, и на
#: loopback это проходит. Буфер меньше датаграммы означал бы её молчаливое
#: усечение, поэтому берётся практический максимум полезной нагрузки UDP.
#: Буфер выделяется один раз при открытии, не на кадр.
RECV_BUFFER_BYTES = 65535

#: Тип обратного вызова: сырые байты датаграммы и момент `perf_counter`.
TapCallback = Callable[[bytes, float], None]


@dataclass(frozen=True)
class TransportStats:
    """Снимок счётчиков транспорта.

    Неизменяемая копия: читатель (UI, диагностика) не должен видеть, как
    значения меняются под ним посреди отрисовки.
    """

    commands_sent: int = 0
    bytes_sent: int = 0
    datagrams_received: int = 0
    bytes_received: int = 0
    dropped_queue_full: int = 0
    """Датаграммы, вытесненные из нашей очереди медленным потребителем."""

    foreign_datagrams: int = 0
    """Датаграммы с адреса, не совпавшего с `device_ip` (и портом, если включена строгая сверка)."""

    queue_peak: int = 0
    """Пиковая заполненность очереди за время жизни транспорта."""

    last_rx_mono: float = 0.0
    """Момент `perf_counter` последнего приёма. 0.0 — не принято ничего."""

    icmp_resets: int = 0
    """Сколько раз ядро отдало ConnectionResetError вместо датаграммы (Windows, ICMP)."""

    errors: Mapping[str, int] = field(default_factory=dict)
    """Ошибки транспорта по имени класса исключения."""


@dataclass(slots=True)
class _Counters:
    """Изменяемые счётчики.

    Блокировка не нужна: у каждого счётчика ровно один писатель — `send`
    пишет счётчики отправки, приёмный поток пишет счётчики приёма. Читатель
    может увидеть значение, отставшее на одну датаграмму, и это допустимо:
    счётчики диагностические, а взятие блокировки в горячем пути — нет.
    Исключение — словарь ошибок: у него писателей несколько, и он защищён
    отдельной блокировкой на редком пути.
    """

    commands_sent: int = 0
    bytes_sent: int = 0
    datagrams_received: int = 0
    bytes_received: int = 0
    dropped_queue_full: int = 0
    foreign_datagrams: int = 0
    queue_peak: int = 0
    last_rx_mono: float = 0.0
    icmp_resets: int = 0


def disable_udp_conn_reset(sock: socket.socket) -> bool:
    """Отключает WSAECONNRESET на UDP-сокете. Возвращает True, если применилось.

    Ловушка Windows: если удалённая сторона недоступна, стек превращает
    пришедший ICMP «port unreachable» в `ConnectionResetError` на следующем
    `recvfrom` — то есть приёмный поток умирает от ошибки, не относящейся
    к принимаемым данным. Лечится `SIO_UDP_CONNRESET` со значением False.

    На остальных платформах делать нечего: `socket.ioctl` там отсутствует,
    и функция возвращает False, ничего не трогая.
    """
    if sys.platform != "win32":
        return False
    try:  # pragma: no cover — ветка исполняется только на Windows
        sock.ioctl(SIO_UDP_CONNRESET, False)
    except OSError:
        return False
    return True


class UdpTransport:
    """UDP-обмен с прибором: отправка команд и приём датаграмм в фоне.

    Контракт `tap`
    --------------
    `tap(data, t_mono)` вызывается из отдельного потока `dispatch`, по одной
    датаграмме за раз, в порядке приёма. `data` принадлежит вызывающему
    целиком: это копия, приёмный буфер не переиспользуется, удерживать её
    можно сколько угодно.

    Можно: разобрать кадр, положить результат в свою очередь, обновить счётчики.

    Нельзя:
      * блокироваться надолго. Пока `tap` работает, очередь растёт; когда она
        упирается в `rx_queue_capacity`, самая старая датаграмма вытесняется
        и увеличивается `dropped_queue_full`. Приём при этом **не встаёт**:
        потеря контролируемая и посчитанная, но она есть;
      * писать в файлы и журналы — это работа `packet_log` и `recorder`,
        которым датаграмма передаётся дальше через их собственные очереди;
      * трогать Qt. Поток не является потоком UI (KB_03);
      * рассчитывать на то, что исключение будет замечено. Исключение из `tap`
        не роняет поток: оно попадает в `stats.errors` по имени класса, и всё.

    Порядок операций гарантирован, доставка — нет: UDP есть UDP, а очередь
    конечна.
    """

    def __init__(self, endpoint: Endpoint, tap: TapCallback) -> None:
        self._endpoint = endpoint
        self._tap = tap
        self._sock: socket.socket | None = None
        self._queue: deque[tuple[bytes, float]] = deque(maxlen=endpoint.rx_queue_capacity)
        self._counters = _Counters()
        self._errors: dict[str, int] = {}
        self._error_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._wakeup = threading.Event()
        self._rx_thread: threading.Thread | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._rcvbuf_actual = 0
        self._conn_reset_disabled = False

    # --- Жизненный цикл ----------------------------------------------------------------

    def open(self) -> None:
        """Открывает сокет и запускает потоки. Повторный вызов ничего не делает."""
        if self._sock is not None:
            return
        endpoint = self._endpoint
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Порядок важен: ioctl применяется к свежесозданному сокету, до bind.
            self._conn_reset_disabled = disable_udp_conn_reset(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, endpoint.rcvbuf_bytes)
            sock.settimeout(endpoint.rx_poll_timeout_s)
            sock.bind(endpoint.bind_address)
        except OSError:
            sock.close()
            raise
        self._rcvbuf_actual = int(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        self._sock = sock
        self._shutdown.clear()
        self._wakeup.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="fbg-rx", daemon=True)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="fbg-tap", daemon=True
        )
        self._rx_thread.start()
        self._dispatch_thread.start()

    def close(self) -> None:
        """Останавливает потоки и освобождает порт. Повторный вызов безвреден.

        Ждать приходится не дольше `rx_poll_timeout_s`: приёмный поток выходит
        по таймауту `recvfrom`, а не по закрытию сокета. Сокет закрывается
        **после** присоединения потоков — закрывать его под работающим
        `recvfrom` значит работать с освобождённым дескриптором.

        Датаграммы, оставшиеся в очереди, отбрасываются: доставлять их некуда,
        потребитель уже сворачивается.
        """
        self._shutdown.set()
        self._wakeup.set()
        for thread in (self._rx_thread, self._dispatch_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=self._join_timeout_s())
        self._rx_thread = self._dispatch_thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._queue.clear()

    def _join_timeout_s(self) -> float:
        """Запас к периоду опроса: поток обязан выйти за один цикл."""
        return self._endpoint.rx_poll_timeout_s + 5.0

    def __enter__(self) -> "UdpTransport":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- Состояние ---------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True, если сокет открыт."""
        return self._sock is not None

    @property
    def endpoint(self) -> Endpoint:
        """Настройки, с которыми создан транспорт."""
        return self._endpoint

    @property
    def local_address(self) -> tuple[str, int]:
        """Фактический адрес приёма. Осмыслен только после `open`.

        Отличается от `endpoint.bind_address`, если запрошен эфемерный порт:
        так биндятся тесты, чтобы не занимать 8001.
        """
        if self._sock is None:
            raise RuntimeError("транспорт закрыт: адрес приёма ещё не назначен")
        host, port = self._sock.getsockname()[:2]
        return str(host), int(port)

    @property
    def rcvbuf_actual(self) -> int:
        """Приёмный буфер, фактически выданный ядром.

        Может отличаться от запрошенного в обе стороны: Linux удваивает
        значение под служебные структуры и обрезает его по `net.core.rmem_max`.
        """
        return self._rcvbuf_actual

    @property
    def conn_reset_disabled(self) -> bool:
        """True, если `SIO_UDP_CONNRESET` применился (только Windows)."""
        return self._conn_reset_disabled

    @property
    def queue_depth(self) -> int:
        """Текущая заполненность очереди, датаграмм."""
        return len(self._queue)

    def stats(self) -> TransportStats:
        """Снимок счётчиков."""
        counters = self._counters
        with self._error_lock:
            errors = dict(self._errors)
        return TransportStats(
            commands_sent=counters.commands_sent,
            bytes_sent=counters.bytes_sent,
            datagrams_received=counters.datagrams_received,
            bytes_received=counters.bytes_received,
            dropped_queue_full=counters.dropped_queue_full,
            foreign_datagrams=counters.foreign_datagrams,
            queue_peak=counters.queue_peak,
            last_rx_mono=counters.last_rx_mono,
            icmp_resets=counters.icmp_resets,
            errors=errors,
        )

    # --- Отправка ----------------------------------------------------------------------

    def send(self, payload: bytes) -> bool:
        """Отправляет команду прибору. True — датаграмма отдана ядру.

        True не означает, что прибор её получил: подтверждений в UDP нет,
        корреляцией ответа занимается `session`. False означает отказ на
        локальной стороне (нет маршрута, сокет закрыли гонкой) — причина
        попадает в `stats.errors`.

        Отправка идёт из приёмного сокета: прибор отвечает на прописанный
        Destination IP:Port, и с отдельного сокета ответ не пришёл бы (KB_01).
        """
        sock = self._sock
        if sock is None:
            raise RuntimeError("транспорт закрыт: вызовите open() перед send()")
        try:
            sent = sock.sendto(payload, self._endpoint.device_address)
        except OSError as exc:
            self._record_error(exc)
            return False
        self._counters.commands_sent += 1
        self._counters.bytes_sent += sent
        return True

    # --- Приём -------------------------------------------------------------------------

    def _rx_loop(self) -> None:
        """Принимает датаграммы и кладёт их в очередь. Ничего не разбирает.

        В этом цикле запрещено логирование (KB_05) — только инкремент счётчиков.
        Буфер приёма переиспользуется: `recvfrom_into` пишет в него, наружу
        уходит копия ровно принятой длины. Полностью без аллокаций обойтись
        нельзя — копия датаграммы и кортеж адреса от `recvfrom_into` создаются
        на каждый кадр; при 2 кГц это измеримо и укладывается в бюджет,
        см. `tests/test_transport_load.py`.
        """
        sock = self._sock
        assert sock is not None
        endpoint = self._endpoint
        counters = self._counters
        queue = self._queue
        capacity = endpoint.rx_queue_capacity
        device_ip = endpoint.device_ip
        device_port = endpoint.device_port if endpoint.strict_source_port else None
        shutdown = self._shutdown.is_set
        wake = self._wakeup.set
        buffer = bytearray(RECV_BUFFER_BYTES)
        view = memoryview(buffer)
        perf_counter = time.perf_counter

        while not shutdown():
            try:
                nbytes, address = sock.recvfrom_into(buffer)
            except TimeoutError:
                continue
            except ConnectionResetError:
                # Windows: ICMP «port unreachable» от недоступного прибора.
                # Датаграммы это не касается, приём продолжается.
                counters.icmp_resets += 1
                continue
            except OSError as exc:
                if shutdown():
                    return
                # Прочие ошибки сокета повторились бы на каждой итерации
                # и превратили бы цикл в busy loop, поэтому поток выходит.
                self._record_error(exc)
                return

            t_mono = perf_counter()
            if address[0] != device_ip or (device_port is not None and address[1] != device_port):
                counters.foreign_datagrams += 1
                continue

            counters.datagrams_received += 1
            counters.bytes_received += nbytes
            counters.last_rx_mono = t_mono
            if len(queue) >= capacity:
                # deque(maxlen=…) вытеснит старейшую сам; здесь только учёт.
                counters.dropped_queue_full += 1
            queue.append((bytes(view[:nbytes]), t_mono))
            depth = len(queue)
            if depth > counters.queue_peak:
                counters.queue_peak = depth
            wake()

    def _dispatch_loop(self) -> None:
        """Отдаёт принятые датаграммы в `tap`. Медленный `tap` тормозит только этот поток."""
        queue = self._queue
        tap = self._tap
        wakeup = self._wakeup
        poll = self._endpoint.rx_poll_timeout_s
        shutdown = self._shutdown.is_set

        while not shutdown():
            try:
                data, t_mono = queue.popleft()
            except IndexError:
                wakeup.wait(poll)
                wakeup.clear()
                continue
            try:
                tap(data, t_mono)
            # Падение потребителя не должно валить приём: оно учитывается и всё.
            except Exception as exc:
                self._record_error(exc)

    def _record_error(self, exc: BaseException) -> None:
        """Учитывает ошибку по имени класса. Редкий путь, поэтому под блокировкой."""
        name = type(exc).__name__
        with self._error_lock:
            self._errors[name] = self._errors.get(name, 0) + 1
