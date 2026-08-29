"""Тесты проводки журнала пакетов к сессии (Р54).

Здесь проверяется единственная правка `fbg/core` этого чата: развилка в
`Session._on_datagram` и вызовы `log_tx` рядом с отправкой. Сам журнал
написан в чате №8 и не меняется.

Три свойства, ради которых проводка и делается именно так:

* сырые байты попадают в журнал **до** разбора и не искажаются (KB_05 №3);
* сессия без журнала работает ровно как раньше — журнал необязателен;
* отказ журнала обмен не прерывает (Р50), но и не молчит: он считается
  в `SessionStats.tap_errors`.

`fbg/core` при этом `fbg/io` не импортирует: проводка сделана двумя
колбэками, а совпадение их сигнатур с `PacketLog.log_rx` / `log_tx` —
удобство подключения, а не зависимость.
"""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from fbg.core import codec
from fbg.core.endpoint import Endpoint
from fbg.core.profile import DeviceProfile
from fbg.core.session import Session, SessionConfig
from fbg.io.packet_log import Direction, PacketLog, PacketLogConfig
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene

#: Те же короткие сроки, что в тестах сессии: прогон не должен ждать реальных пауз.
ENDPOINT_KWARGS: dict[str, object] = {
    "local_ip": "127.0.0.1",
    "local_port": 0,
    "read_timeout_s": 0.15,
    "write_timeout_s": 0.2,
    "retries": 1,
    "rx_poll_timeout_s": 0.02,
}

#: Keepalive практически выключен: служебный трафик мешал бы считать записи.
QUIET = SessionConfig(
    keepalive_period_s=30.0,
    stream_stall_floor_s=0.3,
    backoff_schedule=(0.05, 0.1),
    retry_pause_s=0.02,
    reassembly_timeout_s=0.3,
    watchdog_tick_s=0.02,
    settle_before_readback_s=0.01,
)


class Collector:
    """Простейший приёмник байтов вместо журнала: помнит всё, что ему дали."""

    def __init__(self) -> None:
        self.rx: list[tuple[bytes, float]] = []
        self.tx: list[tuple[bytes, float]] = []
        self._lock = threading.Lock()

    def log_rx(self, data: bytes, t_mono: float) -> None:
        """Сигнатура совпадает с `PacketLog.log_rx` и с `TapCallback` транспорта."""
        with self._lock:
            self.rx.append((data, t_mono))

    def log_tx(self, data: bytes, t_mono: float) -> None:
        """Сигнатура совпадает с `PacketLog.log_tx`."""
        with self._lock:
            self.tx.append((data, t_mono))

    def pairs(self, source: list[tuple[bytes, float]]) -> list[tuple[int, int]]:
        """Пары (ID, FC) записанного, по порядку."""
        with self._lock:
            return [(data[0], data[1]) for data, _ in source if len(data) >= 2]


class Rig:
    """Симулятор плюс сессия с подключённым журналом.

    Порядок как в реальности и как в `tests/test_session.py`: прибор
    поднимается первым, сессия открывает приёмный порт, и только потом
    симулятору сообщается адрес ответа — прибор отвечает на прописанный
    адрес назначения, а не на source-порт запроса (KB_01).
    """

    def __init__(
        self,
        *,
        log_rx: object = None,
        log_tx: object = None,
        rate_hz: float = 200.0,
    ) -> None:
        self.profile = DeviceProfile()
        self.sim = DeviceSimulator(
            profile=self.profile,
            scene=Scene(self.profile, [Grating(0, 0, 1545.0), Grating(1, 0, 1560.0)]),
            reply_to=("127.0.0.1", 1),
            frame_rate_hz=rate_hz,
        )
        self.sim.start()
        host, port = self.sim.address
        self.session = Session(
            Endpoint(device_ip=host, device_port=port, **ENDPOINT_KWARGS),  # type: ignore[arg-type]
            self.profile,
            QUIET,
            log_rx=log_rx,  # type: ignore[arg-type]
            log_tx=log_tx,  # type: ignore[arg-type]
        )
        self.session._transport.open()
        self.sim.reply_to = self.session.local_address

    def close(self) -> None:
        """Сначала замолкает прибор, потом закрывается сессия."""
        self.sim.stop()
        self.session.disconnect()


@pytest.fixture
def collector() -> Collector:
    """Приёмник байтов."""
    return Collector()


@pytest.fixture
def rig(collector: Collector) -> Iterator[Rig]:
    """Подключённый стенд с приёмником байтов вместо журнала."""
    stand = Rig(log_rx=collector.log_rx, log_tx=collector.log_tx)
    try:
        assert stand.session.connect().ok
        yield stand
    finally:
        stand.close()


def wait_until(predicate: object, timeout: float = 5.0) -> bool:
    """Ждёт условия. False — не дождались."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.005)
    return bool(predicate())  # type: ignore[operator]


# --------------------------------------------------------------------------------------
# Развилка получает и TX, и RX
# --------------------------------------------------------------------------------------


def test_журнал_получает_и_отправленное_и_принятое(rig: Rig, collector: Collector) -> None:
    """Подключение — это пять чтений плюс Stop, и все они видны с обеих сторон."""
    отправлено = collector.pairs(collector.tx)
    принято = collector.pairs(collector.rx)
    assert (codec.ID_MODE, codec.FC_STOP) in отправлено
    for fc in (
        codec.FC_VERSION,
        codec.FC_SERIAL,
        codec.FC_MODULE_PARAMS,
        codec.FC_SWEEP,
        codec.FC_CHANNEL_SETUP,
    ):
        assert (codec.ID_READ, fc) in отправлено, f"команда 10 {fc:02X} не попала в журнал"
        assert (codec.ID_READ, fc) in принято, f"ответ 10 {fc:02X} не попал в журнал"


def test_stop_при_подключении_попадает_в_журнал_первым(rig: Rig, collector: Collector) -> None:
    """Правило KB_05 №6 видно в журнале: Stop уходит раньше опроса конфигурации."""
    assert collector.pairs(collector.tx)[0] == (codec.ID_MODE, codec.FC_STOP)


def test_stop_при_отключении_тоже_попадает_в_журнал(collector: Collector) -> None:
    """Stop в `finally` — тоже трафик, и в записи провода он обязан быть."""
    stand = Rig(log_rx=collector.log_rx, log_tx=collector.log_tx)
    assert stand.session.connect().ok
    stand.sim.stop()
    stand.session.disconnect()
    assert collector.pairs(collector.tx)[-1] == (codec.ID_MODE, codec.FC_STOP)


def test_телеметрия_попадает_в_журнал_сырой(rig: Rig, collector: Collector) -> None:
    """Кадр `30 02` идёт и в журнал, и по существующему пути — развилка, не замена."""
    полученные: list[bytes] = []
    rig.session._on_telemetry = lambda data, _t: полученные.append(data)
    assert rig.session.start_stream().ok
    try:
        assert wait_until(lambda: len(полученные) >= 3)
    finally:
        rig.session.stop_stream()

    из_журнала = [data for data, _ in collector.rx if data[:2] == bytes([0x30, 0x02])]
    assert из_журнала, "кадр телеметрии не попал в журнал"
    assert полученные[0] in из_журнала


def test_сырые_байты_не_искажены(rig: Rig, collector: Collector) -> None:
    """Журнал получает ту же датаграмму, что и корреляция: ни копии, ни обрезки.

    Проверяется на ответе `10 04`, у которого известна и длина, и содержимое:
    он разбирается кодеком из **тех же** байтов, что легли в журнал.
    """
    ответ = rig.session.read_module_params()
    assert ответ.ok
    кадры = [data for data, _ in collector.rx if data[:2] == bytes([codec.ID_READ, 0x04])]
    assert кадры
    последний = кадры[-1]
    assert codec.parse_module_params(последний).unwrap() == ответ.unwrap()


def test_метки_времени_растут(rig: Rig, collector: Collector) -> None:
    """Обе метки — `perf_counter`, и по ним журнал сшивается с файлом измерений."""
    метки = [t for _, t in collector.rx] + [t for _, t in collector.tx]
    assert all(t > 0 for t in метки)
    assert sorted(t for _, t in collector.tx) == [t for _, t in collector.tx]


# --------------------------------------------------------------------------------------
# Журнал необязателен
# --------------------------------------------------------------------------------------


def test_сессия_без_журнала_работает_как_раньше() -> None:
    """Ни один колбэк не задан — поведение прежнее, счётчик отказов нулевой."""
    stand = Rig()
    try:
        assert stand.session.connect().ok
        assert stand.session.read_version().unwrap() > 0
        assert stand.session.stats().tap_errors == 0
    finally:
        stand.close()


def test_подключён_только_приём(collector: Collector) -> None:
    """Половинная проводка допустима: колбэки независимы."""
    stand = Rig(log_rx=collector.log_rx)
    try:
        assert stand.session.connect().ok
        assert collector.rx and not collector.tx
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Отказ журнала (Р50)
# --------------------------------------------------------------------------------------


def test_падающий_журнал_не_роняет_подключение() -> None:
    """Диагностический модуль не имеет права остановить протокол.

    Журнал вторичен по отношению к обмену: его баг обязан стоить записи
    в журнале, а не связи с прибором.
    """

    def взорваться(data: bytes, t_mono: float) -> None:
        raise RuntimeError("журналу плохо")

    stand = Rig(log_rx=взорваться, log_tx=взорваться)
    try:
        assert stand.session.connect().ok
        assert stand.session.read_version().unwrap() > 0
    finally:
        stand.close()


def test_отказ_журнала_считается_а_не_молчит() -> None:
    """Молчаливой потери в тракте быть не должно нигде (KB_05 №13).

    Число отказов — единственный признак того, что журнал неполон; без него
    пустой журнал был бы неотличим от отсутствия трафика.
    """

    def взорваться(data: bytes, t_mono: float) -> None:
        raise RuntimeError("журналу плохо")

    stand = Rig(log_tx=взорваться)
    try:
        assert stand.session.connect().ok
        # Подключение — это шесть команд: Stop и пять чтений.
        assert stand.session.stats().tap_errors >= 6
    finally:
        stand.close()


def test_падающий_журнал_не_ломает_поток_телеметрии() -> None:
    """Приёмный тракт продолжает раздавать кадры потребителю."""

    def взорваться(data: bytes, t_mono: float) -> None:
        raise RuntimeError("журналу плохо")

    полученные: list[bytes] = []
    stand = Rig(log_rx=взорваться)
    stand.session._on_telemetry = lambda data, _t: полученные.append(data)
    try:
        assert stand.session.connect().ok
        assert stand.session.start_stream().ok
        assert wait_until(lambda: len(полученные) >= 5)
        stand.session.stop_stream()
        assert stand.session.stats().tap_errors >= 5
    finally:
        stand.close()


# --------------------------------------------------------------------------------------
# Сквозной прогон с настоящим журналом
# --------------------------------------------------------------------------------------


def test_сквозной_обмен_виден_в_файле_журнала(tmp_path: Path) -> None:
    """Сессия с настоящим `PacketLog` против симулятора: файл содержит обе стороны.

    Это проверка проводки целиком — от `transport.send` и tap транспорта
    до строки на диске. Телеметрия в файл при умолчаниях не идёт вовсе
    (`telemetry_stride = 0`), и в этом тесте она и не нужна: проверяются
    команды и ответы.
    """
    log = PacketLog(
        DeviceProfile(),
        PacketLogConfig(directory=tmp_path, serial=94401220, firmware="4.10", poll_period_s=0.01),
    )
    log.open()
    log.start()
    stand = Rig(log_rx=log.log_rx, log_tx=log.log_tx)
    try:
        assert stand.session.connect().ok
        assert stand.session.read_sweep().ok
    finally:
        stand.close()
        log.close()

    path = log.path
    assert path is not None
    text = path.read_text(encoding="ascii")
    строки = [line for line in text.splitlines() if not line.startswith("#")]
    данные = [line for line in строки[1:] if line]

    направления = {line.split(";")[1] for line in данные}
    assert направления == {str(Direction.TX), str(Direction.RX)}

    пары = {line.split(";")[5] for line in данные}
    assert "30 01" in пары, "Stop не записан"
    assert "10 05" in пары, "обмен 10 05 не записан"

    # Байты в журнале — те же, что ушли на провод: команда 10 05 записывается целиком.
    ожидаемая = codec.build_read_sweep().hex(" ").upper()
    assert any(f";{ожидаемая};" in line for line in данные), "сырые байты команды искажены"

    assert log.stats.error is None
    assert stand.session.stats().tap_errors == 0
