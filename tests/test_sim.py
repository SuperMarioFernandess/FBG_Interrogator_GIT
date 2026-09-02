"""Тесты симулятора прибора.

Симулятор собирает ответы модулем `fbg.sim.encode`, а проверяются они
кодеком `fbg.core.codec` — двумя независимыми реализациями раскладок из KB_02.
Если бы симулятор вызывал функции кодека наоборот, эти тесты проверяли бы
согласованность кодека с самим собой, и общая ошибка в понимании протокола
осталась бы невидимой.

⚠️ Тесты фиксируют **поведение симулятора**, а не факты о приборе. После
скрининга 27.08.2026 часть воспроизводимого поведения стала наблюдённой —
раскладка кадра (N4), единицы частоты (D1), код «пик не найден» (N3),
раскладка `30 03` и два ответа на одну команду (N14), ориентация массива
АЦП (D9). Скрининг 01.09.2026 добавил D4 и Р62/Р63: `20 06` молчит,
0x10/0x20 проходят во время потока, а режимная 0x30 вытесняет его.
Открытой остаётся знаковость температуры (N2b).
"""

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.frames import GainSetting, ParseErrorKind, SweepConfig
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.sim.device_sim import (
    FACTORY_THRESHOLD_AUTO,
    DeviceSimulator,
    Faults,
    Pacer,
    SimState,
)
from fbg.sim.encode import MeasurementEncoder, encode_measurement
from fbg.sim.scene import Grating, Scene, scene_two_gratings

#: Щедрый таймаут ожидания ответа: тесты не должны мигать на загруженной машине.
REPLY_TIMEOUT_S = 2.0

#: Темп потока в функциональных тестах. Нагрузочные 2000 Гц — в `test_load.py`.
TEST_FRAME_RATE_HZ = 200.0


class Harness:
    """Клиентская сторона: сокет, на который прибор шлёт ответы, плюс отправитель.

    Воспроизводит асимметрию настоящего прибора — он отвечает на прописанный
    адрес назначения, а не на source-порт запроса (KB_01, раздел «Сеть»),
    поэтому команды уходят с одного сокета, а ответы приходят на другой.
    """

    def __init__(self, simulator: DeviceSimulator, inbox: socket.socket) -> None:
        self.sim = simulator
        self.inbox = inbox
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: bytes) -> None:
        """Отправляет команду прибору, не ожидая ответа."""
        self.sender.sendto(command, self.sim.address)

    def receive(self, timeout: float = REPLY_TIMEOUT_S) -> bytes | None:
        """Ждёт одну датаграмму. None — прибор промолчал."""
        self.inbox.settimeout(timeout)
        try:
            return self.inbox.recv(65535)
        except TimeoutError:
            return None

    def ask(self, command: bytes, timeout: float = REPLY_TIMEOUT_S) -> bytes | None:
        """Отправляет команду и возвращает ответ. None — прибор промолчал."""
        self.send(command)
        return self.receive(timeout)

    def drain(self, seconds: float = 0.2) -> int:
        """Вычитывает и считает всё, что накопилось в приёмном буфере."""
        deadline = time.perf_counter() + seconds
        count = 0
        while time.perf_counter() < deadline:
            if self.receive(timeout=0.05) is None:
                continue
            count += 1
        return count

    def close(self) -> None:
        self.sender.close()


@pytest.fixture
def profile() -> DeviceProfile:
    """Профиль прибора по умолчанию: 4 канала, 30 решёток, 2000 Гц."""
    return DeviceProfile()


@pytest.fixture
def scene(profile: DeviceProfile) -> Scene:
    """Типовая сцена скрининга: две решётки на канале 1, одна на канале 2."""
    return Scene(
        profile,
        [
            Grating(channel=0, position=0, wavelength_nm=1545.0),
            Grating(channel=0, position=1, wavelength_nm=1550.0),
            Grating(channel=1, position=0, wavelength_nm=1560.0),
        ],
    )


@pytest.fixture
def harness(profile: DeviceProfile, scene: Scene) -> Iterator[Harness]:
    """Поднимает симулятор на эфемерном порту loopback и глушит его после теста."""
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    inbox.bind(("127.0.0.1", 0))

    simulator = DeviceSimulator(
        profile=profile,
        scene=scene,
        reply_to=inbox.getsockname(),
        frame_rate_hz=TEST_FRAME_RATE_HZ,
    )
    simulator.start()
    harness = Harness(simulator, inbox)
    try:
        yield harness
    finally:
        simulator.stop()
        harness.close()
        inbox.close()


# ======================================================================================
# Жизненный цикл
# ======================================================================================


def test_симулятор_биндится_на_эфемерный_порт_loopback(harness: Harness) -> None:
    """Тесты не занимают заводской адрес 192.168.0.19:4567."""
    host, port = harness.sim.address
    assert host == "127.0.0.1"
    assert port != 4567
    assert port > 0


def test_остановка_освобождает_порт(profile: DeviceProfile, scene: Scene) -> None:
    """После `stop` порт можно занять заново — иначе тесты цеплялись бы друг за друга."""
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", 0))
    simulator = DeviceSimulator(profile=profile, scene=scene, reply_to=inbox.getsockname())
    simulator.start()
    port = simulator.port
    simulator.stop()

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()
        inbox.close()


def test_повторный_старт_запрещён(profile: DeviceProfile, scene: Scene) -> None:
    """Двойной `start` — баг вызывающего, а не штатная ситуация."""
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", 0))
    simulator = DeviceSimulator(profile=profile, scene=scene, reply_to=inbox.getsockname())
    simulator.start()
    try:
        with pytest.raises(RuntimeError, match="уже запущен"):
            simulator.start()
    finally:
        simulator.stop()
        inbox.close()


def test_повторная_остановка_безвредна(profile: DeviceProfile, scene: Scene) -> None:
    """`stop` в `finally` может вызваться дважды — это не должно ломаться (KB_05 №6)."""
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", 0))
    simulator = DeviceSimulator(profile=profile, scene=scene, reply_to=inbox.getsockname())
    simulator.start()
    simulator.stop()
    simulator.stop()
    inbox.close()


def test_симулятор_работает_контекстным_менеджером(profile: DeviceProfile, scene: Scene) -> None:
    """`with` гарантирует остановку потока даже при исключении."""
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", 0))
    with DeviceSimulator(profile=profile, scene=scene, reply_to=inbox.getsockname()) as simulator:
        assert simulator.port > 0
    assert simulator._sock is None
    inbox.close()


def test_прибор_отвечает_на_прописанный_адрес_а_не_на_source_порт(harness: Harness) -> None:
    """Ключевая асимметрия прибора: ответ приходит не туда, откуда ушёл запрос.

    Команда уходит с эфемерного порта `sender`, а ответ прибор кладёт
    на прописанный `reply_to` (KB_01). Отсюда требование R4/R5 к транспорту:
    приём идёт на отдельном сокете с фиксированным портом.
    """
    harness.send(codec.build_read_version())
    assert harness.receive() is not None

    harness.sender.settimeout(0.3)
    with pytest.raises(TimeoutError):
        harness.sender.recv(1024)


# ======================================================================================
# Команды чтения (ID = 0x10) — пять штук
# ======================================================================================


def test_10_01_версия_прошивки(harness: Harness) -> None:
    """Версия совпадает с прочитанной на приборе SN 94401220: v4.10."""
    reply = harness.ask(codec.build_read_version())
    assert reply is not None
    assert codec.parse_version(reply).unwrap() == 410


def test_10_03_серийный_номер(harness: Harness) -> None:
    """Серийный номер совпадает с шильдиком: 94 401 220."""
    reply = harness.ask(codec.build_read_serial())
    assert reply is not None
    assert codec.parse_serial(reply).unwrap() == 94_401_220


def test_10_04_параметры_модуля(harness: Harness, profile: DeviceProfile) -> None:
    """Скорость, каналы, решётки и интервал пиков — как в ответе прибора."""
    reply = harness.ask(codec.build_read_module_params())
    assert reply is not None
    params = codec.parse_module_params(reply).unwrap()
    assert params.speed_code == 0x00CA
    assert params.speed_hz == 2000
    assert params.channels == profile.channels
    assert params.fbg_per_channel == profile.fbg_per_channel
    assert params.peak_gap_ghz == 30


def test_10_05_параметры_развёртки(harness: Harness) -> None:
    """Развёртка отдаётся в сырых параметрах и пересчитывается в паспортные ГГц."""
    reply = harness.ask(codec.build_read_sweep())
    assert reply is not None
    sweep = codec.parse_sweep_params(reply, harness.sim.profile).unwrap()
    assert (sweep.start_param, sweep.step_param) == (1, 2)
    assert (sweep.stop_param, sweep.adc_step_param) == (5101, 2)
    # База развёртки 196250 (D8): заводские параметры 1 / 5101 дают 196249 / 191149,
    # а круглые границы получаются при параметрах 0 / 5100.
    assert sweep.start_ghz == 196249
    assert sweep.stop_ghz == 191149
    assert sweep.adc_points == 2551


def test_10_06_пороги_и_усиления(harness: Harness) -> None:
    """Заводская конфигурация: все пороги авто, усиление авто уровень 5."""
    reply = harness.ask(codec.build_read_channel_setup())
    assert reply is not None
    setups = codec.parse_channel_setup(reply, harness.sim.profile).unwrap()
    assert len(setups) == 4
    for index, setup in enumerate(setups):
        assert setup.channel == index
        assert setup.threshold_auto
        assert setup.gain == GainSetting(manual=False, level=5)


def test_ответ_10_06_совпадает_с_вектором_прибора(harness: Harness) -> None:
    """Байты заводского ответа 10 06 совпадают с реальным захватом из KB_02."""
    reply = harness.ask(codec.build_read_channel_setup())
    assert reply == bytes.fromhex("100600 14FFFF0005FFFF0005FFFF0005FFFF0005".replace(" ", ""))


# ======================================================================================
# Команды записи (ID = 0x20) — пять штук, и read-back
# ======================================================================================


def test_20_01_развёртка_записывается_и_читается_обратно(
    harness: Harness, profile: DeviceProfile
) -> None:
    """Состояние настоящее: записанная развёртка возвращается командой 10 05."""
    wanted = SweepConfig.from_params(11, 4, 5000, 4, profile)
    reply = harness.ask(codec.build_set_sweep(wanted, profile))
    assert reply is not None
    assert codec.parse_write_ack(reply).unwrap() is True

    read_back = codec.parse_sweep_params(
        harness.ask(codec.build_read_sweep()) or b"", profile
    ).unwrap()
    assert (read_back.start_param, read_back.step_param) == (11, 4)
    assert (read_back.stop_param, read_back.adc_step_param) == (5000, 4)


@pytest.mark.parametrize("padding", [0, 4])
def test_20_01_принимает_точный_кадр_и_padding_штатного_по(
    harness: Harness, profile: DeviceProfile, padding: int
) -> None:
    """D3: LEN=12, на проводе наблюдались 12 байт и 16 с четырьмя нулями."""
    command = codec.build_set_sweep(SweepConfig.from_params(1, 2, 5101, 2, profile), profile)
    command += b"\x00" * padding
    assert len(command) == 12 + padding
    reply = harness.ask(command)
    assert reply is not None
    assert codec.parse_write_ack(reply).unwrap() is True


def test_20_02_порог_записывается_и_читается_обратно(
    harness: Harness, profile: DeviceProfile
) -> None:
    """Порог канала 3 = 1200 возвращается командой 10 06."""
    reply = harness.ask(codec.build_set_threshold(2, 1200, profile))
    assert reply is not None
    assert codec.parse_write_ack(reply).unwrap() is True

    setups = codec.parse_channel_setup(
        harness.ask(codec.build_read_channel_setup()) or b"", profile
    ).unwrap()
    assert setups[2].threshold == 1200
    assert setups[0].threshold_auto, "остальные каналы задеты быть не должны"


def test_20_02_автоматический_порог_возвращает_none(
    harness: Harness, profile: DeviceProfile
) -> None:
    """FF FF означает авторасчёт, и читается обратно именно как авторасчёт."""
    harness.ask(codec.build_set_threshold(1, 4000, profile))
    harness.ask(codec.build_set_threshold(1, None, profile))
    setups = codec.parse_channel_setup(
        harness.ask(codec.build_read_channel_setup()) or b"", profile
    ).unwrap()
    assert setups[1].threshold is None


def test_20_03_усиление_записывается_и_читается_обратно(
    harness: Harness, profile: DeviceProfile
) -> None:
    """Ручное усиление уровня 2 на канале 4 возвращается командой 10 06."""
    gain = GainSetting(manual=True, level=2)
    reply = harness.ask(codec.build_set_gain(3, gain, profile))
    assert reply is not None
    assert codec.parse_write_ack(reply).unwrap() is True

    setups = codec.parse_channel_setup(
        harness.ask(codec.build_read_channel_setup()) or b"", profile
    ).unwrap()
    assert setups[3].gain == gain


def test_20_04_интервал_пиков_записывается_и_читается_обратно(harness: Harness) -> None:
    """Интервал 40 ГГц возвращается командой 10 04."""
    reply = harness.ask(codec.build_set_peak_gap(40))
    assert reply is not None
    assert codec.parse_write_ack(reply).unwrap() is True

    params = codec.parse_module_params(
        harness.ask(codec.build_read_module_params()) or b""
    ).unwrap()
    assert params.peak_gap_ghz == 40


def test_20_06_не_отвечает_ничего(harness: Harness) -> None:
    """Гипотеза D4: на «сохранить пороги» прибор не отвечает.

    Проверяется отсутствием датаграммы в течение таймаута, а не быстрым
    опросом: молчание нельзя отличить от медленного ответа мгновенно.
    """
    assert harness.ask(codec.build_save_thresholds(), timeout=0.5) is None


def test_20_06_сохраняет_текущие_пороги(harness: Harness, profile: DeviceProfile) -> None:
    """Ответа нет, но состояние меняется: пороги уходят в «энергонезависимую» память."""
    harness.ask(codec.build_set_threshold(0, 777, profile))
    harness.ask(codec.build_save_thresholds(), timeout=0.3)
    assert harness.sim.state.saved_thresholds is not None
    assert harness.sim.state.saved_thresholds[0] == 777


def test_недопустимый_аргумент_даёт_отказ(harness: Harness) -> None:
    """Порог сверх 14 бит подтверждается кодом 0, а не 1.

    ⚠️ Это **гипотеза симулятора**, а не наблюдение: реакция прибора
    на аргумент вне диапазона — открытый вопрос N10, сценарий G8 из KB_06.
    Тест фиксирует поведение симулятора, чтобы сессию было на чём проверять.
    """
    # Кодек такой кадр собрать не даст, поэтому байты собираются вручную.
    reply = harness.ask(bytes([0x20, 0x02, 0x06, 0x00, 0x40, 0x00]))
    assert reply is not None
    assert codec.parse_write_ack(reply).unwrap() is False


# ======================================================================================
# Команды режимов (ID = 0x30) — четыре штуки
# ======================================================================================


def test_30_01_стоп_подтверждается(harness: Harness, profile: DeviceProfile) -> None:
    """Ответ на Stop совпадает с реальным вектором 30 01 00 00 00 08 00 01."""
    reply = harness.ask(codec.build_stop())
    assert reply == bytes.fromhex("3001000000080001")
    assert codec.parse_stop_ack(reply, profile).unwrap() is True


def test_30_01_работает_из_любого_состояния(harness: Harness, profile: DeviceProfile) -> None:
    """Stop обязан срабатывать и в простое, и во время потока (KB_05 №6)."""
    assert codec.parse_stop_ack(harness.ask(codec.build_stop()) or b"", profile).unwrap() is True

    harness.send(codec.build_start_stream())
    time.sleep(0.1)
    assert harness.sim.streaming

    harness.send(codec.build_stop())
    time.sleep(0.2)
    assert not harness.sim.streaming
    assert harness.sim.sim_state is SimState.IDLE


def test_30_07_сырые_отсчёты_ацп(harness: Harness, profile: DeviceProfile) -> None:
    """Спектр канала: 2551 точка, 14 бит, размер ответа совпадает с оценкой KB_01."""
    reply = harness.ask(codec.build_read_raw_adc(0, profile))
    assert reply is not None

    # Раскладка KB_02: ID+FC(2) LEN(4) Канал(2) Усиление(2) ADC(2)×2551 = 5112.
    # ⚠️ В KB_01 стоит оценка «≈ 2551 × 2 + 8 = 5110»: там забыты два байта
    # ID и FC. Кодек считает заголовок как 2 + mode_len_width + 4 = 10 байт,
    # то есть согласен с KB_02. Расхождение внутри файлов знаний, а не с прибором:
    # захвата ответа 30 07 не существует, вопрос D5 открыт.
    header = 2 + profile.mode_len_width + 4
    assert len(reply) == header + profile.adc_points * 2 == 5112

    block = codec.parse_raw_adc(reply, profile).unwrap()
    assert block.channel == 0
    assert block.points == profile.adc_points == 2551
    assert block.gain == GainSetting(manual=False, level=5)
    assert block.adc.max() <= profile.adc_max


def test_30_07_видит_пики_сцены(harness: Harness, profile: DeviceProfile) -> None:
    """В спектре канала 1 два пика сцены, в канале 4 — только шумовая полка."""
    busy = codec.parse_raw_adc(
        harness.ask(codec.build_read_raw_adc(0, profile)) or b"", profile
    ).unwrap()
    empty = codec.parse_raw_adc(
        harness.ask(codec.build_read_raw_adc(3, profile)) or b"", profile
    ).unwrap()
    assert busy.adc.max() > 5 * empty.adc.max()


def test_30_07_молчит_на_несуществующий_канал(harness: Harness) -> None:
    """Сценарий G7: реакция прибора неизвестна, симулятор ничего не утверждает.

    Молчание выбрано именно потому, что это единственный вариант, который
    не фиксирует в тестах выдуманный ответ. Вопрос N10 открыт.
    """
    assert harness.ask(bytes([0x30, 0x07, 0x06, 0x00, 0x00, 0x05]), timeout=0.5) is None


def test_30_03_порождает_два_ответа(harness: Harness, profile: DeviceProfile) -> None:
    """✅ N14: сначала отдельная датаграмма 30 02, затем сам ответ 30 03.

    Скрининг показал, что кадр телеметрии в тело `30 03` не входит и приходит
    непосредственно перед ним. Симулятор это воспроизводит, и порядок здесь
    часть проверки: сессия обязана пережить незапрошенный `30 02`.
    """
    harness.send(codec.build_debug_once())

    first = harness.receive()
    second = harness.receive()
    assert first is not None and second is not None

    assert codec.classify(first) == (codec.ID_MODE, codec.FC_STREAM)
    assert len(first) == profile.frame_size
    assert codec.parse_measurement(first, profile).ok

    assert codec.classify(second) == (codec.ID_MODE, codec.FC_DEBUG)
    assert len(second) == 20430
    assert int.from_bytes(second[2:6], "big") == 0x00004FCE

    time.sleep(0.2)
    assert not harness.sim.streaming
    assert harness.sim.sim_state is SimState.IDLE
    assert SimState.DEBUG in harness.sim.state_history


def test_30_03_тело_раскладывается_на_блоки_каналов(
    harness: Harness, profile: DeviceProfile
) -> None:
    """✅ N14: [Канал(2) Усиление(2) ADC(2)×2551] × 4, частот в теле нет."""
    harness.send(codec.build_debug_once())
    harness.receive()  # кадр телеметрии 30 02
    reply = harness.receive()
    assert reply is not None

    debug = codec.parse_debug_once(reply, profile).unwrap()
    assert debug.channels == profile.channels
    for index, block in enumerate(debug.blocks):
        assert block.channel == index
        assert block.points == profile.adc_points
    # Канал 1 несёт решётки сцены, поэтому его спектр заметно выше полки.
    assert debug.blocks[0].adc.max() > debug.blocks[3].adc.max() * 5


def test_30_03_отдаёт_тот_же_спектр_что_и_30_07(harness: Harness, profile: DeviceProfile) -> None:
    """Блок канала в 30 03 и ответ 30 07 — одна и та же раскладка блока.

    Совпадают заголовок блока и число отсчётов; сами значения различаются,
    потому что сцена шумит от вызова к вызову.
    """
    harness.send(codec.build_debug_once())
    harness.receive()
    debug = codec.parse_debug_once(harness.receive() or b"", profile).unwrap()
    raw = codec.parse_raw_adc(
        harness.ask(codec.build_read_raw_adc(1, profile)) or b"", profile
    ).unwrap()

    assert debug.blocks[1].channel == raw.channel == 1
    assert debug.blocks[1].gain == raw.gain
    assert debug.blocks[1].points == raw.points


def test_неизвестная_команда_остаётся_без_ответа(harness: Harness) -> None:
    """N10: мусор `AA BB CC DD` игнорируется, прибор остаётся жив."""
    assert harness.ask(bytes.fromhex("AABBCCDD"), timeout=0.5) is None
    assert harness.sim.stats.unknown_requests >= 1


# ======================================================================================
# Поток телеметрии
# ======================================================================================


def test_чтения_и_записи_не_останавливают_поток(profile: DeviceProfile, scene: Scene) -> None:
    """Р62: симулятор не должен врать, что 0x10/0x20 вытесняют Streaming."""
    sim = DeviceSimulator(profile=profile, scene=scene, reply_to=("127.0.0.1", 1))
    sim._handle(codec.build_start_stream())
    assert sim.streaming

    assert sim._handle(codec.build_read_module_params())
    assert sim._handle(codec.build_set_threshold(0, 1000, profile))
    assert sim.state.thresholds[0] == 1000
    assert sim.streaming


@pytest.mark.parametrize(
    "command",
    [codec.build_debug_once(), codec.build_read_raw_adc(0, DeviceProfile())],
)
def test_режимная_команда_вытесняет_поток_до_stop(
    profile: DeviceProfile, scene: Scene, command: bytes
) -> None:
    """Р62/Р63: после 30 03/30 07 простой повтор 30 02 поток не возвращает."""
    sim = DeviceSimulator(profile=profile, scene=scene, reply_to=("127.0.0.1", 1))
    sim._handle(codec.build_start_stream())
    assert sim.streaming

    assert sim._handle(command)
    assert not sim.streaming

    sim._handle(codec.build_start_stream())
    assert not sim.streaming

    assert sim._handle(codec.build_stop())
    sim._handle(codec.build_start_stream())
    assert sim.streaming


def test_поток_стартует_и_останавливается(harness: Harness) -> None:
    """30 02 запускает поток без подтверждения, 30 01 его глушит."""
    assert harness.ask(codec.build_start_stream(), timeout=0.15) is not None, "пошли кадры"
    assert harness.sim.streaming

    harness.send(codec.build_stop())
    time.sleep(0.3)
    harness.drain(seconds=0.3)

    assert not harness.sim.streaming
    assert harness.receive(timeout=0.5) is None, "после Stop кадров быть не должно"


def test_кадр_телеметрии_494_байта_и_len_верный(harness: Harness, profile: DeviceProfile) -> None:
    """Размер кадра и поле LEN совпадают с расчётом KB_01: 494 и 0x000001EE."""
    harness.send(codec.build_start_stream())
    frame = harness.receive()
    harness.send(codec.build_stop())

    assert frame is not None
    assert len(frame) == profile.frame_size == 494
    assert frame[:2] == bytes([0x30, 0x02])
    assert int.from_bytes(frame[2:6], "big") == len(frame) == 0x1EE


def test_кадр_разбирается_кодеком_и_восстанавливает_длины_волн(
    harness: Harness, profile: DeviceProfile
) -> None:
    """Круг замкнулся: сцена → байты симулятора → кодек → заданные λ.

    Допуск 0.02 нм покрывает джиттер ±2 пм и квантование поля частоты.
    """
    harness.send(codec.build_start_stream())
    frame = harness.receive()
    harness.send(codec.build_stop())
    assert frame is not None

    measurement = codec.parse_measurement(frame, profile, t_mono=1.25).unwrap()
    assert measurement.t_mono == 1.25
    assert measurement.freq_divisor == 10
    assert measurement.index_mismatches == 0

    wavelengths = measurement.wavelength_nm()
    assert wavelengths[0, 0] == pytest.approx(1545.0, abs=0.02)
    assert wavelengths[0, 1] == pytest.approx(1550.0, abs=0.02)
    assert wavelengths[1, 0] == pytest.approx(1560.0, abs=0.02)
    assert np.isnan(wavelengths[2, 0]), "канал без решёток — NaN, а не последнее значение"
    assert np.count_nonzero(np.isfinite(measurement.freq_ghz)) == 3


@pytest.mark.parametrize(("divisor", "tolerance_nm"), [(1, 0.02), (10, 0.005)])
def test_обе_гипотезы_единиц_частоты(
    profile: DeviceProfile, divisor: int, tolerance_nm: float
) -> None:
    """Симулятор умеет отдавать обе гипотезы D1, автодетект кодека их различает.

    Допуск разный не случайно: гипотеза A квантует поле шагом 8 пм,
    гипотеза B — 0.8 пм. Это и есть довод в пользу B из KB_01.

    Разбор идёт профилем со **снятым** делителем: после скрининга умолчание
    равно 10, и при нём гипотеза A не проверялась бы вовсе — кадр не прошёл бы
    валидацию по диапазону. Автодетект остаётся страховкой на случай прибора
    с другой прошивкой, и проверять его надо явно.
    """
    scene = Scene(profile, [Grating(0, 0, 1550.0)], divisor=divisor, jitter_pm=0.0)
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", 0))
    simulator = DeviceSimulator(
        profile=profile,
        scene=scene,
        reply_to=inbox.getsockname(),
        frame_rate_hz=TEST_FRAME_RATE_HZ,
    )
    simulator.start()
    harness = Harness(simulator, inbox)
    try:
        harness.send(codec.build_start_stream())
        frame = harness.receive()
        harness.send(codec.build_stop())
        assert frame is not None

        autodetect = replace(profile, freq_divisor=None)
        measurement = codec.parse_measurement(frame, autodetect).unwrap()
        assert measurement.freq_divisor == divisor
        assert measurement.wavelength_nm()[0, 0] == pytest.approx(1550.0, abs=tolerance_nm)
    finally:
        simulator.stop()
        harness.close()
        inbox.close()


def test_темп_потока_соответствует_заданному(harness: Harness) -> None:
    """Отправитель выдерживает заданный темп, а не «сколько получится»."""
    harness.send(codec.build_start_stream())
    time.sleep(1.0)
    harness.send(codec.build_stop())
    time.sleep(0.2)

    report = harness.sim.pace
    assert report.frames > 0
    assert report.rate_hz == pytest.approx(TEST_FRAME_RATE_HZ, rel=0.05)


def test_код_скорости_в_30_02_меняет_настройку_прибора(harness: Harness) -> None:
    """Скорость из команды старта запоминается и читается обратно командой 10 04."""
    harness.send(codec.build_start_stream(500))
    time.sleep(0.1)
    harness.send(codec.build_stop())
    time.sleep(0.2)
    harness.drain(seconds=0.2)

    params = codec.parse_module_params(
        harness.ask(codec.build_read_module_params()) or b""
    ).unwrap()
    assert params.speed_code == 0x01F5
    assert params.speed_hz == 500


# ======================================================================================
# Внесение сбоев
# ======================================================================================


def test_сбой_потеря_кадров(harness: Harness) -> None:
    """Заданная доля кадров не уходит в сеть — стимул для оценки потерь в сессии."""
    harness.sim.faults.frame_drop_probability = 0.5
    harness.send(codec.build_start_stream())
    time.sleep(1.0)
    harness.send(codec.build_stop())
    time.sleep(0.2)

    stats = harness.sim.stats
    total = stats.frames_sent + stats.frames_dropped
    assert total > 50
    assert 0.3 < stats.frames_dropped / total < 0.7


def test_сбой_задержка_ответа(harness: Harness) -> None:
    """Ответ приходит позже таймаута сессии — проверка логики повторов."""
    harness.sim.faults.response_delay_s = 0.4

    started = time.perf_counter()
    assert harness.ask(codec.build_read_version(), timeout=0.2) is None, "в таймаут не уложились"
    reply = harness.receive(timeout=REPLY_TIMEOUT_S)
    elapsed = time.perf_counter() - started

    assert reply is not None
    assert elapsed >= 0.4
    assert codec.parse_version(reply).unwrap() == 410


def test_сбой_полное_молчание(harness: Harness) -> None:
    """Прибор замолкает на заданное время и сам возвращается к жизни."""
    harness.sim.go_silent(0.6)
    assert harness.ask(codec.build_read_version(), timeout=0.3) is None

    time.sleep(0.6)
    assert harness.ask(codec.build_read_version()) is not None


def test_сбой_молчание_глушит_и_телеметрию(harness: Harness) -> None:
    """Молчание должно быть полным, иначе им не проверить watchdog потока."""
    harness.send(codec.build_start_stream())
    assert harness.receive() is not None

    harness.sim.go_silent(0.5)
    harness.drain(seconds=0.1)
    assert harness.receive(timeout=0.25) is None

    time.sleep(0.5)
    assert harness.receive() is not None
    harness.send(codec.build_stop())


def test_сбой_неверный_len(harness: Harness) -> None:
    """Ответ цел, но поле LEN не совпадает с длиной — сценарий G6."""
    harness.sim.faults.bad_len = True
    reply = harness.ask(codec.build_read_version())

    assert reply is not None
    assert len(reply) == 8, "испорчен только LEN, длина кадра прежняя"
    result = codec.parse_version(reply)
    assert not result.ok
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_сбой_неверный_len_в_ответе_режима(harness: Harness, profile: DeviceProfile) -> None:
    """Порча LEN учитывает разную ширину поля у ответов 0x10 и 0x30."""
    harness.sim.faults.bad_len = True
    reply = harness.ask(codec.build_stop())

    assert reply is not None
    assert len(reply) == 8
    result = codec.parse_stop_ack(reply, profile)
    assert not result.ok
    assert result.error is not None
    assert result.error.kind is ParseErrorKind.LEN_MISMATCH


def test_неверный_len_запроса_молча_игнорируется(harness: Harness) -> None:
    """N10: `10 01 FF 00` ответа не даёт, но симулятор остаётся жив."""
    assert harness.ask(bytes.fromhex("10 01 FF 00"), timeout=0.05) is None
    reply = harness.ask(codec.build_read_version())
    assert reply is not None


def test_сбой_ответ_мусором(harness: Harness, profile: DeviceProfile) -> None:
    """Вместо кадра приходит мусор — диспетчер обязан его отвергнуть, а не упасть."""
    harness.sim.faults.garbage = True
    reply = harness.ask(codec.build_read_version())

    assert reply is not None
    assert not codec.parse_any(reply, profile).ok


def test_сбой_перезагрузка_сбрасывает_настройки(harness: Harness, profile: DeviceProfile) -> None:
    """«Перезагрузка» посреди работы теряет настройки и глушит поток.

    Это сценарий G2 из KB_06: сессия обязана заметить расхождение
    при перечитывании конфигурации и предложить восстановление (KB_03).
    """
    harness.ask(codec.build_set_threshold(2, 1200, profile))
    harness.ask(codec.build_set_peak_gap(40))
    harness.send(codec.build_start_stream())
    time.sleep(0.1)
    assert harness.sim.streaming

    harness.sim.reboot()
    time.sleep(0.2)
    harness.drain(seconds=0.2)

    assert not harness.sim.streaming
    assert harness.sim.stats.reboots == 1

    setups = codec.parse_channel_setup(
        harness.ask(codec.build_read_channel_setup()) or b"", profile
    ).unwrap()
    assert setups[2].threshold_auto, "порог должен вернуться к заводскому"
    params = codec.parse_module_params(
        harness.ask(codec.build_read_module_params()) or b""
    ).unwrap()
    assert params.peak_gap_ghz == 30


def test_сбои_по_умолчанию_выключены() -> None:
    """Чистый `Faults` ничего не ломает: сбои включаются явно."""
    faults = Faults()
    assert faults.frame_drop_probability == 0.0
    assert faults.response_delay_s == 0.0
    assert not faults.bad_len
    assert not faults.garbage
    assert not faults.is_silent(time.perf_counter())


# ======================================================================================
# Сцена: физическая модель
# ======================================================================================


def test_сцена_нагрев_сдвигает_длину_волны(profile: DeviceProfile) -> None:
    """10 пм/°C из KB_01: нагрев на 50 °C сдвигает пик примерно на 0.5 нм."""
    scene = Scene(profile, [Grating(0, 0, 1550.0)], jitter_pm=0.0, reference_temp_c=25.0)
    before = scene.wavelength_of(scene.find(0, 0), noise=False)

    scene.heat(0, 0, 75.0)
    after = scene.wavelength_of(scene.find(0, 0), noise=False)

    assert after - before == pytest.approx(0.5, abs=1e-9)


def test_сцена_нагрев_одной_решётки_не_трогает_соседнюю(profile: DeviceProfile) -> None:
    """Сценарий скрининга 6.5: греется одна решётка канала, вторая остаётся опорной."""
    scene = Scene(
        profile,
        [Grating(0, 0, 1545.0), Grating(0, 1, 1550.0)],
        jitter_pm=0.0,
    )
    scene.heat(0, 0, 45.0)
    raw = scene.sample_freq_raw(noise=False)

    heated_nm = C_NM_GHZ / (int(raw[0, 0]) / scene.divisor)
    reference_nm = C_NM_GHZ / (int(raw[0, 1]) / scene.divisor)
    assert heated_nm == pytest.approx(1545.2, abs=0.005)
    assert reference_nm == pytest.approx(1550.0, abs=0.005)


def test_сцена_джиттер_укладывается_в_паспортную_повторяемость(profile: DeviceProfile) -> None:
    """Разброс положения пика соответствует паспортным ±2 пм (KB_01)."""
    scene = Scene(profile, [Grating(0, 0, 1550.0)], jitter_pm=2.0, seed=42)
    grating = scene.find(0, 0)
    samples = np.array([scene.wavelength_of(grating) for _ in range(2000)])

    assert samples.mean() == pytest.approx(1550.0, abs=0.0005)
    assert samples.std() == pytest.approx(2e-3, rel=0.15), "σ ≈ 2 пм"


def test_сцена_отключение_линии_гасит_весь_канал(profile: DeviceProfile) -> None:
    """Сценарий скрининга 6.6: отсоединённая оптика — все позиции канала пусты."""
    scene = Scene(profile, [Grating(0, 0, 1545.0), Grating(1, 0, 1560.0)])
    scene.disconnect_channel(0)

    raw = scene.sample_freq_raw()
    assert int(raw[0, 0]) == scene.missing_raw
    assert int(raw[1, 0]) != scene.missing_raw
    assert not scene.is_connected(0)

    scene.connect_channel(0)
    assert int(scene.sample_freq_raw()[0, 0]) != scene.missing_raw


def test_отключение_линии_даёт_nan_после_разбора(harness: Harness, profile: DeviceProfile) -> None:
    """Сквозная проверка правила KB_05 №7: пропавший пик становится NaN."""
    harness.send(codec.build_start_stream())
    assert harness.receive() is not None

    harness.sim.scene.disconnect_channel(0)
    time.sleep(0.1)
    harness.drain(seconds=0.1)
    frame = harness.receive()
    harness.send(codec.build_stop())

    assert frame is not None
    measurement = codec.parse_measurement(frame, profile).unwrap()
    assert np.isnan(measurement.freq_ghz[0, 0])
    assert np.isfinite(measurement.freq_ghz[1, 0]), "соседний канал остался живым"


def test_решётка_вне_развёртки_невидима(profile: DeviceProfile) -> None:
    """Прибор не видит того, что вне 1527.60…1568.36 нм — это не «пик не найден» наугад."""
    scene = Scene(profile, [Grating(0, 0, 1600.0)], jitter_pm=0.0)
    assert int(scene.sample_freq_raw()[0, 0]) == scene.missing_raw


def test_сцена_температура_корпуса_одинакова_во_всех_каналах(profile: DeviceProfile) -> None:
    """Корпус один: масштаб 0.01 °C даёт 2500 при 25 °C (гипотеза N2)."""
    scene = Scene(profile, [], case_temp_c=25.0)
    temp = scene.sample_temp_raw()
    assert temp.tolist() == [2500] * profile.channels


def test_спектр_адc_имеет_пик_нужной_ширины(profile: DeviceProfile) -> None:
    """FWHM ≈ 25 ГГц ≈ 12–13 отсчётов при шаге 2 ГГц (KB_01)."""
    scene = Scene(profile, [Grating(0, 0, 1550.0)], floor_noise_adc=0.0)
    spectrum = scene.spectrum(0, gain_level=5).astype(np.float64)

    half = (spectrum.max() + scene.floor_adc) / 2
    width_points = int(np.count_nonzero(spectrum >= half))
    assert 11 <= width_points <= 15


def test_спектр_зависит_от_усиления(profile: DeviceProfile) -> None:
    """Меньшее усиление — меньшая амплитуда, по отношению коэффициентов из KB_02."""
    scene = Scene(profile, [Grating(0, 0, 1550.0)], floor_noise_adc=0.0, peak_adc=8000.0)
    # Сравниваются высоты пиков над шумовой полкой: сама полка от усиления не зависит.
    high = int(scene.spectrum(0, gain_level=5).max()) - scene.floor_adc
    low = int(scene.spectrum(0, gain_level=0).max()) - scene.floor_adc
    assert low < high
    assert low / high == pytest.approx(2.9059e-6 / 2.36161e-5, rel=0.05)


def test_спектр_не_превышает_разрядность_ацп(profile: DeviceProfile) -> None:
    """АЦП 14 бит: значения не выходят за 0…16383 даже при завышенной амплитуде."""
    scene = Scene(profile, [Grating(0, 0, 1550.0)], peak_adc=1e6)
    spectrum = scene.spectrum(0, gain_level=5)
    assert spectrum.min() >= 0
    assert spectrum.max() <= profile.adc_max


def test_сцена_отвергает_решётку_вне_конфигурации(profile: DeviceProfile) -> None:
    """Позиция за границами прибора — баг вызывающего, а не штатная ситуация."""
    with pytest.raises(ValueError, match="канал"):
        Scene(profile, [Grating(9, 0, 1550.0)])
    with pytest.raises(ValueError, match="позиция"):
        Scene(profile, [Grating(0, 99, 1550.0)])


# ======================================================================================
# Кодирование кадра: быстрый путь против буквального
# ======================================================================================


def test_быстрый_энкодер_совпадает_с_побайтовым(profile: DeviceProfile) -> None:
    """Предсобранный буфер даёт ровно те же байты, что и буквальная сборка.

    Это главная защита оптимизации: патч 120 полей через numpy-view легко
    ошибается смещением на байт, и без такой сверки ошибка была бы не видна.
    """
    freq_raw, temp_raw = scene_two_gratings(profile, 10)
    encoder = MeasurementEncoder(profile)
    encoder.update(freq_raw, temp_raw)
    assert encoder.to_bytes() == encode_measurement(profile, freq_raw, temp_raw)


def test_быстрый_энкодер_переиспользует_буфер(profile: DeviceProfile) -> None:
    """Второй кадр не тянет за собой хвосты первого."""
    encoder = MeasurementEncoder(profile)
    first, temp = scene_two_gratings(profile, 10)
    encoder.update(first, temp)

    second = np.zeros_like(first)
    second[0, 5] = 1_950_000
    encoder.update(second, temp)
    assert encoder.to_bytes() == encode_measurement(profile, second, temp)


def test_быстрый_энкодер_ловит_переполнение_поля(profile: DeviceProfile) -> None:
    """Поле частоты трёхбайтовое: молча отбросить старший байт нельзя."""
    encoder = MeasurementEncoder(profile)
    freq_raw = np.zeros((profile.channels, profile.fbg_per_channel), dtype=np.uint32)
    freq_raw[0, 0] = 0x1_00_00_00
    with pytest.raises(ValueError, match="три байта"):
        encoder.update(freq_raw, np.zeros(profile.channels, dtype=np.int32))


@pytest.mark.parametrize("signed", [True, False])
def test_энкодер_уважает_знаковость_температуры(signed: bool) -> None:
    """Знаковость поля температуры — параметр профиля (вопрос N2b), не константа."""
    profile = DeviceProfile(case_temp_signed=signed)
    encoder = MeasurementEncoder(profile)
    freq_raw = np.zeros((profile.channels, profile.fbg_per_channel), dtype=np.uint32)
    temp_raw = np.full(profile.channels, -1500 if signed else 1500, dtype=np.int64)

    encoder.update(freq_raw, temp_raw)
    assert encoder.to_bytes() == encode_measurement(profile, freq_raw, temp_raw)


# ======================================================================================
# Выдерживание темпа
# ======================================================================================


def test_pacer_отвергает_неположительный_темп() -> None:
    """Нулевой темп — баг вызывающего."""
    with pytest.raises(ValueError, match="положительным"):
        Pacer(0.0)


def test_pacer_калибрует_запас_по_фактическому_сну() -> None:
    """Запас берётся из измерения, а не из константы под конкретную ОС."""
    pacer = Pacer(1000.0)
    margin = pacer.calibrate()
    assert margin >= Pacer.MIN_MARGIN_S
    assert margin == pacer.margin_s


def test_pacer_выдерживает_темп_точнее_чем_сон() -> None:
    """Момент считается от абсолютной базы, поэтому ошибка шага не накапливается.

    Наивный `sleep(period)` промахнулся бы на величину оверслипа каждый шаг
    и дал бы заметно меньший темп — ради этого Pacer и написан.
    """
    pacer = Pacer(500.0)
    pacer.start()
    for _ in range(250):
        pacer.wait()

    assert pacer.report.rate_hz == pytest.approx(500.0, rel=0.05)
    assert pacer.report.resyncs == 0


def test_pacer_пересинхронизируется_после_долгой_паузы() -> None:
    """После паузы кадры не выпаливаются залпом: база сдвигается.

    Иначе приёмный буфер сокета переполнился бы очередью «догоняющих» кадров,
    и нагрузочный тест мерил бы размер SO_RCVBUF, а не приёмник.
    """
    pacer = Pacer(2000.0, max_lag_periods=5)
    pacer.start()
    pacer.wait()
    time.sleep(0.05)  # 100 периодов
    pacer.wait()

    assert pacer.report.resyncs == 1
    assert pacer.report.late_frames >= 1


def test_ресинхронизация_не_завышает_темп() -> None:
    """Длительность сеанса считается от старта, а не от сдвинутой базы.

    Регрессия: при отсчёте от базы пересинхронизация обнуляла знаменатель,
    оставляя накопленный счётчик кадров, и темп 2000 Гц показывался как 5500.
    """
    pacer = Pacer(1000.0, max_lag_periods=5)
    pacer.start()
    for _ in range(20):
        pacer.wait()
    time.sleep(0.05)  # 50 периодов — гарантированная ресинхронизация
    for _ in range(20):
        pacer.wait()

    assert pacer.report.resyncs >= 1
    assert pacer.report.rate_hz < 1000.0, "пропущенные периоды снижают темп, а не повышают"
    # 20 периодов + пауза 50 мс + 19 периодов после сдвига базы ≈ 89 мс.
    assert pacer.report.elapsed_s >= 0.08


def test_pacer_собирает_перцентили_когда_попросили() -> None:
    """σ при тяжёлом хвосте обманчива, поэтому отчёт умеет в перцентили."""
    pacer = Pacer(1000.0, deviation_capacity=200)
    pacer.start()
    for _ in range(100):
        pacer.wait()

    assert pacer.report.percentile_us(50) < pacer.report.max_deviation_us + 1
    assert "перцентили" in pacer.report.describe()


def test_отчёт_без_сбора_отклонений_честно_отказывает() -> None:
    """Перцентили не выдумываются из среднего и σ."""
    pacer = Pacer(1000.0)
    pacer.start()
    pacer.wait()
    with pytest.raises(RuntimeError, match="deviation_capacity"):
        pacer.report.percentile_us(50)


# ======================================================================================
# Состояние прибора
# ======================================================================================


def test_заводское_состояние_совпадает_с_прибором(harness: Harness) -> None:
    """Значения по умолчанию — те, что прочитаны с прибора SN 94401220."""
    state = harness.sim.state
    assert state.version_raw == 410
    assert state.serial == 94_401_220
    assert state.speed_code == 0x00CA
    assert state.thresholds == [FACTORY_THRESHOLD_AUTO] * 4
    assert state.gain_levels == [5] * 4


def test_симулятор_не_тянет_qt() -> None:
    """Симулятор — часть проекта, а не UI: PySide6 он не импортирует.

    Проверка ушла в подпроцесс в чате №10 по той же причине, что и у дымового
    теста: с появлением `fbg/ui` в прогоне живут тесты, импортирующие Qt
    намеренно, и `sys.modules` текущего процесса перестал что-либо говорить
    о зависимостях симулятора — сборка тестовых модулей затягивает PySide6
    ещё до запуска первого теста.
    """
    program = (
        "import sys\n"
        "import fbg.sim.device_sim, fbg.sim.encode, fbg.sim.scene\n"
        "sys.exit('симулятор импортировал PySide6' if 'PySide6' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout


# ======================================================================================
# Недокументированная команда 10 02 (N17) и ориентация спектра (D9)
# ======================================================================================


def test_10_02_недокументированная_команда(harness: Harness) -> None:
    """✅ Симулятор отдаёт ровно те байты, что пришли с прибора (N17).

    Смысл полей неизвестен, вычислять их не из чего, поэтому это литерал
    из захвата. Команду мы не используем ни в опросе, ни в watchdog.
    """
    reply = harness.ask(codec.build_read_undocumented())
    assert reply is not None
    assert reply == bytes.fromhex("10020014" + "05DC0A80" + "0000" * 6)

    response = codec.parse_undocumented(reply).unwrap()
    assert response.words[:2] == (1500, 2688)


def test_спектр_ориентирован_от_stop_к_start(profile: DeviceProfile) -> None:
    """✅ D9: индекс 0 — нижняя частота, длина волны убывает с индексом.

    Решётка ставится ближе к длинноволновому краю развёртки; её пик обязан
    оказаться в первой половине массива. До скрининга ось шла в обратную
    сторону, и этот же пик попал бы во вторую половину.
    """
    scene = Scene(profile, [Grating(0, 0, 1565.0)], jitter_pm=0.0)
    axis = scene.freq_axis_ghz()

    assert axis[0] == profile.stop_ghz
    assert axis[-1] == profile.start_ghz
    assert C_NM_GHZ / axis[0] > C_NM_GHZ / axis[-1]

    peak_index = int(np.argmax(scene.spectrum(0, 5)))
    assert peak_index < profile.adc_points // 2
    assert profile.adc_index_to_nm(peak_index) == pytest.approx(1565.0, abs=0.02)


def test_позиция_пика_согласована_с_профилем(profile: DeviceProfile) -> None:
    """Пересчёт индекса в профиле и ось сцены описывают одно и то же."""
    scene = Scene(profile, [Grating(0, 0, 1550.0)], jitter_pm=0.0)
    peak_index = int(np.argmax(scene.spectrum(0, 5)))
    expected = profile.ghz_to_adc_index(C_NM_GHZ / 1550.0)
    assert peak_index == pytest.approx(expected, abs=1.0)
