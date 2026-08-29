"""Нагрузочный тест журнала: 2000 датаграмм/с, 60 секунд, полный режим.

Полный режим — ровно тот случай, когда журнал может утопить приём: на каждый
кадр приходится примерно 1.5 КБ текста, и записывается он на тот же диск,
куда пишутся измерения. Поэтому вопрос «создаёт ли журнал обратное давление»
здесь меряется, а не предполагается.

Тракт нарочно короткий — симулятор → транспорт → журнал. Сессия и pipeline
в него не входят: они уже измерены чатами №6 и №7, а здесь проверяется вклад
самого журнала, и лишние звенья только размыли бы его.

Критерий тот же, что у транспорта (KB_05): потери менее 0.1 % за 60 секунд,
считаются на выходе `tap`. Полнота проверяется **по номерам записей в самих
файлах**, а не по счётчикам в памяти: счётчик может ошибаться заодно с кодом,
который он считает.

Маркер `slow`: прогон длится больше минуты, отдельная job в CI.
"""

import time
from pathlib import Path

import pytest

from fbg.core import codec
from fbg.core.profile import DeviceProfile
from fbg.io.packet_log import (
    COLUMNS,
    SEPARATOR,
    Direction,
    PacketLog,
    PacketLogConfig,
    PacketRecord,
    format_record,
    records_from_file,
)
from tests.test_packet_log import files_of, make_frame
from tests.test_transport import Rig, wait_until

pytestmark = pytest.mark.slow

#: Паспортный темп прибора: 2000 Гц, код 0x00CA, ✅ прочитан командой 10 04.
TARGET_RATE_HZ = 2000

#: Длительность основного прогона (KB_05: критерий измеряется за 60 секунд).
LOAD_SECONDS = 60.0

#: Допустимая доля потерь, KB_05.
MAX_LOSS_FRACTION = 0.001

#: Ротация под нагрузкой: 64 МБ дают несколько файлов за минуту полного
#: режима и проверяют, что смена файла на ходу записей не роняет.
ROTATE_BYTES = 64 << 20


class LoggingSink:
    """Потребитель `tap`: считает датаграммы и отдаёт их журналу.

    Именно так журнал подключается в приложении — к tap транспорта, до
    всякого разбора (KB_05 №3). Своей работы у стока нет: всё, что он делает
    сверх счётчика, — вызов `log_rx`, и в этом смысл замера.
    """

    def __init__(self, log: PacketLog, frame_size: int) -> None:
        self.log = log
        self.frame_size = frame_size
        self.frames = 0
        self.other = 0

    def __call__(self, data: bytes, t_mono: float) -> None:
        if len(data) == self.frame_size:
            self.frames += 1
        else:
            self.other += 1
        self.log.log_rx(data, t_mono)


def _run(log: PacketLog, seconds: float, rate_hz: int) -> tuple[Rig, LoggingSink]:
    """Гоняет поток заданное время через транспорт с подключённым журналом."""
    profile = DeviceProfile()
    sink = LoggingSink(log, profile.frame_size)
    log.start()
    rig = Rig(tap=sink, rate_hz=float(rate_hz))  # type: ignore[arg-type]
    try:
        # Отправка логируется так же, как будет в приложении: `log_tx` рядом
        # с `send`. Без неё под нагрузкой проверялось бы только одно
        # направление, а сброс отсчёта телеметрии по команде Start —
        # вообще ни одно.
        start = codec.build_start_stream(rate_hz)
        rig.transport.send(start)
        log.log_tx(start)
        time.sleep(seconds)
        stop = codec.build_stop()
        rig.transport.send(stop)
        log.log_tx(stop)
        assert wait_until(lambda: not rig.sim.streaming, timeout=2.0)
        # Долёт последних датаграмм и разбор очереди журнала.
        time.sleep(1.0)
    finally:
        rig.close()
        log.stop()
    return rig, sink


def _report(name: str, rig: Rig, sink: LoggingSink, log: PacketLog, seconds: float) -> float:
    """Печатает метрики прогона и возвращает долю потерь на приёме."""
    transport = rig.transport.stats()
    stats = log.stats
    sent = rig.sim.stats.frames_sent
    loss = 1.0 - sink.frames / sent if sent else 1.0
    directory = log.config.directory
    paths = files_of(directory) if directory is not None else []
    volume = sum(path.stat().st_size for path in paths)

    print(f"\n--- {name} ---")
    print(f"темп отправки: {rig.sim.pace.describe()}")
    print(
        f"отправлено {sent}, вынуто из сокета {transport.datagrams_received}, "
        f"дошло до журнала {sink.frames} (прочих датаграмм {sink.other})"
    )
    print(
        f"потери на приёме {loss * 100:.4f} % · "
        f"вытеснено очередью транспорта {transport.dropped_queue_full} · "
        f"пик очереди транспорта {transport.queue_peak} из {rig.endpoint.rx_queue_capacity}"
    )
    print(
        f"журнал: принято {stats.records_in}, записано {stats.records_written}, "
        f"вытеснено своей очередью {stats.dropped_queue_full}, "
        f"отмечено потерянными {stats.lost_records}"
    )
    print(
        f"телеметрия: видел {stats.telemetry_seen}, пропустил {stats.telemetry_admitted}, "
        f"отсеял {stats.telemetry_skipped}"
    )
    if volume:
        print(
            f"файлов {len(paths)}, объём {volume / (1 << 20):.1f} МБ "
            f"({volume / seconds / (1 << 20):.2f} МБ/с), "
            f"запись в среднем {volume // max(stats.records_written, 1)} Б"
        )
    else:
        print("файлов нет: журнал работал только в памяти")
    print(f"ошибка записи журнала: {stats.error or 'нет'}")
    print(f"ошибки транспорта: {dict(transport.errors) or 'нет'}")
    return loss


def _sequence_numbers(directory: Path) -> list[int]:
    """Номера записей из всех файлов подряд, в порядке имён файлов."""
    numbers: list[int] = []
    for path in files_of(directory):
        numbers += [int(line.split(SEPARATOR, 1)[0]) for line in records_from_file(path)]
    return numbers


def test_полный_режим_2000_записей_в_секунду_60_секунд(tmp_path: Path) -> None:
    """Полный режим на паспортном темпе: сколько это стоит и что теряется.

    Утверждается ровно одно: журнал не создаёт обратного давления на приём.
    Если бы `log_rx` делал что-нибудь дорогое — форматировал hex, писал файл, —
    диспетчер транспорта встал бы, очередь транспорта переполнилась,
    и потери вылезли бы на приёме. Поэтому потери меряются **до** журнала,
    на выходе `tap`.

    Отставание самого журнала потерей приёма не является и меряется отдельно:
    его очередь вытесняет старейшее и считает вытесненное, а разрыв номеров
    превращается в отметку `LostRecords` в файле. Число печатается, а не
    утверждается: оно зависит от диска, и наше дело — чтобы потеря была
    видимой и ограниченной журналом, а не молчаливой и не общей.
    """
    profile = DeviceProfile()
    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path,
            serial=94401220,
            firmware="4.10",
            telemetry_stride=1,
            telemetry_limit=None,
            keep_files=None,
            rotate_bytes=ROTATE_BYTES,
        ),
    )
    rig, sink = _run(log, LOAD_SECONDS, TARGET_RATE_HZ)
    loss = _report("полный режим, 2000 Гц", rig, sink, log, LOAD_SECONDS)

    stats = log.stats
    transport = rig.transport.stats()
    numbers = _sequence_numbers(tmp_path)
    paths = files_of(tmp_path)

    assert rig.sim.pace.rate_hz == pytest.approx(TARGET_RATE_HZ, rel=0.02), (
        "отправитель не выдержал 2000 Гц — сравнивать нечего"
    )
    assert transport.errors == {}, f"ошибки транспорта: {dict(transport.errors)}"
    assert stats.error is None, f"журнал отвалился: {stats.error}"
    assert len(paths) >= 2, "ротация под нагрузкой не сработала — стык файлов не проверен"
    assert loss < MAX_LOSS_FRACTION, (
        f"потери на приёме {loss * 100:.4f} % — журнал создал обратное давление"
    )
    # Полнота по самим файлам: номера обязаны строго возрастать и через стык
    # файлов тоже. Пропуски допустимы (их отмечает `LostRecords`), а вот
    # повтор или перестановка означали бы порчу журнала.
    assert numbers == sorted(set(numbers)), "номера записей в файлах обязаны строго возрастать"
    assert stats.lost_records == stats.dropped_queue_full, (
        "каждая вытесненная запись обязана быть отмечена в журнале"
    )
    assert stats.queue_depth == 0, "остановка обязана дописать очередь до конца"
    assert len(numbers) == stats.records_written, (
        "число строк в файлах обязано совпасть со счётчиком записанного"
    )


def test_умолчание_почти_ничего_не_стоит(tmp_path: Path) -> None:
    """Тот же поток при `telemetry_stride = 0`: в журнале только команды.

    Это рабочий режим, и он же ответ на риск R7: телеметрия в журнал
    не пишется, объём файла остаётся в килобайтах при мегабайтах трафика.
    """
    profile = DeviceProfile()
    seconds = 15.0
    log = PacketLog(profile, PacketLogConfig(directory=tmp_path, serial=94401220, firmware="4.10"))
    rig, sink = _run(log, seconds, TARGET_RATE_HZ)
    _report("умолчание (телеметрия не пишется), 2000 Гц", rig, sink, log, seconds)

    stats = log.stats
    volume = sum(path.stat().st_size for path in files_of(tmp_path))
    lines = records_from_file(files_of(tmp_path)[0])

    assert stats.error is None
    assert stats.telemetry_admitted == 0
    assert stats.dropped_queue_full == 0, "без телеметрии очередь журнала переполниться не может"
    assert [column for column in lines if column.split(SEPARATOR)[1] == "TX"], (
        "отправленные команды обязаны попасть в журнал при любых настройках телеметрии"
    )
    assert volume < 100 << 10, f"журнал в режиме умолчания занял {volume} байт — это много"
    assert all(line.split(SEPARATOR)[1] != "RX" or "Telemetry" not in line for line in lines)


def test_потолок_журнала(tmp_path: Path) -> None:
    """Сколько записей в секунду журнал способен отформатировать и положить на диск.

    Худший случай формата — кадр телеметрии: 494 байта дают 1481 символ
    одного только hex. Меряется ровно то, чем занят поток журнала:
    форматирование строки и `write` в настоящий файл, без сети и без очередей.
    Запись на диск входит в замер намеренно — без неё цифра описывала бы
    только скорость `%`-форматирования и была бы вдвое оптимистичнее того,
    что журнал делает на самом деле.

    Число печатается всегда, а не только при падении: от него зависит,
    имеет ли смысл полный режим на паспортном темпе (KB_05).
    """
    profile = DeviceProfile()
    frame = make_frame(profile)
    total = 20_000
    records = [
        PacketRecord(index + 1, Direction.RX, index * 0.0005, frame, "Telemetry ch=4 filled=2")
        for index in range(1000)
    ]

    path = tmp_path / "ceiling.log"
    format_seconds = 0.0
    volume = 0
    started = time.perf_counter()
    with path.open("wb", buffering=1 << 20) as handle:
        for _ in range(total // len(records)):
            mark = time.perf_counter()
            text = "".join(format_record(record, "14:32:10.123") for record in records)
            format_seconds += time.perf_counter() - mark
            payload = text.encode("ascii")
            handle.write(payload)
            volume += len(payload)
        handle.flush()
    elapsed = time.perf_counter() - started

    rate = total / elapsed
    print("\n--- потолок журнала, кадр телеметрии ---")
    print(
        f"{total} записей по {volume // total} Б за {elapsed:.2f} с: "
        f"{rate:,.0f} записей/с, {volume / elapsed / (1 << 20):.0f} МБ/с"
    )
    print(
        f"из них форматирование {format_seconds:.2f} с "
        f"({format_seconds / elapsed * 100:.0f} %), запись {elapsed - format_seconds:.2f} с"
    )
    per_second = volume / total * TARGET_RATE_HZ / (1 << 20)
    print(f"объём полного режима на {TARGET_RATE_HZ} Гц: {per_second:.2f} МБ/с")
    print(f"запас над паспортным темпом: ×{rate / TARGET_RATE_HZ:.1f}")

    assert rate > 2 * TARGET_RATE_HZ, (
        f"журнал форматирует {rate:.0f} записей/с — запаса над 2000 Гц нет"
    )
    columns = format_record(records[0], "14:32:10.123").rstrip("\n").split(SEPARATOR)
    assert len(columns) == len(COLUMNS)
    # Не `records_from_file`: она снимает шапку журнала, а здесь файл
    # состоит из одних записей.
    written = path.read_text(encoding="ascii").splitlines()
    assert len(written) == total, "строки обязаны быть все на месте"
    assert len(written[-1].split(SEPARATOR)) == len(COLUMNS), "последняя строка обязана быть целой"


def test_объём_растёт_линейно_по_stride(tmp_path: Path) -> None:
    """Промежуточные режимы стоят ровно во столько раз меньше, во сколько реже пишут.

    Проверяется на живом потоке, а не на арифметике: `stride` применяется
    во входном потоке, и если бы он считал не то, объём это показал бы.
    """
    profile = DeviceProfile()
    seconds = 10.0
    volumes: dict[int, int] = {}
    for stride in (1, 10):
        directory = tmp_path / f"stride{stride}"
        log = PacketLog(
            profile,
            PacketLogConfig(
                directory=directory,
                telemetry_stride=stride,
                telemetry_limit=None,
                keep_files=None,
            ),
        )
        rig, sink = _run(log, seconds, TARGET_RATE_HZ)
        _report(f"stride={stride}, 2000 Гц", rig, sink, log, seconds)
        volumes[stride] = sum(path.stat().st_size for path in files_of(directory))

    ratio = volumes[1] / volumes[10]
    print(f"\nотношение объёмов stride=1 к stride=10: {ratio:.1f}")
    assert 7.0 < ratio < 13.0, f"объём при stride=10 отличается в {ratio:.1f} раза, ожидалось ~10"
