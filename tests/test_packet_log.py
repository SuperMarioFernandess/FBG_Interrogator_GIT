"""Тесты журнала обмена.

Датаграммы берутся из эталонных векторов KB_02 (ответы реального прибора)
и собираются `fbg.sim.encode` — тем же независимым от кодека кодом, что
у симулятора (KB_05 №11). Сеть здесь почти не участвует: она закрыта тестами
транспорта, а журналу важен не способ доставки байтов, а то, что он с ними
делает.

Поток журнала в модульных тестах не запускается: `pump()` зовётся вручную,
иначе проверялся бы планировщик ОС. Метки времени задаются явно.

Главное, что проверяется, — что байты не теряются ни при каких условиях
(KB_05 №3): ни когда датаграмма не разбирается, ни когда расшифровка падает
с исключением, ни когда файл писать некуда.
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fbg.core import codec
from fbg.core.pipeline import Pipeline, PipelineConfig
from fbg.core.profile import DeviceProfile
from fbg.io.packet_log import (
    COLUMNS,
    SEPARATOR,
    Direction,
    PacketLog,
    PacketLogConfig,
    PacketRecord,
    describe,
    filter_records,
    format_hex,
    format_id_fc,
    records_from_file,
)
from fbg.io.recorder import Recorder, RecorderConfig
from fbg.sim.encode import encode_measurement, nm_to_raw

#: Решётки стенда, ✅ скрининг: прибор распознаёт две из четырёх.
STAND_NM = (1544.80, 1551.51)

#: ✅ Ответы реального прибора, KB_02 «Эталонные векторы».
REPLY_VERSION = bytes.fromhex("100100080000019A")
REPLY_SERIAL = bytes.fromhex("1003000805A072C4")
REPLY_MODULE_PARAMS = bytes.fromhex("1004000C00CA0004001E001E")
REPLY_STOP_ACK = bytes.fromhex("3001000000080001")
REPLY_WRITE_ACK = bytes.fromhex("200200060001")

#: Датаграмма с парой, которой нет в протоколе: журнал обязан записать её всегда.
GARBAGE = bytes.fromhex("3F7F00040102")


# --------------------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------------------


def make_frame(profile: DeviceProfile, *, filled: int = 2) -> bytes:
    """Кадр телеметрии с заданным числом распознанных решёток в канале 1."""
    divisor = profile.freq_divisor or 10
    freq = np.zeros((profile.channels, profile.fbg_per_channel), dtype=np.uint32)
    for position in range(filled):
        freq[0, position] = nm_to_raw(STAND_NM[position % len(STAND_NM)] - position * 0.01, divisor)
    temp = np.full(profile.channels, 1685, dtype=np.int32)
    return encode_measurement(profile, freq, temp)


def feed_telemetry(log: PacketLog, profile: DeviceProfile, count: int, *, start_s: float) -> None:
    """Подаёт `count` кадров телеметрии с шагом 500 мкс, как на 2 кГц."""
    frame = make_frame(profile)
    for index in range(count):
        log.log_rx(frame, start_s + index * 0.0005)


def files_of(directory: Path, prefix: str = "packets") -> list[Path]:
    """Файлы журнала в порядке появления — по именам, а не по времени создания.

    Метки времени файловой системы источником порядка не являются (KB_05 №27):
    на Windows их разрешение грубее интервала между ротациями, что и вскрыл
    CI чата №7 на файлах измерений.
    """
    return sorted(directory.glob(f"{prefix}_*.log"))


def fields(line: str) -> list[str]:
    """Колонки строки записи."""
    return line.split(SEPARATOR)


def column(line: str, name: str) -> str:
    """Значение колонки по имени."""
    return fields(line)[COLUMNS.index(name)]


class BrokenFile:
    """Файловый объект, у которого запись всегда отказывает.

    Заменяет собой открытый файл журнала: заполненный диск в тесте
    не воспроизвести, а важно именно поведение при отказе `write`.
    """

    def __init__(self) -> None:
        self.closed = False

    def write(self, payload: bytes) -> int:
        raise OSError(28, "No space left on device")

    def flush(self) -> None:
        raise OSError(28, "No space left on device")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def profile() -> DeviceProfile:
    """Профиль прибора со стенда."""
    return DeviceProfile()


@pytest.fixture
def log(profile: DeviceProfile, tmp_path: Path) -> PacketLog:
    """Журнал с файлом, поток не запущен: `pump` зовётся вручную."""
    instance = PacketLog(profile, PacketLogConfig(directory=tmp_path))
    instance.open()
    return instance


# --------------------------------------------------------------------------------------
# Оба направления
# --------------------------------------------------------------------------------------


def test_записывает_отправленную_и_принятую_датаграмму(log: PacketLog, tmp_path: Path) -> None:
    """Оба направления попадают в файл, каждое со своей меткой и расшифровкой."""
    log.log_tx(codec.build_read_module_params(), 1.0)
    log.log_rx(REPLY_MODULE_PARAMS, 1.002)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    assert len(lines) == 2

    request, reply = lines
    assert column(request, "dir") == "TX"
    assert column(request, "id_fc") == "10 04"
    assert column(request, "hex") == "10 04 04 00"
    assert column(request, "decode") == "ReadModuleParams"
    assert column(request, "t_mono") == "1.000000"

    assert column(reply, "dir") == "RX"
    assert column(reply, "len") == "12"
    assert column(reply, "hex") == format_hex(REPLY_MODULE_PARAMS)
    assert column(reply, "decode") == "ModuleParams speed=2000Hz ch=4 fbg=30 gap=30GHz"


def test_номера_записей_сквозные_и_возрастают(log: PacketLog, tmp_path: Path) -> None:
    """`seq` присваивается на входе и не сбрасывается ничем."""
    log.log_tx(codec.build_stop(), 0.5)
    log.log_rx(REPLY_STOP_ACK, 0.51)
    log.log_tx(codec.build_read_version(), 0.6)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    assert [int(column(line, "seq")) for line in lines] == [1, 2, 3]


def test_запрос_и_ответ_одной_пары_расшифрованы_по_разному(profile: DeviceProfile) -> None:
    """Пара (ID, FC) у запроса и ответа одна — различает их направление.

    Без этого `10 01 04 00` и `10 01 00 08 …` в журнале выглядели бы
    одинаково, и журнал перестал бы отвечать на вопрос «кто это сказал».
    """
    assert describe(codec.build_read_version(), Direction.TX, profile) == "ReadVersion"
    assert describe(REPLY_VERSION, Direction.RX, profile) == "Version 4.10"
    assert describe(REPLY_SERIAL, Direction.RX, profile) == "Serial 94401220"
    assert describe(REPLY_WRITE_ACK, Direction.RX, profile) == "SetThresholdAck ok"
    assert describe(REPLY_STOP_ACK, Direction.RX, profile) == "StopAck ok"


def test_отказ_прибора_отличается_от_подтверждения(profile: DeviceProfile) -> None:
    """`00 00` — отказ, `00 01` — успех; в журнале это разные слова, а не 0 и 1.

    `bool` — подкласс `int`, и без явной проверки порядка подтверждение
    записи превратилось бы в «1» рядом с серийным номером «94401220».
    """
    refused = bytes.fromhex("200200060000")
    assert describe(refused, Direction.RX, profile) == "SetThresholdAck refused"


def test_расшифровка_телеметрии_даёт_число_заполненных_позиций(profile: DeviceProfile) -> None:
    """«Telemetry ch=4 filled=2» — ровно случай стенда: две решётки из четырёх."""
    assert describe(make_frame(profile), Direction.RX, profile) == "Telemetry ch=4 filled=2"
    assert (
        describe(make_frame(profile, filled=0), Direction.RX, profile) == "Telemetry ch=4 filled=0"
    )


# --------------------------------------------------------------------------------------
# Нерасшифрованное и упавшая расшифровка
# --------------------------------------------------------------------------------------


def test_нерасшифрованная_датаграмма_записывается_с_байтами_и_пометкой(
    log: PacketLog, tmp_path: Path
) -> None:
    """Пара, которой нет в протоколе, попадает в журнал целиком.

    Именно такие датаграммы журнал и заводился показывать: через pipeline
    они не прошли бы вовсе.
    """
    log.log_rx(GARBAGE, 2.0)
    log.pump()
    log.close()

    line = records_from_file(files_of(tmp_path)[0])[0]
    assert column(line, "hex") == format_hex(GARBAGE)
    assert column(line, "decode") == "UnknownCommand 3F 7F"
    assert column(line, "id_fc") == "3F 7F"
    assert column(line, "len") == str(len(GARBAGE))


def test_испорченный_кадр_телеметрии_записывается_с_видом_ошибки(
    log: PacketLog, profile: DeviceProfile, tmp_path: Path
) -> None:
    """Отказ разбора — результат расшифровки, а не причина потерять байты."""
    broken = make_frame(profile)[:-1]
    log.log_rx(broken, 3.0)
    log.pump()
    log.close()

    line = records_from_file(files_of(tmp_path)[0])[0]
    assert column(line, "hex") == format_hex(broken)
    assert column(line, "decode") == "Telemetry ParseError LEN_MISMATCH"


def test_датаграмма_без_заголовка_записывается(log: PacketLog, tmp_path: Path) -> None:
    """Один байт — не пара (ID, FC), но байт остаётся в журнале."""
    log.log_rx(b"\x30", 4.0)
    log.pump()
    log.close()

    line = records_from_file(files_of(tmp_path)[0])[0]
    assert column(line, "id_fc") == "--"
    assert column(line, "hex") == "30"
    assert column(line, "decode") == "NoHeader len=1"


def test_исключение_в_расшифровке_не_теряет_запись(
    log: PacketLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Упавшая расшифровка даёт пометку, а байты пишутся как ни в чём не бывало.

    Это прямое требование KB_05 №3: расшифровка — украшение поверх байтов,
    а не условие записи. Стимул грубый намеренно — важно, что журнал переживёт
    **любое** исключение из разбора чужих байтов, а не заранее известное.
    """

    def explode(*args: object, **kwargs: object) -> str:
        raise ZeroDivisionError("расшифровка сломалась")

    monkeypatch.setattr("fbg.io.packet_log.describe", explode)
    log.log_rx(REPLY_VERSION, 5.0)
    log.pump()
    log.close()

    line = records_from_file(files_of(tmp_path)[0])[0]
    assert column(line, "hex") == format_hex(REPLY_VERSION)
    assert column(line, "decode") == "DecodeFailed ZeroDivisionError"
    assert log.stats.decode_errors == 1
    assert log.stats.records_written == 1


# --------------------------------------------------------------------------------------
# Фильтрация телеметрии
# --------------------------------------------------------------------------------------


def test_по_умолчанию_телеметрия_в_журнал_не_идёт(
    log: PacketLog, profile: DeviceProfile, tmp_path: Path
) -> None:
    """Умолчание `stride = 0`: команды и ответы есть, потока нет.

    При 2 кГц полный журнал даёт около 3 МБ/с почти одинаковых кадров —
    поэтому умолчание именно такое (риск R7).
    """
    log.log_tx(codec.build_start_stream(2000), 1.0)
    feed_telemetry(log, profile, 100, start_s=1.1)
    log.log_rx(REPLY_STOP_ACK, 2.0)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    assert [column(line, "decode") for line in lines] == ["StartStream", "StopAck ok"]
    stats = log.stats
    assert stats.telemetry_seen == 100
    assert stats.telemetry_admitted == 0
    assert stats.telemetry_skipped == 100


def test_stride_пишет_каждый_n_й_кадр(profile: DeviceProfile, tmp_path: Path) -> None:
    """`stride = 10` при 2 кГц даёт 200 записей в секунду вместо 2000."""
    log = PacketLog(
        profile,
        PacketLogConfig(directory=tmp_path, telemetry_stride=10, telemetry_limit=None),
    )
    log.open()
    feed_telemetry(log, profile, 100, start_s=0.0)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    assert len(lines) == 10
    assert log.stats.telemetry_admitted == 10
    assert log.stats.telemetry_skipped == 90


def test_лимит_отсекает_телеметрию_и_отмечает_это_в_файле(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Полный режим с лимитом: первые N кадров, затем строка `NOTE`.

    Лимит виден в самом журнале, а не только в счётчике: иначе оператор,
    читающий файл, решил бы, что поток прекратился.
    """
    log = PacketLog(
        profile, PacketLogConfig(directory=tmp_path, telemetry_stride=1, telemetry_limit=5)
    )
    log.open()
    feed_telemetry(log, profile, 50, start_s=0.0)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    telemetry = [line for line in lines if column(line, "dir") == "RX"]
    notes = [line for line in lines if column(line, "dir") == "NOTE"]

    assert len(telemetry) == 5
    assert len(notes) == 1
    assert column(notes[0], "decode") == "TelemetryLimitReached limit=5"
    assert column(notes[0], "hex") == ""
    assert log.stats.telemetry_skipped == 45


def test_лимит_обнуляется_при_старте_потока(profile: DeviceProfile, tmp_path: Path) -> None:
    """Второй запуск потока снова пишется: журнал видит Start в отправленном.

    Без сброса лимит, выбранный один раз, сделал бы второй сеанс невидимым,
    а сравнивать поток «до» и «после» правки настроек — обычная работа
    на стенде.
    """
    log = PacketLog(
        profile, PacketLogConfig(directory=tmp_path, telemetry_stride=1, telemetry_limit=3)
    )
    log.open()
    log.log_tx(codec.build_start_stream(2000), 0.0)
    feed_telemetry(log, profile, 10, start_s=0.1)
    log.log_tx(codec.build_stop(), 1.0)
    log.log_tx(codec.build_start_stream(2000), 2.0)
    feed_telemetry(log, profile, 10, start_s=2.1)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    assert len([line for line in lines if column(line, "dir") == "RX"]) == 6
    assert log.stats.telemetry_admitted == 3


def test_сброс_лимита_отключается_настройкой(profile: DeviceProfile, tmp_path: Path) -> None:
    """При `telemetry_limit_resets_on_start = False` лимит общий на весь сеанс."""
    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path,
            telemetry_stride=1,
            telemetry_limit=3,
            telemetry_limit_resets_on_start=False,
        ),
    )
    log.open()
    log.log_tx(codec.build_start_stream(2000), 0.0)
    feed_telemetry(log, profile, 10, start_s=0.1)
    log.log_tx(codec.build_start_stream(2000), 2.0)
    feed_telemetry(log, profile, 10, start_s=2.1)
    log.pump()
    log.close()

    assert log.stats.telemetry_admitted == 3


def test_фильтр_телеметрии_не_трогает_аномальный_кадр(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Кадр `30 02` неверной длины пишется даже при выключенной телеметрии.

    Свойство, ради которого фильтр смотрит и на пару, и на длину: настройка
    объёма не имеет права спрятать аномалию. Именно неразобравшиеся
    датаграммы интереснее всего при отладке протокола.
    """
    log = PacketLog(profile, PacketLogConfig(directory=tmp_path, telemetry_stride=0))
    log.open()
    feed_telemetry(log, profile, 20, start_s=0.0)
    log.log_rx(make_frame(profile)[:-1], 1.0)
    log.log_rx(GARBAGE, 1.1)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    assert [column(line, "decode") for line in lines] == [
        "Telemetry ParseError LEN_MISMATCH",
        "UnknownCommand 3F 7F",
    ]
    assert log.stats.telemetry_seen == 20, "аномальный кадр телеметрией не считается"


# --------------------------------------------------------------------------------------
# Кольцевой буфер
# --------------------------------------------------------------------------------------


def test_кольцо_держит_фиксированный_размер_и_вытесняет_старейшее(
    profile: DeviceProfile,
) -> None:
    """Кольцо для панели журнала: размер фиксирован, вытесняется самое старое."""
    log = PacketLog(
        profile,
        PacketLogConfig(ring_capacity=8, telemetry_stride=1, telemetry_limit=None),
    )
    feed_telemetry(log, profile, 40, start_s=0.0)
    log.pump()

    snapshot = log.snapshot()
    assert len(snapshot) == 8
    assert [record.seq for record in snapshot] == list(range(33, 41))
    assert log.stats.ring_size == 8


def test_снимок_кольца_не_меняется_под_читателем(profile: DeviceProfile) -> None:
    """Снимок — копия: дальнейшие записи его не трогают."""
    log = PacketLog(profile, PacketLogConfig(ring_capacity=100))
    log.log_rx(REPLY_VERSION, 1.0)
    log.pump()
    taken = log.snapshot()

    log.log_rx(REPLY_SERIAL, 2.0)
    log.pump()

    assert len(taken) == 1
    assert len(log.snapshot()) == 2
    assert taken[0].decoded == "Version 4.10"


def test_снимок_с_ограничением_отдаёт_последние(profile: DeviceProfile) -> None:
    """`snapshot(limit)` — свежий хвост, как и нужно панели."""
    log = PacketLog(profile, PacketLogConfig(ring_capacity=100))
    for index in range(10):
        log.log_rx(REPLY_VERSION, float(index))
    log.pump()

    tail = log.snapshot(limit=3)
    assert [record.seq for record in tail] == [8, 9, 10]
    assert len(log.snapshot(limit=999)) == 10


def test_журнал_без_директории_работает_только_в_памяти(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Панель диагностики без записи на диск — рабочий режим, а не вырожденный."""
    log = PacketLog(profile, PacketLogConfig(directory=None))
    log.open()
    log.log_rx(REPLY_VERSION, 1.0)
    log.pump()
    log.close()

    assert log.path is None
    assert log.stats.files == 0
    assert log.stats.bytes_written == 0
    assert len(log.snapshot()) == 1
    assert list(tmp_path.iterdir()) == []


def test_потеря_записи_из_очереди_отмечена_в_журнале(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Отставший журнал теряет записи — и говорит об этом строкой `LostRecords`.

    Номера присваиваются на входе, до очереди, поэтому разрыв виден потоку
    журнала и превращается в отметку. Молчаливой потери в тракте быть
    не должно нигде (KB_05 №13, №22).
    """
    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path, queue_capacity=4, telemetry_stride=1, telemetry_limit=None
        ),
    )
    log.open()
    feed_telemetry(log, profile, 20, start_s=0.0)
    log.pump()
    log.close()

    lines = records_from_file(files_of(tmp_path)[0])
    notes = [line for line in lines if column(line, "dir") == "NOTE"]
    assert log.stats.dropped_queue_full == 16
    assert log.stats.lost_records == 16
    assert len(notes) == 1
    assert column(notes[0], "decode") == "LostRecords count=16"


# --------------------------------------------------------------------------------------
# Ротация и имена файлов
# --------------------------------------------------------------------------------------


def test_ротация_по_размеру(profile: DeviceProfile, tmp_path: Path) -> None:
    """Файл сменяется по достижении лимита, и записи не теряются."""
    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path,
            rotate_bytes=2048,
            keep_files=None,
            telemetry_stride=1,
            telemetry_limit=None,
        ),
    )
    log.open()
    feed_telemetry(log, profile, 30, start_s=0.0)
    log.pump()
    log.close()

    paths = files_of(tmp_path)
    assert len(paths) > 1, "ротация по размеру не сработала"
    numbers = [int(column(line, "seq")) for path in paths for line in records_from_file(path)]
    assert numbers == list(range(1, 31)), "при смене файла записи обязаны остаться все"


def test_ротация_по_времени(profile: DeviceProfile, tmp_path: Path) -> None:
    """Ротация по времени включается настройкой; по умолчанию она выключена.

    Умолчание `None` обосновано: в штатном режиме журнал растёт медленно,
    и ротация по времени плодила бы почти пустые файлы.
    """
    assert PacketLogConfig().rotate_seconds is None

    log = PacketLog(profile, PacketLogConfig(directory=tmp_path, rotate_seconds=0.05))
    log.open()
    log.log_rx(REPLY_VERSION, 1.0)
    log.pump()
    time.sleep(0.08)
    log.log_rx(REPLY_SERIAL, 2.0)
    log.pump()
    log.close()

    paths = files_of(tmp_path)
    assert len(paths) == 2
    assert len(records_from_file(paths[0])) == 1
    assert len(records_from_file(paths[1])) == 1


def test_имена_файлов_сортируются_в_порядке_создания(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Алфавитная сортировка имён обязана совпасть с порядком создания.

    Проверяется по полю `file_part` в шапке, а не по времени создания файла:
    на Windows его разрешение грубее интервала между ротациями (KB_05 №27,
    Р46). Ротация здесь частая намеренно — так в имена попадает номер
    совпадения, дополненный нулями, и без дополнения `_10` встал бы
    перед `_2`.
    """
    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path,
            rotate_bytes=1200,
            keep_files=None,
            telemetry_stride=1,
            telemetry_limit=None,
        ),
    )
    log.open()
    feed_telemetry(log, profile, 60, start_s=0.0)
    log.pump()
    log.close()

    paths = files_of(tmp_path)
    assert len(paths) >= 10, "нужно больше девяти файлов, иначе двузначный номер не проверен"
    parts = [_file_part(path) for path in paths]
    assert parts == sorted(parts), f"порядок имён разошёлся с порядком создания: {parts}"
    assert parts == list(range(1, len(paths) + 1))


def test_удаление_старых_файлов_не_ломает_порядок_имён(
    profile: DeviceProfile, tmp_path: Path
) -> None:
    """Имя удалённой части не достаётся следующему файлу.

    Дефект, найденный этим тестом: подбор номера через `exists()` выдавал
    освободившееся имя заново, и `sorted(glob(...))` выстраивал уцелевшие
    части в неверном порядке — без единого признака, что что-то не так.
    Ровно тот же класс отказа, что и Р46, но причина другая: там номер
    не дополнялся нулями, здесь — переиспользовался.

    Стимул считается заранее, а не берётся «побольше»: переиспользование
    имён циклично, и при большинстве длин прогона уцелевшая тройка случайно
    оказывается упорядоченной. Первая версия теста подавала сорок записей
    и проходила на сломанном коде. Пять записей при `keep_files=3` дают
    части 4, 5, 6 в именах `_004`, базовое и `_002` — то есть порядок
    [5, 6, 4], сломанный однозначно. Тест проверен мутацией: со старым
    подбором имени он краснеет, с исправленным зелёный.
    """
    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path,
            rotate_bytes=500,
            keep_files=3,
            telemetry_stride=1,
            telemetry_limit=None,
        ),
    )
    log.open()
    feed_telemetry(log, profile, 5, start_s=0.0)
    log.pump()
    log.close()

    paths = files_of(tmp_path)
    assert len(paths) == 3
    parts = [_file_part(path) for path in paths]
    assert parts == sorted(parts), f"алфавитный порядок имён разошёлся с порядком создания: {parts}"
    assert parts == [4, 5, 6], "стимул перестал попадать в сломанный случай — пересчитать"


def _file_part(path: Path) -> int:
    """Номер части из шапки файла."""
    for line in path.read_text(encoding="ascii").splitlines():
        if "file_part=" in line:
            return int(line.split("file_part=")[1].split()[0])
    raise AssertionError(f"в шапке {path.name} нет поля file_part")


def test_старые_файлы_удаляются_но_только_свои(profile: DeviceProfile, tmp_path: Path) -> None:
    """`keep_files` держит N последних файлов этого запуска и не трогает чужие.

    Удалять по маске значило бы стирать журналы прошлых запусков, которые
    оператор мог сохранить намеренно. Стёртый чужой файл невосстановим,
    а накопление между запусками предсказуемо.
    """
    stranger = tmp_path / "packets_20200101_000000.log"
    stranger.write_bytes(b"chapter from a previous run\n")

    log = PacketLog(
        profile,
        PacketLogConfig(
            directory=tmp_path,
            rotate_bytes=1200,
            keep_files=3,
            telemetry_stride=1,
            telemetry_limit=None,
        ),
    )
    log.open()
    feed_telemetry(log, profile, 40, start_s=0.0)
    log.pump()
    log.close()

    assert stranger.exists(), "чужой файл журнала удалять нельзя"
    own = [path for path in files_of(tmp_path) if path != stranger]
    assert len(own) == 3


# --------------------------------------------------------------------------------------
# Формат файла и целевая платформа
# --------------------------------------------------------------------------------------


def test_файл_читается_средствами_целевой_платформы(
    log: PacketLog, profile: DeviceProfile, tmp_path: Path
) -> None:
    """Файл открывается без указания кодировки — то есть в кодировке ОС.

    Проверка имеет смысл только на целевой платформе (риск R12): на Linux
    кодировка по умолчанию UTF-8 и пройдёт что угодно, а на Windows это
    cp1252, и любая кириллица уронила бы разбор `UnicodeDecodeError`.
    Windows-job здесь не дублирование, а единственная настоящая проверка.
    Ровно так файл откроет Блокнот и импортирует Excel.
    """
    log.log_tx(codec.build_read_module_params(), 1.0)
    log.log_rx(REPLY_MODULE_PARAMS, 1.002)
    log.log_rx(make_frame(profile)[:-1], 1.1)
    log.pump()
    log.close()

    path = files_of(tmp_path)[0]
    # Без `encoding=`: локальная кодировка ОС, как у наивной постобработки.
    with path.open() as handle:
        text = handle.read()
    assert "ReadModuleParams" in text
    assert path.read_bytes().isascii(), "файл журнала обязан остаться чисто ASCII"


def test_перевод_строки_одинаков_на_всех_платформах(log: PacketLog, tmp_path: Path) -> None:
    """Файл открыт в двоичном режиме, поэтому CR в него не попадает.

    Иначе на Windows строки кончались бы на CRLF, объём считался бы неверно,
    а сравнение файлов между платформами теряло бы смысл.
    """
    log.log_rx(REPLY_VERSION, 1.0)
    log.pump()
    log.close()

    raw = files_of(tmp_path)[0].read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


def test_шапка_начинается_с_имён_колонок(log: PacketLog, tmp_path: Path) -> None:
    """Имена колонок — первой строкой, комментарии следом (Р43).

    Порядок непривычный, но он же выбран для файла измерений и по той же
    причине: так файл читается инструментами с умолчаниями.
    """
    log.log_rx(REPLY_VERSION, 1.0)
    log.pump()
    log.close()

    lines = files_of(tmp_path)[0].read_text(encoding="ascii").splitlines()
    assert lines[0] == SEPARATOR.join(COLUMNS)
    assert lines[1].startswith("# fbg-interrogator packet log")
    assert any("telemetry_stride=" in line for line in lines if line.startswith("#"))


def test_каждая_запись_состоит_из_ожидаемых_колонок(
    log: PacketLog, profile: DeviceProfile, tmp_path: Path
) -> None:
    """Число колонок постоянно, разделитель внутри расшифровки не появляется."""
    log.log_tx(codec.build_set_threshold(1, 1200, profile), 1.0)
    log.log_rx(REPLY_WRITE_ACK, 1.01)
    log.log_rx(GARBAGE, 1.02)
    log.pump()
    log.close()

    for line in records_from_file(files_of(tmp_path)[0]):
        assert len(fields(line)) == len(COLUMNS), f"колонки разъехались: {line}"


def test_локальное_время_записывается_с_миллисекундами(log: PacketLog, tmp_path: Path) -> None:
    """Колонка `t_local` — для чтения глазами, `t_mono` — для сверки с CSV данных.

    Две метки, а не одна: сопоставить журнал с файлом измерений можно только
    по `perf_counter`, а найти нужное место в журнале по нему невозможно.
    """
    log.log_rx(REPLY_VERSION, 12.5)
    log.pump()
    log.close()

    line = records_from_file(files_of(tmp_path)[0])[0]
    stamp = column(line, "t_local")
    hours, minutes, rest = stamp.split(":")
    seconds, millis = rest.split(".")
    assert 0 <= int(hours) <= 23
    assert 0 <= int(minutes) <= 59
    assert 0 <= int(seconds) <= 59
    assert len(millis) == 3
    assert column(line, "t_mono") == "12.500000"


def test_hex_не_обрезается_никогда(profile: DeviceProfile, tmp_path: Path) -> None:
    """Длинный ответ пишется целиком: правило KB_05 №3 исключений не имеет."""
    log = PacketLog(
        profile, PacketLogConfig(directory=tmp_path, telemetry_stride=1, telemetry_limit=None)
    )
    log.open()
    long_reply = make_frame(profile)
    log.log_rx(long_reply, 1.0)
    log.pump()
    log.close()

    line = records_from_file(files_of(tmp_path)[0])[0]
    assert column(line, "hex") == format_hex(long_reply)
    assert bytes.fromhex(column(line, "hex")) == long_reply, "байты обязаны читаться обратно"


def test_форматирование_идентификатора_пары(profile: DeviceProfile) -> None:
    """Пара в одной колонке: две колонки Excel превратил бы в числа."""
    assert format_id_fc(REPLY_MODULE_PARAMS) == "10 04"
    assert format_id_fc(b"\x30") == "--"
    assert format_id_fc(b"") == "--"


# --------------------------------------------------------------------------------------
# Экспорт с фильтром
# --------------------------------------------------------------------------------------


@pytest.fixture
def filled(profile: DeviceProfile) -> PacketLog:
    """Журнал в памяти с разнородным обменом: три направления, разные пары."""
    log = PacketLog(profile, PacketLogConfig(telemetry_stride=1, telemetry_limit=None))
    log.log_tx(codec.build_read_version(), 1.0)
    log.log_rx(REPLY_VERSION, 1.1)
    log.log_tx(codec.build_read_module_params(), 2.0)
    log.log_rx(REPLY_MODULE_PARAMS, 2.1)
    log.log_rx(make_frame(profile), 3.0)
    log.log_rx(GARBAGE, 4.0)
    log.pump()
    return log


def test_экспорт_по_направлению(filled: PacketLog, tmp_path: Path) -> None:
    """Фильтр по направлению оставляет только отправленное."""
    path = tmp_path / "tx.log"
    assert filled.export(path, direction=Direction.TX) == 2
    lines = records_from_file(path)
    assert [column(line, "decode") for line in lines] == ["ReadVersion", "ReadModuleParams"]


def test_экспорт_по_паре_id_fc(filled: PacketLog, tmp_path: Path) -> None:
    """Фильтр по (ID, FC) берёт обе стороны обмена одной команды."""
    path = tmp_path / "version.log"
    assert filled.export(path, id_fc=(0x10, 0x01)) == 2
    lines = records_from_file(path)
    assert [column(line, "dir") for line in lines] == ["TX", "RX"]


def test_экспорт_по_интервалу_времени(filled: PacketLog, tmp_path: Path) -> None:
    """Границы интервала включительные."""
    path = tmp_path / "window.log"
    assert filled.export(path, t_from=2.0, t_to=3.0) == 3
    lines = records_from_file(path)
    assert [column(line, "t_mono") for line in lines] == ["2.000000", "2.100000", "3.000000"]


def test_экспорт_комбинирует_фильтры_и_описывает_их_в_шапке(
    filled: PacketLog, tmp_path: Path
) -> None:
    """Условия складываются через «и», а шапка говорит, что именно отобрано.

    Без записи фильтра выгрузка через неделю читается как полный журнал,
    в котором почему-то нет половины обмена.
    """
    path = tmp_path / "narrow.log"
    assert filled.export(path, direction=Direction.RX, t_from=2.0, t_to=3.0) == 2

    text = path.read_text(encoding="ascii")
    assert "export filter: dir=RX id_fc=any t_from=2.000000 t_to=3.000000" in text
    assert path.read_bytes().isascii()


def test_экспорт_пустого_отбора_даёт_файл_с_одной_шапкой(filled: PacketLog, tmp_path: Path) -> None:
    """Пустой результат — это файл без записей, а не отсутствие файла."""
    path = tmp_path / "none.log"
    assert filled.export(path, id_fc=(0x20, 0x01)) == 0
    assert records_from_file(path) == []
    assert path.read_text(encoding="ascii").startswith(SEPARATOR.join(COLUMNS))


def test_отбор_записей_чистой_функцией() -> None:
    """`filter_records` работает без журнала и без файлов."""
    records = (
        PacketRecord(1, Direction.TX, 1.0, REPLY_VERSION, "x"),
        PacketRecord(2, Direction.RX, 2.0, REPLY_SERIAL, "y"),
        PacketRecord(3, Direction.NOTE, 3.0, b"", "z"),
    )
    assert len(filter_records(records)) == 3
    assert [r.seq for r in filter_records(records, direction=Direction.RX)] == [2]
    assert [r.seq for r in filter_records(records, id_fc=(0x10, 0x03))] == [2]
    assert [r.seq for r in filter_records(records, t_from=2.0)] == [2, 3]
    assert [r.seq for r in filter_records(records, t_to=1.0)] == [1]


# --------------------------------------------------------------------------------------
# Отказ записи
# --------------------------------------------------------------------------------------


def test_ошибка_записи_не_роняет_журнал_и_оставляет_кольцо_живым(
    log: PacketLog, profile: DeviceProfile
) -> None:
    """Диск отказал — файл закрыт, причина записана, кольцо продолжает работать.

    Этим журнал отличается от `recorder` (Р47), который после отказа
    завершает свой поток: там писать некуда и смысла продолжать нет,
    а здесь панель диагностики нужна как раз тогда, когда что-то отказало.
    """
    reasons: list[str] = []
    log._on_error = reasons.append
    log._file = BrokenFile()

    log.log_rx(REPLY_VERSION, 1.0)
    log.log_rx(REPLY_SERIAL, 2.0)
    log.pump()

    stats = log.stats
    assert stats.error is not None and "No space left" in stats.error
    assert len(reasons) == 1, "об отказе сообщается один раз, а не на каждой записи"
    assert stats.records_written == 0
    assert [record.decoded for record in log.snapshot()] == ["Version 4.10", "Serial 94401220"]


def test_ошибка_записи_журнала_не_роняет_recorder(profile: DeviceProfile, tmp_path: Path) -> None:
    """Отказ журнала не касается записи измерений: разные потоки и разные файлы.

    Журнал вторичен по отношению к данным, и проверяется здесь именно
    это: измерения продолжают писаться, когда журнал уже мёртв.
    """
    data_dir = tmp_path / "data"
    pipeline = Pipeline(profile, PipelineConfig(history_frames=1024, ui_period_s=0.01))
    recorder = Recorder(pipeline, RecorderConfig(directory=data_dir))
    log = PacketLog(profile, PacketLogConfig(directory=tmp_path / "logs"))
    log.open()
    log._file = BrokenFile()
    recorder.open()
    try:
        frame = make_frame(profile)
        for index in range(20):
            t_mono = 1.0 + index * 0.0005
            log.log_rx(frame, t_mono)
            pipeline.on_telemetry(frame, t_mono)
        log.pump()
        recorder.pump()
    finally:
        recorder.close()
        log.close()

    assert log.stats.error is not None, "журнал обязан был отказать — иначе тест ничего не значит"
    assert recorder.stats.error is None
    assert recorder.stats.rows == 20


def test_ошибка_записи_не_роняет_приём(profile: DeviceProfile, tmp_path: Path) -> None:
    """Транспорт продолжает принимать датаграммы при мёртвом журнале.

    Журнал подключён к tap ровно так, как будет в приложении, и отказ
    файла остаётся внутри его собственного потока.
    """
    from tests.test_transport import Rig, wait_until

    log = PacketLog(profile, PacketLogConfig(directory=tmp_path, telemetry_stride=1))
    log.open()
    log._file = BrokenFile()
    log.start()
    rig = Rig(tap=log.log_rx)  # type: ignore[arg-type]
    try:
        rig.transport.send(codec.build_start_stream(200))
        assert wait_until(lambda: log.stats.records_in >= 20, timeout=5.0)
        rig.transport.send(codec.build_stop())
        assert wait_until(lambda: not rig.sim.streaming, timeout=2.0)
    finally:
        rig.close()
        log.stop()

    stats = rig.transport.stats()
    assert stats.errors == {}, f"ошибки транспорта: {dict(stats.errors)}"
    assert stats.datagrams_received >= 20
    assert log.stats.error is not None
    assert len(log.snapshot()) >= 20, "кольцо обязано наполняться и при мёртвом файле"


# --------------------------------------------------------------------------------------
# Поток и конфигурация
# --------------------------------------------------------------------------------------


def test_поток_журнала_переносит_записи_сам(profile: DeviceProfile, tmp_path: Path) -> None:
    """`start`/`stop` без единого ручного `pump`, и после остановки потоков нет."""
    before = {thread.name for thread in threading.enumerate()}
    log = PacketLog(profile, PacketLogConfig(directory=tmp_path, poll_period_s=0.005))
    with log:
        assert log.is_running
        log.log_rx(REPLY_VERSION, 1.0)
        log.log_rx(REPLY_SERIAL, 2.0)
        deadline = time.perf_counter() + 5.0
        while log.stats.records_written < 2 and time.perf_counter() < deadline:
            time.sleep(0.005)

    assert log.stats.records_written == 2
    assert not log.is_running
    assert {thread.name for thread in threading.enumerate()} == before
    assert len(records_from_file(files_of(tmp_path)[0])) == 2


def test_остановка_дописывает_остаток_очереди(profile: DeviceProfile, tmp_path: Path) -> None:
    """Записи, поданные перед самой остановкой, не теряются."""
    log = PacketLog(profile, PacketLogConfig(directory=tmp_path))
    log.open()
    for index in range(50):
        log.log_rx(REPLY_VERSION, float(index))
    log.close()

    assert len(records_from_file(files_of(tmp_path)[0])) == 50


def test_некорректная_конфигурация_отвергается(tmp_path: Path) -> None:
    """Некорректные параметры — баг вызывающего, значит ValueError (KB_05)."""
    with pytest.raises(ValueError, match="telemetry_stride"):
        PacketLogConfig(telemetry_stride=-1)
    with pytest.raises(ValueError, match="telemetry_limit"):
        PacketLogConfig(telemetry_limit=-1)
    with pytest.raises(ValueError, match="ring_capacity"):
        PacketLogConfig(ring_capacity=0)
    with pytest.raises(ValueError, match="queue_capacity"):
        PacketLogConfig(queue_capacity=0)
    with pytest.raises(ValueError, match="rotate_bytes"):
        PacketLogConfig(rotate_bytes=0)
    with pytest.raises(ValueError, match="rotate_seconds"):
        PacketLogConfig(rotate_seconds=0.0)
    with pytest.raises(ValueError, match="keep_files"):
        PacketLogConfig(keep_files=0)
    with pytest.raises(ValueError, match="prefix"):
        PacketLogConfig(prefix="")


def test_не_ascii_в_конфигурации_отвергается(tmp_path: Path) -> None:
    """Кириллица в полях шапки сделала бы файл нечитаемым на Windows.

    Решение то же, что у файла измерений (Р44), но причина шире: журнал
    открывают Блокнотом и импортируют в Excel, а кодировку по умолчанию
    там выбирает ОС, и договориться с ней нельзя.
    """
    with pytest.raises(ValueError, match="ASCII"):
        PacketLogConfig(directory=tmp_path, firmware="версия 4.10")
    with pytest.raises(ValueError, match="ASCII"):
        PacketLogConfig(directory=tmp_path, prefix="пакеты")


def test_расшифровка_приводится_к_ascii_без_разделителя(profile: DeviceProfile) -> None:
    """Расшифровка не имеет права испортить колонки или кодировку файла.

    Замена, а не отказ: сорванная расшифровка — не причина терять байты.
    """
    record = PacketRecord(1, Direction.NOTE, 1.0, b"", "плохо; очень\nплохо")
    from fbg.io.packet_log import format_record

    line = format_record(record, "00:00:01.000")
    assert line.isascii()
    assert len(line.rstrip("\n").split(SEPARATOR)) == len(COLUMNS)
