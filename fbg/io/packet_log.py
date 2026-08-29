"""Журнал обмена: сырые байты, расшифровка, ротация, фильтры, экспорт.

Второй модуль `fbg/io` и второй потребитель, которому разрешены файловый I/O
и собственный поток. От `recorder` отличается источником: тот **забирает**
разобранные кадры курсором кольца, а журналу нужны байты, которых в кольце
нет вовсе.

Почему tap транспорта, а не pipeline
------------------------------------
Две причины, и обе следуют из назначения журнала.

Правило KB_05 №3 требует сырых байтов **до** разбора, а pipeline отдаёт уже
разобранные кадры: переразобрать историю новым профилем по ним нельзя.

Журнал обязан видеть то, что кодек разобрать **не смог**. Именно такие
датаграммы интереснее всего при отладке протокола, и через pipeline они
не прошли бы вовсе — он их отбрасывает. Отсюда следствие, определяющее весь
модуль: **расшифровка необязательна.** Она украшение поверх байтов, а не
условие записи. Не разобралось — пишем байты и пометку.

Кто кого ждёт
-------------
Никто. `log_rx` и `log_tx` вызываются из чужих потоков (диспетчер транспорта,
поток, отправляющий команду) и делают минимум: применяют дешёвый фильтр
телеметрии и кладут байты в свою очередь. Расшифровка, форматирование hex,
открытие файла и запись живут в потоке `fbg-packet-log`. В приёмном тракте
не делается ничего дороже нескольких десятков микросекунд (KB_05 №23),
и затык диска обратного давления на приём не создаёт.

Объём при 2 кГц
---------------
Кадр телеметрии в журнале — около 1.5 КБ текста (1481 символ одного только
hex). На паспортных 2000 Гц это ~3 МБ/с журнала поверх 1 МБ/с данных, причём
кадры почти одинаковы. Поэтому по умолчанию телеметрия в журнал **не идёт**
(`telemetry_stride = 0`), а полнота включается явно и с лимитом.

Режимов-перечислений нет, вместо них три числа — они покрывают все нужные
случаи и не изобретают лишнего:

============================  ==========================================
Что нужно                     Настройка
============================  ==========================================
только команды и ответы       `stride=0` (умолчание)
первые N кадров после Start   `stride=1, limit=N`
поток целиком, но редко       `stride=200, limit=None` (10 Гц при 2 кГц)
всё подряд, осознанно         `stride=1, limit=None`
============================  ==========================================

Отсчёт `stride` и лимит обнуляются, когда журнал видит **отправленную**
команду `30 02` (Start): о старте потока он узнаёт из самого обмена,
внешней координации для этого не нужно.

Что фильтр не трогает никогда
-----------------------------
Фильтр применяется только к датаграмме, которая **выглядит** телеметрией:
пара `30 02` и длина ровно `profile.frame_size`. Всё остальное — команды,
ответы, мусор, кадр телеметрии неверной длины — пишется безусловно, при любых
настройках. Проверка стоит два байта и сравнение длины, разбора не требует,
и ровно она обеспечивает главное свойство журнала: аномалия не может быть
отфильтрована настройкой объёма.

Цена честная и названа: кадр правильной длины с мусором внутри при
`stride = 0` в журнал не попадёт. Такие кадры считает `pipeline.parse_errors`;
увидеть их байты — повод включить `stride`.

Отвергнутый режим «только изменившиеся кадры»
---------------------------------------------
Отбор по изменению числа заполненных позиций требует полного разбора **каждого**
кадра ради решения о записи: это 21 мкс на кадр, второй разбор тех же байтов
поверх pipeline и постоянные 4 % ядра в режиме, который почти ничего не пишет.
Вдобавок он заставил бы журнал байтов фильтровать по семантике, надёжность
которой сам же и проверяет. Изменение числа пиков — событие уровня метрик,
и место ему в pipeline, а не здесь.

Кодировка
---------
Файл чисто ASCII, метки расшифровки английские. Причина та же, что у Р44,
но следствие сильнее: журнал открывают в Блокноте и импортируют в Excel,
а обе программы на Windows решают про кодировку сами. UTF-8 без BOM Excel
прочтёт как ANSI и покажет кракозябры; UTF-8 с BOM ломает `grep`, наивный
`open(..., encoding="ascii")` и склейку файлов; cp1251 нечитаем всюду, кроме
Windows; транслитерация неоднозначна и уродлива. ASCII читается везде
одинаково и без единой настройки, а переводить в метках вида `Telemetry`
или `ReadModuleParams` нечего — это ярлыки, не проза. Русский остаётся
в докстрингах и в будущей панели UI, которая строит текст из полей записи,
а не из строки файла.

Перевод строки — `\\n`, файл открыт в двоичном режиме. Блокнот Windows 10
показывает LF корректно с версии 1809, а одинаковый перевод строки на всех
платформах делает объём файла считаемым точно и сравнение файлов осмысленным.

Надёжность
----------
Журнал вторичен по отношению к данным. Ошибка записи (диск заполнен, папка
исчезла) закрывает файл, запоминает причину в `PacketLogStats.error` и зовёт
`on_error`, но поток **не останавливает**: кольцевой буфер продолжает
наполняться, и панель диагностики остаётся живой ровно тогда, когда она
нужнее всего. Этим журнал отличается от `recorder` (Р47), у которого после
отказа писать некуда и смысла продолжать нет.

Потеря записи из-за отставания журнала видна в самом журнале: номера `seq`
проставляются на входе, до очереди, и разрыв в них превращается в строку
`NOTE … LostRecords count=N`. Молчаливой потери в тракте быть не должно
нигде (KB_05 №13, №22).
"""

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from fbg.core import codec
from fbg.core.frames import (
    AdcBlock,
    DebugResponse,
    MeasurementFrame,
    ModuleParams,
    SweepConfig,
    UndocumentedResponse,
)
from fbg.core.profile import DeviceProfile

#: Разделитель колонок — тот же, что в файле измерений.
SEPARATOR = ";"

#: Знаков в колонке `t_mono`. Микросекунда — разрешение `perf_counter`.
TIME_DECIMALS = 6

#: Ротация по размеру, байты (KB_03: журнал — 100 МБ × 10 файлов).
DEFAULT_ROTATE_BYTES = 100 << 20

#: Сколько файлов журнала, созданных **этим экземпляром**, хранить.
DEFAULT_KEEP_FILES = 10

#: Ёмкость кольцевого буфера для панели журнала (чат №10).
DEFAULT_RING_CAPACITY = 2000

#: Ёмкость очереди между входом и потоком журнала.
DEFAULT_QUEUE_CAPACITY = 8192

#: Потолок телеметрии в журнале: 20 000 кадров — 10 секунд потока при 2 кГц
#: и около 30 МБ текста. Дальше журнал перестал бы быть журналом.
DEFAULT_TELEMETRY_LIMIT = 20_000

#: Пауза потока журнала, когда записей нет.
DEFAULT_POLL_PERIOD_S = 0.01

#: Как часто буфер файла сбрасывается в ОС.
DEFAULT_FLUSH_PERIOD_S = 1.0

#: Сколько записей поток журнала забирает из очереди за один заход.
DEFAULT_BATCH_LIMIT = 512

#: Размер буфера файла.
FILE_BUFFER_BYTES = 1 << 20

#: Значение колонки `id_fc` для записи, у которой пары (ID, FC) нет.
NO_ID_FC = "--"

#: Имена колонок журнала. `hex` стоит **до** `decode`: сырые байты первичны
#: (KB_05 №3), а расшифровка — последняя колонка переменной длины, которая
#: может быть и пустой, и пометкой об ошибке, не сдвигая ничего левее.
COLUMNS: tuple[str, ...] = ("seq", "dir", "t_mono", "t_local", "len", "id_fc", "hex", "decode")


class Direction(StrEnum):
    """Направление записи. Значение пишется в файл как есть."""

    TX = "TX"
    """Датаграмма, отправленная прибору."""

    RX = "RX"
    """Датаграмма, принятая от прибора."""

    NOTE = "NOTE"
    """Событие самого журнала, а не трафик: потеря записи, исчерпание лимита."""


# --------------------------------------------------------------------------------------
# Имена пар (ID, FC)
# --------------------------------------------------------------------------------------

#: Имена команд, которые отправляем мы. Пары — из `codec.KNOWN_COMMANDS`;
#: имена английские (см. раздел «Кодировка» в шапке модуля).
TX_NAMES: dict[tuple[int, int], str] = {
    (codec.ID_READ, codec.FC_VERSION): "ReadVersion",
    (codec.ID_READ, codec.FC_UNDOCUMENTED): "ReadUndocumented",
    (codec.ID_READ, codec.FC_SERIAL): "ReadSerial",
    (codec.ID_READ, codec.FC_MODULE_PARAMS): "ReadModuleParams",
    (codec.ID_READ, codec.FC_SWEEP): "ReadSweep",
    (codec.ID_READ, codec.FC_CHANNEL_SETUP): "ReadChannelSetup",
    (codec.ID_WRITE, codec.FC_SET_SWEEP): "SetSweep",
    (codec.ID_WRITE, codec.FC_SET_THRESHOLD): "SetThreshold",
    (codec.ID_WRITE, codec.FC_SET_GAIN): "SetGain",
    (codec.ID_WRITE, codec.FC_SET_PEAK_GAP): "SetPeakGap",
    (codec.ID_WRITE, codec.FC_SAVE_THRESHOLDS): "SaveThresholds",
    (codec.ID_MODE, codec.FC_STOP): "Stop",
    (codec.ID_MODE, codec.FC_STREAM): "StartStream",
    (codec.ID_MODE, codec.FC_DEBUG): "DebugOnce",
    (codec.ID_MODE, codec.FC_RAW_ADC): "ReadRawAdc",
}

#: Имена того, что приходит от прибора. Отличаются от `TX_NAMES` не для красоты:
#: пара (ID, FC) у запроса и ответа одна, и без направления `10 04 04 00`
#: и `10 04 00 0C …` в журнале выглядели бы одинаково.
RX_NAMES: dict[tuple[int, int], str] = {
    (codec.ID_READ, codec.FC_VERSION): "Version",
    (codec.ID_READ, codec.FC_UNDOCUMENTED): "Undocumented",
    (codec.ID_READ, codec.FC_SERIAL): "Serial",
    (codec.ID_READ, codec.FC_MODULE_PARAMS): "ModuleParams",
    (codec.ID_READ, codec.FC_SWEEP): "Sweep",
    (codec.ID_READ, codec.FC_CHANNEL_SETUP): "ChannelSetup",
    (codec.ID_WRITE, codec.FC_SET_SWEEP): "SetSweepAck",
    (codec.ID_WRITE, codec.FC_SET_THRESHOLD): "SetThresholdAck",
    (codec.ID_WRITE, codec.FC_SET_GAIN): "SetGainAck",
    (codec.ID_WRITE, codec.FC_SET_PEAK_GAP): "SetPeakGapAck",
    (codec.ID_WRITE, codec.FC_SAVE_THRESHOLDS): "SaveThresholdsAck",
    (codec.ID_MODE, codec.FC_STOP): "StopAck",
    (codec.ID_MODE, codec.FC_STREAM): "Telemetry",
    (codec.ID_MODE, codec.FC_DEBUG): "Debug",
    (codec.ID_MODE, codec.FC_RAW_ADC): "RawAdc",
}

#: Пара телеметрии — самая частая проверка модуля, поэтому вынесена байтами.
TELEMETRY_PREFIX = bytes([codec.ID_MODE, codec.FC_STREAM])


# --------------------------------------------------------------------------------------
# Конфигурация и состояние
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketLogConfig:
    """Параметры журнала.

    `directory = None` — журнал работает только в памяти: кольцевой буфер
    наполняется, файлы не создаются. Это рабочий режим панели диагностики,
    когда писать на диск незачем.
    """

    directory: Path | None = None
    """Куда складывать файлы. None — не писать на диск вовсе."""

    app_version: str = "0.1.0"
    device_model: str = "GC-97001C-03-01-A-F"
    serial: int | None = None
    firmware: str | None = None

    telemetry_stride: int = 0
    """Писать каждый N-й кадр телеметрии. 0 — не писать (умолчание), 1 — все."""

    telemetry_limit: int | None = DEFAULT_TELEMETRY_LIMIT
    """Потолок числа кадров телеметрии. None — без потолка, осознанно."""

    telemetry_limit_resets_on_start: bool = True
    """Обнулять отсчёт `stride` и лимит, увидев отправленную команду 30 02."""

    ring_capacity: int = DEFAULT_RING_CAPACITY
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY
    rotate_bytes: int | None = DEFAULT_ROTATE_BYTES

    rotate_seconds: float | None = None
    """Ротация по времени. По умолчанию выключена: в штатном режиме журнал
    растёт медленно, и часовая ротация плодила бы почти пустые файлы,
    а в полном режиме 100 МБ набираются раньше любого разумного срока."""

    keep_files: int | None = DEFAULT_KEEP_FILES
    """Сколько своих файлов хранить. None — не удалять ничего."""

    poll_period_s: float = DEFAULT_POLL_PERIOD_S
    flush_period_s: float = DEFAULT_FLUSH_PERIOD_S
    batch_limit: int = DEFAULT_BATCH_LIMIT
    prefix: str = "packets"
    """Начало имени файла: `packets_YYYYMMDD_HHMMSS.log` (KB_05, именование)."""

    def __post_init__(self) -> None:
        """Проверяет параметры: некорректные — баг вызывающего, значит ValueError."""
        if self.telemetry_stride < 0:
            raise ValueError(f"telemetry_stride={self.telemetry_stride} должен быть ≥ 0")
        if self.telemetry_limit is not None and self.telemetry_limit < 0:
            raise ValueError("telemetry_limit должен быть ≥ 0 либо None")
        for name, value in (
            ("ring_capacity", self.ring_capacity),
            ("queue_capacity", self.queue_capacity),
            ("batch_limit", self.batch_limit),
        ):
            if value < 1:
                raise ValueError(f"{name}={value} должен быть ≥ 1")
        if self.rotate_bytes is not None and self.rotate_bytes < 1:
            raise ValueError("rotate_bytes должен быть положительным либо None")
        if self.rotate_seconds is not None and self.rotate_seconds <= 0:
            raise ValueError("rotate_seconds должен быть положительным либо None")
        if self.keep_files is not None and self.keep_files < 1:
            raise ValueError("keep_files должен быть ≥ 1 либо None")
        for name, value in (
            ("poll_period_s", self.poll_period_s),
            ("flush_period_s", self.flush_period_s),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} должен быть положительным")
        if not self.prefix:
            raise ValueError("prefix не может быть пустым")
        # Файл журнала обязан остаться чисто ASCII: его открывают Блокнотом
        # и импортируют в Excel, а кодировку по умолчанию там выбирает ОС.
        for name, text in (
            ("app_version", self.app_version),
            ("device_model", self.device_model),
            ("firmware", self.firmware),
            ("prefix", self.prefix),
        ):
            if text is not None and not text.isascii():
                raise ValueError(
                    f"{name}={text!r} содержит не-ASCII: файл журнала обязан остаться "
                    "чисто ASCII, иначе он нечитаем в Блокноте и в Excel на Windows"
                )


@dataclass(frozen=True, slots=True)
class PacketRecord:
    """Одна запись журнала: байты и, если получилось, их расшифровка."""

    seq: int
    """Сквозной номер, присвоенный на входе — **до** очереди и до записи."""

    direction: Direction
    t_mono: float
    """Момент `perf_counter`: у RX — снятый транспортом сразу после `recvfrom`."""

    data: bytes
    """Сырая датаграмма целиком. Пусто у записей `NOTE`."""

    decoded: str
    """Короткая расшифровка либо пометка о том, почему её нет."""

    @property
    def id_fc(self) -> tuple[int, int] | None:
        """Пара (ID, FC) или None, если байтов на неё не хватило."""
        return codec.classify(self.data)


@dataclass(frozen=True)
class PacketLogStats:
    """Снимок состояния журнала. Читается из потока UI."""

    records_in: int
    """Сколько записей принято на входе, включая события `NOTE`."""

    records_written: int
    """Сколько записей отдано файлу."""

    bytes_written: int
    telemetry_seen: int
    """Сколько датаграмм, выглядящих телеметрией, прошло через вход."""

    telemetry_admitted: int
    """Сколько из них фильтр пропустил в очередь."""

    telemetry_skipped: int
    """Сколько отсеяно фильтром: `stride` или исчерпанный лимит."""

    dropped_queue_full: int
    """Записи, вытесненные из очереди отставшим потоком журнала."""

    lost_records: int
    """Сколько вытесненных записей отмечено строками `LostRecords`."""

    decode_errors: int
    """Расшифровок, упавших с исключением. Байты при этом записаны."""

    queue_depth: int
    ring_size: int
    files: int
    path: Path | None
    error: str | None
    """Причина остановки записи в файл. Кольцевой буфер при этом жив."""


# --------------------------------------------------------------------------------------
# Расшифровка
# --------------------------------------------------------------------------------------


def _ascii(text: str) -> str:
    """Приводит строку к одной ASCII-строке без разделителя колонок.

    Замена, а не отказ: расшифровка не имеет права испортить запись байтов.
    """
    flat = text.replace(SEPARATOR, ",").replace("\r", " ").replace("\n", " ")
    return flat.encode("ascii", "replace").decode("ascii")


def _summarize(
    data: bytes,
    key: tuple[int, int],
    profile: DeviceProfile,
    out: MeasurementFrame | None,
) -> str:
    """Краткая сводка по разобранному ответу прибора.

    Разбор делает `codec`, а не журнал: своя реализация означала бы вторую
    трактовку протокола, расходящуюся с рабочей незаметно для обеих.
    """
    if key == (codec.ID_MODE, codec.FC_STREAM):
        # `parse_any` создал бы буфер кадра на каждую датаграмму; в полном
        # режиме это 2000 аллокаций в секунду на пустом месте.
        result = codec.parse_measurement(data, profile, out=out)
    else:
        result = codec.parse_any(data, profile)
    if result.error is not None:
        return f"ParseError {result.error.kind.name}"

    value = result.value
    if isinstance(value, MeasurementFrame):
        channels, fbg = value.freq_ghz.shape
        return f"ch={channels} filled={channels * fbg - value.missing}"
    if isinstance(value, ModuleParams):
        speed = "unknown" if value.speed_hz is None else f"{value.speed_hz}Hz"
        return (
            f"speed={speed} ch={value.channels} "
            f"fbg={value.fbg_per_channel} gap={value.peak_gap_ghz}GHz"
        )
    if isinstance(value, SweepConfig):
        return (
            f"start={value.start_ghz}GHz stop={value.stop_ghz}GHz "
            f"step={value.step_param} adc_step={value.adc_step_param}"
        )
    if isinstance(value, AdcBlock):
        return f"ch={value.channel + 1} points={value.points}"
    if isinstance(value, DebugResponse):
        points = value.blocks[0].points if value.blocks else 0
        return f"channels={value.channels} points={points}"
    if isinstance(value, UndocumentedResponse):
        # Смысл полей неизвестен (N17), поэтому только их количество.
        return f"words={len(value.words)}"
    if isinstance(value, tuple):
        return f"channels={len(value)}"
    # bool проверяется до int: bool — его подкласс, и подтверждение записи
    # иначе превратилось бы в «1».
    if isinstance(value, bool):
        return "ok" if value else "refused"
    if isinstance(value, int):
        if key == (codec.ID_READ, codec.FC_VERSION):
            return codec.format_version(value)
        return str(value)
    return ""


def describe(
    data: bytes,
    direction: Direction,
    profile: DeviceProfile,
    *,
    out: MeasurementFrame | None = None,
) -> str:
    """Короткая расшифровка датаграммы: «ReadModuleParams», «Telemetry ch=4 filled=2».

    Отправленные команды получают только имя: обратного разбора собственных
    команд в кодеке нет, а придумывать его здесь значило бы завести вторую
    трактовку полей (KB_05 №10). Байты команды видны в соседней колонке.

    Принятое расшифровывается через `codec`: имя пары плюс сводка по значению
    либо вид ошибки разбора. Ошибка — нормальный результат, а не отказ:
    именно неразобравшиеся датаграммы журнал и заводился показывать.

    `out` — переиспользуемый буфер кадра телеметрии, как в `parse_measurement`.
    """
    key = codec.classify(data)
    if key is None:
        return f"NoHeader len={len(data)}"
    names = TX_NAMES if direction is Direction.TX else RX_NAMES
    name = names.get(key)
    if name is None:
        return f"UnknownCommand {key[0]:02X} {key[1]:02X}"
    if direction is Direction.TX:
        return name
    summary = _summarize(data, key, profile, out)
    return f"{name} {summary}" if summary else name


# --------------------------------------------------------------------------------------
# Формат файла
# --------------------------------------------------------------------------------------


def format_hex(data: bytes) -> str:
    """Байты через пробел, в верхнем регистре — как во всех наших записях.

    Не обрезается никогда: правило KB_05 №3 требует сырых байтов целиком.
    Ответ `30 03` даёт строку в 61 КБ, и это осознанная цена — такие ответы
    приходят по одному на команду, а не потоком.
    """
    return data.hex(" ").upper()


def format_id_fc(data: bytes) -> str:
    """Колонка `id_fc`: «10 04» либо «--», если пары в датаграмме нет.

    Одной колонкой, а не двумя: Excel превратил бы отдельный «10» в число
    десять, хотя это шестнадцатеричный 0x10.
    """
    key = codec.classify(data)
    return NO_ID_FC if key is None else f"{key[0]:02X} {key[1]:02X}"


def format_record(record: PacketRecord, t_local: str) -> str:
    """Строка файла. Порядок колонок — `COLUMNS`."""
    return (
        SEPARATOR.join(
            (
                str(record.seq),
                record.direction.value,
                f"{record.t_mono:.{TIME_DECIMALS}f}",
                t_local,
                str(len(record.data)),
                format_id_fc(record.data),
                format_hex(record.data),
                _ascii(record.decoded),
            )
        )
        + "\n"
    )


def build_header(
    profile: DeviceProfile,
    config: PacketLogConfig,
    *,
    t_wall_start: datetime,
    t_mono_start: float,
    t_wall_file: datetime,
    part: int,
    note: str | None = None,
) -> str:
    """Шапка файла: строка имён колонок, затем блок комментариев.

    Порядок тот же, что у файла измерений (Р43), и по той же причине:
    `numpy.genfromtxt(names=True)` берёт имена из первой строки, а
    `pandas.read_csv(sep=';', comment='#')` пропускает комментарии. Журнал
    читают в основном глазами, но раз формат всё равно табличный, пусть
    он читается и инструментами.
    """
    serial = "unknown" if config.serial is None else str(config.serial)
    limit = "none" if config.telemetry_limit is None else str(config.telemetry_limit)
    lines = [
        f"fbg-interrogator packet log {config.app_version}",
        f"device={config.device_model} sn={serial} fw={config.firmware or 'unknown'}",
        f"t_wall_start={t_wall_start.isoformat()} t_mono_start={t_mono_start:.{TIME_DECIMALS}f}",
        f"t_wall_file={t_wall_file.isoformat()} file_part={part}",
        "t_mono=perf_counter_seconds t_local=local_clock_HH:MM:SS.mmm",
        f"telemetry_stride={config.telemetry_stride} telemetry_limit={limit}",
        "telemetry_stride=0 means telemetry is not logged; commands and replies always are",
        "raw bytes are logged before decoding and never truncated",
        'decode is best effort: "ParseError ..." and "DecodeFailed ..." keep the bytes',
        "dir=NOTE lines are log events, not device traffic",
        "ascii only on purpose: this file is opened in Notepad and imported into Excel",
    ]
    if note is not None:
        lines.append(note)
    comments = "".join(f"# {_ascii(line)}\n" for line in lines)
    return SEPARATOR.join(COLUMNS) + "\n" + comments


def filter_records(
    records: Iterable[PacketRecord],
    *,
    direction: Direction | None = None,
    id_fc: tuple[int, int] | None = None,
    t_from: float | None = None,
    t_to: float | None = None,
) -> tuple[PacketRecord, ...]:
    """Отбор записей: по направлению, по паре (ID, FC), по интервалу `t_mono`.

    Границы интервала включительные. Условия комбинируются через «и».
    """
    selected = []
    for record in records:
        if direction is not None and record.direction is not direction:
            continue
        if id_fc is not None and record.id_fc != id_fc:
            continue
        if t_from is not None and record.t_mono < t_from:
            continue
        if t_to is not None and record.t_mono > t_to:
            continue
        selected.append(record)
    return tuple(selected)


class _LocalClock:
    """Локальное время `HH:MM:SS.mmm` по метке `perf_counter`.

    Привязка к настенным часам снимается один раз (Р45), поэтому колонка
    согласована с `t_mono` и не дёргается вслед за NTP.

    Строка целой секунды кэшируется: `localtime` и `strftime` стоят несколько
    микросекунд, а в полном режиме журнал форматирует 2000 записей в секунду,
    из которых новая секунда наступает один раз.
    """

    def __init__(self, wall_offset: float) -> None:
        self._offset = wall_offset
        self._whole = -1
        self._prefix = "00:00:00"

    def format(self, t_mono: float) -> str:
        """Форматирует метку. Не потокобезопасен: зовётся только из потока журнала."""
        seconds = t_mono + self._offset
        whole = int(seconds)
        if whole != self._whole:
            self._whole = whole
            self._prefix = time.strftime("%H:%M:%S", time.localtime(whole))
        return f"{self._prefix}.{int((seconds - whole) * 1000.0):03d}"


# --------------------------------------------------------------------------------------
# Журнал
# --------------------------------------------------------------------------------------


class PacketLog:
    """Журнал обмена с прибором.

    Использование::

        log = PacketLog(profile, PacketLogConfig(directory=Path("logs")))
        log.start()
        transport = UdpTransport(endpoint, log.log_rx)   # либо развилка рядом
        ...
        log.stop()

    `log_rx` имеет ровно сигнатуру `TapCallback`, поэтому подключается к tap
    транспорта напрямую. Когда у tap уже есть потребитель, ставится развилка::

        transport = UdpTransport(endpoint, lambda data, t: (log.log_rx(data, t),
                                                            session_tap(data, t)))

    Отправленное журнал сам не видит — его сообщает вызывающий: `log_tx`
    рядом с `transport.send`.

    Тесты и одиночные шаги обходятся без потока: `open()` готовит файл,
    `pump()` переносит очередную пачку, `close()` дописывает и закрывает.
    """

    def __init__(
        self,
        profile: DeviceProfile,
        config: PacketLogConfig | None = None,
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._profile = profile
        self._config = config or PacketLogConfig()
        self._on_error = on_error
        self._frame_size = profile.frame_size
        self._buffer = MeasurementFrame.for_profile(profile)

        self._queue: deque[tuple[int, Direction, float, bytes, str | None]] = deque(
            maxlen=self._config.queue_capacity
        )
        self._ring: deque[PacketRecord] = deque(maxlen=self._config.ring_capacity)

        # Вход трогают два чужих потока — диспетчер транспорта и тот, кто
        # отправляет команду, — и оба меняют одни и те же счётчики политики.
        # Секция короткая: инкремент, сравнение, `append`.
        self._in_lock = threading.Lock()
        self._ring_lock = threading.Lock()

        self._seq = 0
        self._telemetry_seen = 0
        self._telemetry_admitted = 0
        self._telemetry_skipped = 0
        self._dropped = 0
        self._limit_announced = False

        self._last_seq = 0
        self._lost = 0
        self._records_written = 0
        self._decode_errors = 0
        self._bytes = 0

        self._file = None  # type: ignore[var-annotated]
        self._path: Path | None = None
        self._own_files: list[Path] = []
        self._stem = ""
        self._stem_index = 0
        self._files = 0
        self._file_bytes = 0
        self._file_opened_mono = 0.0
        self._last_flush_mono = 0.0
        self._error: str | None = None

        self._t_wall_start = datetime.now().astimezone()
        self._t_mono_start = time.perf_counter()
        self._clock = _LocalClock(time.time() - time.perf_counter())

        self._stop_flag = threading.Event()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None

    # --- Состояние ---------------------------------------------------------------------

    @property
    def config(self) -> PacketLogConfig:
        """Параметры журнала."""
        return self._config

    @property
    def path(self) -> Path | None:
        """Текущий файл. None — журнал пишется только в память."""
        return self._path

    @property
    def is_running(self) -> bool:
        """True, если поток журнала запущен."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def stats(self) -> PacketLogStats:
        """Снимок состояния."""
        with self._ring_lock:
            ring_size = len(self._ring)
        return PacketLogStats(
            records_in=self._seq,
            records_written=self._records_written,
            bytes_written=self._bytes,
            telemetry_seen=self._telemetry_seen,
            telemetry_admitted=self._telemetry_admitted,
            telemetry_skipped=self._telemetry_skipped,
            dropped_queue_full=self._dropped,
            lost_records=self._lost,
            decode_errors=self._decode_errors,
            queue_depth=len(self._queue),
            ring_size=ring_size,
            files=self._files,
            path=self._path,
            error=self._error,
        )

    def snapshot(self, limit: int | None = None) -> tuple[PacketRecord, ...]:
        """Копия кольцевого буфера, старейшие записи первыми.

        Неизменяемая копия под короткой блокировкой: читатель (панель журнала,
        чат №10) не должен видеть, как кольцо меняется у него под руками.
        Блокировка здесь допустима — в приёмном потоке её нет, а писатель
        кольца ровно один, поток журнала.
        """
        with self._ring_lock:
            if limit is None or limit >= len(self._ring):
                return tuple(self._ring)
            return tuple(self._ring)[-limit:]

    # --- Вход --------------------------------------------------------------------------

    def log_rx(self, data: bytes, t_mono: float) -> None:
        """Принятая датаграмма. Сигнатура совпадает с `TapCallback` транспорта.

        Дёшево по построению: фильтр телеметрии — сравнение двух байт и длины,
        дальше `append` в очередь. Ни hex, ни разбора, ни файлов (KB_05 №23).
        """
        with self._in_lock:
            if self._looks_like_telemetry(data) and not self._admit_telemetry():
                return
            self._enqueue(Direction.RX, t_mono, data, None)
        self._wakeup.set()

    def log_tx(self, data: bytes, t_mono: float | None = None) -> None:
        """Отправленная датаграмма. Метка по умолчанию снимается здесь же."""
        stamp = time.perf_counter() if t_mono is None else t_mono
        with self._in_lock:
            self._note_start_command(data)
            self._enqueue(Direction.TX, stamp, data, None)
        self._wakeup.set()

    def _looks_like_telemetry(self, data: bytes) -> bool:
        """True, если датаграмма выглядит штатным кадром телеметрии.

        Требуется и пара `30 02`, и точная длина: кадр телеметрии неверной
        длины — аномалия, и отфильтровать её настройкой объёма нельзя.
        """
        return len(data) == self._frame_size and data[:2] == TELEMETRY_PREFIX

    def _admit_telemetry(self) -> bool:
        """Решает судьбу кадра телеметрии. Вызывается под `_in_lock`."""
        self._telemetry_seen += 1
        stride = self._config.telemetry_stride
        if stride == 0 or (self._telemetry_seen - 1) % stride != 0:
            self._telemetry_skipped += 1
            return False
        limit = self._config.telemetry_limit
        if limit is not None and self._telemetry_admitted >= limit:
            self._telemetry_skipped += 1
            if not self._limit_announced:
                self._limit_announced = True
                self._enqueue(
                    Direction.NOTE,
                    time.perf_counter(),
                    b"",
                    f"TelemetryLimitReached limit={limit}",
                )
            return False
        self._telemetry_admitted += 1
        return True

    def _note_start_command(self, data: bytes) -> None:
        """Сбрасывает отсчёт телеметрии, увидев отправленный Start (30 02).

        О старте потока журнал узнаёт из самого обмена: внешняя координация
        для этого не нужна, а без сброса лимит, выбранный один раз, сделал бы
        второй запуск потока невидимым.
        """
        if not self._config.telemetry_limit_resets_on_start:
            return
        if data[:2] == TELEMETRY_PREFIX:
            self._telemetry_seen = 0
            self._telemetry_admitted = 0
            self._limit_announced = False

    def _enqueue(self, direction: Direction, t_mono: float, data: bytes, note: str | None) -> None:
        """Кладёт запись в очередь. Вызывается под `_in_lock`."""
        self._seq += 1
        if len(self._queue) >= self._queue.maxlen:  # type: ignore[operator]
            # deque(maxlen=…) вытеснит старейшую сам; здесь только учёт.
            self._dropped += 1
        self._queue.append((self._seq, direction, t_mono, data, note))

    # --- Жизненный цикл ----------------------------------------------------------------

    def open(self) -> None:
        """Открывает первый файл. При `directory = None` не делает ничего.

        `OSError` не перехватывается: не открывшаяся папка — отказ **старта**
        журнала, о котором вызывающий узнаёт сразу, а не через поле `error`.
        """
        if self._file is not None or self._config.directory is None:
            return
        self._config.directory.mkdir(parents=True, exist_ok=True)
        self._t_wall_start = datetime.now().astimezone()
        self._t_mono_start = time.perf_counter()
        self._clock = _LocalClock(time.time() - time.perf_counter())
        self._error = None
        self._open_file()

    def start(self) -> None:
        """Открывает журнал и запускает поток. Повторный вызов ничего не делает."""
        if self.is_running:
            return
        self.open()
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, name="fbg-packet-log", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Останавливает поток и закрывает файл, дописав остаток очереди."""
        self._stop_flag.set()
        self._wakeup.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._config.poll_period_s + 10.0)
        self._thread = None
        self.close()

    def close(self) -> None:
        """Дописывает остаток очереди и закрывает файл."""
        while self._queue and self.pump():
            pass
        file = self._file
        if file is None:
            self._path = None
            return
        try:
            file.flush()
        except OSError as exc:
            self._fail(exc)
        finally:
            self._close_file()

    def __enter__(self) -> "PacketLog":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # --- Перенос записей ---------------------------------------------------------------

    def pump(self) -> int:
        """Обрабатывает одну пачку записей. Возвращает их число.

        Исключений не бросает: ошибка записи закрывает файл и запоминает
        причину, а кольцевой буфер продолжает наполняться.
        """
        batch = self._take()
        if not batch:
            self._maybe_flush()
            return 0
        for seq, direction, t_mono, data, note in batch:
            self._handle(seq, direction, t_mono, data, note)
        self._maybe_flush()
        return len(batch)

    def _take(self) -> list[tuple[int, Direction, float, bytes, str | None]]:
        """Забирает из очереди до `batch_limit` записей одним заходом."""
        batch: list[tuple[int, Direction, float, bytes, str | None]] = []
        limit = self._config.batch_limit
        with self._in_lock:
            while self._queue and len(batch) < limit:
                batch.append(self._queue.popleft())
        return batch

    def _loop(self) -> None:
        """Поток журнала: разбирает очередь, спит, когда она пуста."""
        while not self._stop_flag.is_set():
            if self.pump() == 0:
                self._wakeup.wait(self._config.poll_period_s)
                self._wakeup.clear()

    def _handle(
        self, seq: int, direction: Direction, t_mono: float, data: bytes, note: str | None
    ) -> None:
        """Расшифровывает запись, кладёт в кольцо и пишет в файл."""
        if seq != self._last_seq + 1:
            # Разрыв номеров означает, что очередь вытеснила записи, пока
            # поток журнала отставал. Молчаливой потери быть не должно.
            lost = seq - self._last_seq - 1
            self._lost += lost
            self._emit(
                PacketRecord(
                    seq=self._last_seq + 1,
                    direction=Direction.NOTE,
                    t_mono=t_mono,
                    data=b"",
                    decoded=f"LostRecords count={lost}",
                )
            )
        self._last_seq = seq

        if note is not None:
            decoded = note
        else:
            try:
                decoded = describe(data, direction, self._profile, out=self._buffer)
            # Отказ расшифровки — ошибка расшифровки, а не причина потерять
            # байты (KB_05 №3). Ловится всё: разбор чужих байтов может упасть
            # где угодно, а журнал обязан пережить это без потери записи.
            except Exception as exc:
                self._decode_errors += 1
                decoded = f"DecodeFailed {type(exc).__name__}"
        self._emit(
            PacketRecord(seq=seq, direction=direction, t_mono=t_mono, data=data, decoded=decoded)
        )

    def _emit(self, record: PacketRecord) -> None:
        """Кладёт готовую запись в кольцо и, если есть куда, в файл."""
        with self._ring_lock:
            self._ring.append(record)
        if self._file is None:
            return
        try:
            self._rotate_if_needed()
            self._write(format_record(record, self._clock.format(record.t_mono)))
        except OSError as exc:
            self._fail(exc)
            return
        self._records_written += 1

    # --- Файлы -------------------------------------------------------------------------

    def _write(self, text: str) -> None:
        """Отдаёт текст файлу одним вызовом и учитывает объём."""
        assert self._file is not None
        # Строго ASCII: файл открывают Блокнотом и импортируют в Excel,
        # а кодировку по умолчанию там выбирает ОС. Поля конфигурации
        # проверены в `__post_init__`, расшифровка приведена `_ascii`,
        # поэтому недостижимого `strict` здесь нет — есть гарантия.
        payload = text.encode("ascii")
        self._file.write(payload)
        self._file_bytes += len(payload)
        self._bytes += len(payload)

    def _maybe_flush(self) -> None:
        """Сбрасывает буфер не чаще `flush_period_s`."""
        if self._file is None:
            return
        now = time.monotonic()
        if now - self._last_flush_mono < self._config.flush_period_s:
            return
        self._last_flush_mono = now
        try:
            self._file.flush()
        except OSError as exc:
            self._fail(exc)

    def _rotate_if_needed(self) -> None:
        """Открывает новый файл, если исчерпан лимит по размеру или времени."""
        rotate_bytes = self._config.rotate_bytes
        rotate_seconds = self._config.rotate_seconds
        by_size = rotate_bytes is not None and self._file_bytes >= rotate_bytes
        by_time = (
            rotate_seconds is not None
            and time.monotonic() - self._file_opened_mono >= rotate_seconds
        )
        if by_size or by_time:
            self._close_file()
            self._open_file()

    def _open_file(self) -> None:
        """Открывает очередной файл, пишет шапку и подчищает старые свои файлы."""
        directory = self._config.directory
        assert directory is not None
        now = datetime.now().astimezone()
        path = self._unique_path(now)
        # Двоичный режим: перевод строки одинаков на всех платформах,
        # а объём файла считается точно.
        self._file = path.open("wb", buffering=FILE_BUFFER_BYTES)
        self._path = path
        self._own_files.append(path)
        self._file_bytes = 0
        self._file_opened_mono = time.monotonic()
        self._last_flush_mono = self._file_opened_mono
        self._files += 1
        self._write(
            build_header(
                self._profile,
                self._config,
                t_wall_start=self._t_wall_start,
                t_mono_start=self._t_mono_start,
                t_wall_file=now,
                part=self._files,
            )
        )
        self._trim_old_files()

    def _unique_path(self, now: datetime) -> Path:
        """Имя `packets_YYYYMMDD_HHMMSS.log`; при совпадении добавляется номер.

        Номер дополняется нулями до трёх знаков (KB_05 №27, Р46): без этого
        `_10` сортируется перед `_2`, и порядок частей журнала перестаёт
        читаться из имён. Опираться на время создания файла нельзя — на Windows
        его разрешение грубее интервала между ротациями, что и вскрыл CI
        чата №7. Порядок дублируется полем `file_part` в шапке.

        Номер берётся из **счётчика, который не откатывается**, а не только
        из проверки существования файла. Разница видна ровно в связке
        с `keep_files`: удалённая часть освобождает своё имя, и подбор
        по `exists()` выдал бы его следующему файлу — алфавитный порядок
        разошёлся бы с порядком создания молча. Проверка `exists()` при этом
        остаётся: она защищает от чужого файла с тем же именем, оставшегося
        от прошлого запуска.
        """
        directory = self._config.directory
        assert directory is not None
        stem = f"{self._config.prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
        if stem != self._stem:
            self._stem = stem
            self._stem_index = 0
        while True:
            self._stem_index += 1
            name = f"{stem}.log" if self._stem_index == 1 else f"{stem}_{self._stem_index:03d}.log"
            path = directory / name
            if not path.exists():
                return path

    def _trim_old_files(self) -> None:
        """Удаляет самые ранние файлы **этого** экземпляра сверх `keep_files`.

        Только свои: удалять по маске значило бы стирать журналы прошлых
        запусков, которые оператор мог сохранить намеренно. Цена решения —
        накопление между запусками; она предсказуема, а стёртый чужой файл
        невосстановим.
        """
        keep = self._config.keep_files
        if keep is None:
            return
        while len(self._own_files) > keep:
            victim = self._own_files.pop(0)
            # Не удалось удалить — файл занят или уже исчез; журнал это
            # переживает, ронять из-за уборки нечего.
            with contextlib.suppress(OSError):
                victim.unlink()

    def _close_file(self) -> None:
        """Закрывает текущий файл, не трогая счётчики."""
        file = self._file
        self._file = None
        if file is None:
            return
        try:
            file.close()
        except OSError as exc:
            self._fail(exc)

    def _fail(self, exc: OSError) -> None:
        """Останавливает запись в файл, сохранив причину.

        Поток при этом **не** завершается, в отличие от `recorder` (Р47):
        журнал — диагностика, и кольцевой буфер обязан продолжать работать,
        даже когда писать на диск некуда. Приём данных и запись измерений
        отказ журнала не касается вовсе: они в других потоках.
        """
        if self._error is None:
            self._error = f"{type(exc).__name__}: {exc}"
        file = self._file
        self._file = None
        if file is not None:
            with contextlib.suppress(OSError):
                file.close()
        if self._on_error is not None:
            self._on_error(self._error)

    # --- Экспорт -----------------------------------------------------------------------

    def export(
        self,
        path: Path,
        *,
        direction: Direction | None = None,
        id_fc: tuple[int, int] | None = None,
        t_from: float | None = None,
        t_to: float | None = None,
    ) -> int:
        """Выгружает записи кольца в файл с фильтром. Возвращает их число.

        Экспортируется то, что видно в панели, — кольцевой буфер. Файлы
        журнала уже лежат на диске в том же формате, и пересобирать их
        незачем: для них есть `sorted(glob(...))` и обычный текстовый поиск.
        """
        records = filter_records(
            self.snapshot(), direction=direction, id_fc=id_fc, t_from=t_from, t_to=t_to
        )
        clock = _LocalClock(time.time() - time.perf_counter())
        parts = [
            build_header(
                self._profile,
                self._config,
                t_wall_start=self._t_wall_start,
                t_mono_start=self._t_mono_start,
                t_wall_file=datetime.now().astimezone(),
                part=0,
                note=_export_note(direction, id_fc, t_from, t_to),
            )
        ]
        parts += [format_record(record, clock.format(record.t_mono)) for record in records]
        path.write_bytes("".join(parts).encode("ascii"))
        return len(records)


def _export_note(
    direction: Direction | None,
    id_fc: tuple[int, int] | None,
    t_from: float | None,
    t_to: float | None,
) -> str:
    """Строка шапки экспорта, описывающая применённый фильтр."""
    parts: list[str] = []
    parts.append(f"dir={direction.value if direction is not None else 'any'}")
    parts.append(f"id_fc={f'{id_fc[0]:02X} {id_fc[1]:02X}' if id_fc is not None else 'any'}")
    parts.append(f"t_from={'any' if t_from is None else f'{t_from:.{TIME_DECIMALS}f}'}")
    parts.append(f"t_to={'any' if t_to is None else f'{t_to:.{TIME_DECIMALS}f}'}")
    return "export filter: " + " ".join(parts)


def records_from_file(path: Path) -> Sequence[str]:
    """Строки записей файла: без имён колонок и без комментариев.

    Нужна тестам и постобработке; читает в ASCII намеренно — если файл
    перестанет быть ASCII, это должно упасть здесь, а не в Блокноте.
    """
    lines = path.read_text(encoding="ascii").splitlines()
    return [line for line in lines[1:] if line and not line.startswith("#")]
