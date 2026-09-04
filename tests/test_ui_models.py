"""Тесты моделей панелей и лестницы диагностики. Qt здесь нет.

Всё, что панель показывает, строится чистыми функциями из снимка, поэтому
проверяется без окна. Виджету остаётся расставить готовые строки по ячейкам,
и вот это уже проверяется отдельно, тестами с маркером `ui`.
"""

import math
import time

import numpy as np
import pytest

from fbg.core.calibration import ReadingStatus, Sensor, SensorReading, SensorType
from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting, ModuleParams, SweepConfig
from fbg.core.pipeline import PipelineMetrics
from fbg.core.profile import DeviceProfile
from fbg.core.session import (
    DeviceConfig,
    SessionError,
    SessionErrorKind,
    SessionState,
    SessionStats,
    StreamRecoveryOutcome,
)
from fbg.core.transport import TransportStats
from fbg.io import packet_log as packet_log_module
from fbg.io.packet_log import Direction, PacketLogStats, PacketRecord
from fbg.ui import diagnostics, models, texts
from fbg.ui.diagnostics import Verdict
from fbg.ui.models import AppSnapshot, ProfileDifference
from fbg.ui.texts import Tone

PROFILE = DeviceProfile()

DEVICE = DeviceConfig(
    version_raw=410,
    serial=94401220,
    module=ModuleParams(
        speed_code=0x00CA, speed_hz=2000, channels=4, fbg_per_channel=30, peak_gap_ghz=30
    ),
    sweep=SweepConfig.from_params(1, 2, 5101, 2, PROFILE),
    channels=(
        ChannelSetup(0, None, GainSetting(False, 5)),
        ChannelSetup(1, 4001, GainSetting(True, 4)),
    ),
)


def snapshot(**kwargs: object) -> AppSnapshot:
    """Снимок с умолчаниями: тесты задают только то, что проверяют."""
    base: dict[str, object] = {
        "endpoint": Endpoint(),
        "profile": PROFILE,
        "state": SessionState.DISCONNECTED,
    }
    base.update(kwargs)
    return AppSnapshot(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Состояние сессии
# --------------------------------------------------------------------------------------


def test_подписи_есть_у_всех_состояний_автомата() -> None:
    """Состояние без подписи означало бы пустой индикатор при живой сессии."""
    assert set(texts.STATE_LABELS) == set(SessionState)


def test_degraded_и_reconnecting_выглядят_иначе_чем_disconnected() -> None:
    """Связь потеряна и восстанавливается — это не «не подключались».

    Одинаковый вид означал бы, что оператор за стендом читает идущий backoff
    как «кнопка не сработала» и жмёт её ещё раз.
    """
    disconnected = models.state_view(snapshot(state=SessionState.DISCONNECTED))
    degraded = models.state_view(snapshot(state=SessionState.DEGRADED))
    reconnecting = models.state_view(snapshot(state=SessionState.RECONNECTING))
    assert disconnected.tone is Tone.NEUTRAL
    assert degraded.tone is Tone.WARN and reconnecting.tone is Tone.WARN
    assert degraded.tone is not disconnected.tone
    assert texts.TONE_COLORS[Tone.WARN] != texts.TONE_COLORS[Tone.NEUTRAL]


def test_три_этапа_восстановления_потока_различимы_в_подписи() -> None:
    """Оператор различает потерю, самовозобновление и наш перезапуск."""
    lost = models.state_view(
        snapshot(
            state=SessionState.DEGRADED,
            session=SessionStats(stream_resume_wait_in_s=12.3),
        )
    )
    resumed = models.state_view(
        snapshot(
            state=SessionState.STREAMING,
            stream_recovery_outcome=StreamRecoveryOutcome.RESUMED,
        )
    )
    restarted = models.state_view(
        snapshot(
            state=SessionState.STREAMING,
            stream_recovery_outcome=StreamRecoveryOutcome.RESTARTED,
        )
    )

    assert texts.STREAM_CONNECTION_LOST in lost.text
    assert "12.3" in lost.text
    assert texts.STREAM_RECOVERED_RESUMED in resumed.text
    assert texts.STREAM_RECOVERED_RESTARTED in restarted.text
    assert len({lost.text, resumed.text, restarted.text}) == 3


def test_активная_попытка_восстановления_видна_и_отменяема() -> None:
    """Во время сетевого опроса UI не теряет номер активной попытки."""
    view = models.state_view(
        snapshot(
            state=SessionState.DEGRADED,
            session=SessionStats(recovery_attempt=2),
        )
    )
    assert "попытка 2 выполняется" in view.text
    assert texts.RECOVERY_CANCEL in view.text


def test_reconnecting_показывает_номер_срок_и_отмену() -> None:
    """Backoff не выглядит как зависание: видны попытка, срок и способ отмены."""
    view = models.state_view(
        snapshot(
            state=SessionState.RECONNECTING,
            session=SessionStats(recovery_attempt=3, next_attempt_in_s=1.25),
        )
    )
    assert "попытка 3" in view.text
    assert "1.2" in view.text
    assert texts.RECOVERY_CANCEL in view.text


def test_расхождение_конфигурации_видно_в_подписи() -> None:
    """Прибор мог быть перезагружен по питанию и потерять записанное."""
    view = models.state_view(
        snapshot(state=SessionState.IDLE, config_mismatch=("развёртка: было …",))
    )
    assert texts.CONFIG_MISMATCH in view.text


def test_строка_состояния_собирает_темп_журнал_и_запись() -> None:
    """Строка состояния окна — то, что видно, не открывая вкладок."""
    metrics = PipelineMetrics(
        frames=10,
        parse_errors=0,
        frame_rate_hz=1999.7,
        expected_rate_hz=2000.0,
        loss_estimate=0.00015,
        filled_by_channel=(2, 0, 0, 0),
        ingest_lag_s=0.0001,
        ui_latency_s=0.005,
        ui_updates=3,
        ui_gates=3,
        history_frames=20000,
        history_used=10,
        history_bytes=1,
        evicted=0,
    )
    line = models.status_line(
        snapshot(
            state=SessionState.STREAMING,
            metrics=metrics,
            log=PacketLogStats(
                records_in=7,
                records_written=7,
                bytes_written=1,
                telemetry_seen=0,
                telemetry_admitted=0,
                telemetry_skipped=0,
                dropped_queue_full=0,
                lost_records=0,
                decode_errors=0,
                queue_depth=0,
                ring_size=7,
                files=1,
                path=None,
                error=None,
            ),
        )
    )
    assert "Идёт поток телеметрии" in line
    assert "1999.7 Гц" in line
    assert "журнал: 7" in line


# --------------------------------------------------------------------------------------
# Панель прибора
# --------------------------------------------------------------------------------------


def _rows(sections: tuple[models.InfoSection, ...], title: str) -> dict[str, models.InfoRow]:
    """Строки одной группы по подписи."""
    section = next(item for item in sections if item.title == title)
    return {row.label: row for row in section.rows}


def test_до_подключения_показывается_профиль_настроек() -> None:
    """Прибор не опрошен — показывается то, что помнят настройки, и честный прочерк."""
    sections = models.device_sections(snapshot())
    device = _rows(sections, texts.SECTION_DEVICE)
    assert device[texts.ROW_SERIAL].value == texts.UNKNOWN
    assert device[texts.ROW_FIRMWARE].value == texts.UNKNOWN
    sweep = _rows(sections, texts.SECTION_SWEEP)
    assert sweep[texts.ROW_SPEED].value == "2 000"
    assert "параметр 1" in sweep[texts.ROW_START].value


def test_после_подключения_показывается_то_что_сказал_прибор() -> None:
    """Значения прибора и значения профиля не смешиваются в одной строке.

    Именно на их расхождении держится сверка геометрии: если бы панель
    показывала смесь, расхождение стало бы невидимым.
    """
    sections = models.device_sections(snapshot(state=SessionState.IDLE, device=DEVICE))
    device = _rows(sections, texts.SECTION_DEVICE)
    assert device[texts.ROW_SERIAL].value == "94401220"
    assert device[texts.ROW_FIRMWARE].value == "4.10"
    sweep = _rows(sections, texts.SECTION_SWEEP)
    assert "196 249 ГГц" in sweep[texts.ROW_START].value
    assert "1527.6127 нм" in sweep[texts.ROW_START].value
    assert sweep[texts.ROW_ADC_POINTS].value == "2 551"


def test_каналы_показываются_с_порогом_и_усилением() -> None:
    """Пороги и усиления — то, ради чего в панель заглядывают перед запуском."""
    sections = models.device_sections(snapshot(state=SessionState.IDLE, device=DEVICE))
    channels = _rows(sections, texts.SECTION_CHANNELS)
    # Номер канала в протоколе 0-based, человеку показывается 1-based.
    assert "авто (FFFF)" in channels["Канал 1"].value
    assert "авто, уровень 5" in channels["Канал 1"].value
    assert "4001" in channels["Канал 2"].value
    assert "вручную, уровень 4" in channels["Канал 2"].value


def test_неподтверждённое_значение_помечается() -> None:
    """KB_05 №16: успех записи без read-back'а успехом не объявляется."""
    sections = models.device_sections(
        snapshot(state=SessionState.IDLE, device=DEVICE, unconfirmed=frozenset({"threshold:1"}))
    )
    channels = _rows(sections, texts.SECTION_CHANNELS)
    assert channels["Канал 2"].note == "не подтверждено"
    assert channels["Канал 1"].note == ""


def test_оценка_потерь_помечена_оценкой() -> None:
    """В протоколе нет ни счётчиков кадров, ни номеров: это оценка, не измерение."""
    metrics = PipelineMetrics(
        frames=100,
        parse_errors=1,
        frame_rate_hz=1900.0,
        expected_rate_hz=2000.0,
        loss_estimate=0.05,
        filled_by_channel=(2, 0, 0, 0),
        ingest_lag_s=0.0,
        ui_latency_s=0.0,
        ui_updates=0,
        ui_gates=0,
        history_frames=20000,
        history_used=100,
        history_bytes=1,
        evicted=17,
    )
    quality = _rows(models.device_sections(snapshot(metrics=metrics)), texts.SECTION_QUALITY)
    assert quality[texts.ROW_LOSS].value == "5.00 %"
    assert quality[texts.ROW_LOSS].note == texts.LOSS_IS_AN_ESTIMATE


def test_вытеснение_из_кольца_помечено_как_не_потеря() -> None:
    """Правило KB_05 №22: `evicted` растёт при любом потоке и потерей не является."""
    metrics = PipelineMetrics(
        frames=100,
        parse_errors=0,
        frame_rate_hz=2000.0,
        expected_rate_hz=2000.0,
        loss_estimate=0.0,
        filled_by_channel=(2,),
        ingest_lag_s=0.0,
        ui_latency_s=0.0,
        ui_updates=0,
        ui_gates=0,
        history_frames=20000,
        history_used=100,
        history_bytes=1,
        evicted=100001,
    )
    quality = _rows(models.device_sections(snapshot(metrics=metrics)), texts.SECTION_QUALITY)
    assert quality[texts.ROW_EVICTED].note == texts.EVICTED_IS_NOT_LOSS


def test_счётчики_связи_попадают_в_панель() -> None:
    """Таймауты, повторы, orphan_responses и tap_errors — по требованию чата."""
    stats = SessionStats(retries=3, timeouts=2, orphan_responses=5, tap_errors=1)
    transport = TransportStats(commands_sent=9, datagrams_received=8, foreign_datagrams=4)
    quality = _rows(
        models.device_sections(snapshot(session=stats, transport=transport)),
        texts.SECTION_QUALITY,
    )
    assert quality[texts.ROW_TIMEOUTS].value == "2"
    assert quality[texts.ROW_RETRIES].value == "3"
    assert quality[texts.ROW_ORPHANS].value == "5"
    assert quality[texts.ROW_TAP_ERRORS].value == "1"
    assert quality[texts.ROW_FOREIGN].value == "4"


def test_расхождение_профиля_превращается_в_текст() -> None:
    """Сообщение называет поле, оба значения и то, что настройки не тронуты."""
    lines = models.profile_mismatch_lines(
        snapshot(profile_mismatch=(ProfileDifference("fbg_per_channel", 25, 30),))
    )
    assert lines
    assert "fbg_per_channel" in lines[1]
    assert "25" in lines[1] and "30" in lines[1]
    assert "не перезаписаны" in lines[0]
    assert models.profile_mismatch_lines(snapshot()) == ()


# --------------------------------------------------------------------------------------
# Таблица журнала
# --------------------------------------------------------------------------------------


def record(
    seq: int = 1, data: bytes = b"\x10\x04\x04\x00", t_mono: float = 12.345678
) -> PacketRecord:
    """Запись журнала для теста."""
    return PacketRecord(
        seq=seq, direction=Direction.TX, t_mono=t_mono, data=data, decoded="ReadModuleParams"
    )


def test_время_панели_совпадает_с_временем_файла() -> None:
    """Панель и файл журнала обязаны показывать одно и то же время.

    Формат `t_local` реализован в двух местах — в файле его пишет
    `packet_log`, в ячейке рисует панель, — поэтому совпадение закреплено
    тестом: разъехаться молча они не должны.
    """
    offset = time.time() - time.perf_counter()
    clock = packet_log_module._LocalClock(offset)
    for t_mono in (0.0, 12.345678, 1234.999):
        assert models.local_time(t_mono, offset) == clock.format(t_mono)


def test_ячейки_строки_журнала() -> None:
    """Порядок колонок тот же, что в файле: hex до расшифровки (KB_05 №3)."""
    row = models.packet_row(record(), wall_offset=0.0)
    assert len(row) == len(texts.LOG_COLUMNS)
    assert row[0] == "1"
    assert row[1] == "TX"
    assert row[2] == "12.345678"
    assert row[4] == "4"
    assert row[5] == "10 04"
    assert row[6] == "10 04 04 00"
    assert row[7] == "ReadModuleParams"


def test_несуществующая_колонка_это_баг_вызывающего() -> None:
    """Номер колонки задаёт таблица; выход за границы — программная ошибка."""
    with pytest.raises(IndexError):
        models.packet_cell(record(), models.LOG_COLUMN_COUNT, 0.0)


def test_короткие_байты_показываются_целиком() -> None:
    """Обычная команда в ячейку помещается вся."""
    data = bytes(range(models.HEX_DISPLAY_BYTES))
    assert models.format_hex_cell(data) == data.hex(" ").upper()


def test_длинные_байты_обрезаются_с_явной_пометкой() -> None:
    """Обрезается **показ**; в файле и в экспорте байты целы (KB_05 №3).

    Ответ `30 03` — это 20430 байт, то есть 61 КБ в одной ячейке: таблица
    на таком встаёт, а прочитать его глазами всё равно нельзя. Пометка
    обязательна: молча укороченный hex читался бы как настоящая датаграмма.
    """
    data = bytes(20430)
    cell = models.format_hex_cell(data)
    assert cell.startswith("00 00 00")
    assert "20430" in cell
    assert len(cell) < 300


def test_список_пар_строится_по_встретившемуся() -> None:
    """В журнале интереснее всего пары, которых нет в списке известных команд."""
    records = (
        record(1, b"\x10\x04\x04\x00"),
        record(2, b"\x30\x02\x00\x00"),
        record(3, b"\x10\x04\x00\x0c"),
        record(4, b"\xaa"),
    )
    assert models.id_fc_choices(records) == ((0x10, 0x04), (0x30, 0x02))
    assert models.format_id_fc_pair((0x30, 0x02)) == "30 02"


def test_пара_разбирается_обратно_из_строки() -> None:
    """Данные элемента списка Qt хранит через `QVariant` и типы Python теряет.

    Кортеж возвращается списком, `StrEnum` — обычной строкой, поэтому в списках
    панели лежат строки, а разбор живёт здесь, где проверяется без окна.
    """
    for pair in ((0x10, 0x04), (0x30, 0x02), (0x20, 0x06)):
        assert models.parse_id_fc_pair(models.format_id_fc_pair(pair)) == pair


def test_имя_файла_экспорта_ascii() -> None:
    """Файл журнала ASCII, и имя выгрузки тоже (KB_05 №29)."""
    name = models.export_suggested_name()
    assert name.isascii() and name.startswith("packets_export_") and name.endswith(".log")


# --------------------------------------------------------------------------------------
# Диагностика
# --------------------------------------------------------------------------------------


def verdicts(diagnosis: diagnostics.Diagnosis) -> dict[str, Verdict]:
    """Вердикты по заголовкам пунктов."""
    return {check.title: check.verdict for check in diagnosis.checks}


def test_приём_не_на_нулевой_адрес_названо_вероятной_причиной() -> None:
    """Р29: bind на конкретный адрес — молчание при полностью исправной связи.

    Это самый дорогой класс отказа, потому что выглядит как «сеть не работает»,
    и единственный пункт лестницы, который проверяется кодом полностью.
    """
    bad = Endpoint(local_ip="192.168.0.14")
    assert verdicts(diagnostics.diagnose(bad))["Приём на 0.0.0.0"] is Verdict.SUSPECT
    good = Endpoint()
    assert verdicts(diagnostics.diagnose(good))["Приём на 0.0.0.0"] is Verdict.OK


def test_команды_уходят_ответов_нет_указывает_на_брандмауэр() -> None:
    """Риск R5: команды уходят и без правила, а ответы блокируются."""
    stats = TransportStats(commands_sent=4, bytes_sent=24)
    diagnosis = diagnostics.diagnose(
        Endpoint(), stats=stats, error=SessionError(SessionErrorKind.TIMEOUT, "нет ответа")
    )
    assert "Команды уходят, ответов нет" in diagnosis.headline
    marks = verdicts(diagnosis)
    assert marks["Команды уходят"] is Verdict.OK
    assert marks["Ответы приходят"] is Verdict.SUSPECT
    assert "UDP/8001" in diagnostics.format_diagnosis(diagnosis)


def test_чужие_датаграммы_меняют_вывод() -> None:
    """Отвечает кто-то, но не тот прибор: либо не тот IP, либо второй прибор."""
    stats = TransportStats(commands_sent=4, foreign_datagrams=12)
    diagnosis = diagnostics.diagnose(
        Endpoint(), stats=stats, error=SessionError(SessionErrorKind.TIMEOUT, "нет ответа")
    )
    assert "отвечает кто-то другой" in diagnosis.headline.lower()
    assert verdicts(diagnosis)["Ответы приходят"] is Verdict.SUSPECT


def test_icmp_сбросы_означают_недоступный_прибор() -> None:
    """Windows превращает ICMP «port unreachable» в ошибку приёмного сокета."""
    stats = TransportStats(commands_sent=4, icmp_resets=3)
    diagnosis = diagnostics.diagnose(
        Endpoint(), stats=stats, error=SessionError(SessionErrorKind.TIMEOUT, "нет ответа")
    )
    assert "недоступном адресе" in diagnosis.headline
    assert verdicts(diagnosis)["ICMP «port unreachable»"] is Verdict.SUSPECT


def test_принятые_датаграммы_снимают_подозрение_с_брандмауэра() -> None:
    """Что-то от прибора уже приходило — значит порт открыт."""
    stats = TransportStats(commands_sent=4, datagrams_received=9)
    diagnosis = diagnostics.diagnose(Endpoint(), stats=stats)
    assert verdicts(diagnosis)["Ответы приходят"] is Verdict.OK


def test_адрес_компьютера_вне_подсети_прибора() -> None:
    """Риск R4: ПК сменил IP, и прибор шлёт «в никуда»."""
    marks = verdicts(diagnostics.diagnose(Endpoint(), local_addresses=("10.0.0.5",)))
    assert marks["IP компьютера"] is Verdict.SUSPECT
    marks = verdicts(diagnostics.diagnose(Endpoint(), local_addresses=(texts.EXPECTED_LOCAL_IP,)))
    assert marks["IP компьютера"] is Verdict.OK


def test_неизвестные_адреса_оставляют_проверку_человеку() -> None:
    """Не сумели узнать адреса — это вопрос человеку, а не вывод о неисправности."""
    marks = verdicts(diagnostics.diagnose(Endpoint(), local_addresses=()))
    assert marks["IP компьютера"] is Verdict.CHECK


def test_отказ_отправки_отличается_от_молчания_прибора() -> None:
    """Датаграмма не ушла из сокета — это не «прибор не ответил»."""
    diagnosis = diagnostics.diagnose(
        Endpoint(), error=SessionError(SessionErrorKind.SEND_FAILED, "сеть недоступна")
    )
    assert "не ушла из сокета" in diagnosis.headline


def test_лестница_всегда_заканчивается_wireshark() -> None:
    """Последняя ступень — то, что кодом не проверяется в принципе."""
    diagnosis = diagnostics.diagnose(Endpoint())
    assert diagnosis.checks[-1].title == "Wireshark"
    assert texts.WIRESHARK_FILTER in diagnostics.format_diagnosis(diagnosis)


def test_успешное_подключение_не_выглядит_отказом() -> None:
    """Без ошибки лестница остаётся, но вывод у неё другой."""
    diagnosis = diagnostics.diagnose(Endpoint(), stats=TransportStats(datagrams_received=5))
    assert diagnosis.headline == "Связь установлена."
    assert diagnosis.suspects == ()


def test_адреса_компьютера_добываются_без_исключений() -> None:
    """Диагностика не имеет права падать тогда, когда всё остальное уже сломалось."""
    addresses = diagnostics.local_ipv4_addresses()
    assert isinstance(addresses, tuple)
    assert all(isinstance(address, str) for address in addresses)


# --------------------------------------------------------------------------------------
# Панель датчиков
# --------------------------------------------------------------------------------------


def _sensor(sensor_id: str, channel: int = 0, unit_type: SensorType = SensorType.TEMPERATURE) -> Sensor:
    return Sensor(
        id=sensor_id,
        name=f"Датчик {sensor_id}",
        channel=channel,
        type=unit_type,
        expected_nm=1545.0 + channel,
        window_nm=0.2,
        value0=25.0,
        k1=100.0,
    )


def _reading(sensor_id: str, status: ReadingStatus, value: float = 25.0) -> SensorReading:
    found = status in (ReadingStatus.OK, ReadingStatus.OUT_OF_LIMITS, ReadingStatus.REFERENCE_MISSING)
    return SensorReading(
        sensor_id=sensor_id,
        status=status,
        wavelength_nm=1545.0 if found else math.nan,
        value=value if status in (ReadingStatus.OK, ReadingStatus.OUT_OF_LIMITS) else math.nan,
        position=0 if found else -1,
        candidates=2 if status is ReadingStatus.AMBIGUOUS else (1 if found else 0),
    )


def test_датчик_без_пика_остаётся_строкой_со_статусом() -> None:
    """Исчезающая строка скрыла бы потерю датчика так же, как last-known-value."""
    sensor = _sensor("T1")
    model = models.sensor_panel_model(snapshot(sensors=(sensor,), sensor_readings=()))
    assert len(model.rows) == 1
    assert model.rows[0].sensor.id == "T1"
    assert model.rows[0].status is ReadingStatus.PEAK_NOT_FOUND
    assert math.isnan(model.rows[0].value)


def test_все_пять_статусов_доходят_до_модели_без_слияния() -> None:
    sensors = tuple(_sensor(f"S{index}", channel=index % 4) for index in range(5))
    statuses = tuple(ReadingStatus)
    readings = tuple(_reading(sensor.id, status) for sensor, status in zip(sensors, statuses, strict=True))
    model = models.sensor_panel_model(snapshot(sensors=sensors, sensor_readings=readings))
    assert tuple(row.status for row in model.rows) == statuses


def test_карта_пиков_строится_только_из_телеметрии() -> None:
    """Р77: карта использует λ из UiSnapshot, не ADC и не команду 30 07."""
    wavelength = np.full((PROFILE.channels, PROFILE.fbg_per_channel), np.nan)
    wavelength[0, :3] = (1551.0, 1545.0, 1548.0)
    ui = type("Ui", (), {"wavelength_nm": wavelength})()
    model = models.sensor_panel_model(snapshot(ui=ui))
    assert np.array_equal(model.peaks_by_channel[0], np.array([1545.0, 1548.0, 1551.0]))
    assert all(block.size == 0 for block in model.peaks_by_channel[1:])


def test_фильтр_датчиков_не_зависит_от_статуса() -> None:
    alpha = Sensor(
        id="A", name="Балка", channel=0, type=SensorType.TEMPERATURE,
        expected_nm=1545.0, window_nm=0.2
    )
    beta = Sensor(
        id="B", name="Свая", channel=1, type=SensorType.STRAIN_UE,
        expected_nm=1550.0, window_nm=0.2
    )
    model = models.sensor_panel_model(snapshot(sensors=(alpha, beta)), filter_text="свая")
    assert [row.sensor.id for row in model.rows] == ["B"]
    assert set(model.units) == {"°C", "µε"}
