"""Тесты настроек: чтение, запись, версия формата и устойчивость к правке руками.

Главное, что здесь проверяется, — что **ни один** способ испортить файл
не мешает приложению запуститься, и при этом ни одна порча не проходит молча.
Файл правят Блокнотом, и опечатка не повод не стартовать; но опечатка,
о которой не сказали, хуже отказа.

Второй сюжет — что попадает в файл из `DeviceProfile`. Поля, закрытые
скринингом, настройкой не являются: их изменение означает, что прибор другой
(KB_01). Тесты фиксируют именно это решение, а не текущий состав полей.
"""

import json
from pathlib import Path

import pytest

from fbg.core.endpoint import Endpoint
from fbg.core.frames import ModuleParams, SweepConfig
from fbg.core.pipeline import PipelineConfig
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionConfig
from fbg.io import config as cfg
from fbg.io.config import (
    CONFIG_VERSION,
    PROFILE_DEVICE_FIELDS,
    PROFILE_OPERATOR_FIELDS,
    PROFILE_SETTABLE,
    AppConfig,
    IssueKind,
    load,
    load_sensors,
    profile_from_device,
    save,
    save_sensors,
)
from fbg.io.packet_log import PacketLogConfig
from fbg.io.recorder import RecorderConfig


def write(path: Path, payload: object) -> Path:
    """Кладёт готовый JSON в файл настроек."""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def minimal(**sections: object) -> dict[str, object]:
    """Файл настроек текущей версии с указанными разделами."""
    return {"version": CONFIG_VERSION, **sections}


def kinds(result: object) -> set[IssueKind]:
    """Виды замечаний в результате чтения."""
    return {issue.kind for issue in result.issues}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Отсутствующий файл
# --------------------------------------------------------------------------------------


def test_файла_нет_это_не_ошибка(tmp_path: Path) -> None:
    """Первый запуск: умолчания, ни замечаний, ни созданного файла."""
    target = tmp_path / "fbg_config.json"
    result = load(target)
    assert result.ok and not result.existed
    assert result.config == AppConfig()
    assert not target.exists(), "load не должен создавать файл: пишет только save"


def test_файл_создаётся_при_первом_сохранении(tmp_path: Path) -> None:
    """Пишем на диск тогда, когда пользователь что-то поменял, а не при старте."""
    target = tmp_path / "sub" / "fbg_config.json"
    save(AppConfig(), target)
    assert target.exists()
    assert load(target).ok


def test_папка_под_файл_создаётся(tmp_path: Path) -> None:
    """Настройки могут лежать глубже, чем существующая папка."""
    target = tmp_path / "a" / "b" / "c.json"
    assert save(AppConfig(), target) == target


# --------------------------------------------------------------------------------------
# Испорченный файл
# --------------------------------------------------------------------------------------


def test_битый_json_не_роняет_приложение(tmp_path: Path) -> None:
    """Оборванная правка Блокнотом — это умолчания и замечание, а не падение."""
    target = tmp_path / "c.json"
    target.write_text('{"version": 1, "endpoint": {', encoding="utf-8")
    result = load(target)
    assert result.config == AppConfig()
    assert kinds(result) == {IssueKind.FILE_UNREADABLE}
    assert not result.readable


def test_чужая_кодировка_не_роняет_приложение(tmp_path: Path) -> None:
    """Старый Блокнот мог сохранить cp1251, и это тоже штатный случай.

    Путь с русскими буквами в cp1251 даст `UnicodeDecodeError` при чтении
    в UTF-8 — отдельный от JSON класс отказа, и ловить его нужно отдельно.
    """
    target = tmp_path / "c.json"
    payload = json.dumps({"version": 1, "device_model": "Прибор"}, ensure_ascii=False)
    target.write_bytes(payload.encode("cp1251"))
    result = load(target)
    assert result.config == AppConfig()
    assert kinds(result) == {IssueKind.FILE_UNREADABLE}


def test_json_не_объект_на_верхнем_уровне(tmp_path: Path) -> None:
    """Массив вместо объекта — тоже порча, а не «пустые настройки»."""
    result = load(write(tmp_path / "c.json", [1, 2, 3]))
    assert result.config == AppConfig()
    assert kinds(result) == {IssueKind.FILE_UNREADABLE}


def test_неизвестный_раздел_игнорируется_с_сообщением(tmp_path: Path) -> None:
    """Раздел из будущей версии или опечатка в имени: пропускаем, но называем."""
    result = load(write(tmp_path / "c.json", minimal(эндпоинт={"device_port": 1})))
    assert result.config == AppConfig()
    assert kinds(result) == {IssueKind.UNKNOWN_SECTION}
    assert "эндпоинт" in str(result.issues[0])


def test_неизвестное_поле_игнорируется_с_сообщением(tmp_path: Path) -> None:
    """Опечатка в имени поля не должна стирать соседние правильные поля."""
    payload = minimal(endpoint={"device_ip": "10.0.0.5", "device_prt": 9000})
    result = load(write(tmp_path / "c.json", payload))
    assert result.config.endpoint.device_ip == "10.0.0.5"
    assert result.config.endpoint.device_port == Endpoint().device_port
    assert kinds(result) == {IssueKind.UNKNOWN_FIELD}
    assert "endpoint.device_prt" in str(result.issues[0])


def test_поле_неверного_типа_игнорируется_с_сообщением(tmp_path: Path) -> None:
    """Строка вместо числа — умолчание и замечание, а не исключение."""
    payload = minimal(endpoint={"device_ip": "10.0.0.5", "device_port": "четыре"})
    result = load(write(tmp_path / "c.json", payload))
    assert result.config.endpoint.device_ip == "10.0.0.5"
    assert result.config.endpoint.device_port == Endpoint().device_port
    assert kinds(result) == {IssueKind.WRONG_TYPE}


def test_true_не_проходит_как_единица(tmp_path: Path) -> None:
    """В Python `True` — это `int`, и без явной проверки `retries: true` дало бы 1.

    Проверка не педантизм: `retries = True` тихо превратило бы три попытки
    в две, и увидеть это можно было бы только по числу датаграмм в захвате.
    """
    result = load(write(tmp_path / "c.json", minimal(endpoint={"retries": True})))
    assert result.config.endpoint.retries == Endpoint().retries
    assert kinds(result) == {IssueKind.WRONG_TYPE}


def test_число_с_дробной_частью_не_проходит_в_целое_поле(tmp_path: Path) -> None:
    """`retries: 2.5` — опечатка, а не «округлить как-нибудь»."""
    result = load(write(tmp_path / "c.json", minimal(endpoint={"retries": 2.5})))
    assert result.config.endpoint.retries == Endpoint().retries
    assert kinds(result) == {IssueKind.WRONG_TYPE}


def test_целое_записанное_как_2_0_принимается(tmp_path: Path) -> None:
    """JSON не различает 2 и 2.0, и отвергать второе было бы придирками."""
    result = load(write(tmp_path / "c.json", minimal(endpoint={"retries": 2.0})))
    assert result.config.endpoint.retries == 2
    assert result.ok


def test_целое_принимается_в_вещественное_поле(tmp_path: Path) -> None:
    """`read_timeout_s: 1` — нормальная запись секунды."""
    result = load(write(tmp_path / "c.json", minimal(endpoint={"read_timeout_s": 1})))
    assert result.config.endpoint.read_timeout_s == pytest.approx(1.0)
    assert result.ok


def test_раздел_не_объект_игнорируется_целиком(tmp_path: Path) -> None:
    """`"endpoint": 5` — умолчания раздела и замечание."""
    result = load(write(tmp_path / "c.json", minimal(endpoint=5)))
    assert result.config.endpoint == Endpoint()
    assert kinds(result) == {IssueKind.WRONG_TYPE}


def test_отвергнутое_проверкой_значение_стоит_одного_поля(tmp_path: Path) -> None:
    """Опечатка в `retries` не должна стирать правильный `device_ip` рядом.

    `Endpoint.__post_init__` отвергает отрицательные повторы, и наивная сборка
    «весь раздел разом» потеряла бы вместе с ними весь раздел. Поэтому при
    отказе поля применяются по одному, и виновное называется поимённо.
    """
    payload = minimal(endpoint={"device_ip": "10.0.0.5", "retries": -3, "local_port": 9100})
    result = load(write(tmp_path / "c.json", payload))
    assert result.config.endpoint.device_ip == "10.0.0.5"
    assert result.config.endpoint.local_port == 9100
    assert result.config.endpoint.retries == Endpoint().retries
    assert kinds(result) == {IssueKind.REJECTED_VALUE}
    assert "endpoint.retries" in str(result.issues[0])


def test_несколько_отвергнутых_значений_называются_каждое(tmp_path: Path) -> None:
    """Замечание на поле, а не на раздел: чинить придётся оба."""
    payload = minimal(endpoint={"retries": -3, "rcvbuf_bytes": 0, "device_ip": "10.0.0.5"})
    result = load(write(tmp_path / "c.json", payload))
    места = {issue.location for issue in result.issues}
    assert места == {"endpoint.retries", "endpoint.rcvbuf_bytes"}
    assert result.config.endpoint.device_ip == "10.0.0.5"


def test_не_ascii_в_шапке_файла_данных_отвергается(tmp_path: Path) -> None:
    """Правило Р44 живёт в `RecorderConfig`, и настройки его не обходят.

    Русское `device_model` сделало бы файл данных нечитаемым для
    `numpy.genfromtxt` на Windows, поэтому оно отвергается здесь же.
    """
    payload = minimal(recorder={"prefix": "данные"})
    result = load(write(tmp_path / "c.json", payload))
    assert result.config.recorder.prefix == RecorderConfig(directory=Path(".")).prefix
    assert kinds(result) == {IssueKind.REJECTED_VALUE}


# --------------------------------------------------------------------------------------
# Версия формата
# --------------------------------------------------------------------------------------


def test_версия_новее_не_читается_вовсе(tmp_path: Path) -> None:
    """Частичное чтение файла из будущего опаснее умолчаний.

    Незнакомые поля молча пропали бы, и приложение работало бы на смеси
    чужих настроек с нашими умолчаниями, ничего об этом не сообщая.
    """
    payload = {"version": CONFIG_VERSION + 1, "endpoint": {"device_ip": "10.0.0.5"}}
    result = load(write(tmp_path / "c.json", payload))
    assert result.config == AppConfig()
    assert result.file_version == CONFIG_VERSION + 1
    assert kinds(result) == {IssueKind.VERSION_NEWER}
    assert not result.readable


def test_версия_отсутствует_или_бессмысленна(tmp_path: Path) -> None:
    """Файл без версии читать нельзя: неизвестно, по каким правилам."""
    for payload in ({}, {"version": 0}, {"version": "1"}, {"version": True}):
        result = load(write(tmp_path / "c.json", payload))
        assert result.config == AppConfig()
        assert kinds(result) == {IssueKind.VERSION_INVALID}


def test_файл_новее_не_затирается_а_откладывается(tmp_path: Path) -> None:
    """Стирание — единственное необратимое действие этого модуля.

    Файл, который мы не поняли, переезжает в `*.bad`, а не исчезает: его
    написала более новая версия приложения, и он ещё пригодится.
    """
    target = write(tmp_path / "c.json", {"version": CONFIG_VERSION + 5, "х": 1})
    save(AppConfig(), target)
    отложенный = json.loads((tmp_path / "c.json.bad").read_text(encoding="utf-8"))
    assert отложенный["version"] == CONFIG_VERSION + 5
    assert load(target).ok


def test_битый_файл_тоже_откладывается(tmp_path: Path) -> None:
    """Причина та же: там лежит то, чего мы не понимаем."""
    target = tmp_path / "c.json"
    target.write_text("{не json", encoding="utf-8")
    save(AppConfig(), target)
    assert (tmp_path / "c.json.bad").read_text(encoding="utf-8") == "{не json"


def test_исправный_файл_перезаписывается_без_копии(tmp_path: Path) -> None:
    """`*.bad` — реакция на непонятное, а не побочный продукт каждой записи."""
    target = tmp_path / "c.json"
    save(AppConfig(), target)
    save(AppConfig(), target)
    assert not (tmp_path / "c.json.bad").exists()


def test_временный_файл_после_записи_не_остаётся(tmp_path: Path) -> None:
    """Запись атомарная: оборванное сохранение не оставляет полуфайла."""
    target = tmp_path / "c.json"
    save(AppConfig(), target)
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_механизм_миграции_работает(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Преобразований пока нет — версия одна, — но механизм не заглушка.

    Проверяется подстановкой шага: файл версии 1 с чужим именем поля доводится
    до версии 2 и читается. Без такого теста цикл `_migrate` был бы мёртвым
    кодом, о работоспособности которого узнали бы в момент, когда он понадобится.
    """
    monkeypatch.setattr(cfg, "CONFIG_VERSION", 2)
    monkeypatch.setattr(
        cfg,
        "MIGRATIONS",
        {1: lambda data: {"version": data["version"], "endpoint": {"device_ip": data["ip"]}}},
    )
    result = load(write(tmp_path / "c.json", {"version": 1, "ip": "10.0.0.7"}))
    assert result.config.endpoint.device_ip == "10.0.0.7"
    assert result.ok


def test_нет_шага_миграции_это_замечание_а_не_падение(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Разрыв в цепочке преобразований называется, а не игнорируется."""
    monkeypatch.setattr(cfg, "CONFIG_VERSION", 3)
    result = load(write(tmp_path / "c.json", {"version": 1}))
    assert IssueKind.VERSION_INVALID in kinds(result)


# --------------------------------------------------------------------------------------
# Круговорот
# --------------------------------------------------------------------------------------


def test_круговорот_save_load_сохраняет_всё(tmp_path: Path) -> None:
    """Сохранили — прочитали — получили то же самое, включая пути и кортежи."""
    original = AppConfig(
        endpoint=Endpoint(device_ip="10.0.0.5", local_port=9100, strict_source_port=True),
        profile=DeviceProfile(channels=2, fbg_per_channel=8, case_temp_signed=False),
        session=SessionConfig(keepalive_period_s=3.5, backoff_schedule=(0.5, 1.0)),
        pipeline=PipelineConfig(history_frames=500, expected_rate_hz=1000.0),
        recorder=RecorderConfig(directory=tmp_path / "данные", decimation=20, fbg_limit=4),
        packet_log=PacketLogConfig(directory=tmp_path / "журнал", telemetry_stride=10),
        calibration_path=tmp_path / "sensors.json",
        device_model="GC-97001C-03-01-A-F",
        serial=94401220,
        firmware="4.10",
    )
    target = save(original, tmp_path / "c.json")
    result = load(target)
    assert result.ok, result.issues
    assert result.config == original


def test_круговорот_умолчаний(tmp_path: Path) -> None:
    """Файл, записанный из умолчаний, читается обратно в умолчания."""
    save(AppConfig(), tmp_path / "c.json")
    assert load(tmp_path / "c.json").config == AppConfig()


def test_в_файл_пишутся_только_отклонения(tmp_path: Path) -> None:
    """Файл остаётся коротким и читаемым, а изменившееся умолчание доезжает.

    Поле без умолчания (`RecorderConfig.directory`) пишется всегда: сравнивать
    его не с чем.
    """
    save(AppConfig(endpoint=Endpoint(device_ip="10.0.0.5")), tmp_path / "c.json")
    raw = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert raw["endpoint"] == {"device_ip": "10.0.0.5"}
    assert raw["profile"] == {}
    assert "directory" in raw["recorder"]


def test_путь_с_русскими_буквами_переживает_круговорот(tmp_path: Path) -> None:
    """`C:\\Данные` — обычный путь на русской Windows, и он обязан сохраняться."""
    directory = tmp_path / "Данные стенда"
    config = AppConfig(recorder=RecorderConfig(directory=directory))
    save(config, tmp_path / "c.json")
    assert load(tmp_path / "c.json").config.recorder.directory == directory


def test_файл_настроек_читаемый_глазами(tmp_path: Path) -> None:
    """Его правят Блокнотом, поэтому отступы есть, а кириллица не экранирована."""
    save(AppConfig(device_model="Прибор"), tmp_path / "c.json")
    text = (tmp_path / "c.json").read_text(encoding="utf-8")
    assert "\n  " in text and "Прибор" in text


# --------------------------------------------------------------------------------------
# DeviceProfile: три владельца числа
# --------------------------------------------------------------------------------------


def test_состав_настраиваемых_полей_профиля() -> None:
    """Ровно два множества, и они не пересекаются.

    Тест охраняет **решение**, а не список: добавление поля сюда означает
    смену владельца числа и принимается при закрытии или открытии вопроса
    в KB_04, а не по удобству.
    """
    assert {"case_temp_signed"} == PROFILE_OPERATOR_FIELDS
    assert not (PROFILE_DEVICE_FIELDS & PROFILE_OPERATOR_FIELDS)
    assert PROFILE_SETTABLE == PROFILE_DEVICE_FIELDS | PROFILE_OPERATOR_FIELDS
    assert {f.name for f in DeviceProfile.__dataclass_fields__.values()} >= PROFILE_SETTABLE


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sweep_base_ghz", 196251),
        ("freq_divisor", 1),
        ("case_temp_scale", 0.1),
        ("adc_index_ascending_freq", False),
        ("mode_len_width", 2),
        ("set_sweep_frame_len", 11),
        ("adc_max", 4095),
    ],
)
def test_поле_закрытое_скринингом_настройкой_не_является(
    tmp_path: Path, field_name: str, value: object
) -> None:
    """Их изменение означает, что прибор другой, а не что настройка другая.

    Цена ошибки асимметрична: `sweep_base_ghz`, сдвинутый на единицу, даёт
    систематические 8 пм, которые не видны ни в одном числе на экране
    и всплывут через месяц. Правило KB_05 №8 выполнено тем, что все они —
    поля датакласса: это и делает правку однострочной, а не наличие в JSON.
    """
    result = load(write(tmp_path / "c.json", minimal(profile={field_name: value})))
    assert getattr(result.config.profile, field_name) == getattr(DeviceProfile(), field_name)
    assert kinds(result) == {IssueKind.NOT_SETTABLE}
    assert "скринингом" in str(result.issues[0])


def test_поле_оператора_настройкой_является(tmp_path: Path) -> None:
    """`case_temp_signed` — единственный незакрытый вопрос профиля (N2b).

    Проверяется он охлаждением прибора ниже нуля, то есть экспериментом,
    который ставит человек на стенде. Ему настройка и нужна.
    """
    result = load(write(tmp_path / "c.json", minimal(profile={"case_temp_signed": False})))
    assert result.config.profile.case_temp_signed is False
    assert result.ok


def test_геометрия_прибора_читается_из_файла_как_кэш(tmp_path: Path) -> None:
    """Кольцо, буферы и фильтр журнала строятся из профиля **до** подключения.

    Без кэша последнего опроса первое подключение к прибору другой геометрии
    шло бы на чужих размерах.
    """
    payload = minimal(profile={"channels": 8, "fbg_per_channel": 16})
    result = load(write(tmp_path / "c.json", payload))
    assert (result.config.profile.channels, result.config.profile.fbg_per_channel) == (8, 16)
    assert result.ok


def test_профиль_обновляется_ответами_прибора() -> None:
    """`profile_from_device` трогает только то, что прибор сообщил сам."""
    module = ModuleParams(
        speed_code=0x00CA, speed_hz=1000, channels=8, fbg_per_channel=16, peak_gap_ghz=40
    )
    sweep = SweepConfig.from_params(
        start_param=3, step_param=4, stop_param=4000, adc_step_param=4, profile=DeviceProfile()
    )
    updated = profile_from_device(DeviceProfile(), module, sweep)
    assert (updated.channels, updated.fbg_per_channel) == (8, 16)
    assert (updated.sweep_speed_hz, updated.peak_gap_ghz) == (1000, 40)
    assert (updated.start_param, updated.stop_param) == (3, 4000)
    # Интерпретация байтов — результат скрининга, прибор о ней не сообщает.
    assert updated.sweep_base_ghz == DeviceProfile().sweep_base_ghz
    assert updated.freq_divisor == DeviceProfile().freq_divisor


def test_неразобранная_скорость_не_обнуляет_прежнюю() -> None:
    """`speed_hz = None` бывает при коде вне таблицы (вопрос D7).

    Подставить сюда ноль значило бы испортить окно тишины watchdog'а, которое
    считается от периода развёртки.
    """
    module = ModuleParams(
        speed_code=0x0999, speed_hz=None, channels=4, fbg_per_channel=30, peak_gap_ghz=30
    )
    sweep = SweepConfig.from_params(
        start_param=1, step_param=2, stop_param=5101, adc_step_param=2, profile=DeviceProfile()
    )
    updated = profile_from_device(DeviceProfile(), module, sweep)
    assert updated.sweep_speed_hz == DeviceProfile().sweep_speed_hz


# --------------------------------------------------------------------------------------
# Идентификация прибора: одно поле, две шапки
# --------------------------------------------------------------------------------------


def test_идентификация_подставляется_в_обе_шапки(tmp_path: Path) -> None:
    """Хранится один раз, подставляется в две шапки методами `AppConfig`.

    Р53 запрещает абстракцию «`DeviceConfig` → шапка файла» — её здесь нет:
    подстановка это `replace` двух полей. Зато одно поле в файле вместо двух
    копий избавляет от файла, где `serial` в разных разделах разный.
    """
    config = AppConfig(
        recorder=RecorderConfig(directory=tmp_path),
        packet_log=PacketLogConfig(directory=tmp_path),
    ).with_device(serial=94401220, firmware="4.10")
    assert config.recorder_config().serial == 94401220
    assert config.packet_log_config().serial == 94401220
    assert config.recorder_config().firmware == "4.10"
    assert config.packet_log_config().device_model == config.device_model


def test_идентификация_в_разделах_не_хранится(tmp_path: Path) -> None:
    """Иначе один серийный номер лежал бы в файле дважды и мог бы разойтись."""
    config = AppConfig(recorder=RecorderConfig(directory=tmp_path)).with_device(
        serial=1, firmware="4.10"
    )
    save(config, tmp_path / "c.json")
    raw = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert raw["serial"] == 1 and raw["firmware"] == "4.10"
    assert "serial" not in raw["recorder"] and "firmware" not in raw["packet_log"]


def test_до_первого_опроса_идентификация_неизвестна() -> None:
    """Журнал создаётся до подключения, и в его первом файле честное `unknown`.

    На момент записи прибор ещё не спрошен; подставить туда что-нибудь
    правдоподобное было бы враньём в шапке файла.
    """
    config = AppConfig()
    assert config.serial is None and config.firmware is None
    assert config.packet_log_config().serial is None


# --------------------------------------------------------------------------------------
# Файл калибровок
# --------------------------------------------------------------------------------------


def test_калибровок_нет_это_не_замечание(tmp_path: Path) -> None:
    """Приложение обязано работать без калибровок: сырые нм пишутся всегда."""
    sensors, issues = load_sensors(tmp_path / "нет.json")
    assert sensors == () and issues == ()


def test_круговорот_датчиков(tmp_path: Path) -> None:
    """Набор переживает запись и чтение целиком."""
    from fbg.core.calibration import Sensor, SensorType

    original = (
        Sensor(
            id="T1",
            name="Температура",
            channel=0,
            type=SensorType.TEMPERATURE,
            expected_nm=1544.787,
            window_nm=0.35,
            k1=100.0,
        ),
    )
    path = save_sensors(original, tmp_path / "sensors.json")
    sensors, issues = load_sensors(path)
    assert issues == () and sensors == original


def test_испорченная_запись_стоит_одной_записи(tmp_path: Path) -> None:
    """Одна опечатка не должна отменять остальные датчики."""
    payload = {
        "sensors": [
            {"id": "A", "channel": 0, "type": 0, "expected_nm": 1545.0, "window_nm": 0.3},
            {"id": "B", "channel": 0, "type": 99, "expected_nm": 1555.0, "window_nm": 0.3},
            {"id": "C", "channel": 0, "type": 0, "expected_nm": 1565.0, "window_nm": 0.3},
        ]
    }
    sensors, issues = load_sensors(write(tmp_path / "s.json", payload))
    assert [s.id for s in sensors] == ["A", "C"]
    assert len(issues) == 1 and "sensors[1]" in issues[0].location


def test_битый_файл_калибровок_не_роняет_приложение(tmp_path: Path) -> None:
    """Тот же принцип, что у настроек: правят руками, значит ломают."""
    path = tmp_path / "s.json"
    path.write_text("{сломано", encoding="utf-8")
    sensors, issues = load_sensors(path)
    assert sensors == () and len(issues) == 1
    assert issues[0].kind is IssueKind.FILE_UNREADABLE


def test_старая_форма_калибровки_не_мигрирует_молча(tmp_path: Path) -> None:
    """Абсолютные c0/c1/c2 считаются непонятым файлом, а не новой моделью."""
    path = write(
        tmp_path / "sensors.json",
        {
            "sensors": [
                {
                    "id": "T1",
                    "channel": 0,
                    "type": 0,
                    "expected_nm": 1544.8,
                    "window_nm": 0.3,
                    "c0": -154455.0,
                    "c1": 100.0,
                    "c2": 0.0,
                }
            ]
        },
    )
    sensors, issues = load_sensors(path)
    assert sensors == ()
    assert len(issues) == 1 and issues[0].kind is IssueKind.FILE_UNREADABLE
    assert "c0/c1/c2" in issues[0].message


def test_сохранение_новой_формы_откладывает_старый_файл(tmp_path: Path) -> None:
    """KB_05 №33: непонятое не затирается при первом сохранении новой модели."""
    from fbg.core.calibration import Sensor, SensorType

    path = tmp_path / "sensors.json"
    old = (
        '{"sensors":[{"id":"T1","channel":0,"type":0,"expected_nm":1544.8,'
        '"window_nm":0.3,"c0":-154455,"c1":100,"c2":0}]}\n'
    )
    path.write_text(old, encoding="utf-8")
    sensor = Sensor(
        id="T1",
        name="Температура",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=1544.8,
        window_nm=0.3,
        value0=25.0,
        k1=100.0,
    )

    save_sensors((sensor,), path)

    backup = tmp_path / "sensors.json.bad"
    assert backup.read_text(encoding="utf-8") == old
    sensors, issues = load_sensors(path)
    assert issues == () and sensors == (sensor,)


def test_сохранение_старой_формы_не_затирает_предыдущий_bad(tmp_path: Path) -> None:
    """Повторная карантинизация сохраняет уже отложенный непонятный файл."""
    from fbg.core.calibration import Sensor, SensorType

    path = tmp_path / "sensors.json"
    old = (
        '{"sensors":[{"id":"T1","channel":0,"type":0,"expected_nm":1544.8,'
        '"window_nm":0.3,"c0":-154455,"c1":100,"c2":0}]}\n'
    )
    first_bad = tmp_path / "sensors.json.bad"
    first_bad.write_text("previous backup\n", encoding="utf-8")
    path.write_text(old, encoding="utf-8")
    sensor = Sensor(
        id="T1",
        name="Температура",
        channel=0,
        type=SensorType.TEMPERATURE,
        expected_nm=1544.8,
        window_nm=0.3,
        value0=25.0,
        k1=100.0,
    )

    save_sensors((sensor,), path)

    assert first_bad.read_text(encoding="utf-8") == "previous backup\n"
    assert (tmp_path / "sensors.json.bad.2").read_text(encoding="utf-8") == old
