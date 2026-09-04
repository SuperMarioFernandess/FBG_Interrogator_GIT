"""Настройки приложения: один JSON рядом с исполняемым файлом.

Ни базы, ни реестра (Р10): пользователь один, прибор один, а файл, лежащий
рядом с exe, переносится вместе с ним и правится Блокнотом.

Три требования, и все три — про то, что файл правят руками
----------------------------------------------------------
**Файла нет — не ошибка.** `load` возвращает умолчания и говорит, что файла
не было. Создаётся файл при первом `save`, а не при старте: приложение,
которое пишет на диск раньше, чем пользователь что-то поменял, — плохой сосед.

**Испорченный файл не роняет приложение.** Битый JSON, чужая кодировка,
неизвестное поле, поле неверного типа, значение, отвергнутое проверкой
секции, — всё это замечания в `LoadResult.issues`, а не исключения.
Опечатка в одном поле стоит **одного** поля: секция собирается со всеми
принятыми значениями, а отвергнутое заменяется умолчанием. Молчания при
этом нет нигде — каждое проигнорированное поле названо (KB_05 №13).

**Версия формата в файле.** Не абстракция впрок: профиль уже менялся дважды —
D8 сдвинул базу развёртки, D1 зафиксировал делитель, — и это не последний раз.
Файл версии **новее** текущей не читается вовсе: частичное чтение файла
из будущего опаснее умолчаний, потому что новые поля молча пропадут. Такой
файл при следующем `save` не затирается, а переименовывается в `*.bad`.

Что попадает в файл из `DeviceProfile`, и почему не всё
------------------------------------------------------
Поля профиля делятся по признаку «кто владелец числа», и владельцев трое.

**Владелец — прибор** (`channels`, `fbg_per_channel`, `sweep_speed_hz`,
`peak_gap_ghz`, `start_param`, `step_param`, `stop_param`, `adc_step_param`).
Прибор сообщает их ответами `10 04` и `10 05` при каждом подключении.
В файле они лежат **кэшем последнего опроса**, и обновляет их приложение
через `profile_from_device` после успешного `Probing`, а не пользователь.
Кэш нужен потому, что кольцо истории, буферы кадра и фильтр телеметрии
журнала строятся из профиля **до** подключения: без кэша первое подключение
к прибору другой геометрии шло бы на чужих размерах.

**Владелец — скрининг** (`sweep_base_ghz`, `freq_divisor`, `case_temp_scale`,
`peak_missing_codes`, `adc_index_ascending_freq`, `mode_len_width`,
`set_sweep_frame_len`, `set_sweep_len_field`, `adc_max`, `threshold_auto`,
`gain_max_level`). Это не настройки, а результат разбора захвата: их
изменение означает, что прибор **другой**, и правильный путь — новый скрининг
и новые умолчания в `profile.py`. В файл они не пишутся, а если кто-то впишет
их руками, поле игнорируется с явной причиной, а не молча. Довод — цена
ошибки: `sweep_base_ghz`, сдвинутый на единицу, даёт систематические 8 пм,
которые не видны ни в одном числе на экране и всплывут через месяц.
Правило KB_05 №8 при этом выполнено тем, что все они — **поля датакласса**:
именно это делает правку однострочной, а не наличие их в JSON.

**Владелец — оператор** (`case_temp_signed`). Единственное поле профиля,
чей вопрос ещё открыт (N2b): знаковое ли поле температуры корпуса, проверяется
охлаждением прибора ниже нуля, и это эксперимент, который ставит человек
на стенде. Ему настройка и нужна.

Список `PROFILE_OPERATOR_FIELDS` — одна строка, и добавление в него нового
поля есть решение о том, кто владеет числом. Оно принимается при закрытии
или открытии вопроса в KB_04, а не по удобству.

Кто заполняет `serial` и `firmware`
-----------------------------------
Идентификация прибора хранится **один раз**, полями `AppConfig`, и оттуда
подставляется в `RecorderConfig` и `PacketLogConfig` методами
`recorder_config()` и `packet_log_config()`. Р53 запрещает абстракцию
«`DeviceConfig` → шапка файла» — её здесь и нет: подстановка это `replace`
двух полей, а не тип и не слой. Зато одно поле в файле вместо двух копий
избавляет от файла, где `serial` в двух секциях разный.

Известны эти значения только после `Probing`, поэтому порядок такой:

* журнал пакетов создаётся **до** подключения — он обязан записать сам обмен
  подключения, — и в шапке его первого файла стоит `unknown`. Это честно:
  на момент записи мы прибор ещё не спросили;
* приложение после успешного `connect()` зовёт `AppConfig.with_device(...)`
  из `Session.device_config` и сохраняет настройки. Со следующего запуска
  шапка журнала верна с первого байта;
* `recorder` создаётся в момент начала записи, то есть уже после `Probing`,
  и его шапка верна всегда — метаданные ему передаются живыми.
"""

import dataclasses
import json
import os
import sys
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path

from fbg.core.calibration import SENSORS_KEY, Sensor, sensor_from_json, sensors_to_json
from fbg.core.endpoint import Endpoint
from fbg.core.frames import ModuleParams, SweepConfig
from fbg.core.pipeline import PipelineConfig
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.io.recorder import RecorderConfig

#: Версия формата файла настроек. Растёт, когда старый файл перестаёт читаться
#: без преобразования. Файл с большим числом не читается вовсе.
CONFIG_VERSION = 1

#: Имя файла настроек рядом с исполняемым файлом.
CONFIG_FILENAME = "fbg_config.json"

#: Модель прибора для шапок файлов (KB_01, «Идентификация»).
DEFAULT_MODEL = "GC-97001C-03-01-A-F"

#: Поля `DeviceProfile`, которые сообщает сам прибор ответами 10 04 и 10 05.
#: В файле лежат кэшем последнего `Probing`; обновляет их `profile_from_device`.
PROFILE_DEVICE_FIELDS: frozenset[str] = frozenset(
    {
        "channels",
        "fbg_per_channel",
        "sweep_speed_hz",
        "peak_gap_ghz",
        "start_param",
        "step_param",
        "stop_param",
        "adc_step_param",
    }
)

#: Поля `DeviceProfile`, которыми распоряжается оператор. Сейчас ровно одно:
#: `case_temp_signed` — единственный незакрытый вопрос профиля (N2b, KB_04).
PROFILE_OPERATOR_FIELDS: frozenset[str] = frozenset({"case_temp_signed"})

#: Всё, что вообще может стоять в секции `profile` файла настроек.
PROFILE_SETTABLE: frozenset[str] = PROFILE_DEVICE_FIELDS | PROFILE_OPERATOR_FIELDS

#: Поля секций, которыми владеет верхний уровень `AppConfig`: в секцию они
#: не пишутся и из секции не читаются, иначе один серийный номер лежал бы
#: в файле в двух местах и мог бы разойтись.
IDENTITY_FIELDS: frozenset[str] = frozenset({"device_model", "serial", "firmware"})

#: Преобразования файла из версии N в N+1. Пусто: версия пока одна, и писать
#: миграцию не из чего. Механизм при этом рабочий, а не заглушка, — он
#: проверяется тестом, который подставляет сюда шаг.
MIGRATIONS: dict[int, typing.Callable[[dict[str, object]], dict[str, object]]] = {}


class IssueKind(StrEnum):
    """Что именно не так с прочитанным."""

    FILE_UNREADABLE = "file_unreadable"
    """Файл не открылся, не разобрался как JSON или оказался не объектом."""

    VERSION_NEWER = "version_newer"
    """Файл от более новой версии приложения: не читается целиком."""

    VERSION_INVALID = "version_invalid"
    UNKNOWN_SECTION = "unknown_section"
    UNKNOWN_FIELD = "unknown_field"
    WRONG_TYPE = "wrong_type"
    NOT_SETTABLE = "not_settable"
    """Поле известно, но настройкой не является: владелец числа не оператор."""

    REJECTED_VALUE = "rejected_value"
    """Значение отвергнуто проверкой секции: подставлено умолчание."""


@dataclass(frozen=True)
class ConfigIssue:
    """Одно замечание к прочитанному файлу."""

    kind: IssueKind
    location: str
    """Где: `«секция.поле»` либо имя секции, либо имя файла."""

    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


def default_base_dir() -> Path:
    """Папка, рядом с которой живут настройки, данные и журналы.

    В собранном PyInstaller'ом exe это папка самого exe (`sys.frozen`),
    при запуске из исходников — текущая рабочая папка. Различать их нужно
    именно так: у собранного приложения `__file__` указывает внутрь временной
    распаковки, которая исчезает после выхода.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def default_path(base_dir: Path | None = None) -> Path:
    """Путь к файлу настроек."""
    return (base_dir or default_base_dir()) / CONFIG_FILENAME


def _default_data_dir() -> Path:
    return default_base_dir() / "data"


def _default_log_dir() -> Path:
    return default_base_dir() / "logs"


def _default_recorder() -> RecorderConfig:
    return RecorderConfig(directory=_default_data_dir())


def _default_packet_log() -> PacketLogConfig:
    return PacketLogConfig(directory=_default_log_dir())


@dataclass(frozen=True)
class AppConfig:
    """Все настройки приложения одним объектом.

    Секции — те же датаклассы, что читают модули: `Endpoint` читает транспорт,
    `SessionConfig` — сессия, и так далее. Отдельного «плоского» представления
    нет намеренно: оно означало бы второе описание тех же полей, которое
    рассинхронизируется при первой же правке.
    """

    endpoint: Endpoint = field(default_factory=Endpoint)
    profile: DeviceProfile = field(default_factory=DeviceProfile)
    session: SessionConfig = field(default_factory=SessionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    recorder: RecorderConfig = field(default_factory=_default_recorder)
    packet_log: PacketLogConfig = field(default_factory=_default_packet_log)

    calibration_path: Path = field(default_factory=lambda: default_base_dir() / "sensors.json")
    """Файл калибровок. Имена и типы датчиков живут там, а не здесь: иначе
    у одного датчика было бы два имени в двух файлах."""

    device_model: str = DEFAULT_MODEL
    serial: int | None = None
    firmware: str | None = None
    """Идентификация прибора: кэш последнего успешного `Probing`, нужный
    шапкам файлов до того, как прибор ответит (см. докстринг модуля)."""

    def recorder_config(self) -> RecorderConfig:
        """`RecorderConfig` с подставленной идентификацией прибора."""
        return replace(
            self.recorder,
            device_model=self.device_model,
            serial=self.serial,
            firmware=self.firmware,
        )

    def packet_log_config(self) -> PacketLogConfig:
        """`PacketLogConfig` с подставленной идентификацией прибора."""
        return replace(
            self.packet_log,
            device_model=self.device_model,
            serial=self.serial,
            firmware=self.firmware,
        )

    def with_device(self, *, serial: int | None, firmware: str | None) -> "AppConfig":
        """Запоминает, что ответил прибор при опросе конфигурации."""
        return replace(self, serial=serial, firmware=firmware)

    def with_profile(self, profile: DeviceProfile) -> "AppConfig":
        """Заменяет профиль целиком."""
        return replace(self, profile=profile)


def profile_from_device(
    profile: DeviceProfile, module: ModuleParams, sweep: SweepConfig
) -> DeviceProfile:
    """Обновляет геометрию профиля тем, что прибор сообщил в `10 04` и `10 05`.

    Трогаются только поля из `PROFILE_DEVICE_FIELDS`: интерпретация байтов —
    результат скрининга, и прибор о ней ничего не сообщает.

    `ModuleParams.speed_hz` бывает `None`, если код скорости не разобран по
    таблице (вопрос D7). Тогда прежнее значение сохраняется: подставить сюда
    ноль значило бы испортить окно тишины watchdog'а, которое считается
    от периода развёртки.
    """
    return replace(
        profile,
        channels=module.channels,
        fbg_per_channel=module.fbg_per_channel,
        peak_gap_ghz=module.peak_gap_ghz,
        sweep_speed_hz=module.speed_hz if module.speed_hz is not None else profile.sweep_speed_hz,
        start_param=sweep.start_param,
        step_param=sweep.step_param,
        stop_param=sweep.stop_param,
        adc_step_param=sweep.adc_step_param,
    )


@dataclass(frozen=True)
class LoadResult:
    """Что получилось прочитать и что при этом не понравилось."""

    config: AppConfig
    issues: tuple[ConfigIssue, ...] = ()
    path: Path | None = None
    existed: bool = False
    """Был ли файл на диске. False — первый запуск, и это не ошибка."""

    file_version: int | None = None
    readable: bool = True
    """Удалось ли прочитать файл как настройки этой версии. False означает,
    что при сохранении файл будет отложен в `*.bad`, а не затёрт."""

    @property
    def ok(self) -> bool:
        """True, если замечаний нет вовсе."""
        return not self.issues


# --------------------------------------------------------------------------------------
# Приведение типов: ровно то, что встречается в секциях, и ничего сверх
# --------------------------------------------------------------------------------------


class _TypeError(ValueError):
    """Значение не подошло под объявленный тип поля."""


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    """Разбирает `T | None` на `(T, допустим ли None)`."""
    origin = typing.get_origin(annotation)
    if origin is not types.UnionType and origin is not typing.Union:
        return annotation, False
    args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    if len(args) != 1:
        raise _TypeError(f"тип {annotation!r} не поддерживается файлом настроек")
    return args[0], True


def _coerce(value: object, annotation: object) -> object:
    """Приводит значение из JSON к типу поля секции.

    Поддерживаются ровно те типы, что встречаются в шести секциях: `str`,
    `int`, `float`, `bool`, `Path`, `tuple[float, ...]` и любой из них
    в паре с `None`. Незнакомый тип — не «пропустить как есть», а отказ:
    молча положить в поле объект неизвестной формы хуже, чем умолчание.

    `bool` проверяется раньше `int` намеренно: в Python `True` — это `int`,
    и без отдельной проверки `retries: true` прошло бы как `retries = 1`.
    """
    inner, optional = _unwrap_optional(annotation)
    if value is None:
        if optional:
            return None
        raise _TypeError("null недопустим для этого поля")

    if inner is bool:
        if not isinstance(value, bool):
            raise _TypeError(f"ожидалось true/false, получено {value!r}")
        return value
    if inner is int:
        if isinstance(value, bool):
            raise _TypeError(f"ожидалось целое, получено {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise _TypeError(f"ожидалось целое, получено {value!r}")
    if inner is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _TypeError(f"ожидалось число, получено {value!r}")
        return float(value)
    if inner is str:
        if not isinstance(value, str):
            raise _TypeError(f"ожидалась строка, получено {value!r}")
        return value
    if inner is Path:
        if not isinstance(value, str):
            raise _TypeError(f"ожидался путь строкой, получено {value!r}")
        return Path(value)
    if typing.get_origin(inner) is tuple:
        if not isinstance(value, list):
            raise _TypeError(f"ожидался список, получено {value!r}")
        item_type = typing.get_args(inner)[0]
        return tuple(_coerce(item, item_type) for item in value)
    raise _TypeError(f"тип {inner!r} не поддерживается файлом настроек")


def _to_json(value: object) -> object:
    """Обратное преобразование: типы секций → то, что переварит `json.dump`."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_json(item) for item in value]
    return value


# --------------------------------------------------------------------------------------
# Секции
# --------------------------------------------------------------------------------------


def _section_to_json(instance: object, *, skip: frozenset[str] = frozenset()) -> dict[str, object]:
    """Отклонения секции от умолчаний плюс поля, у которых умолчания нет.

    Пишутся не все поля, а разница: файл остаётся коротким и читаемым, а
    умолчание, изменившееся в новой версии приложения, доезжает до тех,
    кто его не трогал. Поле без умолчания (`RecorderConfig.directory`)
    пишется всегда — сравнивать его не с чем.
    """
    data: dict[str, object] = {}
    for info in fields(instance):  # type: ignore[arg-type]
        if info.name in skip or not info.init:
            continue
        value = getattr(instance, info.name)
        if info.default is not dataclasses.MISSING and value == info.default:
            continue
        if (
            info.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            and value == info.default_factory()  # type: ignore[misc]
        ):
            continue
        data[info.name] = _to_json(value)
    return data


def _build_section[T](
    cls: type[T],
    default: T,
    raw: object,
    section: str,
    issues: list[ConfigIssue],
    *,
    allowed: frozenset[str] | None = None,
    not_settable_hint: str = "",
) -> T:
    """Собирает секцию из умолчаний и принятых полей файла.

    Отвергнутое значение стоит **одно поле**, а не всю секцию: сначала
    делается попытка применить принятые поля разом, и только если проверка
    секции их отвергла (`__post_init__` бросил `ValueError`), поля применяются
    по одному, и виновное называется. Опечатка в `retries` не должна стирать
    правильный `device_ip`, набранный рядом.
    """
    if raw is None:
        return default
    if not isinstance(raw, Mapping):
        issues.append(
            ConfigIssue(IssueKind.WRONG_TYPE, section, f"ожидался объект, получено {raw!r}")
        )
        return default

    hints = typing.get_type_hints(cls)
    known = {info.name for info in fields(cls) if info.init}  # type: ignore[arg-type]
    accepted: dict[str, object] = {}

    for key, value in raw.items():
        location = f"{section}.{key}"
        if key not in known:
            issues.append(
                ConfigIssue(IssueKind.UNKNOWN_FIELD, location, "поле неизвестно и пропущено")
            )
            continue
        if allowed is not None and key not in allowed:
            issues.append(ConfigIssue(IssueKind.NOT_SETTABLE, location, not_settable_hint))
            continue
        try:
            accepted[key] = _coerce(value, hints[key])
        except _TypeError as exc:
            issues.append(
                ConfigIssue(IssueKind.WRONG_TYPE, location, f"{exc}; оставлено умолчание")
            )

    if not accepted:
        return default
    try:
        return replace(default, **accepted)  # type: ignore[type-var, return-value]
    except (ValueError, TypeError):
        pass

    result = default
    for key, value in accepted.items():
        try:
            result = replace(result, **{key: value})  # type: ignore[type-var]
        except (ValueError, TypeError) as exc:
            issues.append(
                ConfigIssue(
                    IssueKind.REJECTED_VALUE,
                    f"{section}.{key}",
                    f"значение {value!r} отвергнуто проверкой: {exc}; оставлено умолчание",
                )
            )
    return result


_PROFILE_HINT = (
    "поле профиля закрыто скринингом и настройкой не является: его изменение "
    "означает, что прибор другой — см. KB_01 и умолчания fbg/core/profile.py"
)


# --------------------------------------------------------------------------------------
# Чтение и запись
# --------------------------------------------------------------------------------------


def to_json(config: AppConfig) -> dict[str, object]:
    """Содержимое файла настроек как словарь."""
    defaults = AppConfig()
    return {
        "version": CONFIG_VERSION,
        "device_model": config.device_model,
        "serial": config.serial,
        "firmware": config.firmware,
        "calibration_path": str(config.calibration_path),
        "endpoint": _section_to_json(config.endpoint),
        "profile": {
            name: getattr(config.profile, name)
            for name in sorted(PROFILE_SETTABLE)
            if getattr(config.profile, name) != getattr(defaults.profile, name)
        },
        "session": _section_to_json(config.session),
        "pipeline": _section_to_json(config.pipeline),
        "recorder": _section_to_json(config.recorder, skip=IDENTITY_FIELDS),
        "packet_log": _section_to_json(config.packet_log, skip=IDENTITY_FIELDS),
    }


_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "device_model",
        "serial",
        "firmware",
        "calibration_path",
        "endpoint",
        "profile",
        "session",
        "pipeline",
        "recorder",
        "packet_log",
    }
)


def from_json(raw: Mapping[str, object]) -> tuple[AppConfig, list[ConfigIssue]]:
    """Собирает настройки из уже разобранного JSON. Версию проверяет `load`."""
    issues: list[ConfigIssue] = []
    defaults = AppConfig()

    for key in raw:
        if key not in _TOP_LEVEL_KEYS:
            issues.append(
                ConfigIssue(IssueKind.UNKNOWN_SECTION, key, "раздел неизвестен и пропущен")
            )

    def scalar(key: str, annotation: object, fallback: object) -> object:
        if key not in raw:
            return fallback
        try:
            return _coerce(raw[key], annotation)
        except _TypeError as exc:
            issues.append(ConfigIssue(IssueKind.WRONG_TYPE, key, f"{exc}; оставлено умолчание"))
            return fallback

    config = AppConfig(
        endpoint=_build_section(
            Endpoint, defaults.endpoint, raw.get("endpoint"), "endpoint", issues
        ),
        profile=_build_section(
            DeviceProfile,
            defaults.profile,
            raw.get("profile"),
            "profile",
            issues,
            allowed=PROFILE_SETTABLE,
            not_settable_hint=_PROFILE_HINT,
        ),
        session=_build_section(
            SessionConfig, defaults.session, raw.get("session"), "session", issues
        ),
        pipeline=_build_section(
            PipelineConfig, defaults.pipeline, raw.get("pipeline"), "pipeline", issues
        ),
        recorder=_build_section(
            RecorderConfig, defaults.recorder, raw.get("recorder"), "recorder", issues
        ),
        packet_log=_build_section(
            PacketLogConfig, defaults.packet_log, raw.get("packet_log"), "packet_log", issues
        ),
        calibration_path=typing.cast(
            Path, scalar("calibration_path", Path, defaults.calibration_path)
        ),
        device_model=typing.cast(str, scalar("device_model", str, defaults.device_model)),
        serial=typing.cast(int | None, scalar("serial", int | None, defaults.serial)),
        firmware=typing.cast(str | None, scalar("firmware", str | None, defaults.firmware)),
    )
    return config, issues


def _migrate(
    raw: dict[str, object], from_version: int, issues: list[ConfigIssue]
) -> dict[str, object]:
    """Проводит файл от его версии до текущей шагами из `MIGRATIONS`."""
    data = raw
    version = from_version
    while version < CONFIG_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            issues.append(
                ConfigIssue(
                    IssueKind.VERSION_INVALID,
                    "version",
                    f"нет преобразования из версии {version} в {version + 1}",
                )
            )
            return data
        data = step(data)
        version += 1
    return data


def load(path: Path | None = None) -> LoadResult:
    """Читает настройки. Ничего не создаёт и не бросает исключений.

    Отсутствие файла, битый JSON, чужая кодировка, права — всё это замечания
    и умолчания, а не отказ старта: пользователь правит этот файл Блокнотом.
    """
    target = path or default_path()
    issues: list[ConfigIssue] = []

    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LoadResult(config=AppConfig(), path=target, existed=False)
    except (OSError, UnicodeDecodeError) as exc:
        issues.append(
            ConfigIssue(
                IssueKind.FILE_UNREADABLE,
                target.name,
                f"файл не прочитан ({exc}); приложение запущено на умолчаниях",
            )
        )
        return LoadResult(
            config=AppConfig(), issues=tuple(issues), path=target, existed=True, readable=False
        )

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        issues.append(
            ConfigIssue(
                IssueKind.FILE_UNREADABLE,
                target.name,
                f"не разобрался как JSON (строка {exc.lineno}, позиция {exc.colno}: {exc.msg}); "
                "приложение запущено на умолчаниях",
            )
        )
        return LoadResult(
            config=AppConfig(), issues=tuple(issues), path=target, existed=True, readable=False
        )

    if not isinstance(raw, dict):
        issues.append(
            ConfigIssue(
                IssueKind.FILE_UNREADABLE,
                target.name,
                f"на верхнем уровне ожидался объект, получен {type(raw).__name__}",
            )
        )
        return LoadResult(
            config=AppConfig(), issues=tuple(issues), path=target, existed=True, readable=False
        )

    raw_version = raw.get("version")
    if isinstance(raw_version, bool) or not isinstance(raw_version, int) or raw_version < 1:
        issues.append(
            ConfigIssue(
                IssueKind.VERSION_INVALID,
                "version",
                f"версия формата {raw_version!r} не читается; приложение запущено на умолчаниях",
            )
        )
        return LoadResult(
            config=AppConfig(), issues=tuple(issues), path=target, existed=True, readable=False
        )

    if raw_version > CONFIG_VERSION:
        issues.append(
            ConfigIssue(
                IssueKind.VERSION_NEWER,
                "version",
                f"файл версии {raw_version}, приложение понимает {CONFIG_VERSION}; "
                "файл не читается целиком, чтобы не потерять его новые поля",
            )
        )
        return LoadResult(
            config=AppConfig(),
            issues=tuple(issues),
            path=target,
            existed=True,
            file_version=raw_version,
            readable=False,
        )

    data = _migrate(raw, raw_version, issues)
    config, section_issues = from_json(data)
    issues.extend(section_issues)
    return LoadResult(
        config=config,
        issues=tuple(issues),
        path=target,
        existed=True,
        file_version=raw_version,
        readable=True,
    )


def save(config: AppConfig, path: Path | None = None) -> Path:
    """Сохраняет настройки. Возвращает путь записанного файла.

    Запись атомарная — через временный файл рядом и `os.replace`, — поэтому
    оборванное сохранение не оставляет полуфайла настроек.

    Файл, который `load` прочитать не смог (битый JSON, версия новее),
    не затирается, а переименовывается в `<имя>.bad`. Причина одна и та же
    для обоих случаев: там лежит то, чего мы не понимаем, а стирание —
    единственное необратимое действие, доступное этому модулю.

    `OSError` не перехватывается: не записавшиеся настройки — отказ, о котором
    пользователь обязан узнать, а не поле в отчёте.
    """
    target = path or default_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not load(target).readable:
        target.replace(target.with_name(target.name + ".bad"))

    payload = json.dumps(to_json(config), ensure_ascii=False, indent=2, sort_keys=False)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


# --------------------------------------------------------------------------------------
# Файл калибровок
# --------------------------------------------------------------------------------------


def load_sensors(path: Path) -> tuple[tuple[Sensor, ...], tuple[ConfigIssue, ...]]:
    """Читает набор датчиков. Испорченная запись стоит одной записи.

    Отсутствие файла — не ошибка и даже не замечание: калибровок может
    не быть вовсе, и приложение обязано работать без них (сырые нанометры
    пишутся всегда, правило №4).
    """
    issues: list[ConfigIssue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (), ()
    except (OSError, UnicodeDecodeError) as exc:
        return (), (ConfigIssue(IssueKind.FILE_UNREADABLE, path.name, f"файл не прочитан: {exc}"),)

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return (), (
            ConfigIssue(IssueKind.FILE_UNREADABLE, path.name, f"не разобрался как JSON: {exc.msg}"),
        )

    if not isinstance(raw, Mapping) or not isinstance(raw.get(SENSORS_KEY), list):
        return (), (
            ConfigIssue(
                IssueKind.FILE_UNREADABLE,
                path.name,
                f"ожидался объект с массивом {SENSORS_KEY!r}",
            ),
        )

    # Файлы до чата №15 использовали абсолютный полином c0+c1·λ+c2·λ².
    # Автоматической миграции нет: коэффициенты могли быть введены вручную,
    # а молча принять их как новую опорную форму означало бы получить
    # правдоподобные, но неверные величины. Такой файл считается целиком
    # непонятым и при следующем сохранении откладывается по KB_05 №33.
    legacy_fields = {"c0", "c1", "c2"}
    if any(
        isinstance(item, Mapping) and bool(legacy_fields.intersection(item))
        for item in raw[SENSORS_KEY]
    ):
        return (), (
            ConfigIssue(
                IssueKind.FILE_UNREADABLE,
                path.name,
                "устаревшая абсолютная форма калибровки c0/c1/c2; "
                "автоматическая миграция не выполняется",
            ),
        )

    sensors: list[Sensor] = []
    for index, item in enumerate(raw[SENSORS_KEY]):
        location = f"{SENSORS_KEY}[{index}]"
        if not isinstance(item, Mapping):
            issues.append(ConfigIssue(IssueKind.WRONG_TYPE, location, "ожидался объект"))
            continue
        try:
            sensors.append(sensor_from_json(item))
        except ValueError as exc:
            issues.append(ConfigIssue(IssueKind.REJECTED_VALUE, location, str(exc)))
    return tuple(sensors), tuple(issues)


def save_sensors(sensors: Sequence[Sensor], path: Path) -> Path:
    """Записывает датчики атомарно; непонятый прежний файл сначала откладывает."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _loaded, issues = load_sensors(path)
        if any(issue.kind is IssueKind.FILE_UNREADABLE for issue in issues):
            backup = path.with_name(path.name + ".bad")
            suffix = 2
            while backup.exists():
                backup = path.with_name(f"{path.name}.bad.{suffix}")
                suffix += 1
            path.replace(backup)
    payload = json.dumps(sensors_to_json(sensors), ensure_ascii=False, indent=2)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
