"""Диагностика неудачного подключения — лестница из KB_01, а не «ошибка связи».

Человек за стендом должен понять, что делать, не открывая документацию.
Поэтому текст не один, а список проверок, и каждая из них по возможности
**уже сделана кодом**: счётчики транспорта отвечают на половину вопросов
лестницы сами.

Что именно спрашивается и почему (KB_01, раздел «Сеть»):

* прибор отвечает на `255.255.255.255:8001` **широковещательно**, с порта
  4567 — значит bind обязан быть на `0.0.0.0`, иначе Windows эти датаграммы
  приложению не отдаст, и отказ будет выглядеть как «сеть не работает» при
  полностью исправной сети (Р29);
* IP компьютера на стенде — `192.168.0.14`, прибор настроен на эту подсеть;
* входящий UDP/8001 обязан быть разрешён брандмауэром (риск R5);
* остаток проверяется Wireshark: уходит ли наш пакет и приходит ли ответ.

Разделение модуля на чистую часть и одну грязную функцию сделано ради тестов:
`diagnose` получает адреса компьютера параметром и проверяется без сети,
а `local_ipv4_addresses` их добывает и ошибается молча — не сумели узнать
адреса, значит проверка остаётся на человеке, а не превращается в отказ.
"""

import socket
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from fbg.core.endpoint import Endpoint
from fbg.core.session import SessionError, SessionErrorKind
from fbg.core.transport import TransportStats
from fbg.ui import texts


class Verdict(StrEnum):
    """Что известно про пункт лестницы."""

    OK = "ok"
    """Проверено кодом и претензий нет."""

    SUSPECT = "suspect"
    """Проверено кодом, и это вероятная причина отказа."""

    CHECK = "check"
    """Кодом не проверяется: делает человек."""


@dataclass(frozen=True)
class Check:
    """Один пункт лестницы."""

    title: str
    detail: str
    verdict: Verdict


@dataclass(frozen=True)
class Diagnosis:
    """Итог: короткий вывод и лестница проверок."""

    headline: str
    checks: tuple[Check, ...] = ()

    @property
    def suspects(self) -> tuple[Check, ...]:
        """Пункты, названные вероятной причиной."""
        return tuple(check for check in self.checks if check.verdict is Verdict.SUSPECT)


#: Значок пункта лестницы. Три исхода и все различимы: проверено кодом,
#: названо вероятной причиной, остаётся человеку.
VERDICT_MARKS: dict[Verdict, str] = {
    Verdict.OK: "[ok]",
    Verdict.SUSPECT: "[!!]",
    Verdict.CHECK: "[ ? ]",
}


def format_diagnosis(diagnosis: Diagnosis) -> str:
    """Текст лестницы целиком — то, что читает человек за стендом.

    Живёт здесь, а не в панели: это подстановка строк, и проверяется она
    без окна, как и всё остальное в этом модуле.
    """
    lines = [diagnosis.headline, ""]
    for check in diagnosis.checks:
        lines.append(f"{VERDICT_MARKS[check.verdict]} {check.title}")
        lines.append(f"      {check.detail}")
    return "\n".join(lines)


def local_ipv4_addresses() -> tuple[str, ...]:
    """IPv4-адреса этого компьютера. Пустой кортеж — узнать не удалось.

    Риск R4 из KB_03: ПК сменил IP, и прибор шлёт «в никуда». Полного обхода
    интерфейсов в стандартной библиотеке нет, поэтому берётся то, что доступно
    переносимо: адреса, на которые разрешается имя хоста. Этого достаточно,
    чтобы заметить компьютер в чужой подсети, и недостаточно, чтобы на этом
    что-то утверждать наверняка — отсюда `Verdict.CHECK` при пустом ответе.

    Ошибки глотаются намеренно: диагностика не имеет права падать в тот
    момент, когда всё остальное уже не работает.
    """
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
    except OSError:
        return ()
    addresses = {str(info[4][0]) for info in infos}
    return tuple(sorted(addresses))


def _same_subnet(first: str, second: str) -> bool:
    """Совпадают ли первые три октета — грубая проверка «та же подсеть /24».

    Маску никто не сообщал, и узнать её переносимо нельзя. /24 берётся потому,
    что подсеть стенда именно такая (`192.168.0.0/24`, KB_01); ошибка этой
    оценки стоит лишнего пункта в списке, а не неверного вывода — вердикт
    из неё выводится только вместе со счётчиками.
    """
    left = first.split(".")
    right = second.split(".")
    if len(left) != 4 or len(right) != 4:
        return False
    return left[:3] == right[:3]


def _bind_check(endpoint: Endpoint) -> Check:
    """Bind на `0.0.0.0` — единственный пункт, который проверяется полностью."""
    if endpoint.local_ip == "0.0.0.0":
        return Check(
            "Приём на 0.0.0.0",
            f"порт {endpoint.local_port}, адрес любой — так и должно быть",
            Verdict.OK,
        )
    return Check(
        "Приём на 0.0.0.0",
        f"в настройках стоит {endpoint.local_ip}. {texts.LOCAL_IP_LOCKED}. "
        "Пока адрес не 0.0.0.0, ответы прибора до приложения не дойдут вовсе",
        Verdict.SUSPECT,
    )


def _address_check(endpoint: Endpoint, local_addresses: Sequence[str]) -> Check:
    """IP компьютера: та ли подсеть."""
    title = "IP компьютера"
    if not local_addresses:
        return Check(
            title,
            f"адреса компьютера определить не удалось. На стенде IP ПК — "
            f"{texts.EXPECTED_LOCAL_IP}; проверьте `ipconfig`",
            Verdict.CHECK,
        )
    listed = ", ".join(local_addresses)
    if texts.EXPECTED_LOCAL_IP in local_addresses:
        return Check(title, f"{texts.EXPECTED_LOCAL_IP} на месте (найдено: {listed})", Verdict.OK)
    if any(_same_subnet(address, endpoint.device_ip) for address in local_addresses):
        return Check(
            title,
            f"подсеть прибора {endpoint.device_ip} видна (найдено: {listed}), "
            f"но стендового {texts.EXPECTED_LOCAL_IP} среди адресов нет",
            Verdict.CHECK,
        )
    return Check(
        title,
        f"ни один адрес компьютера не лежит в подсети прибора {endpoint.device_ip} "
        f"(найдено: {listed}). На стенде IP ПК — {texts.EXPECTED_LOCAL_IP}",
        Verdict.SUSPECT,
    )


def _send_check(stats: TransportStats | None) -> Check:
    """Уходят ли команды из сокета."""
    title = "Команды уходят"
    if stats is None or stats.commands_sent == 0:
        return Check(title, "ни одной команды ещё не отправлено", Verdict.CHECK)
    return Check(
        title,
        f"отправлено {stats.commands_sent}, {stats.bytes_sent} Б — сокет их принял",
        Verdict.OK,
    )


def _receive_check(endpoint: Endpoint, stats: TransportStats | None) -> Check:
    """Приходят ли ответы. Здесь же ловится брандмауэр (риск R5)."""
    title = "Ответы приходят"
    if stats is None:
        return Check(title, "связь ещё не открывалась", Verdict.CHECK)
    if stats.datagrams_received > 0:
        return Check(
            title,
            f"принято {stats.datagrams_received} датаграмм — "
            f"порт {endpoint.local_port} открыт и брандмауэр не мешает",
            Verdict.OK,
        )
    if stats.foreign_datagrams > 0:
        return Check(
            title,
            f"от прибора {endpoint.device_ip} — ничего, но {stats.foreign_datagrams} датаграмм "
            "пришло с другого адреса. Либо в настройках не тот IP прибора, "
            "либо в подсети работает второй прибор",
            Verdict.SUSPECT,
        )
    return Check(
        title,
        f"не принято ни одной датаграммы. Проверьте правило брандмауэра Windows "
        f"на **входящий** UDP/{endpoint.local_port}: команды уходят и без него, "
        "а ответы блокируются",
        Verdict.SUSPECT,
    )


def _broadcast_note(endpoint: Endpoint) -> Check:
    """Напоминание про широковещание — оно объясняет остальные пункты."""
    return Check(
        "Прибор отвечает широковещательно",
        f"все ответы уходят на 255.255.255.255:{endpoint.local_port} с порта "
        f"{endpoint.device_port}, а не на IP компьютера (✅ скрининг, 4937 ответов). "
        "Отсюда и требование к приёму на 0.0.0.0, и то, что трафик прибора "
        "видят все узлы подсети",
        Verdict.CHECK,
    )


def _icmp_note(stats: TransportStats | None) -> Check | None:
    """ICMP-сбросы: прибор недоступен, и Windows сообщает об этом ошибкой сокета."""
    if stats is None or stats.icmp_resets == 0:
        return None
    return Check(
        "ICMP «port unreachable»",
        f"{stats.icmp_resets} раз ядро вернуло ConnectionResetError вместо датаграммы. "
        "Так выглядит недоступный или выключенный прибор: команда ушла, "
        "а по адресу назначения никого нет",
        Verdict.SUSPECT,
    )


def _wireshark_check() -> Check:
    """Последняя ступень: то, что кодом не проверяется в принципе."""
    return Check(
        "Wireshark",
        f"фильтр `{texts.WIRESHARK_FILTER}`. Уходит ли наш пакет на "
        "прибор и приходит ли ответ — это разделяет отказ отправки, "
        "молчание прибора и потерю ответа по дороге",
        Verdict.CHECK,
    )


def _headline(endpoint: Endpoint, stats: TransportStats | None, error: SessionError | None) -> str:
    """Короткий вывод: то единственное, что оператор прочитает наверняка."""
    if error is None:
        return "Связь установлена."
    if error.kind is SessionErrorKind.SEND_FAILED:
        return "Датаграмма не ушла из сокета: отказала отправка, а не прибор."
    if error.kind is SessionErrorKind.WRONG_STATE:
        return f"Команда недопустима в текущем состоянии: {error.message}"
    if stats is not None and stats.commands_sent > 0 and stats.datagrams_received == 0:
        if stats.icmp_resets > 0:
            return "Команды уходят, ответов нет, и ядро сообщает о недоступном адресе."
        if stats.foreign_datagrams > 0:
            return "Команды уходят; отвечает кто-то другой, не прибор из настроек."
        return (
            f"Команды уходят, ответов нет. Так выглядит закрытый входящий "
            f"UDP/{endpoint.local_port} и так же — молчащий прибор."
        )
    if error.kind is SessionErrorKind.TIMEOUT:
        return "Прибор не ответил за отведённое время."
    return f"Подключиться не удалось: {error.message}"


def diagnose(
    endpoint: Endpoint,
    *,
    stats: TransportStats | None = None,
    error: SessionError | None = None,
    local_addresses: Sequence[str] = (),
) -> Diagnosis:
    """Собирает лестницу проверок под конкретный отказ.

    Порядок пунктов — от того, что проверено кодом, к тому, что делает
    человек: сначала ответы, которые уже есть, потом вопросы.
    """
    checks: list[Check] = [
        _bind_check(endpoint),
        _address_check(endpoint, local_addresses),
        _send_check(stats),
        _receive_check(endpoint, stats),
    ]
    icmp = _icmp_note(stats)
    if icmp is not None:
        checks.append(icmp)
    checks.append(_broadcast_note(endpoint))
    checks.append(_wireshark_check())
    return Diagnosis(headline=_headline(endpoint, stats, error), checks=tuple(checks))
