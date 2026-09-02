"""Приёмный тракт телеметрии: разбор, кольцевая история, децимация для UI, метрики.

Pipeline — единственный потребитель телеметрии сессии. Сессия отдаёт сырые байты
кадра колбэком `on_telemetry` из своего потока-диспетчера; всё, что происходит
с кадром дальше, происходит здесь.

Три потребителя, три механизма
------------------------------
Требования потребителей противоположны, поэтому общего механизма у них нет.

===================  ==========================  ==========================
Потребитель          Что ему нужно               Чем обслуживается
===================  ==========================  ==========================
запись в файл        все кадры, по порядку,      `FrameCursor` — курсор
                     потеря только с отметкой    по кольцу с номерами
таблица              последний кадр и агрегаты,  `UiSnapshot` — публикуемый
                     ~20 Гц                      снимок, децимация по времени
график               выбранные слоты истории    `TraceHistorySnapshot` — копия
                     по запросу UI               только выбранных столбцов
метрики связи        темп и счётчики             `PipelineMetrics` — расчёт
                                                 из того же кольца по запросу
===================  ==========================  ==========================

Общее хранилище одно — кольцо, но **дисциплины чтения разные**, и это главное
проектное решение модуля. Писателю нужна полнота: он идёт по кольцу
последовательно, своим номером кадра, и по разрыву номеров узнаёт, сколько
кадров его обогнало. UI нужна свежесть: таблица получает готовый снимок последнего кадра,
собранный отдельным потоком не чаще `ui_period_s`, а график по таймеру
запрашивает отдельную копию только выбранных столбцов истории. Само кольцо
виджетам не отдаётся. Один буфер, из которого оба читали бы одинаково, обслужил бы
плохо обоих: очередь на выброс старейшего теряла бы кадры у писателя,
а очередь без выброса упиралась бы в UI и останавливала приём.

Почему у писателя курсор, а не очередь
--------------------------------------
Очередь кадров для писателя означала бы копию каждого кадра при постановке
в очередь — вторую копию тех же данных при 2000 кадрах/с. Кольцо уже хранит
все кадры в порядке приёма; курсор поверх него даёт ровно то, чего требует
запись — полноту, порядок и **обнаружение потери** по разрыву номеров, —
не заводя второй копии и не заставляя поток приёма ждать читателя.

Что делает поток приёма
-----------------------
`on_telemetry` вызывается из потока-диспетчера транспорта, и в нём нет ничего
долгого: разбор кадра (21 мкс, замер чата №1), запись строк в предвыделенные
массивы кольца, инкремент счётчиков и — не чаще `ui_period_s` — `Event.set()`
для потока-публикатора. Ни файлов, ни Qt, ни ожидания на блокировках.
Агрегаты считает поток-публикатор, а не поток приёма.

Согласованность чтения без блокировок
-------------------------------------
Кольцо пишет ровно один поток; номер `written` увеличивается **после** записи
строки. Читатели работают по схеме seqlock: берут диапазон номеров, копируют
данные, затем проверяют, что начало диапазона всё ещё не вытеснено. Гонка
физически требует, чтобы за время копирования поток приёма прошёл почти всё
кольцо (при 20 000 кадрах и 2 кГц — 10 секунд), поэтому проверка практически
никогда не срабатывает; она нужна для корректности, а не для типичного случая.
Блокировки нет намеренно: она стояла бы в потоке приёма.

Чего pipeline не делает
-----------------------
Не привязывает датчик к позиции в кадре. Позиции — слоты, заполняемые
по возрастанию λ по мере обнаружения (KB_04, N15; решение Р30): 1551.5 нм
на стенде лежит в позиции 1, будучи четвёртой решёткой линии. Привязка
к физическому датчику делается по длине волны с допуском и относится
к калибровке. Здесь отдаётся то, что пришло: позиция, длина волны,
признак валидности.

Не пишет файлов и не трогает Qt. Не считает переменное число валидных пиков
ошибкой: это штатное поведение прибора, а не сбой связи.
"""

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np

from fbg.core import codec
from fbg.core.frames import MeasurementFrame
from fbg.core.profile import C_NM_GHZ, DeviceProfile

#: Размер кольца по умолчанию, кадров.
#:
#: 20 000 кадров — 10 секунд потока при 2000 Гц и 20.3 МБ памяти
#: (см. `RingHistory.nbytes`). Пять минут истории, о которых легко подумать
#: «пусть будет с запасом», стоили бы 610 МБ: 600 000 кадров по 1016 байт.
#:
#: Кольцо — не архив, а развязка между потоком приёма и писателем: архивом
#: служит файл. Десяти секунд хватает на паузу диска, ротацию файла и любой
#: разумный затык записи; если писатель отстал сильнее, это отказ, который
#: нужно показать отметкой разрыва, а не прятать за большим буфером.
DEFAULT_HISTORY_FRAMES = 20_000

#: Период обновления UI по умолчанию — 20 Гц (KB_03, риск R8).
DEFAULT_UI_PERIOD_S = 0.05

#: Окно скользящих агрегатов по умолчанию, секунды.
DEFAULT_AGGREGATE_WINDOW_S = 1.0

#: Окно измерения фактического темпа, секунды и кадры (Р65).
#:
#: Реальный поток 2 кГц приходит пачками: мгновенный темп на коротком окне
#: гуляет при нулевой сквозной потере. Оператору нужна долгосрочная скорость,
#: поэтому метрика использует всю доступную десятисекундную историю. Для
#: медленных приборов ограничение по времени по-прежнему не даёт усреднять
#: минуты данных.
DEFAULT_RATE_WINDOW_S = 10.0
DEFAULT_RATE_WINDOW_FRAMES = DEFAULT_HISTORY_FRAMES


@dataclass(frozen=True)
class PipelineConfig:
    """Параметры приёмного тракта.

    Значения по умолчанию рассчитаны на паспортные 2000 Гц и 4×30 позиций;
    при другой конфигурации прибора меняется параметр, а не код.
    """

    history_frames: int = DEFAULT_HISTORY_FRAMES
    """Глубина кольцевой истории в кадрах."""

    ui_period_s: float = DEFAULT_UI_PERIOD_S
    """Минимальный интервал между снимками для UI. Децимация по времени, не по счёту."""

    aggregate_window_s: float = DEFAULT_AGGREGATE_WINDOW_S
    """Окно скользящих агрегатов."""

    rate_window_s: float = DEFAULT_RATE_WINDOW_S
    """Окно измерения фактического темпа кадров."""

    rate_window_frames: int = DEFAULT_RATE_WINDOW_FRAMES
    """Потолок того же окна в кадрах."""

    expected_rate_hz: float | None = None
    """Ожидаемый темп, Гц. Берётся из прочитанной скорости развёртки (`10 04`)."""

    def __post_init__(self) -> None:
        """Проверяет согласованность параметров: некорректные — баг вызывающего."""
        if self.history_frames < 1:
            raise ValueError(f"history_frames={self.history_frames} должен быть ≥ 1")
        for name, value in (
            ("ui_period_s", self.ui_period_s),
            ("aggregate_window_s", self.aggregate_window_s),
            ("rate_window_s", self.rate_window_s),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} должен быть положительным")
        if self.rate_window_frames < 2:
            raise ValueError("rate_window_frames должен быть ≥ 2: темп считается по интервалам")
        if self.expected_rate_hz is not None and self.expected_rate_hz <= 0:
            raise ValueError("expected_rate_hz должен быть положительным либо None")


# --------------------------------------------------------------------------------------
# Кольцевая история
# --------------------------------------------------------------------------------------


class RingHistory:
    """Кольцевая история кадров: фиксированные массивы, без аллокаций на кадр.

    Пишет ровно один поток — тот, из которого приходит телеметрия. Номер
    `written` увеличивается **после** заполнения строки: читатель, увидевший
    номер, видит и данные.

    Логический номер кадра сквозной и не сбрасывается при обороте кольца.
    Кадр с номером `seq` доступен, пока `seq >= written - capacity`.
    """

    __slots__ = ("capacity", "case_temp_c", "filled", "freq_ghz", "lag_s", "t_mono", "written")

    def __init__(self, capacity: int, channels: int, fbg_per_channel: int) -> None:
        self.capacity = capacity
        self.t_mono = np.zeros(capacity, dtype=np.float64)
        """Момент приёма датаграммы (`perf_counter`), проставленный транспортом."""

        self.freq_ghz = np.full((capacity, channels, fbg_per_channel), np.nan, dtype=np.float64)
        """Частоты позиций, ГГц. NaN — пик не найден (KB_05 №7)."""

        self.case_temp_c = np.full((capacity, channels), np.nan, dtype=np.float64)
        self.filled = np.zeros((capacity, channels), dtype=np.int16)
        """Сколько позиций канала оказалось заполнено в этом кадре."""

        self.lag_s = np.zeros(capacity, dtype=np.float64)
        """Задержка от приёма датаграммы до конца обработки кадра приёмным потоком."""

        self.written = 0
        """Сколько кадров всего записано. Сквозной номер следующего кадра."""

    @property
    def nbytes(self) -> int:
        """Фактический объём кольца в байтах."""
        return int(
            self.t_mono.nbytes
            + self.freq_ghz.nbytes
            + self.case_temp_c.nbytes
            + self.filled.nbytes
            + self.lag_s.nbytes
        )

    @property
    def used(self) -> int:
        """Сколько кадров сейчас доступно для чтения."""
        return min(self.written, self.capacity)

    @property
    def oldest_seq(self) -> int:
        """Номер самого старого доступного кадра."""
        return max(0, self.written - self.capacity)

    def holds(self, seq: int) -> bool:
        """True, если кадр с этим номером ещё не вытеснен."""
        return self.oldest_seq <= seq < self.written

    def segments(self, start: int, stop: int) -> tuple[tuple[int, int], ...]:
        """Физические отрезки массивов для логического диапазона `[start, stop)`.

        Один отрезок, если диапазон не пересекает стык кольца, иначе два.
        Пустой диапазон даёт пустой кортеж.
        """
        if stop <= start:
            return ()
        capacity = self.capacity
        first = start % capacity
        count = stop - start
        if first + count <= capacity:
            return ((first, first + count),)
        return ((first, capacity), (0, count - (capacity - first)))

    def append(
        self,
        t_mono: float,
        freq_ghz: np.ndarray,
        case_temp_c: np.ndarray,
        filled: np.ndarray,
        lag_s: float,
    ) -> int:
        """Записывает кадр и возвращает его номер. Вызывается только из потока приёма."""
        index = self.written % self.capacity
        self.t_mono[index] = t_mono
        self.freq_ghz[index] = freq_ghz
        self.case_temp_c[index] = case_temp_c
        self.filled[index] = filled
        self.lag_s[index] = lag_s
        seq = self.written
        # Инкремент строго последним: читатель, увидевший номер, увидит и данные.
        self.written = seq + 1
        return seq


# --------------------------------------------------------------------------------------
# Агрегаты
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregates:
    """Скользящие агрегаты по окну истории, по каждой позиции отдельно.

    Считаются в ГГц — в тех единицах, в которых кадр пришёл, — и переводятся
    в нанометры на выходе. `mean_nm` при этом равна `c / mean_ghz`, а не
    среднему нанометров: для разброса ±0.05 нм около 1545 нм эти две величины
    расходятся примерно на 1.4·10⁻⁶ нм, то есть на четыре порядка ниже шага
    квантования прибора (8.01 пм). Минимум и максимум переводятся точно —
    преобразование монотонно, поэтому минимальной частоте соответствует
    максимальная длина волны.

    NaN не превращается в ноль ни в одной величине: позиция, не давшая пика
    ни в одном кадре окна, остаётся NaN и в среднем, и в размахе.
    """

    frames: int
    """Сколько кадров попало в окно."""

    span_s: float
    """Фактическая длительность окна, секунды."""

    valid: np.ndarray
    """Сколько кадров дали пик в этой позиции, форма (каналы, решётки)."""

    mean_ghz: np.ndarray
    min_ghz: np.ndarray
    max_ghz: np.ndarray

    @property
    def peak_to_peak_ghz(self) -> np.ndarray:
        """Размах частоты по окну, ГГц."""
        return self.max_ghz - self.min_ghz

    @property
    def mean_nm(self) -> np.ndarray:
        """Средняя длина волны, нм (см. оговорку в описании класса)."""
        return C_NM_GHZ / self.mean_ghz

    @property
    def min_nm(self) -> np.ndarray:
        """Минимальная длина волны: соответствует **максимальной** частоте."""
        return C_NM_GHZ / self.max_ghz

    @property
    def max_nm(self) -> np.ndarray:
        """Максимальная длина волны: соответствует **минимальной** частоте."""
        return C_NM_GHZ / self.min_ghz

    @property
    def peak_to_peak_nm(self) -> np.ndarray:
        """Размах длины волны по окну, нм."""
        return self.max_nm - self.min_nm


@dataclass(frozen=True)
class UiSnapshot:
    """Снимок для графика и таблицы: последний кадр плюс агрегаты.

    Неизменяем и самодостаточен: UI читает его, не касаясь кольца и не мешая
    ни приёму, ни записи. История здесь намеренно отсутствует: для графика
    есть отдельный узкий :class:`TraceHistorySnapshot` только выбранных слотов.

    Позиции — слоты кадра, а не номера датчиков (решение Р30): `freq_ghz[c][i]`
    означает «в канале c позицию i занял пик с такой частотой», и какой это
    физически датчик, решает калибровка по длине волны.
    """

    seq: int
    """Сквозной номер кадра в истории."""

    t_mono: float
    """Момент приёма датаграммы."""

    published_mono: float
    """Момент публикации снимка."""

    latency_s: float
    """Задержка от приёма кадра до публикации снимка."""

    freq_ghz: np.ndarray
    case_temp_c: np.ndarray
    filled: np.ndarray
    """Число заполненных позиций по каналам."""

    freq_divisor: int
    """Делитель единиц частоты, применённый при разборе кадра (KB_04, D1)."""

    aggregates: Aggregates

    @property
    def wavelength_nm(self) -> np.ndarray:
        """Длины волн последнего кадра, нм. NaN сохраняется в тех же позициях."""
        return C_NM_GHZ / self.freq_ghz

    @property
    def filled_total(self) -> int:
        """Сколько позиций заполнено во всём кадре."""
        return int(self.filled.sum())


@dataclass(frozen=True)
class TraceHistorySnapshot:
    """Копия истории только выбранных позиций для графика UI.

    Это второй вид снимка рядом с :class:`UiSnapshot`: последний кадр нужен
    всем панелям, а многотысячная история — только графику и только для тех
    позиций, которые отметил пользователь. Кольцо наружу не отдаётся (Р36).

    `wavelength_nm[:, i]` соответствует `positions[i]`. NaN сохраняется
    буквально: график обязан показать разрыв и не имеет права интерполировать
    ненайденный пик (KB_05 №7).
    """

    positions: tuple[tuple[int, int], ...]
    """Пары `(channel, position)`, оба индекса 0-based."""

    seq_start: int
    seq_stop: int
    t_mono: np.ndarray
    wavelength_nm: np.ndarray

    @property
    def frames(self) -> int:
        """Сколько кадров вошло в снимок истории."""
        return int(self.t_mono.size)

    @property
    def span_s(self) -> float:
        """Фактическая глубина истории по времени."""
        if self.t_mono.size < 2:
            return 0.0
        return float(self.t_mono[-1] - self.t_mono[0])


# --------------------------------------------------------------------------------------
# Пачка кадров для писателя
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameBatch:
    """Кадры подряд, снятые курсором: копия данных, принадлежащая читателю.

    `gap` — сколько кадров было вытеснено из кольца **до** этой пачки, то есть
    потеряно для этого читателя безвозвратно. Ноль означает, что разрыва нет.
    Потеря не молчаливая: писатель обязан отметить разрыв в файле.
    """

    seq_start: int
    """Логический номер первого прочитанного кадра."""

    seq_stop: int
    """Номер, следующий за последним прочитанным кадром."""

    gap: int
    """Сколько кадров пропущено перед пачкой из-за отставания читателя."""

    seq: np.ndarray
    """Сквозные номера отданных кадров. При `stride > 1` идут с шагом, а не подряд."""

    t_mono: np.ndarray
    freq_ghz: np.ndarray
    """Частоты, форма (кадры, каналы, решётки). NaN — пик не найден."""

    case_temp_c: np.ndarray
    filled: np.ndarray
    ingest_lag_s: np.ndarray
    """Задержка приёмного потока по каждому кадру."""

    taken_mono: float
    """Момент снятия пачки — по нему считается задержка доставки читателю."""

    def __len__(self) -> int:
        return int(self.t_mono.size)

    def wavelength_nm(self) -> np.ndarray:
        """Длины волн, нм — то, что пишется в файл всегда, независимо от калибровки."""
        return C_NM_GHZ / self.freq_ghz

    def latency_s(self) -> np.ndarray:
        """Задержка от приёма каждого кадра до момента снятия пачки, секунды.

        Включает интервал опроса читателя: если писатель забирает кадры раз
        в 100 мс, эта величина не может быть меньше его собственной паузы.
        """
        return self.taken_mono - self.t_mono


class FrameCursor:
    """Последовательный читатель кольца: полнота, порядок, отметка разрыва.

    Заводится тем, кому нужны все кадры — писателем файла. Курсор начинает
    с текущего момента: кадры, принятые до его создания, ему не отдаются.

    Курсор ничем не владеет и никого не блокирует. Если читатель отстал
    настолько, что кольцо его обогнало, потерянные кадры **не восстанавливаются
    и не подменяются**: их число приходит в `FrameBatch.gap`.

    `stride` — децимация записи «каждый N-й кадр». Отбор идёт по сквозному
    номеру (`seq % stride == 0`), а не по позиции в пачке: после разрыва
    выборка остаётся той же самой, а не сдвигается.
    """

    def __init__(self, pipeline: "Pipeline", *, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError(f"stride={stride} должен быть ≥ 1")
        self._pipeline = pipeline
        self._ring = pipeline.history
        self._stride = stride
        self._position = self._ring.written
        self._delivered = 0
        self._gaps = 0
        self._lost = 0
        self._retries = 0

    @property
    def position(self) -> int:
        """Номер кадра, с которого продолжится чтение."""
        return self._position

    @property
    def stride(self) -> int:
        """Шаг децимации записи."""
        return self._stride

    @property
    def delivered(self) -> int:
        """Сколько кадров курсор отдал читателю."""
        return self._delivered

    @property
    def lost(self) -> int:
        """Сколько кадров курсор пропустил из-за отставания."""
        return self._lost

    @property
    def gaps(self) -> int:
        """Сколько раз возникал разрыв."""
        return self._gaps

    @property
    def lag(self) -> int:
        """На сколько кадров читатель отстаёт от приёма прямо сейчас."""
        return max(0, self._ring.written - self._position)

    @property
    def torn_reads(self) -> int:
        """Сколько раз чтение пришлось повторить из-за вытеснения по ходу копирования."""
        return self._retries

    def take(self, limit: int | None = None) -> FrameBatch | None:
        """Снимает очередную пачку кадров. None — новых кадров нет.

        `limit` ограничивает размер пачки в кадрах **до** децимации: он нужен,
        чтобы писатель не получил разом всё кольцо после долгой паузы.
        """
        ring = self._ring
        for _ in range(3):
            end = ring.written
            if limit is not None:
                end = min(end, self._position + limit)
            oldest = ring.oldest_seq
            gap = max(0, oldest - self._position)
            start = max(self._position, oldest)
            if end <= start:
                if gap:
                    # Кольцо обогнало читателя целиком: отдавать нечего,
                    # но разрыв обязан быть учтён, а не потерян.
                    self._register_gap(gap)
                    self._position = start
                return None
            batch = self._copy(start, end, gap)
            if batch is not None:
                self._position = end
                self._delivered += len(batch)
                if gap:
                    self._register_gap(gap)
                # Пачка, целиком выпавшая по децимации, читателю не отдаётся:
                # кадры прочитаны и учтены, писать нечего.
                return batch if len(batch) else None
            self._retries += 1
        # Три подряд неудачных попытки означают, что читатель безнадёжно
        # отстаёт от приёма; честнее пропустить историю, чем крутиться здесь.
        self._position = ring.oldest_seq
        return None

    def _register_gap(self, gap: int) -> None:
        """Учитывает разрыв: число потерянных кадров и сам факт."""
        self._lost += gap
        self._gaps += 1

    def _copy(self, start: int, end: int, gap: int) -> FrameBatch | None:
        """Копирует диапазон и проверяет, что его не вытеснили по ходу копирования."""
        ring = self._ring
        segments = ring.segments(start, end)
        t_mono = _gather(ring.t_mono, segments)
        freq = _gather(ring.freq_ghz, segments)
        temp = _gather(ring.case_temp_c, segments)
        filled = _gather(ring.filled, segments)
        lag = _gather(ring.lag_s, segments)
        taken = time.perf_counter()
        if start < ring.oldest_seq:
            return None
        seq = np.arange(start, end, dtype=np.int64)
        if self._stride > 1:
            keep = seq % self._stride == 0
            seq, t_mono, freq = seq[keep], t_mono[keep], freq[keep]
            temp, filled, lag = temp[keep], filled[keep], lag[keep]
        return FrameBatch(start, end, gap, seq, t_mono, freq, temp, filled, lag, taken)


def _gather(array: np.ndarray, segments: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Собирает логически непрерывный кусок кольца в копию."""
    if len(segments) == 1:
        first, last = segments[0]
        return array[first:last].copy()
    return np.concatenate([array[first:last] for first, last in segments])


# --------------------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineMetrics:
    """Снимок метрик приёмного тракта.

    ⚠️ `loss_estimate` — **оценка, а не измерение**. В протоколе нет ни счётчиков
    кадров, ни последовательных номеров (KB_02), поэтому единственный доступный
    признак потери — просадка темпа относительно прочитанной скорости развёртки.
    Величина одинаково реагирует на потерю датаграмм в сети, на затык приёмника
    и на то, что прибор в действительности идёт не с той скоростью, которую
    сообщил. Принимать её за долю потерянных кадров нельзя.

    Если ожидаемый темп неизвестен (`10 04` ещё не прочитан), оценка равна None.
    Отрицательное значение означает, что факт выше ожидания, то есть неверно
    ожидание, — оно не обрезается нулём, чтобы это было видно.
    """

    frames: int
    """Сколько кадров принято и разобрано."""

    parse_errors: int
    """Сколько датаграмм не разобралось. Не потеря связи, а испорченный кадр."""

    frame_rate_hz: float
    """Фактический темп по меткам времени последних кадров. 0.0 — данных мало."""

    expected_rate_hz: float | None
    """Ожидаемый темп из прочитанной скорости развёртки."""

    loss_estimate: float | None
    """1 − факт/ожидание. Оценка, см. описание класса."""

    filled_by_channel: tuple[int, ...]
    """Число заполненных позиций по каналам в последнем кадре."""

    ingest_lag_s: float
    """Задержка приёмного потока на последнем кадре."""

    ui_latency_s: float
    """Задержка от приёма кадра до публикации последнего снимка UI."""

    ui_updates: int
    """Сколько снимков опубликовано."""

    ui_gates: int
    """Сколько раз децимация по времени пропустила кадр к публикатору."""

    history_frames: int
    history_used: int
    history_bytes: int
    evicted: int
    """Сколько кадров уже вытеснено из кольца.

    **Не потеря.** Вытеснение — штатная работа кольца: кадр уходит из истории,
    когда она заполнена, и для читателя, успевшего его забрать, ничего
    не произошло. Потерей это становится только для отставшего читателя,
    и считает её курсор — `FrameCursor.lost`.
    """

    errors: dict[str, int] = field(default_factory=dict)
    """Отказы разбора по видам `ParseErrorKind`."""


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class Pipeline:
    """Приёмный тракт телеметрии: кольцо, снимки для UI, метрики.

    Использование::

        pipeline = Pipeline(profile)
        session = Session(endpoint, profile, on_telemetry=pipeline.on_telemetry)
        pipeline.start()
        cursor = pipeline.cursor()          # для писателя файла
        ...
        snapshot = pipeline.snapshot()      # для UI, по таймеру 20 Гц
        batch = cursor.take()               # для писателя, в его потоке

    `on_telemetry` работает и без `start`: кадры складываются в кольцо, курсоры
    их получают. Поток, запускаемый `start`, нужен только снимкам UI — он снимает
    с приёмного потока расчёт агрегатов.
    """

    def __init__(
        self,
        profile: DeviceProfile | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._profile = profile or DeviceProfile()
        self._config = config or PipelineConfig()
        channels = self._profile.channels
        fbg = self._profile.fbg_per_channel

        self._history = RingHistory(self._config.history_frames, channels, fbg)
        self._buffer = MeasurementFrame(channels, fbg)
        """Буфер разбора: один на всё время жизни, кодек состояния не хранит (Р13)."""

        self._nan_mask = np.zeros((channels, fbg), dtype=bool)
        self._filled_scratch = np.zeros(channels, dtype=np.int64)
        self._fbg = fbg

        self._frames = 0
        self._parse_errors = 0
        self._errors: dict[str, int] = {}
        self._error_lock = threading.Lock()

        self._expected_rate_hz = self._config.expected_rate_hz
        self._ui_last_gate = -math.inf
        self._ui_gates = 0
        self._ui_updates = 0
        self._ui_latency_s = 0.0
        self._snapshot: UiSnapshot | None = None

        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None

    # --- Состояние ---------------------------------------------------------------------

    @property
    def profile(self) -> DeviceProfile:
        """Профиль, с которым разбираются кадры."""
        return self._profile

    @property
    def config(self) -> PipelineConfig:
        """Параметры тракта."""
        return self._config

    @property
    def history(self) -> RingHistory:
        """Кольцевая история. Читать через курсор; прямой доступ — для диагностики."""
        return self._history

    @property
    def sequence(self) -> int:
        """Сколько кадров принято всего. Номер следующего кадра."""
        return self._history.written

    @property
    def is_running(self) -> bool:
        """True, если поток-публикатор запущен."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def set_expected_rate(self, rate_hz: float | None) -> None:
        """Задаёт ожидаемый темп кадров, Гц — из прочитанной `10 04`.

        Отдельным вызовом, а не параметром конструктора: скорость развёртки
        становится известна только после подключения и может быть изменена
        командой `20 01` посреди работы.
        """
        if rate_hz is not None and rate_hz <= 0:
            raise ValueError(f"rate_hz={rate_hz} должен быть положительным либо None")
        self._expected_rate_hz = rate_hz

    # --- Жизненный цикл ----------------------------------------------------------------

    def start(self) -> None:
        """Запускает поток публикации снимков. Повторный вызов ничего не делает."""
        if self.is_running:
            return
        self._shutdown.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._publish_loop, name="fbg-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Останавливает поток публикации и дожидается его. Повторный вызов безвреден."""
        self._shutdown.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._config.ui_period_s + 5.0)
        self._thread = None

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # --- Приём кадров ------------------------------------------------------------------

    def on_telemetry(self, data: bytes, t_mono: float) -> None:
        """Колбэк сессии: разобрать кадр и положить в кольцо. Ничего долгого.

        Вызывается из потока-диспетчера транспорта. Здесь нет ни файлового
        I/O, ни Qt, ни ожидания на блокировках: разбор (21 мкс), запись строк
        в предвыделенные массивы, счётчики и — не чаще `ui_period_s` —
        `Event.set()` для потока-публикатора.

        Кадр, который не разобрался, в историю не попадает и считается
        в `parse_errors`. Переменное число валидных пиков ошибкой **не является**
        (KB_04, N15): прибор заполняет столько позиций, сколько нашёл.
        """
        parsed = codec.parse_measurement(data, self._profile, t_mono, out=self._buffer)
        if parsed.error is not None:
            self._parse_errors += 1
            name = parsed.error.kind.name
            with self._error_lock:
                self._errors[name] = self._errors.get(name, 0) + 1
            return

        frame = self._buffer
        # Число заполненных позиций по каналам — без аллокаций: маска и счётчик
        # предвыделены, обе операции пишут в готовые буферы.
        np.isnan(frame.freq_ghz, out=self._nan_mask)
        np.add.reduce(self._nan_mask, axis=1, dtype=np.int64, out=self._filled_scratch)
        np.subtract(self._fbg, self._filled_scratch, out=self._filled_scratch)

        self._history.append(
            t_mono,
            frame.freq_ghz,
            frame.case_temp_c,
            self._filled_scratch,
            time.perf_counter() - t_mono,
        )
        self._frames += 1

        # Децимация для UI — по времени кадра, а не по их числу: при 3 Гц
        # и при 2000 Гц публикатор просыпается одинаково часто.
        if t_mono - self._ui_last_gate >= self._config.ui_period_s:
            self._ui_last_gate = t_mono
            self._ui_gates += 1
            self._wake.set()

    # --- Снимки для UI -----------------------------------------------------------------

    def snapshot(self) -> UiSnapshot | None:
        """Последний опубликованный снимок. None — публиковать пока нечего.

        Читается из потока UI. Снимок неизменяем и уже собран, поэтому чтение
        не мешает ни приёму, ни записи, а медленный UI не создаёт подпора:
        пропущенные снимки просто не будут прочитаны.
        """
        return self._snapshot

    def trace_history(
        self,
        positions: Sequence[tuple[int, int]],
        history_s: float,
    ) -> TraceHistorySnapshot:
        """Копирует историю выбранных позиций за последние `history_s` секунд.

        UI не читает :attr:`history` напрямую (Р36). Публичный контракт здесь
        намеренно узкий: копируются только выбранные линии, поэтому обычные
        2–4 кривые не превращают каждый такт интерфейса в копирование всех
        120 позиций кольца.

        Глубина физически ограничена кольцом. Если пользователь запросил
        больше, возвращается вся доступная история — выдумывать прошлое нельзя.
        """
        if history_s <= 0:
            raise ValueError(f"history_s={history_s} должен быть положительным")
        selected = tuple(positions)
        for channel, position in selected:
            if not 0 <= channel < self._profile.channels:
                raise ValueError(f"канал {channel} вне диапазона 0…{self._profile.channels - 1}")
            if not 0 <= position < self._fbg:
                raise ValueError(f"позиция {position} вне диапазона 0…{self._fbg - 1}")

        ring = self._history
        end = ring.written
        if not selected or end == 0:
            return TraceHistorySnapshot(
                positions=selected,
                seq_start=end,
                seq_stop=end,
                t_mono=np.empty(0, dtype=np.float64),
                wavelength_nm=np.empty((0, len(selected)), dtype=np.float64),
            )

        channels = np.fromiter((channel for channel, _ in selected), dtype=np.intp)
        slots = np.fromiter((position for _, position in selected), dtype=np.intp)
        for _ in range(3):
            end = ring.written
            start, stamps = self._window_start(end, history_s, ring.capacity)
            blocks = [
                ring.freq_ghz[first:last][:, channels, slots].copy()
                for first, last in ring.segments(start, end)
            ]
            if not blocks:
                freq = np.empty((0, len(selected)), dtype=np.float64)
            elif len(blocks) == 1:
                freq = blocks[0]
            else:
                freq = np.concatenate(blocks, axis=0)

            # Seqlock-проверка после обеих копий. Если первый кадр успели
            # вытеснить, повторяем с новым диапазоном вместо смешения эпох.
            if start < ring.oldest_seq:
                continue
            nm = np.full(freq.shape, np.nan, dtype=np.float64)
            np.divide(C_NM_GHZ, freq, out=nm, where=np.isfinite(freq) & (freq > 0.0))
            return TraceHistorySnapshot(selected, start, end, stamps, nm)

        # Приём обогнал три копирования подряд. Возвращается честный пустой
        # снимок; UI попробует снова через 100 мс, а приём не ждёт ни секунды.
        end = ring.written
        return TraceHistorySnapshot(
            positions=selected,
            seq_start=end,
            seq_stop=end,
            t_mono=np.empty(0, dtype=np.float64),
            wavelength_nm=np.empty((0, len(selected)), dtype=np.float64),
        )

    def publish_now(self) -> UiSnapshot | None:
        """Собирает и публикует снимок немедленно, в потоке вызывающего.

        Нужна тестам и одиночным обновлениям UI (например, после `debug_once`),
        когда ждать очередного такта публикатора незачем.
        """
        snapshot = self._build_snapshot()
        if snapshot is not None:
            self._snapshot = snapshot
            self._ui_updates += 1
            self._ui_latency_s = snapshot.latency_s
        return snapshot

    def _publish_loop(self) -> None:
        """Поток-публикатор: собирает снимки, разгружая приёмный поток от агрегатов."""
        period = self._config.ui_period_s
        while not self._shutdown.is_set():
            if not self._wake.wait(period):
                continue
            self._wake.clear()
            if self._shutdown.is_set():
                return
            self.publish_now()

    def _build_snapshot(self) -> UiSnapshot | None:
        """Собирает снимок: последний кадр плюс скользящие агрегаты."""
        ring = self._history
        end = ring.written
        if end == 0:
            return None
        index = (end - 1) % ring.capacity
        t_mono = float(ring.t_mono[index])
        freq = ring.freq_ghz[index].copy()
        temp = ring.case_temp_c[index].copy()
        filled = ring.filled[index].copy()
        aggregates, window_start = self._aggregate(end)
        # Проверка после копирования: прочитанное могло быть вытеснено по ходу
        # чтения. Проверяется **начало окна агрегатов**, а не только последний
        # кадр: окно шире одного кадра, и уцелевший последний кадр ничего
        # не говорит о судьбе первого. При 20 000 кадрах и 2 кГц вытеснение
        # потребовало бы секунд, но корректность не должна опираться на то,
        # что «так не бывает».
        if window_start < ring.oldest_seq:
            return None
        published = time.perf_counter()
        return UiSnapshot(
            seq=end - 1,
            t_mono=t_mono,
            published_mono=published,
            latency_s=published - t_mono,
            freq_ghz=freq,
            case_temp_c=temp,
            filled=filled,
            freq_divisor=self._buffer.freq_divisor,
            aggregates=aggregates,
        )

    # --- Агрегаты и темп ---------------------------------------------------------------

    def _window_start(self, end: int, window_s: float, max_frames: int) -> tuple[int, np.ndarray]:
        """Находит начало временного окна и отдаёт метки времени этого окна.

        Окно ограничено и по времени, и по числу кадров: при 3 Гц второе
        ограничение не срабатывает, при 2000 Гц — первое.
        """
        ring = self._history
        count = min(end - ring.oldest_seq, max_frames)
        scan_start = end - count
        stamps = _gather(ring.t_mono, ring.segments(scan_start, end))
        if stamps.size == 0:
            return end, stamps
        cutoff = stamps[-1] - window_s
        first = int(np.searchsorted(stamps, cutoff, side="left"))
        return scan_start + first, stamps[first:]

    def _aggregate(self, end: int) -> tuple[Aggregates, int]:
        """Скользящие агрегаты по окну `aggregate_window_s`, устойчивые к NaN.

        Возвращает ещё и номер первого кадра окна: по нему вызывающий проверяет,
        что окно не вытеснили из кольца по ходу чтения.
        """
        ring = self._history
        start, stamps = self._window_start(end, self._config.aggregate_window_s, ring.capacity)
        shape = (self._profile.channels, self._fbg)
        if stamps.size == 0:
            empty_int = np.zeros(shape, dtype=np.int64)
            empty = np.full(shape, np.nan, dtype=np.float64)
            return Aggregates(0, 0.0, empty_int, empty, empty.copy(), empty.copy()), end

        valid: np.ndarray | None = None
        total: np.ndarray | None = None
        low: np.ndarray | None = None
        high: np.ndarray | None = None
        for first, last in ring.segments(start, end):
            block = ring.freq_ghz[first:last]
            # `nansum` и `fmin`/`fmax` пропускают NaN и не предупреждают
            # на полностью пустом срезе, в отличие от `nanmean` и `nanmin`.
            block_valid = block.shape[0] - np.count_nonzero(np.isnan(block), axis=0)
            block_total = np.nansum(block, axis=0)
            block_low = np.fmin.reduce(block, axis=0)
            block_high = np.fmax.reduce(block, axis=0)
            if valid is None:
                valid, total, low, high = block_valid, block_total, block_low, block_high
            else:
                assert total is not None and low is not None and high is not None
                valid = valid + block_valid
                total = total + block_total
                low = np.fmin(low, block_low)
                high = np.fmax(high, block_high)

        assert valid is not None and total is not None and low is not None and high is not None
        # Позиция, не давшая пика ни в одном кадре окна, остаётся NaN:
        # делить нечего, и подставлять ноль было бы выдумыванием данных.
        counted = np.where(valid > 0, valid, 1)
        mean = np.where(valid > 0, total / counted, np.nan)
        span = float(stamps[-1] - stamps[0]) if stamps.size > 1 else 0.0
        return (
            Aggregates(
                frames=int(stamps.size),
                span_s=span,
                valid=valid,
                mean_ghz=mean,
                min_ghz=low,
                max_ghz=high,
            ),
            start,
        )

    def frame_rate_hz(self) -> float:
        """Фактический темп кадров по меткам времени. 0.0 — кадров меньше двух.

        Считается по интервалам между принятыми кадрами, то есть отражает темп
        **после** всех потерь в тракте: и в сети, и в очереди транспорта.
        """
        ring = self._history
        end = ring.written
        if end - ring.oldest_seq < 2:
            return 0.0
        _, stamps = self._window_start(
            end, self._config.rate_window_s, self._config.rate_window_frames
        )
        if stamps.size < 2:
            # Окно по времени оказалось короче двух кадров: берём последнюю пару,
            # иначе при редком потоке темп был бы не измерим вовсе.
            stamps = _gather(ring.t_mono, ring.segments(end - 2, end))
        elapsed = float(stamps[-1] - stamps[0])
        if elapsed <= 0.0:
            return 0.0
        return float(stamps.size - 1) / elapsed

    def metrics(self) -> PipelineMetrics:
        """Снимок метрик. Считается по запросу, в потоке вызывающего."""
        ring = self._history
        rate = self.frame_rate_hz()
        expected = self._expected_rate_hz
        loss = None if expected is None or rate == 0.0 else max(0.0, 1.0 - rate / expected)
        end = ring.written
        if end:
            filled = tuple(int(value) for value in ring.filled[(end - 1) % ring.capacity])
            lag = float(ring.lag_s[(end - 1) % ring.capacity])
        else:
            filled = tuple(0 for _ in range(self._profile.channels))
            lag = 0.0
        with self._error_lock:
            errors = dict(self._errors)
        return PipelineMetrics(
            frames=self._frames,
            parse_errors=self._parse_errors,
            frame_rate_hz=rate,
            expected_rate_hz=expected,
            loss_estimate=loss,
            filled_by_channel=filled,
            ingest_lag_s=lag,
            ui_latency_s=self._ui_latency_s,
            ui_updates=self._ui_updates,
            ui_gates=self._ui_gates,
            history_frames=ring.capacity,
            history_used=ring.used,
            history_bytes=ring.nbytes,
            evicted=ring.oldest_seq,
            errors=errors,
        )

    # --- Читатели ----------------------------------------------------------------------

    def cursor(self, *, stride: int = 1) -> FrameCursor:
        """Заводит курсор для писателя. `stride` — запись каждого N-го кадра."""
        return FrameCursor(self, stride=stride)
