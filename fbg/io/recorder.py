"""Запись измерений в CSV: свой поток, ротация файлов, децимация, отметка разрывов.

Первый потребитель `FrameCursor` и первый модуль в `fbg/io`. Ему разрешены
файловый I/O и собственный поток; `fbg/core` он не меняет и Qt не тянет.

Кто кого ждёт
-------------
Никто. Recorder **забирает** кадры курсором в своём потоке — pipeline о нём
не знает и в его сторону ничего не вызывает. Отсюда следствие, ради которого
в чате №6 и выбрана схема с кольцом: ни форматирование, ни открытие нового
файла, ни затык диска не создают обратного давления на приёмный поток. Пока
recorder занят, кадры копятся в кольце (10 секунд при 2 кГц), а не теряются.
Смена файла на этом фоне стоит миллисекунды и потому не требует ни второго
файлового потока, ни двойной буферизации: механизм развязки уже есть, второй
был бы лишним.

Если recorder всё же отстал настолько, что кольцо его обогнало, потерянные
кадры **не восстанавливаются и не подменяются** — в файл уходит строка-маркер
`# GAP` с числом потерянных кадров.

Три состояния строки, и все три различимы
-----------------------------------------
Это главное содержательное требование формата (KB_05 №24, решение Р42):

=========================  ==========================================
Что произошло              Как выглядит в файле
=========================  ==========================================
кадр записан               строка с числами
кадр был, пиков нет        строка, где все `*_nm` равны `nan`
кадров не было             строка-комментарий `# GAP frames=…`
кадр сознательно пропущен  ничего; шаг `frame_no` равен `decimation`
=========================  ==========================================

Вторая и третья строки таблицы — разные события, и путать их нельзя: кадр без
пиков штатен (на стенде из четырёх решёток распознаются две, при отключённой
линии ноль), а разрыв означает потерю данных. Четвёртая — децимация: пропуск
по настройке разрывом **не считается** и маркером не отмечается.

Как постобработка отличает децимацию от потери
----------------------------------------------
`frame_no` — сквозной номер кадра **прибора** от старта записи, а не номер
строки. При `decimation=N` соседние записанные строки отличаются ровно на `N`,
и это указано в шапке. Скачок, не кратный `N`, означает потерю.

Оговорка, которую честнее записать, чем умолчать: при `N > 1` потеря кадров,
чьи номера и так не попадали в выборку, по одним номерам **не видна** — шаг
остаётся равным `N`. Поэтому маркер `# GAP` при децимации не украшение,
а единственная полная запись о потере, и арифметика по номерам (Р41) остаётся
достаточной только при `decimation=1`.

Что пишется
-----------
Сырые нанометры, всегда, независимо от калибровки (KB_05 №4): калибровки
в этом модуле нет вовсе. Пик не найден → `nan` в ячейке (KB_05 №7).

Надёжность
----------
Файл открыт в двоичном режиме, пишется строго последовательно и никогда
не перезаписывается — повреждённой середины не бывает по построению. Строки
формируются пачкой и уходят одним `write`, буфер сбрасывается не реже
`flush_period_s`, а также при ротации и остановке. При аварийном завершении
процесса теряется хвост буфера: последняя строка может оказаться недописанной,
все предыдущие целы.

Ошибка записи (диск заполнен, папка исчезла) останавливает **только запись**:
recorder закрывает файл, запоминает причину в `RecorderStats.error`, зовёт
`on_error` и завершает свой поток. Приём данных при этом не трогается — он
в другом потоке и о recorder не знает.
"""

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType

import numpy as np

from fbg.core.pipeline import FrameBatch, FrameCursor, Pipeline
from fbg.core.profile import DeviceProfile

#: Разделитель колонок. Точка остаётся десятичным разделителем.
SEPARATOR = ";"

#: Знаков после запятой в колонках длины волны.
#:
#: Сырое поле частоты хранит десятые доли ГГц (D1), то есть шаг прибора
#: по частоте — 0.1 ГГц. В нанометрах это λ²·δf/c: 0.00082 нм на длинном
#: краю развёртки (1568 нм) и 0.00078 нм на коротком. Четыре знака дают
#: шаг 0.0001 нм — в восемь раз мельче, чем прибор способен различить,
#: то есть значение восстанавливается однозначно. Пятый знак записывал бы
#: шум квантования и стоил бы 120 байт на строку.
NM_DECIMALS = 4

#: Знаков в колонках температуры: масштаб поля 0.01 °C (N2).
TEMP_DECIMALS = 2

#: Знаков в колонках времени. Микросекунда — разрешение `perf_counter`
#: и в полтора раза мельче межкадрового интервала при 2 кГц (500 мкс).
TIME_DECIMALS = 6

#: Ротация по времени, секунды.
DEFAULT_ROTATE_SECONDS = 600.0

#: Ротация по размеру, байты.
DEFAULT_ROTATE_BYTES = 100 << 20

#: Пауза потока записи, когда кадров нет.
DEFAULT_POLL_PERIOD_S = 0.01

#: Как часто буфер сбрасывается в ОС. Ограничивает потерю при падении процесса.
DEFAULT_FLUSH_PERIOD_S = 1.0

#: Потолок одной выборки курсора в кадрах **до** децимации. Ограничивает пик
#: памяти, если поток записи проспал: без него после паузы пришло бы всё кольцо.
DEFAULT_BATCH_LIMIT = 8192

#: Сколько строк форматируется и пишется за один `write`. Ротация по размеру
#: проверяется на границах таких кусков, поэтому кусок не должен быть большим.
DEFAULT_CHUNK_FRAMES = 512

#: Размер буфера файла.
FILE_BUFFER_BYTES = 1 << 20

#: Значение полей шапки, которых приложение не знает. Выдумывать нельзя (№10).
UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecorderConfig:
    """Параметры записи.

    Сведения о приборе (`serial`, `firmware`) сюда кладёт вызывающий из
    `DeviceConfig` сессии: recorder сам прибор не опрашивает и в `fbg.core.session`
    не заглядывает. Неизвестное поле остаётся `None` и попадает в шапку
    как `unknown`.
    """

    directory: Path
    """Куда складывать файлы. Создаётся при открытии записи."""

    app_version: str = "0.1.0"
    device_model: str = "GC-97001C-03-01-A-F"
    serial: int | None = None
    firmware: str | None = None

    decimation: int = 1
    """Писать каждый N-й кадр. При 2 кГц настройка обязательная: 120 000 строк в минуту."""

    fbg_limit: int | None = None
    """Писать только первые N позиций канала. None — все."""

    rotate_seconds: float | None = DEFAULT_ROTATE_SECONDS
    rotate_bytes: int | None = DEFAULT_ROTATE_BYTES
    poll_period_s: float = DEFAULT_POLL_PERIOD_S
    flush_period_s: float = DEFAULT_FLUSH_PERIOD_S
    batch_limit: int = DEFAULT_BATCH_LIMIT
    chunk_frames: int = DEFAULT_CHUNK_FRAMES
    prefix: str = "data"
    """Начало имени файла: `data_YYYYMMDD_HHMMSS.csv` (KB_05, именование)."""

    def __post_init__(self) -> None:
        """Проверяет параметры: некорректные — баг вызывающего, значит ValueError."""
        if self.decimation < 1:
            raise ValueError(f"decimation={self.decimation} должен быть ≥ 1")
        if self.fbg_limit is not None and self.fbg_limit < 1:
            raise ValueError(f"fbg_limit={self.fbg_limit} должен быть ≥ 1 либо None")
        if self.rotate_seconds is not None and self.rotate_seconds <= 0:
            raise ValueError("rotate_seconds должен быть положительным либо None")
        if self.rotate_bytes is not None and self.rotate_bytes < 1:
            raise ValueError("rotate_bytes должен быть положительным либо None")
        for name, value in (
            ("poll_period_s", self.poll_period_s),
            ("flush_period_s", self.flush_period_s),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} должен быть положительным")
        if self.batch_limit < 1:
            raise ValueError(f"batch_limit={self.batch_limit} должен быть ≥ 1")
        if self.chunk_frames < 1:
            raise ValueError(f"chunk_frames={self.chunk_frames} должен быть ≥ 1")
        if not self.prefix:
            raise ValueError("prefix не может быть пустым")


@dataclass(frozen=True)
class RecorderStats:
    """Снимок состояния записи. Читается из потока UI."""

    files: int
    """Сколько файлов открыто с начала записи, включая текущий."""

    rows: int
    """Сколько строк данных записано всего, через все файлы."""

    frames_span: int
    """Сколько кадров прибора охвачено записью: `последний frame_no + 1`."""

    gaps: int
    """Сколько маркеров `# GAP` записано."""

    lost_frames: int
    """Сколько кадров потеряно из-за отставания записи от приёма."""

    pending_gap: int
    """Потерянные кадры, для которых маркер ещё не записан: ждут следующей строки."""

    bytes_written: int
    """Сколько байт отдано файлам, включая шапки и маркеры."""

    path: Path | None
    """Текущий файл. None — запись не открыта."""

    last_frame_no: int | None
    error: str | None
    """Причина остановки записи. None — запись исправна."""


def column_names(channels: int, fbg_written: int) -> tuple[str, ...]:
    """Имена колонок в том порядке, в котором они идут в строке (KB_03)."""
    names = ["frame_no", "t_mono", "t_wall"]
    names += [
        f"ch{channel}_fbg{position}_nm"
        for channel in range(1, channels + 1)
        for position in range(1, fbg_written + 1)
    ]
    names += [f"ch{channel}_temp" for channel in range(1, channels + 1)]
    return tuple(names)


def row_format(channels: int, fbg_written: int) -> str:
    """Шаблон строки данных для `%`-форматирования.

    Форматирование пачки целиком (`шаблон * n % кортеж`) уходит в C одним
    вызовом. Замер: 127 полей × 200 строк — 3.3 мс на полностью заполненном
    кадре, то есть 16 мкс на строку и около 60 000 строк/с одним потоком.
    Это в тридцать раз выше паспортных 2000 Гц, поэтому форматирование,
    вопреки ожиданию, узким местом не является — см. сводку чата №7.
    """
    fields = ["%d", f"%.{TIME_DECIMALS}f", f"%.{TIME_DECIMALS}f"]
    fields += [f"%.{NM_DECIMALS}f"] * (channels * fbg_written)
    fields += [f"%.{TEMP_DECIMALS}f"] * channels
    return SEPARATOR.join(fields) + "\n"


def build_header(
    profile: DeviceProfile,
    config: RecorderConfig,
    *,
    columns: tuple[str, ...],
    fbg_written: int,
    t_wall_start: datetime,
    t_wall_file: datetime,
    part: int,
) -> str:
    """Собирает шапку файла: строка имён колонок плюс блок комментариев.

    Шапка делает файл самодостаточным и повторяется в каждом файле после
    ротации.

    Порядок непривычный — имена колонок идут **до** блока комментариев, —
    и выбран он ради читаемости обеими библиотеками с умолчаниями.
    `numpy.genfromtxt(names=True)` берёт имена из **первой строки файла**
    и снимает с неё ведущий `#`, поэтому при комментариях сверху именами
    оказалась бы строка «fbg-interrogator 0.1.0», и файл не прочитался бы
    без `skip_header` с ручным счётом строк. При таком порядке работают
    и `pandas.read_csv(sep=';', comment='#')`, и `numpy.genfromtxt(names=True,
    delimiter=';')` — без единого дополнительного аргумента.
    """
    serial = UNKNOWN if config.serial is None else str(config.serial)
    firmware = config.firmware or UNKNOWN
    divisor = UNKNOWN if profile.freq_divisor is None else str(profile.freq_divisor)
    lines = [
        f"fbg-interrogator {config.app_version}",
        f"device={config.device_model} sn={serial} fw={firmware}",
        f"sweep_start_ghz={profile.start_ghz} sweep_stop_ghz={profile.stop_ghz} "
        f"step_ghz={profile.step_param} adc_step_ghz={profile.adc_step_param}",
        f"sweep_start_param={profile.start_param} sweep_stop_param={profile.stop_param} "
        f"sweep_base_ghz={profile.sweep_base_ghz}",
        f"sweep_speed_hz={profile.sweep_speed_hz} channels={profile.channels} "
        f"fbg_per_channel={profile.fbg_per_channel} fbg_written={fbg_written}",
        f"freq_divisor={divisor} decimation={config.decimation}",
        f"t_wall_start={t_wall_start.isoformat()}",
        f"t_wall_file={t_wall_file.isoformat()} file_part={part}",
        "t_mono — perf_counter, с от старта процесса; t_wall — Unix epoch, с (UTC)",
        "длина волны сырая, нм, без калибровки; nan — пик не найден",
        f"frame_no — сквозной номер кадра прибора от старта записи, шаг {config.decimation}",
        "шаг frame_no, равный decimation, — сознательный пропуск, потерей он не является;",
        "скачок сверх шага и строка «# GAP frames=…» — потеря кадров",
    ]
    comments = "".join(f"# {line}\n" for line in lines)
    return SEPARATOR.join(columns) + "\n" + comments


def format_rows(
    row_template: str,
    frame_no: np.ndarray,
    t_mono: np.ndarray,
    t_wall: np.ndarray,
    nm: np.ndarray,
    temp: np.ndarray,
) -> str:
    """Форматирует пачку строк целиком, одним вызовом `%`.

    Все колонки сводятся в одну матрицу float64 и разворачиваются в плоский
    список: так строка собирается C-кодом, а не циклом по 127 полям на кадр.
    `frame_no` при этом остаётся точным — до 2⁵³ кадров float64 хранит целые
    без потерь, а это 143 000 лет при 2 кГц.
    """
    rows = frame_no.size
    matrix = np.empty((rows, 3 + nm.shape[1] * nm.shape[2] + temp.shape[1]), dtype=np.float64)
    matrix[:, 0] = frame_no
    matrix[:, 1] = t_mono
    matrix[:, 2] = t_wall
    matrix[:, 3 : 3 + nm.shape[1] * nm.shape[2]] = nm.reshape(rows, -1)
    matrix[:, 3 + nm.shape[1] * nm.shape[2] :] = temp
    return (row_template * rows) % tuple(matrix.ravel().tolist())


def format_gap(frames: int, t_mono_from: float, t_mono_to: float) -> str:
    """Строка-маркер разрыва (решение Р42).

    `t_mono_from` — метка последней записанной строки, `t_mono_to` — первой
    строки после разрыва. Если разрыв случился до первой строки файла либо
    запись остановлена, не дождавшись строки после него, соответствующая
    граница равна `nan`: подставлять вместо неё что-то правдоподобное значило
    бы придумывать данные.
    """
    return (
        f"# GAP frames={frames} "
        f"t_mono_from={t_mono_from:.{TIME_DECIMALS}f} "
        f"t_mono_to={t_mono_to:.{TIME_DECIMALS}f}\n"
    )


class Recorder:
    """Запись потока измерений в CSV.

    Использование::

        recorder = Recorder(pipeline, RecorderConfig(directory=Path("data")))
        recorder.start()
        ...
        recorder.stop()

    Тесты и одиночные шаги обходятся без потока: `open()` готовит файл,
    `pump()` переносит очередную пачку, `close()` дописывает и закрывает.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        config: RecorderConfig,
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._profile = pipeline.profile
        self._config = config
        self._on_error = on_error

        fbg = self._profile.fbg_per_channel
        self._fbg_written = fbg if config.fbg_limit is None else min(config.fbg_limit, fbg)
        self._columns = column_names(self._profile.channels, self._fbg_written)
        self._row_template = row_format(self._profile.channels, self._fbg_written)

        self._cursor: FrameCursor | None = None
        self._file = None  # type: ignore[var-annotated]
        self._path: Path | None = None

        self._origin: int | None = None
        """Сквозной номер прибора, соответствующий `frame_no = 0`."""

        self._last_frame_no: int | None = None
        self._last_t_mono = float("nan")
        self._pending_gap = 0
        self._wall_offset = 0.0
        self._t_wall_start = datetime.now().astimezone()

        self._files = 0
        self._rows = 0
        self._gaps = 0
        self._lost = 0
        self._bytes = 0
        self._file_bytes = 0
        self._file_opened_mono = 0.0
        self._last_flush_mono = 0.0
        self._error: str | None = None

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    # --- Состояние ---------------------------------------------------------------------

    @property
    def config(self) -> RecorderConfig:
        """Параметры записи."""
        return self._config

    @property
    def path(self) -> Path | None:
        """Текущий файл. None — запись не открыта."""
        return self._path

    @property
    def is_open(self) -> bool:
        """True, если файл открыт и запись не остановлена ошибкой."""
        return self._file is not None

    @property
    def is_running(self) -> bool:
        """True, если поток записи запущен."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def stats(self) -> RecorderStats:
        """Снимок состояния записи."""
        return RecorderStats(
            files=self._files,
            rows=self._rows,
            frames_span=0 if self._last_frame_no is None else self._last_frame_no + 1,
            gaps=self._gaps,
            lost_frames=self._lost,
            pending_gap=self._pending_gap,
            bytes_written=self._bytes,
            path=self._path,
            last_frame_no=self._last_frame_no,
            error=self._error,
        )

    # --- Жизненный цикл ----------------------------------------------------------------

    def open(self) -> None:
        """Заводит курсор и открывает первый файл. Повторный вызов ничего не делает.

        `OSError` не перехватывается: не открывшаяся папка — отказ **старта**
        записи, о котором вызывающий узнаёт сразу, а не через поле `error`.
        """
        if self._file is not None:
            return
        self._config.directory.mkdir(parents=True, exist_ok=True)
        self._cursor = self._pipeline.cursor(stride=self._config.decimation)
        self._t_wall_start = datetime.now().astimezone()
        # Настенное время привязывается к монотонному один раз: так колонка
        # t_wall остаётся согласованной с t_mono и не дёргается вслед за NTP.
        self._wall_offset = time.time() - time.perf_counter()
        self._error = None
        self._open_file()

    def start(self) -> None:
        """Открывает запись и запускает поток. Повторный вызов ничего не делает."""
        if self.is_running:
            return
        self.open()
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, name="fbg-recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Останавливает поток и закрывает файл, добрав остаток кольца."""
        self._stop_flag.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._config.poll_period_s + 10.0)
        self._thread = None
        self.close()

    def close(self) -> None:
        """Дописывает остаток, отмечает недописанный разрыв и закрывает файл."""
        if self._file is None:
            return
        # Кольцо ещё держит кадры, принятые между последним `pump` и остановкой.
        # Признак «добрали» — нулевое отставание курсора, а не пустая пачка:
        # пачка бывает пустой и когда она целиком выпала по децимации.
        # Цикл ограничен, чтобы не крутиться, если поток данных ещё идёт.
        for _ in range(1024):
            cursor = self._cursor
            if cursor is None or self._file is None or cursor.lag == 0:
                break
            self.pump()
        if self._file is None:
            return
        if self._pending_gap:
            # Разрыв, после которого строк уже не будет: правая граница
            # неизвестна и остаётся nan.
            self._emit_gap(float("nan"))
        try:
            self._file.flush()
        except OSError as exc:
            self._fail(exc)
        finally:
            self._close_file()

    def __enter__(self) -> "Recorder":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # --- Перенос данных ----------------------------------------------------------------

    def pump(self) -> int:
        """Переносит одну пачку кадров из кольца в файл. Возвращает число строк.

        Ноль означает «писать было нечего» — в том числе когда вся пачка выпала
        по децимации. Исключений не бросает: ошибка записи останавливает запись
        и попадает в `RecorderStats.error`.
        """
        cursor = self._cursor
        if self._file is None or cursor is None:
            return 0

        lost_before = cursor.lost
        position = cursor.position
        batch = cursor.take(limit=self._config.batch_limit)
        lost = cursor.lost - lost_before
        if batch is not None:
            # Курсор считает потерю сам, но в одной редкой ветке (три подряд
            # сорванных чтения) сдвигает позицию молча. Арифметика по номерам
            # ловит и её: молчаливой потери в тракте быть не должно нигде.
            lost = max(lost, batch.seq_start - position)
        if lost:
            self._pending_gap += lost
            self._lost += lost

        if batch is None or len(batch) == 0:
            self._maybe_flush()
            return 0
        try:
            return self._write_batch(batch)
        except OSError as exc:
            self._fail(exc)
            return 0

    def _loop(self) -> None:
        """Поток записи: тянет кадры курсором, спит, когда их нет."""
        while not self._stop_flag.is_set():
            if self._file is None:
                return
            if self.pump() == 0:
                self._stop_flag.wait(self._config.poll_period_s)

    # --- Запись ------------------------------------------------------------------------

    def _write_batch(self, batch: FrameBatch) -> int:
        """Пишет пачку кусками, проверяя ротацию на границах строк."""
        if self._origin is None:
            self._origin = int(batch.seq[0])
        frame_no = batch.seq - self._origin
        t_wall = batch.t_mono + self._wall_offset
        nm = batch.wavelength_nm()
        if self._fbg_written != self._profile.fbg_per_channel:
            nm = nm[:, :, : self._fbg_written]

        total = len(batch)
        chunk = self._config.chunk_frames
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            self._rotate_if_needed()
            if self._pending_gap:
                self._emit_gap(float(batch.t_mono[start]))
            text = format_rows(
                self._row_template,
                frame_no[start:stop],
                batch.t_mono[start:stop],
                t_wall[start:stop],
                nm[start:stop],
                batch.case_temp_c[start:stop],
            )
            self._write(text)
            self._rows += stop - start
        self._last_frame_no = int(frame_no[-1])
        self._last_t_mono = float(batch.t_mono[-1])
        self._maybe_flush()
        return total

    def _emit_gap(self, t_mono_to: float) -> None:
        """Записывает маркер разрыва и обнуляет накопленный счёт."""
        self._write(format_gap(self._pending_gap, self._last_t_mono, t_mono_to))
        self._pending_gap = 0
        self._gaps += 1

    def _write(self, text: str) -> None:
        """Отдаёт текст файлу одним вызовом и учитывает объём."""
        assert self._file is not None
        payload = text.encode("utf-8")
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

    # --- Файлы -------------------------------------------------------------------------

    def _rotate_if_needed(self) -> None:
        """Открывает новый файл, если исчерпан лимит по времени или по размеру."""
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
        """Открывает очередной файл и пишет в него шапку."""
        now = datetime.now().astimezone()
        path = self._unique_path(now)
        # Двоичный режим: перевод строки одинаков на всех платформах,
        # а объём файла считается точно, без пересчёта кодировки на каждой строке.
        self._file = path.open("wb", buffering=FILE_BUFFER_BYTES)
        self._path = path
        self._file_bytes = 0
        self._file_opened_mono = time.monotonic()
        self._last_flush_mono = self._file_opened_mono
        self._files += 1
        self._write(
            build_header(
                self._profile,
                self._config,
                columns=self._columns,
                fbg_written=self._fbg_written,
                t_wall_start=self._t_wall_start,
                t_wall_file=now,
                part=self._files,
            )
        )

    def _unique_path(self, now: datetime) -> Path:
        """Имя `data_YYYYMMDD_HHMMSS.csv`; при совпадении добавляется номер.

        Совпадение возможно при частой ротации: разрешение имени — секунда.
        Дописывать в уже существующий файл нельзя — его шапка описывает другую
        запись.
        """
        stem = f"{self._config.prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
        path = self._config.directory / f"{stem}.csv"
        suffix = 2
        while path.exists():
            path = self._config.directory / f"{stem}_{suffix}.csv"
            suffix += 1
        return path

    def _close_file(self) -> None:
        """Закрывает текущий файл, не трогая счётчики записи."""
        file = self._file
        self._file = None
        if file is None:
            return
        try:
            file.close()
        except OSError as exc:
            self._fail(exc)

    def _fail(self, exc: OSError) -> None:
        """Останавливает запись, сохранив причину. Приём данных не трогается."""
        if self._error is None:
            self._error = f"{type(exc).__name__}: {exc}"
        file = self._file
        self._file = None
        if file is not None:
            # Закрыть не удалось — причина уже записана, добавить нечего.
            with contextlib.suppress(OSError):
                file.close()
        self._stop_flag.set()
        if self._on_error is not None:
            self._on_error(self._error)
