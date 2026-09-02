"""Qt-панель измерения. Маркер `ui`, запуск с QT_QPA_PLATFORM=offscreen.

Проверяется устройство и поток данных, а не внешний вид: один reset таблицы
на кадр, выбор линий, разрыв NaN, состояние записи и отсутствие ссылок графика
на кольцо pipeline. Читаемость и размеры остаются визуальной приёмкой Windows.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fbg.core.endpoint import Endpoint
from fbg.core.pipeline import RingHistory
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.io.recorder import RecorderConfig, RecorderStats
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.main_window import UI_PERIOD_MS, MainWindow
from fbg.ui.panels.measurement import MeasurementPanel
from tests.synthetic import load_vectors

pytestmark = pytest.mark.ui

PROFILE = DeviceProfile()
REAL_FRAME = load_vectors("measurement_real.hex")["measurement_real"]


@pytest.fixture(scope="session")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


@pytest.fixture
def controller(tmp_path: Path) -> Iterator[AppController]:
    app_controller = AppController(
        AppConfig(
            recorder=RecorderConfig(directory=tmp_path / "data"),
            packet_log=PacketLogConfig(directory=None),
        )
    )
    app_controller.start()
    try:
        yield app_controller
    finally:
        app_controller.shutdown()


@pytest.fixture
def panel(application: QApplication, controller: AppController) -> Iterator[MeasurementPanel]:
    widget = MeasurementPanel(controller)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()


def snapshot_with_recording(
    tmp_path: Path,
    *,
    active: bool,
    error: str | None = None,
    gaps: int = 0,
    lost: int = 0,
) -> models.AppSnapshot:
    config = RecorderConfig(directory=tmp_path)
    stats = RecorderStats(
        files=1,
        rows=123,
        frames_span=150,
        gaps=gaps,
        lost_frames=lost,
        pending_gap=0,
        bytes_written=654_321,
        path=tmp_path / "data.csv",
        last_frame_no=149,
        error=error,
    )
    return models.AppSnapshot(
        endpoint=Endpoint(),
        profile=PROFILE,
        state=SessionState.STREAMING,
        recorder_config=config,
        recorder=stats,
        recording=active,
        recording_elapsed_s=12.0,
    )


def test_панель_строится_с_4x30_выбором(panel: MeasurementPanel) -> None:
    assert panel.trace_tree.topLevelItemCount() == 4
    assert (
        sum(
            panel.trace_tree.topLevelItem(channel).childCount()  # type: ignore[union-attr]
            for channel in range(panel.trace_tree.topLevelItemCount())
        )
        == 120
    )
    selected = panel.selected_slots()
    assert selected == tuple(models.SlotRef(0, position) for position in range(4))
    assert panel.table_model.rowCount() == 30
    assert panel.table_model.columnCount() == 9


def test_главный_таймер_обновляет_и_гасит_панель(
    application: QApplication, controller: AppController
) -> None:
    window = MainWindow(controller)
    try:
        assert not window.timer.isActive()
        window.start_updates()
        assert window.timer.isActive()
        assert window.timer.interval() == UI_PERIOD_MS == 100
        window.stop_updates()
        assert not window.timer.isActive()
        titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        assert texts.TAB_MEASUREMENT in titles
    finally:
        window.close()
        window.deleteLater()


def test_таблица_обновляется_одним_model_reset(
    panel: MeasurementPanel, controller: AppController
) -> None:
    resets = 0

    def count_reset() -> None:
        nonlocal resets
        resets += 1

    panel.table_model.modelReset.connect(count_reset)
    controller.pipeline.on_telemetry(REAL_FRAME, time.perf_counter())
    controller.pipeline.publish_now()
    panel.refresh(controller.snapshot())

    assert resets == 1
    assert panel.table_model.data(panel.table_model.index(0, 1)) != texts.UNKNOWN
    assert panel.table_model.data(panel.table_model.index(2, 1)) == texts.UNKNOWN
    assert panel.table_model.data(panel.table_model.index(2, 2)) == texts.TABLE_VALID_NO


def test_график_держит_только_копию_а_не_кольцо(
    panel: MeasurementPanel, controller: AppController
) -> None:
    start = time.perf_counter()
    for index in range(10):
        controller.pipeline.on_telemetry(REAL_FRAME, start + index * 0.0005)
    controller.pipeline.publish_now()
    panel.refresh(controller.snapshot())

    assert panel._curves
    curve = panel._curves[models.SlotRef(0, 0)]
    x_data, y_data = curve.getData()
    assert x_data is not None and y_data is not None
    assert not np.shares_memory(y_data, controller.pipeline.history.freq_ghz)
    assert not isinstance(panel.table_model.model, RingHistory)


def test_снятие_флажка_убирает_линию(panel: MeasurementPanel, controller: AppController) -> None:
    controller.pipeline.on_telemetry(REAL_FRAME, time.perf_counter())
    controller.pipeline.publish_now()
    panel.refresh(controller.snapshot())
    assert models.SlotRef(0, 0) in panel._curves

    channel = panel.trace_tree.topLevelItem(0)
    assert channel is not None
    first = channel.child(0)
    first.setCheckState(0, Qt.CheckState.Unchecked)
    panel.refresh(controller.snapshot())
    assert models.SlotRef(0, 0) not in panel._curves


def test_глубина_истории_ограничена_кольцом_10_секунд_на_2кгц(
    panel: MeasurementPanel, controller: AppController
) -> None:
    snapshot = controller.snapshot()
    panel.refresh(snapshot)
    assert panel.history_spin.maximum() == pytest.approx(10.0)


def test_состояние_записи_управляет_кнопками(panel: MeasurementPanel, tmp_path: Path) -> None:
    stopped = snapshot_with_recording(tmp_path, active=False)
    panel.refresh(stopped)
    assert panel.start_record_button.isEnabled()
    assert not panel.stop_record_button.isEnabled()
    assert panel.record_state.text() == texts.RECORD_IDLE

    active = snapshot_with_recording(tmp_path, active=True)
    panel.refresh(active)
    assert not panel.start_record_button.isEnabled()
    assert panel.stop_record_button.isEnabled()
    assert panel.record_state.text() == texts.RECORD_ACTIVE
    assert "123" in panel.record_rows.text()
    assert panel.record_elapsed.text() == "00:00:12"


def test_ошибка_и_gap_видны_на_панели(panel: MeasurementPanel, tmp_path: Path) -> None:
    snapshot = snapshot_with_recording(
        tmp_path,
        active=False,
        error="OSError: No space left on device",
        gaps=2,
        lost=50,
    )
    panel.refresh(snapshot)

    assert "No space left" in panel.record_error.text()
    assert "# GAP" in panel.record_gaps.text()
    assert "50" in panel.record_gaps.text()


def test_оценка_объёма_показана_до_старта(panel: MeasurementPanel, tmp_path: Path) -> None:
    panel.refresh(snapshot_with_recording(tmp_path, active=False))
    assert "10 минут" not in panel.record_estimate.text()  # подпись уже слева в форме
    assert "МБ" in panel.record_estimate.text() or "ГБ" in panel.record_estimate.text()
    assert texts.RECORD_ESTIMATE_SUFFIX in panel.record_estimate.text()


def test_оценка_объёма_сразу_реагирует_на_децимацию(
    panel: MeasurementPanel, tmp_path: Path
) -> None:
    snapshot = snapshot_with_recording(tmp_path, active=False)
    panel.refresh(snapshot)
    before = panel.record_estimate.text()

    panel.record_decimation.setValue(2)
    panel.refresh(snapshot)
    after = panel.record_estimate.text()

    assert after != before
    assert panel.record_decimation.value() == 2
