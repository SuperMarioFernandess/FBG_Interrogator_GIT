"""UI новой вкладки «Датчики»; запускается только с Qt/offscreen."""

import math
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from fbg.core.calibration import ReadingStatus, Sensor, SensorReading, SensorType
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.io.recalibrate import RecalibrationResult
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.panels import sensors as sensors_module
from fbg.ui.panels.sensors import SensorsPanel

pytestmark = pytest.mark.ui
PROFILE = DeviceProfile()


@pytest.fixture(scope="session")
def application() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def controller(tmp_path: Path) -> Iterator[AppController]:
    ctl = AppController(
        AppConfig(
            calibration_path=tmp_path / "sensors.json",
            packet_log=PacketLogConfig(directory=None),
        )
    )
    ctl.start()
    try:
        yield ctl
    finally:
        ctl.shutdown()


def sensor(
    sensor_id: str,
    *,
    expected_nm: float,
    sensor_type: SensorType = SensorType.TEMPERATURE,
) -> Sensor:
    return Sensor(
        id=sensor_id,
        name=f"Датчик {sensor_id}",
        channel=0,
        type=sensor_type,
        expected_nm=expected_nm,
        window_nm=0.10,
        value0=25.0,
        k1=100.0,
    )


def reading(sensor_id: str, status: ReadingStatus) -> SensorReading:
    found = status in (
        ReadingStatus.OK,
        ReadingStatus.OUT_OF_LIMITS,
        ReadingStatus.REFERENCE_MISSING,
    )
    return SensorReading(
        sensor_id=sensor_id,
        status=status,
        wavelength_nm=1545.0 if found else math.nan,
        value=25.0 if status in (ReadingStatus.OK, ReadingStatus.OUT_OF_LIMITS) else math.nan,
        position=0 if found else -1,
        candidates=2 if status is ReadingStatus.AMBIGUOUS else int(found),
    )


def snapshot(controller: AppController, **kwargs: object) -> models.AppSnapshot:
    base: dict[str, object] = {
        "endpoint": controller.config.endpoint,
        "profile": PROFILE,
        "state": SessionState.STREAMING,
        "sensors": controller.sensors,
        "sensor_version": 1,
    }
    base.update(kwargs)
    return models.AppSnapshot(**base)  # type: ignore[arg-type]


def test_таблица_группируется_по_каналу_и_не_прячет_пропавший_датчик(
    application: QApplication, controller: AppController
) -> None:
    controller.replace_sensors((sensor("T1", expected_nm=1545.0),))
    panel = SensorsPanel(controller)
    try:
        panel.refresh(
            snapshot(
                controller,
                sensor_readings=(reading("T1", ReadingStatus.PEAK_NOT_FOUND),),
            )
        )
        assert panel.sensor_tree.topLevelItemCount() == 1
        group = panel.sensor_tree.topLevelItem(0)
        assert group is not None and group.childCount() == 1
        item = group.child(0)
        assert item is not None
        assert item.text(0) == "Датчик T1"
        assert item.text(6) == texts.SENSOR_STATUS_LABELS[ReadingStatus.PEAK_NOT_FOUND.value]
        assert item.text(4) == texts.UNKNOWN
    finally:
        panel.close()
        panel.deleteLater()


def test_все_пять_статусов_различимы_в_таблице(
    application: QApplication, controller: AppController
) -> None:
    sensors = tuple(sensor(f"S{i}", expected_nm=1544.0 + i * 0.4) for i in range(5))
    controller.replace_sensors(sensors)
    statuses = tuple(ReadingStatus)
    panel = SensorsPanel(controller)
    try:
        panel.refresh(
            snapshot(
                controller,
                sensor_readings=tuple(
                    reading(current.id, status)
                    for current, status in zip(sensors, statuses, strict=True)
                ),
            )
        )
        shown = {panel._items[current.id].text(6) for current in sensors}
        assert shown == {texts.SENSOR_STATUS_LABELS[status.value] for status in statuses}
    finally:
        panel.close()
        panel.deleteLater()


def test_ось_y_разрешает_отметить_только_одну_единицу(
    application: QApplication, controller: AppController
) -> None:
    temperature = sensor("T", expected_nm=1545.0, sensor_type=SensorType.TEMPERATURE)
    strain = sensor("E", expected_nm=1550.0, sensor_type=SensorType.STRAIN_UE)
    controller.replace_sensors((temperature, strain))
    panel = SensorsPanel(controller)
    try:
        panel.refresh(snapshot(controller))
        index = panel.unit_combo.findData("°C")
        assert index >= 0
        panel.unit_combo.setCurrentIndex(index)
        temp_item = panel._items["T"]
        strain_item = panel._items["E"]
        assert bool(temp_item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        assert not bool(strain_item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        assert strain_item.checkState(0) == Qt.CheckState.Unchecked
    finally:
        panel.close()
        panel.deleteLater()


def test_взять_текущую_лямбду_берёт_пик_из_телеметрии_а_не_слот(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        wavelengths = np.full((PROFILE.channels, PROFILE.fbg_per_channel), np.nan)
        wavelengths[0, 4] = 1544.812345  # позиция намеренно не нулевая
        panel.refresh(snapshot(controller, ui=SimpleNamespace(wavelength_nm=wavelengths)))
        assert panel.current_peak_combo.count() == 1
        panel.take_wavelength_button.click()
        assert panel.expected_spin.value() == pytest.approx(1544.812345, abs=1e-6)
    finally:
        panel.close()
        panel.deleteLater()


def test_редактор_сохраняет_опорную_форму_и_валидирует_окна(
    application: QApplication, controller: AppController
) -> None:
    controller.replace_sensors((sensor("A", expected_nm=1545.0),))
    panel = SensorsPanel(controller)
    try:
        panel._new_sensor()
        panel.id_edit.setText("B")
        panel.name_edit.setText("Второй")
        panel.expected_spin.setValue(1545.05)  # пересекается с A ±0.10
        panel.window_spin.setValue(0.10)
        panel.value0_spin.setValue(25.0)
        panel.k1_spin.setValue(100.0)
        panel._save_sensor()
        assert [item.id for item in controller.sensors] == ["A"]
        assert any("пересекаются" in notice for notice in controller.notices)

        panel.expected_spin.setValue(1546.0)
        panel._save_sensor()
        saved = {item.id: item for item in controller.sensors}
        assert saved["B"].value0 == pytest.approx(25.0)
        assert saved["B"].k1 == pytest.approx(100.0)
        assert controller.config.calibration_path.exists()
    finally:
        panel.close()
        panel.deleteLater()


def test_парабола_по_трем_точкам_не_выдаёт_нулевую_невязку(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        panel.expected_spin.setValue(1545.0)
        panel._set_points(
            (
                sensors_module.CalibrationPoint(1544.9, 10.0),
                sensors_module.CalibrationPoint(1545.0, 20.0),
                sensors_module.CalibrationPoint(1545.1, 30.0),
            )
        )
        panel.fit_kind.setCurrentIndex(panel.fit_kind.findData("quadratic"))
        panel._fit_points()
        assert "не менее 4" in panel.fit_residual.text()
    finally:
        panel.close()
        panel.deleteLater()


def test_карта_пиков_при_refresh_не_отправляет_команд(
    application: QApplication, controller: AppController
) -> None:
    controller.replace_sensors((sensor("T1", expected_nm=1545.0),))
    panel = SensorsPanel(controller)
    try:
        wavelengths = np.full((PROFILE.channels, PROFILE.fbg_per_channel), np.nan)
        wavelengths[0, :2] = (1544.9, 1545.1)
        before = controller.session.stats().commands
        panel.refresh(snapshot(controller, ui=SimpleNamespace(wavelength_nm=wavelengths)))
        after = controller.session.stats().commands
        assert after == before
        assert len(panel.peak_scatter.getData()[0]) == 2
        assert texts.SENSOR_PEAK_MAP_HINT in panel.peak_map_hint.text()
    finally:
        panel.close()
        panel.deleteLater()


def test_пересчёт_csv_идёт_в_обычном_рабочем_потоке(
    application: QApplication,
    controller: AppController,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller.replace_sensors((sensor("T1", expected_nm=1545.0),))
    source = tmp_path / "data.csv"
    source.write_text("dummy", encoding="ascii")
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(source), "CSV (*.csv)"),
    )

    def slow_recalibrate(path: Path, sensors: tuple[Sensor, ...]) -> RecalibrationResult:
        assert path == source and sensors == controller.sensors
        entered.set()
        assert release.wait(2.0)
        return RecalibrationResult((source,), (tmp_path / "out.csv",), 10, 1)

    monkeypatch.setattr(sensors_module, "recalibrate_recording", slow_recalibrate)
    panel = SensorsPanel(controller)
    try:
        started = time.perf_counter()
        panel._start_recalibration()
        elapsed = time.perf_counter() - started
        assert entered.wait(1.0)
        assert elapsed < 0.2
        assert panel._recalc_thread is not None and panel._recalc_thread.is_alive()
        release.set()
        panel._recalc_thread.join(2.0)
        panel._poll_recalibration()
        assert "строк 10" in panel.recalibrate_state.text()
        assert "# GAP 1" in panel.recalibrate_state.text()
    finally:
        release.set()
        panel.close()
        panel.deleteLater()
