"""Чат №21 в виджетах: стоимость такта и калибровочные исправления.

Производительность проверяется структурно (№44): сколько точек уходит
в ``setData`` каждой кривой за такт и сколько раз трогаются поля λ₀, — а не
секундомером, который под offscreen меряет не то.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QStyleOptionViewItem

from fbg.core.calibration import CalibrationPoint, Sensor, SensorType
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.io.recorder import RecorderConfig
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.history import HISTORY_INTERVAL_S, VIEW_ROW_LIMIT
from fbg.ui.panels.measurement import MeasurementPanel
from fbg.ui.panels.sensors import SensorsPanel

pytestmark = pytest.mark.ui

PROFILE = DeviceProfile()
LINE = (1538.22, 1544.78, 1549.68, 1551.35, 1559.77)


@pytest.fixture(scope="session")
def application() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


def make_controller(tmp_path: Path) -> AppController:
    config = AppConfig(
        recorder=RecorderConfig(directory=tmp_path / "data"),
        packet_log=PacketLogConfig(directory=None),
        calibration_path=tmp_path / "sensors.json",
    )
    controller = AppController(config, config_path=tmp_path / "fbg_config.json")
    controller.start()
    return controller


@pytest.fixture
def controller(tmp_path: Path) -> Iterator[AppController]:
    ctl = make_controller(tmp_path)
    try:
        yield ctl
    finally:
        ctl.shutdown()


def fill_history(controller: AppController, seconds: float, end_bin: int = 10_000_000) -> None:
    """История самописца измерения на пяти решётках канала 1.

    Заливается часовыми кусками: сутки на все 120 позиций одним массивом —
    это 4 ГБ плотных данных, а продукт копит их по интервалу за такт.
    """
    recorder = controller._measurement_history
    recorder.start()
    rows_total = round(seconds / HISTORY_INTERVAL_S)
    columns = PROFILE.channels * PROFILE.fbg_per_channel
    chunk = 36_000
    for begin in range(end_bin - rows_total, end_bin, chunk):
        bins = np.arange(begin, min(begin + chunk, end_bin), dtype=np.int64)
        rows = bins.size
        start = bins * HISTORY_INTERVAL_S
        mean = np.full((rows, columns), np.nan)
        mean[:, : len(LINE)] = np.asarray(LINE) + 0.01 * np.sin(bins[:, np.newaxis] / 997.0)
        n = np.where(np.isfinite(mean), 200, 0)
        sigma = np.where(np.isfinite(mean), 0.002, np.nan)
        recorder.append_intervals(start, start + 0.1, mean, mean - 0.006, mean + 0.006, sigma, n)


def frame(*peaks: float) -> SimpleNamespace:
    wavelengths = np.full((PROFILE.channels, PROFILE.fbg_per_channel), np.nan)
    wavelengths[0, : len(peaks)] = peaks
    return SimpleNamespace(
        wavelength_nm=wavelengths,
        valid=np.isfinite(wavelengths),
        case_temp_c=np.full(PROFILE.channels, 21.5),
    )


def count_points(panel: MeasurementPanel, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    sizes: list[int] = []
    for curve in panel._curves.values():
        original = curve.setData

        def counted(*args: object, _original=original, **kwargs: object) -> None:
            sizes.append(len(args[0]))  # type: ignore[arg-type]
            _original(*args, **kwargs)

        monkeypatch.setattr(curve, "setData", counted)
    return sizes


# --- часть 1 ------------------------------------------------------------------------


@pytest.mark.parametrize("x_mode", ["follow", "all"])
def test_точек_в_кривую_за_такт_столько_же_на_сутках_сколько_на_минуте(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    x_mode: str,
) -> None:
    per_history: dict[float, list[int]] = {}
    for seconds in (60.0, 86_400.0):
        controller = make_controller(tmp_path / f"h{int(seconds)}")
        panel = MeasurementPanel(controller)
        try:
            fill_history(controller, seconds)
            controls = panel.range_controls
            controls.x_mode.setCurrentIndex(controls.x_mode.findData(x_mode))
            # 30 с плюс поле по 15 с помещаются и в минутную историю: дальше
            # объём копии от длины истории не зависит вовсе.
            controls.follow_s.setValue(30.0)
            panel.refresh(controller.snapshot(include_sensor_data=False))  # линии созданы
            sizes = count_points(panel, monkeypatch)
            fill_history_tail = controller._measurement_history
            start = (10_000_000) * HISTORY_INTERVAL_S
            row = np.full((1, PROFILE.channels * PROFILE.fbg_per_channel), np.nan)
            row[0, : len(LINE)] = LINE
            fill_history_tail.append_intervals(
                np.asarray([start]),
                np.asarray([start + 0.1]),
                row,
                row,
                row,
                np.where(np.isfinite(row), 0.0, np.nan),
                np.where(np.isfinite(row), 200, 0),
            )
            panel.refresh(controller.snapshot(include_sensor_data=False))
            assert len(sizes) == 4  # четыре выбранные по умолчанию позиции
            assert max(sizes) <= VIEW_ROW_LIMIT
            per_history[seconds] = sizes
        finally:
            panel.close()
            panel.deleteLater()
            controller.shutdown()
    if x_mode == "follow":
        assert per_history[60.0] == per_history[86_400.0]
    else:
        assert max(per_history[86_400.0]) <= VIEW_ROW_LIMIT


def test_по_умолчанию_только_линия_и_не_больше_одной_полосы(
    application: QApplication, controller: AppController
) -> None:
    fill_history(controller, 60.0)
    panel = MeasurementPanel(controller)
    try:
        panel.refresh(controller.snapshot(include_sensor_data=False))
        items = panel.plot.getPlotItem().listDataItems()
        assert len(items) == len(panel._curves) == 4
        assert not panel._bands

        for mode in ("sigma", "range"):
            panel.band_mode.setCurrentIndex(panel.band_mode.findData(mode))
            panel.refresh(controller.snapshot(include_sensor_data=False))
            assert set(panel._bands) == set(panel._curves)
            assert len(panel.plot.getPlotItem().listDataItems()) == 4 * 3

        panel.band_mode.setCurrentIndex(panel.band_mode.findData("none"))
        panel.refresh(controller.snapshot(include_sensor_data=False))
        assert not panel._bands
        assert len(panel.plot.getPlotItem().listDataItems()) == 4
    finally:
        panel.close()
        panel.deleteLater()


def test_выбор_полосы_переживает_перезапуск(
    application: QApplication, controller: AppController, tmp_path: Path
) -> None:
    panel = MeasurementPanel(controller)
    try:
        panel.band_mode.setCurrentIndex(panel.band_mode.findData("range"))
    finally:
        panel.close()
        panel.deleteLater()
    from fbg.io import config as config_module

    loaded = config_module.load(tmp_path / "fbg_config.json").config
    restarted = AppController(loaded, config_path=tmp_path / "fbg_config.json")
    restarted.start()
    again = MeasurementPanel(restarted)
    try:
        assert again.band_mode.currentData() == "range"
    finally:
        again.close()
        again.deleteLater()
        restarted.shutdown()


def test_подпись_показывает_фактическое_разрешение(
    application: QApplication, controller: AppController
) -> None:
    fill_history(controller, 86_400.0)
    panel = MeasurementPanel(controller)
    try:
        controls = panel.range_controls
        controls.x_mode.setCurrentIndex(controls.x_mode.findData("all"))
        panel.refresh(controller.snapshot(include_sensor_data=False))
        assert panel.graph_resolution.text().startswith("100 с")
        controls.x_mode.setCurrentIndex(controls.x_mode.findData("follow"))
        panel.refresh(controller.snapshot(include_sensor_data=False))
        assert panel.graph_resolution.text().startswith("0.1 с")
    finally:
        panel.close()
        panel.deleteLater()


def test_ручная_область_мышью_не_перехватывается_тактом(
    application: QApplication, controller: AppController
) -> None:
    """№43 после оптимизации: область копии меняется, область экрана — нет."""
    fill_history(controller, 3600.0)
    panel = MeasurementPanel(controller)
    try:
        panel.refresh(controller.snapshot(include_sensor_data=False))
        panel.plot.setXRange(100.0, 200.0, padding=0.0)
        panel.range_controls._mouse_range_changed()
        for _ in range(10):
            panel.refresh(controller.snapshot(include_sensor_data=False))
        low, high = panel.plot.viewRange()[0]
        assert (low, high) == pytest.approx((100.0, 200.0))
        assert controller._measurement_view.kind == "manual"
        assert panel._graph_model is not None
        assert panel._graph_model.resolution_s == pytest.approx(HISTORY_INTERVAL_S)
    finally:
        panel.close()
        panel.deleteLater()


def test_поле_lambda0_на_такте_без_изменений_не_получает_settext(
    application: QApplication,
    controller: AppController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller.set_measurement_lambda0(0, 2, 1549.60)
    panel = MeasurementPanel(controller)
    try:
        base = controller.snapshot(include_sensor_data=False)
        running = models.AppSnapshot(
            **{**base.__dict__, "ui": frame(*LINE), "state": SessionState.STREAMING}
        )
        panel.refresh(running)
        calls: list[models.SlotRef] = []
        for slot, edit in panel._slot_lambda0.items():
            original = edit.setText

            def counted(text: str, _slot=slot, _original=original) -> None:
                calls.append(_slot)
                _original(text)

            monkeypatch.setattr(edit, "setText", counted)
        current_calls: list[models.SlotRef] = []
        for slot, item in panel._slot_current.items():
            original_item = item.setText

            def counted_item(text: str, _slot=slot, _original=original_item) -> None:
                current_calls.append(_slot)
                _original(text)

            monkeypatch.setattr(item, "setText", counted_item)

        for _ in range(10):
            panel.refresh(running)
        assert calls == []
        assert current_calls == []

        controller.set_measurement_lambda0(0, 1, 1544.70)
        changed = models.AppSnapshot(
            **{
                **running.__dict__,
                "measurement_lambda0_nm": controller.config.measurement_lambda0_nm,
            }
        )
        panel.refresh(changed)
        assert calls == [models.SlotRef(0, 1)]

        # Пик пропал — прочерк ставится ровно в его ячейку, а не во все 120.
        gone = models.AppSnapshot(**{**changed.__dict__, "ui": frame(*LINE[:4])})
        panel.refresh(gone)
        assert current_calls == [models.SlotRef(0, 4)]
        assert panel._slot_current[models.SlotRef(0, 4)].text() == texts.UNKNOWN
    finally:
        panel.close()
        panel.deleteLater()


# --- часть 2 ------------------------------------------------------------------------


def sensors_snapshot(controller: AppController, ui: SimpleNamespace | None) -> models.AppSnapshot:
    return models.AppSnapshot(
        endpoint=controller.config.endpoint,
        profile=PROFILE,
        state=SessionState.STREAMING if ui is not None else SessionState.IDLE,
        sensors=controller.sensors,
        sensor_version=1,
        ui=ui,
    )


def choose(panel: SensorsPanel, wavelength_nm: float) -> None:
    index = panel.current_peak_combo.findText(f"{wavelength_nm:.4f} нм")
    assert index >= 0
    panel.current_peak_combo.activated.emit(index)


def test_воспроизведение_координатора_дрожание_на_квант(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        choose(panel, 1549.68)
        assert panel._selected_peak() == pytest.approx(1549.68)

        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        assert panel._selected_peak() == pytest.approx(1549.68)

        jittered = list(LINE)
        jittered[2] += 0.0008
        for _ in range(10):
            panel.refresh(sensors_snapshot(controller, frame(*jittered)))
            assert panel._selected_peak() == pytest.approx(1549.6808)
            assert panel.current_peak_combo.currentText() == "1549.6808 нм"
    finally:
        panel.close()
        panel.deleteLater()


def test_пропавший_пик_блокирует_кнопки_и_не_переключает_выбор(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        choose(panel, 1549.68)
        without = [value for value in LINE if value != 1549.68]
        panel.refresh(sensors_snapshot(controller, frame(*without)))

        assert panel.current_peak_combo.currentIndex() == -1
        assert "1549.6800" in panel.current_peak_combo.placeholderText()
        assert not panel.add_point_button.isEnabled()
        assert not panel.take_wavelength_button.isEnabled()
        assert "потерян" in panel.peak_status.text()
        with pytest.raises(ValueError):
            panel._selected_peak()
        panel._add_point()
        assert panel._points() == ()

        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        assert panel._selected_peak() == pytest.approx(1549.68)
        assert panel.add_point_button.isEnabled()
    finally:
        panel.close()
        panel.deleteLater()


def test_выбор_следует_за_решёткой_при_сдвиге_индекса(
    application: QApplication, controller: AppController
) -> None:
    """Число пиков то же, но снизу появилась решётка, сверху пропала — индекс сдвинулся."""
    panel = SensorsPanel(controller)
    try:
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        choose(panel, 1549.68)
        shifted = (1530.0, *LINE[:4])
        for _ in range(10):
            panel.refresh(sensors_snapshot(controller, frame(*shifted)))
            assert panel.current_peak_combo.currentIndex() == 3
            assert panel.current_peak_combo.currentText() == "1549.6800 нм"
            assert panel._selected_peak() == pytest.approx(1549.68)
    finally:
        panel.close()
        panel.deleteLater()


def test_вне_потока_выбор_не_используется(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        choose(panel, 1551.35)
        panel.refresh(sensors_snapshot(controller, None))
        assert not panel.add_point_button.isEnabled()
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        assert panel._selected_peak() == pytest.approx(1551.35)
    finally:
        panel.close()
        panel.deleteLater()


def test_смена_канала_сбрасывает_выбор(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        choose(panel, 1549.68)
        panel.channel_combo.setCurrentIndex(1)
        panel.channel_combo.setCurrentIndex(0)
        panel.refresh(sensors_snapshot(controller, frame(*LINE)))
        assert panel.current_peak_combo.currentIndex() == -1
        assert not panel.add_point_button.isEnabled()
    finally:
        panel.close()
        panel.deleteLater()


def test_измеренные_поля_точки_не_редактируются(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        panel._set_points((CalibrationPoint(1549.681234567, 20.0, 100, 0.0021),))
        table = panel.points_table
        for column in (0, 2, 3, 4):
            assert not table.item(0, column).flags() & Qt.ItemFlag.ItemIsEditable
        assert table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
        # Даже если текст ячейки λ кто-то поменял, точка берётся из модели.
        table.item(0, 0).setText("1500")
        assert panel._points()[0].wavelength_nm == 1549.681234567
    finally:
        panel.close()
        panel.deleteLater()


def test_значение_эталона_правится_с_запятой(
    application: QApplication, controller: AppController
) -> None:
    panel = SensorsPanel(controller)
    try:
        point = CalibrationPoint(1549.681234567, 20.0, 100, 0.0021)
        panel._set_points((point,))
        table = panel.points_table
        index = table.model().index(0, 1)
        delegate = table.itemDelegateForColumn(1)
        editor = delegate.createEditor(table.viewport(), QStyleOptionViewItem(), index)
        assert isinstance(editor, QLineEdit)
        delegate.setEditorData(editor, index)
        editor.setText("20,5")
        delegate.setModelData(editor, table.model(), index)

        edited = panel._points()[0]
        assert edited.value == 20.5
        assert (edited.wavelength_nm, edited.n, edited.sigma_nm) == (
            point.wavelength_nm,
            point.n,
            point.sigma_nm,
        )
        assert table.item(0, 1).text() == "20.5"

        editor.setText("двадцать")
        delegate.setModelData(editor, table.model(), index)
        assert panel._points()[0].value == 20.5
        assert any("значение эталона не изменено" in notice for notice in controller.notices)
    finally:
        panel.close()
        panel.deleteLater()


def test_точки_переживают_сохранение_и_перезапуск_без_искажений(
    application: QApplication, controller: AppController, tmp_path: Path
) -> None:
    points = (
        CalibrationPoint(1549.681234567891, 20.123456789012, 187, 0.002345678901234),
        CalibrationPoint(1549.912345678912, 45.5, 1, None),
        CalibrationPoint(1550.143210987654, 70.000000000001, 200, 0.0),
    )
    panel = SensorsPanel(controller)
    try:
        panel._new_sensor()
        panel.id_edit.setText("T1")
        panel.expected_spin.setValue(1549.9)
        panel.window_spin.setValue(0.5)
        panel._set_points(points)
        panel._save_sensor()
    finally:
        panel.close()
        panel.deleteLater()

    restarted = AppController(controller.config)
    restarted.start()
    again = SensorsPanel(restarted)
    try:
        sensor = next(item for item in restarted.sensors if item.id == "T1")
        assert sensor.calibration_points == points
        again._load_sensor(sensor)
        assert again._points() == points
        assert again._sensor_from_editor().calibration_points == points
    finally:
        again.close()
        again.deleteLater()
        restarted.shutdown()


def test_полоса_датчиков_по_умолчанию_выключена(
    application: QApplication, controller: AppController
) -> None:
    controller.replace_sensors(
        (Sensor("T1", "T1", 0, SensorType.TEMPERATURE, 1549.68, 0.35, 25.0, 100.0),)
    )
    panel = SensorsPanel(controller)
    try:
        assert panel.band_mode.currentData() == "none"
        assert [panel.band_mode.itemData(i) for i in range(panel.band_mode.count())] == [
            "none",
            "range",
        ]
    finally:
        panel.close()
        panel.deleteLater()
