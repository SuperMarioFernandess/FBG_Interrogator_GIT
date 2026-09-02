"""Виджет панели настройки. Маркер `ui`, гоняется offscreen.

Проверяется устройство панели и блокировки. Внешний вид, ширины и читаемость
остаётся визуальной приёмкой на Windows.
"""

import os
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting, ModuleParams, SweepConfig
from fbg.core.profile import DeviceProfile
from fbg.core.session import DeviceConfig, Result, SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.panels.device_config import DeviceConfigPanel

pytestmark = pytest.mark.ui

PROFILE = DeviceProfile()
DEVICE = DeviceConfig(
    version_raw=410,
    serial=94_401_220,
    module=ModuleParams(0x00CA, 2000, 4, 30, 30),
    sweep=SweepConfig.from_params(1, 2, 5101, 2, PROFILE),
    channels=tuple(ChannelSetup(channel, None, GainSetting(False, 5)) for channel in range(4)),
)


@pytest.fixture(scope="session")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


@pytest.fixture
def controller() -> Iterator[AppController]:
    app_controller = AppController(AppConfig(packet_log=PacketLogConfig(directory=None)))
    app_controller.start()
    try:
        yield app_controller
    finally:
        app_controller.shutdown()


@pytest.fixture
def panel(application: QApplication, controller: AppController) -> Iterator[DeviceConfigPanel]:
    widget = DeviceConfigPanel(controller)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()


def snapshot(state: SessionState = SessionState.IDLE, **changes: object) -> models.AppSnapshot:
    values: dict[str, object] = {
        "endpoint": Endpoint(),
        "profile": PROFILE,
        "state": state,
        "device": DEVICE,
    }
    values.update(changes)
    return models.AppSnapshot(**values)  # type: ignore[arg-type]


def test_поля_имеют_жёсткие_диапазоны(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    assert panel.channel_box.count() == 4
    assert panel.threshold_spin.minimum() == 0
    assert panel.threshold_spin.maximum() == 16383
    assert panel.gain_level.minimum() == 0
    assert panel.gain_level.maximum() == 5
    assert panel.peak_gap.minimum() == 1
    assert panel.peak_gap.maximum() == 255


def test_автопорог_не_вводится_как_ffff(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    assert panel.threshold_auto.isChecked()
    assert not panel.threshold_spin.isEnabled()
    panel.threshold_auto.setChecked(False)
    assert panel.threshold_spin.isEnabled()
    assert panel.threshold_spin.maximum() != 0xFFFF


@pytest.mark.parametrize(
    ("state", "enabled"),
    [
        (SessionState.DISCONNECTED, False),
        (SessionState.PROBING, False),
        (SessionState.IDLE, True),
        (SessionState.STREAMING, True),
        (SessionState.DEBUG, False),
        (SessionState.DEGRADED, False),
        (SessionState.RECONNECTING, False),
    ],
)
def test_кнопки_блокируются_по_состоянию(
    panel: DeviceConfigPanel, state: SessionState, enabled: bool
) -> None:
    panel.refresh(snapshot(state))
    assert panel.apply_threshold_button.isEnabled() is enabled
    assert panel.apply_gain_button.isEnabled() is enabled
    assert panel.apply_peak_gap_button.isEnabled() is enabled
    assert panel.apply_sweep_button.isEnabled() is enabled
    assert panel.save_button.isEnabled() is enabled
    if state is SessionState.STREAMING:
        assert texts.CONFIG_STREAMING_ALLOWED in panel.availability_label.text()


def test_расхождение_геометрии_блокирует_кнопки(panel: DeviceConfigPanel) -> None:
    mismatch = models.ProfileDifference("channels", 8, 4)
    panel.refresh(snapshot(profile_mismatch=(mismatch,)))
    assert not panel.apply_threshold_button.isEnabled()
    assert not panel.apply_gain_button.isEnabled()
    assert not panel.apply_peak_gap_button.isEnabled()
    assert not panel.apply_sweep_button.isEnabled()
    assert not panel.save_button.isEnabled()
    assert texts.CONFIG_PROFILE_MISMATCH in panel.availability_label.text()


def test_развёртка_заблокирована_во_время_записи(panel: DeviceConfigPanel) -> None:
    """Р67 приходит из `AppSnapshot.recording`, без второго флага панели."""
    panel.refresh(snapshot(SessionState.STREAMING, recording=True))
    assert panel.apply_threshold_button.isEnabled()
    assert panel.apply_gain_button.isEnabled()
    assert not panel.apply_sweep_button.isEnabled()
    assert texts.CONFIG_RECORDING_SWEEP_LOCKED in panel.availability_label.text()


def test_невалидная_развёртка_не_может_быть_отправлена(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    panel.start_param.setValue(5101)
    panel.stop_param.setValue(1)
    assert not panel.apply_sweep_button.isEnabled()
    assert "Настройка не отправлена" in panel.sweep_preview.text()


def test_preview_показывает_точки_и_границы_в_нм(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    text = panel.sweep_preview.text()
    assert "2551" in text
    assert "196249" in text
    assert "191149" in text
    assert "нм" in text


def test_развёртка_предупреждает_о_переподключении(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    assert "подключитесь снова" in panel.sweep_hint.text()


def test_подсказки_используют_последний_спектр(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot(last_spectrum_max_adc=11123, last_spectrum_saturated_points=17))
    panel.threshold_auto.setChecked(False)
    panel.threshold_spin.setValue(12000)
    assert "11123" in panel.threshold_hint.text()
    assert "0 пиков" in panel.threshold_hint.text()
    assert "17" in panel.gain_hint.text()
    assert "Насыщение" in panel.gain_hint.text()


def test_unconfirmed_показывается_у_того_же_поля(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot(unconfirmed=frozenset({"threshold:0", "gain:0", "sweep"})))
    assert panel.threshold_status.text() == texts.UNCONFIRMED
    assert panel.gain_status.text() == texts.UNCONFIRMED
    assert panel.sweep_status.text() == texts.UNCONFIRMED


def test_unconfirmed_не_теряется_когда_поле_редактируется(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    panel.threshold_auto.setChecked(False)
    panel.threshold_spin.setValue(1000)
    panel.start_param.setValue(2)

    panel.refresh(snapshot(unconfirmed=frozenset({"threshold:0", "sweep"})))
    assert panel.threshold_spin.value() == 1000
    assert panel.threshold_status.text() == texts.UNCONFIRMED
    assert panel.start_param.value() == 2
    assert panel.sweep_status.text() == texts.UNCONFIRMED


def test_сохранение_честно_описывает_границы_подтверждения(panel: DeviceConfigPanel) -> None:
    panel.refresh(snapshot())
    hint = panel.save_hint.text()
    assert "только пороги" in hint
    assert "уровень 5" in hint
    assert "перезапуск питания" in hint


def test_автопорог_отправляется_как_none(
    panel: DeviceConfigPanel, controller: AppController, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int | None]] = []

    def fake_set_threshold(channel: int, threshold: int | None) -> Result[ChannelSetup]:
        calls.append((channel, threshold))
        return Result(value=DEVICE.channels[channel])

    monkeypatch.setattr(controller, "set_threshold", fake_set_threshold)
    panel.refresh(snapshot())
    panel.channel_box.setCurrentIndex(2)
    panel.threshold_auto.setChecked(True)
    panel.apply_threshold_button.click()
    assert calls == [(2, None)]
