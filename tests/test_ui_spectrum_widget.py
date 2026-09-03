"""UI панели спектра; запускается только с Qt/offscreen."""

import os
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.panels.spectrum import SpectrumPanel

pytestmark = pytest.mark.ui


@pytest.fixture(scope="session")
def application() -> QApplication:
    existing = QApplication.instance()
    return existing if isinstance(existing, QApplication) else QApplication([])


@pytest.fixture
def controller() -> Iterator[AppController]:
    ctl = AppController(AppConfig(packet_log=PacketLogConfig(directory=None)))
    ctl.start()
    try:
        yield ctl
    finally:
        ctl.shutdown()


def test_панель_имеет_оба_режима(application: QApplication, controller: AppController) -> None:
    panel = SpectrumPanel(controller)
    assert panel.take_button.text() == texts.BUTTON_TAKE_SPECTRUM
    assert panel.start_button.text() == texts.BUTTON_START_SPECTRUM
    assert panel.stop_button.text() == texts.BUTTON_STOP_SPECTRUM
    panel.deleteLater()


def test_запись_блокирует_снятие(application: QApplication, controller: AppController) -> None:
    panel = SpectrumPanel(controller)
    snapshot = models.AppSnapshot(
        endpoint=controller.config.endpoint,
        profile=DeviceProfile(),
        state=SessionState.STREAMING,
        recording=True,
    )
    panel.refresh(snapshot)
    assert not panel.take_button.isEnabled()
    assert not panel.start_button.isEnabled()
    assert texts.SPECTRUM_RECORDING_LOCKED in panel.warning.text()
    panel.deleteLater()


def test_идущий_одиночный_снимок_блокирует_повторный_запуск(
    application: QApplication, controller: AppController
) -> None:
    panel = SpectrumPanel(controller)
    snapshot = models.AppSnapshot(
        endpoint=controller.config.endpoint,
        profile=DeviceProfile(),
        state=SessionState.IDLE,
        spectrum_busy=True,
    )
    panel.refresh(snapshot)
    assert not panel.take_button.isEnabled()
    assert not panel.start_button.isEnabled()
    assert not panel.stop_button.isEnabled()
    panel.deleteLater()


def test_период_показывает_эквивалентную_частоту(
    application: QApplication, controller: AppController
) -> None:
    panel = SpectrumPanel(controller)
    panel.period.setValue(0.25)
    assert panel.frequency_label.text() == "4.000"
    panel.deleteLater()
