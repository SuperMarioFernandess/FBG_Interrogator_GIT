"""UI панели спектра; запускается только с Qt/offscreen."""

import os
from collections.abc import Iterator

import pytest

pytest.importorskip("PySide6", reason="тесты интерфейса требуют Qt")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from fbg.core.frames import AdcBlock, GainSetting
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.models import spectrum_model
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


def _spectrum(regions: int = 1):
    profile = DeviceProfile()
    adc = np.zeros(profile.adc_points, dtype=np.uint16)
    for index in range(regions):
        start = 20 + index * 40
        adc[start : start + 4] = (4000, 8000, 7000, 4000)
    block = AdcBlock(
        channel=0,
        gain=GainSetting(manual=True, level=0),
        adc=adc,
    )
    return spectrum_model(block, profile, 3000)


def test_подпись_различает_заданный_и_фактический_период(
    application: QApplication, controller: AppController
) -> None:
    panel = SpectrumPanel(controller)
    panel.period.setValue(0.25)
    assert panel.frequency_label.text() == "задано 4.000 Гц; факт —"
    snapshot = models.AppSnapshot(
        endpoint=controller.config.endpoint,
        profile=DeviceProfile(),
        state=SessionState.IDLE,
        spectrum_actual_period_s=0.80,
    )
    panel.refresh(snapshot)
    assert panel.frequency_label.text() == "задано 4.000 Гц; факт 1.250 Гц (0.800 с)"
    panel.deleteLater()


def test_неизменившаяся_версия_спектра_не_перерисовывается(
    application: QApplication, controller: AppController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Р76: десять тиков с тем же снимком дают ровно одну отрисовку."""
    panel = SpectrumPanel(controller)
    model = _spectrum()
    calls = 0
    original = panel._show_model

    def counted(current):
        nonlocal calls
        calls += 1
        original(current)

    monkeypatch.setattr(panel, "_show_model", counted)
    snapshot = models.AppSnapshot(
        endpoint=controller.config.endpoint,
        profile=DeviceProfile(),
        state=SessionState.IDLE,
        spectrum=model,
        spectrum_version=1,
    )
    for _ in range(10):
        panel.refresh(snapshot)
    assert calls == 1

    # Содержимое то же самое, но версия новая: это новое измерение и оно рисуется.
    panel.refresh(
        models.AppSnapshot(
            endpoint=controller.config.endpoint,
            profile=DeviceProfile(),
            state=SessionState.IDLE,
            spectrum=model,
            spectrum_version=2,
        )
    )
    assert calls == 2
    panel.deleteLater()


def test_таблица_областей_сохраняет_выделение_и_прокрутку(
    application: QApplication, controller: AppController
) -> None:
    panel = SpectrumPanel(controller)
    first = _spectrum(regions=40)
    panel.refresh(
        models.AppSnapshot(
            endpoint=controller.config.endpoint,
            profile=DeviceProfile(),
            state=SessionState.IDLE,
            spectrum=first,
            spectrum_version=1,
        )
    )
    panel.table.resize(500, 120)
    panel.table.selectRow(20)
    scroll = panel.table.verticalScrollBar()
    scroll.setValue(scroll.maximum())
    before_scroll = scroll.value()
    before_item = panel.table.item(20, 0)

    # Та же геометрия таблицы, но новый снимок. setRowCount/items не пересоздаются.
    panel.refresh(
        models.AppSnapshot(
            endpoint=controller.config.endpoint,
            profile=DeviceProfile(),
            state=SessionState.IDLE,
            spectrum=_spectrum(regions=40),
            spectrum_version=2,
        )
    )

    assert panel.table.item(20, 0) is before_item
    assert panel.table.currentRow() == 20
    assert scroll.value() == before_scroll
    panel.deleteLater()
