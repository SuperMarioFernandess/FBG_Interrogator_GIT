"""Модель и контроллер панели настройки прибора — без Qt.

Здесь лежит опасная часть UI: диапазоны и номера каналов проверяются до
отправки, а состояние `unconfirmed` берётся из уже существующего механизма
сессии. Виджет только раскладывает эти значения по полям.
"""

from pathlib import Path

import pytest

from fbg.core.endpoint import Endpoint
from fbg.core.frames import ChannelSetup, GainSetting, ModuleParams, SweepConfig
from fbg.core.pipeline import PipelineConfig
from fbg.core.profile import C_NM_GHZ, DeviceProfile
from fbg.core.session import DeviceConfig, SessionConfig, SessionErrorKind, SessionState
from fbg.io.config import AppConfig
from fbg.io.packet_log import PacketLogConfig
from fbg.sim.device_sim import DeviceSimulator
from fbg.sim.scene import Grating, Scene
from fbg.ui import models, texts
from fbg.ui.app import AppController
from fbg.ui.models import AppSnapshot

PROFILE = DeviceProfile()
DEVICE = DeviceConfig(
    version_raw=410,
    serial=94_401_220,
    module=ModuleParams(
        speed_code=0x00CA,
        speed_hz=2000,
        channels=4,
        fbg_per_channel=30,
        peak_gap_ghz=30,
    ),
    sweep=SweepConfig.from_params(1, 2, 5101, 2, PROFILE),
    channels=tuple(
        ChannelSetup(channel, None, GainSetting(manual=False, level=5)) for channel in range(4)
    ),
)

QUIET = SessionConfig(
    keepalive_period_s=30.0,
    keepalive_failures_to_degrade=2,
    stream_stall_floor_s=0.3,
    backoff_schedule=(0.05, 0.1),
    retry_pause_s=0.01,
    reassembly_timeout_s=0.3,
    watchdog_tick_s=0.02,
    settle_before_readback_s=0.01,
    auto_reconnect=False,
)


def snapshot(**changes: object) -> AppSnapshot:
    values: dict[str, object] = {
        "endpoint": Endpoint(),
        "profile": PROFILE,
        "state": SessionState.IDLE,
        "device": DEVICE,
    }
    values.update(changes)
    return AppSnapshot(**values)  # type: ignore[arg-type]


class LyingThresholdSimulator(DeviceSimulator):
    """Подтверждает 20 02, но не меняет порог — стимул для read-back mismatch."""

    def _apply_threshold(self, request: bytes) -> bool:
        return True


class SaveReadbackMismatchSimulator(DeviceSimulator):
    """После 20 06 меняет текущий порог: read-back обязан заметить расхождение."""

    def _handle_write(self, fc: int, request: bytes) -> list[bytes]:
        responses = super()._handle_write(fc, request)
        if fc == 0x06:
            self.state.thresholds[0] = 1234
        return responses


class ControllerStand:
    """Мини-стенд для методов `AppController`, без виджетов."""

    def __init__(self, tmp_path: Path, simulator: type[DeviceSimulator] = DeviceSimulator) -> None:
        profile = DeviceProfile()
        self.sim = simulator(
            profile=profile,
            scene=Scene(profile, [Grating(0, 0, 1545.0)]),
            reply_to=("127.0.0.1", 1),
            frame_rate_hz=200.0,
        )
        self.sim.start()
        host, port = self.sim.address
        endpoint = Endpoint(
            device_ip=host,
            device_port=port,
            local_ip="127.0.0.1",
            local_port=0,
            read_timeout_s=0.15,
            write_timeout_s=0.2,
            retries=1,
            rx_poll_timeout_s=0.02,
        )
        self.controller = AppController(
            AppConfig(
                endpoint=endpoint,
                profile=profile,
                session=QUIET,
                pipeline=PipelineConfig(history_frames=256),
                packet_log=PacketLogConfig(directory=None),
            ),
            config_path=tmp_path / "fbg_config.json",
        )
        self.controller.start()
        self.controller.session._transport.open()
        self.sim.reply_to = self.controller.session.local_address
        self.controller.connect().unwrap()

    def close(self) -> None:
        self.sim.stop()
        self.controller.shutdown()


# --------------------------------------------------------------------------------------
# Локальная валидация — до сети
# --------------------------------------------------------------------------------------


def test_номер_канала_проверяется_по_живому_числу_каналов() -> None:
    assert models.validate_channel(0, 4) == 0
    assert models.validate_channel(3, 4) == 3
    with pytest.raises(ValueError, match="R14"):
        models.validate_channel(4, 4)
    with pytest.raises(ValueError, match="R14"):
        models.validate_channel(-1, 4)


def test_порог_строго_0_16383_а_ffff_только_через_авто() -> None:
    assert models.threshold_value(False, 0, PROFILE.adc_max) == 0
    assert models.threshold_value(False, 16383, PROFILE.adc_max) == 16383
    assert models.threshold_value(True, 1234, PROFILE.adc_max) is None
    with pytest.raises(ValueError):
        models.threshold_value(False, 16384, PROFILE.adc_max)
    with pytest.raises(ValueError):
        models.threshold_value(False, 0xFFFF, PROFILE.adc_max)


def test_уровень_усиления_строго_0_5() -> None:
    assert models.gain_value(True, 0, PROFILE.gain_max_level) == GainSetting(True, 0)
    assert models.gain_value(False, 5, PROFILE.gain_max_level) == GainSetting(False, 5)
    with pytest.raises(ValueError):
        models.gain_value(True, 6, PROFILE.gain_max_level)


def test_развёртка_проверяет_инвариант_start_stop() -> None:
    with pytest.raises(ValueError, match="start_param < stop_param"):
        models.sweep_edit_model(PROFILE, 5101, 2, 1, 2)
    with pytest.raises(ValueError, match="start_param < stop_param"):
        models.sweep_edit_model(PROFILE, 1, 2, 1, 2)


def test_развёртка_считает_точки_и_границы_в_нм() -> None:
    sweep = models.sweep_edit_model(PROFILE, 1, 2, 5101, 2)
    assert sweep.start_ghz == 196249
    assert sweep.stop_ghz == 191149
    assert sweep.adc_points == 2551
    assert sweep.start_nm == pytest.approx(C_NM_GHZ / 196249)
    assert sweep.stop_nm == pytest.approx(C_NM_GHZ / 191149)


# --------------------------------------------------------------------------------------
# Модель из снимка и разметка read-back
# --------------------------------------------------------------------------------------


def test_модель_строится_из_снимка_без_qt() -> None:
    model = models.device_config_model(snapshot())
    assert model.enabled
    assert model.channel_count == 4
    assert len(model.channels) == 4
    assert model.channels[0].threshold is None
    assert model.channels[0].gain == GainSetting(False, 5)
    assert model.peak_gap_ghz == 30
    assert model.sweep is not None and model.sweep.adc_points == 2551


def test_модель_несёт_сводку_последнего_спектра_без_qt() -> None:
    model = models.device_config_model(
        snapshot(last_spectrum_max_adc=11123, last_spectrum_saturated_points=17)
    )
    assert model.last_spectrum_max_adc == 11123
    assert model.last_spectrum_saturated_points == 17


def test_подсказка_порога_предупреждает_если_он_выше_максимума() -> None:
    text = texts.threshold_spectrum_hint(11123, 12000)
    assert "11123" in text
    assert "12000" in text
    assert "0 пиков" in text


def test_подсказка_усиления_сообщает_о_насыщении() -> None:
    text = texts.gain_spectrum_hint(17)
    assert "17" in text
    assert "Насыщение" in text


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
def test_запись_разрешена_только_в_idle_и_streaming(state: SessionState, enabled: bool) -> None:
    assert models.device_config_model(snapshot(state=state)).enabled is enabled


def test_расхождение_геометрии_блокирует_редактирование() -> None:
    mismatch = models.ProfileDifference("channels", 8, 4)
    assert not models.device_config_model(snapshot(profile_mismatch=(mismatch,))).enabled


def test_развёртка_блокируется_тем_же_snapshot_recording_что_и_панель_измерения() -> None:
    """Р67: единственный источник состояния записи — `AppSnapshot.recording`."""
    model = models.device_config_model(snapshot(state=SessionState.STREAMING, recording=True))
    assert model.enabled, "порог и усиление во время потока по Р62 остаются доступны"
    assert not model.sweep_enabled


def test_unconfirmed_размечается_существующими_ключами_сессии() -> None:
    model = models.device_config_model(
        snapshot(
            unconfirmed=frozenset(
                {"threshold:0", "gain:1", "peak_gap", "sweep", "saved_thresholds"}
            )
        )
    )
    assert model.channels[0].threshold_unconfirmed
    assert model.channels[1].gain_unconfirmed
    assert model.peak_gap_unconfirmed
    assert model.sweep is not None and model.sweep.unconfirmed
    assert model.saved_thresholds_unconfirmed


# --------------------------------------------------------------------------------------
# Контроллер: валидация, read-back и 20 06
# --------------------------------------------------------------------------------------


def test_контроллер_не_выпускает_опасный_номер_канала(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path)
    try:
        before = stand.sim.stats.requests
        with pytest.raises(ValueError, match="R14"):
            stand.controller.set_threshold(4, 1000)
        assert stand.sim.stats.requests == before
    finally:
        stand.close()


def test_контроллер_не_выпускает_опасный_порог_и_усиление(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path)
    try:
        before = stand.sim.stats.requests
        with pytest.raises(ValueError, match="0…16383"):
            stand.controller.set_threshold(0, 16384)
        with pytest.raises(ValueError, match="0…16383"):
            stand.controller.set_threshold(0, -1)
        with pytest.raises(ValueError, match="0…5"):
            stand.controller.set_gain(0, True, 6)
        with pytest.raises(ValueError, match="0…5"):
            stand.controller.set_gain(0, True, -1)
        assert stand.sim.stats.requests == before
    finally:
        stand.close()


def test_подтверждённая_геометрия_обновляет_кэш_профиля(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path)
    try:
        assert stand.controller.set_peak_gap(40).ok
        assert stand.controller.config.profile.peak_gap_ghz == 40
        assert stand.controller.set_sweep(0, 2, 5100, 2).ok
        profile = stand.controller.config.profile
        assert (profile.start_param, profile.stop_param) == (0, 5100)
        assert stand.controller.session.state is SessionState.DISCONNECTED
        assert "подключитесь снова" in " ".join(stand.controller.notices)
        assert (tmp_path / "fbg_config.json").is_file()
    finally:
        stand.close()


def test_readback_подтверждает_запись_и_снимает_unconfirmed(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path)
    try:
        result = stand.controller.set_threshold(0, 1000)
        assert result.ok and result.unwrap().threshold == 1000
        assert "threshold:0" not in stand.controller.snapshot().unconfirmed
    finally:
        stand.close()


def test_readback_расхождение_остаётся_unconfirmed(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path, LyingThresholdSimulator)
    try:
        result = stand.controller.set_threshold(0, 1000)
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.VERIFICATION_MISMATCH
        assert "threshold:0" in stand.controller.snapshot().unconfirmed
    finally:
        stand.close()


def test_20_06_не_ждёт_ответа_и_проверяет_пороги_чтением(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path)
    try:
        assert stand.controller.set_threshold(0, 1000).ok
        before = stand.sim.stats.requests
        result = stand.controller.save_thresholds()
        assert result.ok
        assert stand.sim.stats.requests - before == 2  # 20 06 без ответа + read-back 10 06
        assert stand.sim.state.saved_thresholds == stand.sim.state.thresholds
        assert "saved_thresholds" not in stand.controller.snapshot().unconfirmed
    finally:
        stand.close()


def test_20_06_readback_расхождение_остаётся_unconfirmed(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path, SaveReadbackMismatchSimulator)
    try:
        assert stand.controller.set_threshold(0, 1000).ok
        result = stand.controller.save_thresholds()
        assert result.error is not None
        assert result.error.kind is SessionErrorKind.VERIFICATION_MISMATCH
        assert "saved_thresholds" in stand.controller.snapshot().unconfirmed
        assert stand.sim.state.saved_thresholds is not None
        assert stand.sim.state.saved_thresholds[0] == 1000
    finally:
        stand.close()


def test_порог_можно_менять_во_время_streaming(tmp_path: Path) -> None:
    stand = ControllerStand(tmp_path)
    try:
        assert stand.controller.start_stream().ok
        result = stand.controller.set_threshold(0, 1000)
        assert result.ok
        assert stand.controller.session.state is SessionState.STREAMING
        assert stand.sim.streaming
    finally:
        stand.close()
