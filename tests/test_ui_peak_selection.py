"""Выбор «Текущий пик» держится за решётку (Р89, №43) — Qt-free часть.

Сценарий координатора: линия заказчика 1538.22 · 1544.78 · 1549.68 · 1551.35
· 1559.77 нм, оператор выбрал 1549.68, пик дрогнул на квант 0.8 пм. Прежний
код искал выбор по λ до девятого знака, не находил и Qt молча ставил первый
пик — 1538.22.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fbg.core.endpoint import Endpoint
from fbg.core.profile import DeviceProfile
from fbg.core.session import SessionState
from fbg.io import config as config_module
from fbg.io.config import AppConfig, IssueKind
from fbg.ui import models
from fbg.ui.models import PeakTrackStatus, follow_peak, parse_user_number, select_peak

LINE = np.asarray([1538.22, 1544.78, 1549.68, 1551.35, 1559.77])
QUANTUM_NM = DeviceProfile().wavelength_quantization_nm(1549.68)


def test_квант_в_тесте_настоящий() -> None:
    assert abs(QUANTUM_NM - 0.0080) < 0.0002


def test_дрожание_на_квант_не_меняет_выбор() -> None:
    track = select_peak(LINE, 2)
    jittered = LINE.copy()
    jittered[2] += 0.0008
    track = follow_peak(track, jittered)
    assert track.status is PeakTrackStatus.FOUND
    assert track.index == 2
    assert track.wavelength_nm == pytest.approx(1549.6808)


def test_выбор_переживает_десять_тактов_дрейфа_и_дрожания() -> None:
    """№43: выставленное человеком переживает десять тактов."""
    rng = np.random.default_rng(1)
    track = select_peak(LINE, 2)
    for tick in range(10):
        frame = LINE + rng.integers(-2, 3, LINE.size) * QUANTUM_NM
        frame[2] += 0.002 * tick  # температурный дрейф выбранной решётки
        track = follow_peak(track, frame)
        assert track.status is PeakTrackStatus.FOUND
        assert track.index == 2
        assert abs(track.wavelength_nm - frame[2]) < 1e-12


def test_пропавший_пик_не_переключает_выбор_на_соседа() -> None:
    track = select_peak(LINE, 2)
    without = np.delete(LINE, 2)
    lost = follow_peak(track, without)
    assert lost.status is PeakTrackStatus.LOST
    assert lost.index == -1
    assert not lost.usable
    assert lost.wavelength_nm == pytest.approx(1549.68)


def test_вернувшийся_пик_восстанавливает_тот_же_выбор() -> None:
    track = follow_peak(select_peak(LINE, 2), np.delete(LINE, 2))
    returned = LINE.copy()
    returned[2] += 0.05  # за время отсутствия решётка сдвинулась
    track = follow_peak(track, returned)
    assert track.status is PeakTrackStatus.FOUND
    assert track.wavelength_nm == pytest.approx(1549.73)


def test_пустой_кадр_вне_потока_теряет_выбор_но_не_стирает() -> None:
    track = follow_peak(select_peak(LINE, 2), np.empty(0))
    assert track.status is PeakTrackStatus.LOST
    assert follow_peak(track, LINE).index == 2


def test_допуск_ограничен_половиной_расстояния_до_соседа() -> None:
    """Паразитный пик 1545.34 рядом с 1544.78 (N19) сужает допуск до 0.28 нм."""
    assert select_peak(LINE, 2).tolerance_nm == pytest.approx(models.PEAK_TRACK_MAX_TOLERANCE_NM)
    crowded = np.asarray([1544.78, 1545.34, 1549.68])
    track = select_peak(crowded, 0)
    assert track.tolerance_nm == pytest.approx(0.28)
    # Пропал выбранный — сосед на 0.56 нм остаётся за допуском.
    assert follow_peak(track, crowded[1:]).status is PeakTrackStatus.LOST


def test_два_пика_в_допуске_это_неоднозначность_а_не_угадывание() -> None:
    track = select_peak(LINE, 2)
    doubled = np.sort(np.r_[LINE, 1549.90])
    result = follow_peak(track, doubled)
    assert result.status is PeakTrackStatus.AMBIGUOUS
    assert not result.usable


def test_без_выбора_ничего_не_выбирается_само() -> None:
    track = follow_peak(models.PeakTrack(), LINE)
    assert track.status is PeakTrackStatus.NONE
    assert track.index == -1


def test_мутация_ближайший_без_допуска_ломает_тест() -> None:
    """Тест решения, а не арифметики: «ближайший пик» перескочил бы на соседа."""
    without = np.delete(LINE, 2)
    nearest = int(np.argmin(np.abs(without - 1549.68)))
    assert without[nearest] == pytest.approx(1551.35)
    assert follow_peak(select_peak(LINE, 2), without).status is PeakTrackStatus.LOST


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("20,5", 20.5),
        ("20.5", 20.5),
        (" -3,25 ", -3.25),
        ("\u22121,5", -1.5),
        ("1e3", 1000.0),
        ("20,", 20.0),
    ],
)
def test_число_с_запятой_и_точкой(text: str, value: float) -> None:
    assert parse_user_number(text) == value


@pytest.mark.parametrize("text", ["", "abc", "1,234.5", "1.2.3", "nan", "inf"])
def test_неверное_число_отвергается(text: str) -> None:
    with pytest.raises(ValueError):
        parse_user_number(text)


def test_текущая_λ_только_у_слотов_с_пиком() -> None:
    from types import SimpleNamespace

    wavelengths = np.full((4, 30), np.nan)
    wavelengths[0, 3] = 1549.6808
    wavelengths[2, 0] = 1538.22
    snapshot = models.AppSnapshot(
        endpoint=Endpoint(),
        profile=DeviceProfile(),
        state=SessionState.STREAMING,
        ui=SimpleNamespace(wavelength_nm=wavelengths),
    )
    assert models.current_wavelength_texts(snapshot) == {
        models.SlotRef(0, 3): "1549.680800",
        models.SlotRef(2, 0): "1538.220000",
    }


# --- выбор полосы сохраняется -------------------------------------------------------


def test_полоса_по_умолчанию_только_линия_и_сохраняется(tmp_path: Path) -> None:
    config = AppConfig()
    assert config.graph_band("measurement") == "none"
    assert config.graph_band("sensors") == "none"
    changed = config.with_graph_band("measurement", "sigma").with_graph_band("sensors", "range")
    path = tmp_path / "fbg_config.json"
    config_module.save(changed, path)
    loaded = config_module.load(path)
    assert loaded.config.graph_band("measurement") == "sigma"
    assert loaded.config.graph_band("sensors") == "range"
    assert not loaded.issues


def test_у_датчиков_нет_полосы_сигма() -> None:
    with pytest.raises(ValueError):
        AppConfig().with_graph_band("sensors", "sigma")


def test_испорченный_выбор_полосы_стоит_одной_вкладки(tmp_path: Path) -> None:
    path = tmp_path / "fbg_config.json"
    config_module.save(AppConfig(), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["graph_bands"] = {"measurement": "range", "sensors": "rainbow", "spectrum": "sigma"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = config_module.load(path)
    assert loaded.config.graph_band("measurement") == "range"
    assert loaded.config.graph_band("sensors") == "none"
    kinds = {issue.kind for issue in loaded.issues}
    assert IssueKind.REJECTED_VALUE in kinds
    assert IssueKind.UNKNOWN_FIELD in kinds
