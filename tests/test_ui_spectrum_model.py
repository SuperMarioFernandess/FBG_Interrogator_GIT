"""Qt-free модель спектра 30 07."""

import math

import numpy as np

from fbg.core.frames import AdcBlock, GainSetting
from fbg.core.profile import DeviceProfile
from fbg.ui.models import adc_to_dbm, spectrum_model


def block(adc: np.ndarray, level: int = 0) -> AdcBlock:
    return AdcBlock(
        channel=0, gain=GainSetting(manual=True, level=level), adc=adc.astype(np.uint16)
    )


def test_модель_имеет_2551_точку_и_убывающую_лямбду() -> None:
    profile = DeviceProfile()
    model = spectrum_model(block(np.zeros(profile.adc_points)), profile, 3000)
    assert model.adc.size == 2551
    assert model.wavelength_nm.size == 2551
    assert model.wavelength_nm[0] > model.wavelength_nm[-1]
    assert math.isclose(model.wavelength_nm[0], profile.adc_index_to_nm(0.0))
    assert math.isclose(model.wavelength_nm[-1], profile.adc_index_to_nm(2550.0))


def test_adc_to_dbm_использует_тёмное_смещение_и_текущий_gain() -> None:
    profile = DeviceProfile()
    adc = np.array([profile.adc_dark_offset, profile.adc_dark_offset + 1000], dtype=np.uint16)
    power = adc_to_dbm(adc, 0, profile)
    assert math.isnan(power[0])
    assert math.isclose(power[1], 10.0 * math.log10(1000 * profile.gain_power_coefficients[0]))
    assert not math.isclose(power[1], adc_to_dbm(adc, 5, profile)[1])


def test_плато_считается_одной_областью() -> None:
    profile = DeviceProfile()
    adc = np.zeros(profile.adc_points, dtype=np.uint16)
    adc[100:110] = 5000
    model = spectrum_model(block(adc), profile, 3000)
    assert len(model.regions) == 1
    assert (model.regions[0].start_index, model.regions[0].stop_index) == (100, 109)


def test_насыщенная_область_не_имеет_положения_и_fwhm() -> None:
    profile = DeviceProfile()
    adc = np.zeros(profile.adc_points, dtype=np.uint16)
    adc[500:510] = 5000
    adc[504:507] = profile.adc_max
    model = spectrum_model(block(adc), profile, 3000)
    region = model.regions[0]
    assert model.saturated_points == 3
    assert region.saturated_points == 3
    assert region.peak_nm is None
    assert region.centroid_nm is None
    assert region.fwhm_nm is None


def test_fwhm_ищется_по_исходному_спектру_а_не_обрезается_порогом() -> None:
    profile = DeviceProfile()
    adc = np.zeros(profile.adc_points, dtype=np.uint16)
    center = 1000
    for offset in range(-20, 21):
        adc[center + offset] = max(0, 10_000 - abs(offset) * 400)
    model = spectrum_model(block(adc), profile, 9000)
    assert len(model.regions) == 1
    assert model.regions[0].fwhm_nm is not None
    assert model.regions[0].fwhm_nm > model.regions[0].width_nm


def test_пять_областей_по_центрам_линии_сессии_4() -> None:
    """Частично реконструированный тест: центры реальны из журнала, ADC синтетический."""
    profile = DeviceProfile()
    adc = np.full(profile.adc_points, 100, dtype=np.uint16)
    expected_nm = (1538.22, 1544.78, 1549.68, 1551.35, 1559.77)
    for wavelength in expected_nm:
        freq = 299_792_458.0 / wavelength
        center = round(profile.ghz_to_adc_index(freq))
        for offset, value in enumerate((3500, 5000, 8000, 5000, 3500), start=-2):
            adc[center + offset] = value
    model = spectrum_model(block(adc), profile, 3000)
    assert len(model.regions) == 5
    found = sorted(region.peak_nm for region in model.regions if region.peak_nm is not None)
    assert len(found) == 5
    for actual, expected in zip(found, sorted(expected_nm), strict=True):
        assert abs(actual - expected) < 0.03


def test_вершина_уточняется_дробным_индексом() -> None:
    profile = DeviceProfile()
    adc = np.zeros(profile.adc_points, dtype=np.uint16)
    center = 900
    adc[center - 1 : center + 2] = (7000, 10000, 9000)
    model = spectrum_model(block(adc), profile, 3000)
    region = model.regions[0]
    assert region.peak_index is not None
    assert center < region.peak_index < center + 0.5
    assert region.peak_nm is not None
    assert math.isclose(
        region.peak_nm, profile.adc_index_to_nm(region.peak_index), rel_tol=0.0, abs_tol=1e-12
    )
