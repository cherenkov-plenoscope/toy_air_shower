import numpy as np
import toy_air_shower as tas


def test_frank_tamm():
    # http://www.desy.de/~niebuhr/Vorlesung/Detektor/Vorlesung_11.pdf
    #
    # 400nm to 700nm
    #
    # medium            n         Photonen / m^{-1}     beta
    # ---------------------------------------------------------------
    # air               1.000295  30                    1
    # isobutan          1.00131   130                   1

    wavelengths = np.linspace(400e-9, 700e-9, 1024)
    spectrum = tas.make_cherenkov_spectrum(wavelengths)
    avg_wavelength = np.average(wavelengths, weights=spectrum)
    avg_photon_energy = (tas.SPEED_OF_LIGHT*tas.PLANCK_ACTION)/avg_wavelength

    dNdz_air = -tas.dE_over_dz(
        q=tas.UNIT_CHARGE,
        beta=1,
        n=1.000295,
        mu=tas.PERMABILITY_AIR,
        wavelength_start=400e-9,
        wavelength_end=700e-9
    )/avg_photon_energy
    assert(np.abs(dNdz_air - 30) < 5)

    dNdz_isobutan = -tas.dE_over_dz(
        q=tas.UNIT_CHARGE,
        beta=1,
        n=1.00131,
        mu=tas.PERMABILITY_AIR,
        wavelength_start=400e-9,
        wavelength_end=700e-9
    )/avg_photon_energy
    assert(np.abs(dNdz_isobutan - 130) < 10)
