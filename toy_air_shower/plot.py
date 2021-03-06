from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import toy_air_shower as tas
import json


def save_shower_figure(particles, cherenkov_photons, path):
    dpi = 200
    fig = plt.figure(figsize=(8, 4.5), dpi=dpi)
    ax = fig.add_axes((0.07, 0.07, 0.92, 0.92))

    start_xs = np.zeros(len(particles))
    end_xs = np.zeros(len(particles))
    childs = np.zeros(len(particles), dtype=np.uint64)

    for idx, particle in enumerate(particles):
        color = "blue" if particle['type'] == "electron" else "green"

        if idx != 0:
            altitude_range = particle['start_altitude'] - particle['end_altitude']
            if childs[particle['mother']] == 0:
                trnansversal_offset = altitude_range
            else:
                trnansversal_offset = - altitude_range
            childs[particle['mother']] += 1
        else:
            trnansversal_offset = 0

        if idx != 0:
            start_xs[idx] = end_xs[particle['mother']]
        else:
            start_xs[idx] = 0

        end_xs[idx] = trnansversal_offset + start_xs[idx]

        start_x = start_xs[idx]
        end_x = end_xs[idx]

        if idx != 0:
            start_y = particle['start_altitude']
        else:
            start_y = 1.1*particle['end_altitude']
        end_y = particle['end_altitude']

        # trajectories
        ax.plot(
            np.array([start_x, end_x])*1e-3,
            np.array([start_y, end_y])*1e-3,
            color=color)
        # interaction marker
        if idx != 0:
            ax.plot(
                np.array([start_x])*1e-3,
                np.array([start_y])*1e-3,
                "ok")

        center_x = np.mean([start_x, end_x])
        center_y = np.mean([start_y, end_y])

        if particle['type'] == 'electron':
            mask = cherenkov_photons[:, tas.IDX_MOTHER] == idx
            num_cherenkov_photons = np.sum(mask)
            text = "{:.0f}MeV, {:d}ch-ph".format(
                1e-6*particle['start_energy']/tas.UNIT_CHARGE,
                num_cherenkov_photons)
        elif particle['type'] == 'gamma':
            text = "{:.0f}MeV".format(
                1e-6*particle['start_energy']/tas.UNIT_CHARGE)

        ax.text(
            x=center_x*1e-3,
            y=center_y*1e-3,
            s=text,
            fontsize=6)


    ax.set_ylabel("altitude / km")
    ax.set_xlabel("transverse / arb. unit")
    ax.yaxis.grid(True)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    fig.savefig(path)


def save_figure_frank_tamm(
    path,
    wavelength_start=250e-9,
    wavelength_end=700e-9
):
    altitudes = np.logspace(
        np.log10(1e3),
        np.log10(100e3),
        128)
    energies = np.logspace(
        np.log10(1e7*tas.UNIT_CHARGE),
        np.log10(1e9*tas.UNIT_CHARGE),
        128)

    cherenkov_yield = np.zeros(shape=(altitudes.shape[0], energies.shape[0]))

    for aidx, altitude in enumerate(altitudes):
        for eidx, energy in enumerate(energies):
            current_n = tas.refraction_in_air(altitude)
            current_beta = tas.natural_velocity(
                kinetic_energy=energy,
                rest_energy=tas.ELECTRON_REST_ENERGY)

            if current_beta < 1./current_n:
                cherenkov_yield[aidx, eidx] = 0.
            else:
                cherenkov_yield[aidx, eidx] = - tas.dE_over_dz(
                    q=tas.UNIT_CHARGE,
                    beta=current_beta,
                    n=current_n,
                    mu=tas.PERMABILITY_AIR,
                    wavelength_start=wavelength_start,
                    wavelength_end=wavelength_end)

    dpi = 200
    fig = plt.figure(figsize=(8, 4.5), dpi=dpi)
    ax = fig.add_axes((0.1, 0.1, 0.9, 0.85))
    im = ax.pcolormesh(
        energies/tas.UNIT_CHARGE, # GeV
        altitudes, # m
        cherenkov_yield/tas.UNIT_CHARGE, # eV/m
        norm=LogNorm())
    ax.loglog()
    ax.set_xlabel(r"E / eV")
    ax.set_ylabel(r"z (above sea level)/ m")
    ax.set_title(
        "Wavelength: {:.0f}nm - {:.0f}nm".format(
            wavelength_start*1e9,
            wavelength_end*1e9))
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    cbar = plt.colorbar(im,format=LogFormatterMathtext())
    cbar.ax.set_ylabel(r'dE/dz / eV m$^{-1}$')
    fig.savefig(path)
