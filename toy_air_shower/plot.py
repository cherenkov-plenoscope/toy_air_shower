from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import toy_air_shower


def save_shower_figure(particles, cherenkov_photons, path):
    plt.Figure()
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
        plt.plot(
            np.array([start_x, end_x])*1e-3,
            np.array([start_y, end_y])*1e-3,
            color=color)
        # interaction marker
        if idx != 0:
            plt.plot(
                np.array([start_x])*1e-3,
                np.array([start_y])*1e-3,
                "ok")

        center_x = np.mean([start_x, end_x])
        center_y = np.mean([start_y, end_y])

        if particle['type'] == 'electron':
            mask = cherenkov_photons[:, IDX_MOTHER] == idx
            num_cherenkov_photons = np.sum(mask)
            text = "{:.0f}MeV, {:d}ch-ph".format(
                1e-6*particle['start_energy']/UNIT_CHARGE,
                num_cherenkov_photons)
        elif particle['type'] == 'gamma':
            text = "{:.0f}MeV".format(
                1e-6*particle['start_energy']/UNIT_CHARGE)

        plt.text(
            x=center_x*1e-3,
            y=center_y*1e-3,
            s=text)


    plt.ylabel("altitude / km")
    plt.xlabel("transverse / arb. unit")
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.savefig(path)


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
        np.log10(1e6*toy_air_shower.UNIT_CHARGE),
        np.log10(1e9*toy_air_shower.UNIT_CHARGE),
        128)

    cherenkov_yield = np.zeros(shape=(altitudes.shape[0], energies.shape[0]))

    for aidx, altitude in enumerate(altitudes):
        for eidx, energy in enumerate(energies):
            current_gamma_factor = energy/(
                toy_air_shower.ELECTRON_MASS *
                toy_air_shower.SPEED_OF_LIGHT**2)
            current_beta = np.sqrt(1. - 1./current_gamma_factor**2)
            current_n = toy_air_shower.refraction_in_air(altitude)
            if current_beta < 1./current_n:
                cherenkov_yield[aidx, eidx] = 0.
            else:
                cherenkov_yield[aidx, eidx] = - toy_air_shower.dE_over_dz(
                    q=toy_air_shower.UNIT_CHARGE,
                    beta=current_beta,
                    n=current_n,
                    mu=toy_air_shower.PERMABILITY_AIR,
                    wavelength_start=wavelength_start,
                    wavelength_end=wavelength_end)

    im = plt.pcolormesh(
        energies/toy_air_shower.UNIT_CHARGE, # GeV
        altitudes, # m
        cherenkov_yield/toy_air_shower.UNIT_CHARGE, # eV/m
        norm=LogNorm())
    plt.loglog()
    plt.xlabel(r"E / eV")
    plt.ylabel(r"z (above sea level)/ m")
    plt.title(
        "Frank-Tamm-formula\n"
        "Energy loss due to Cherenkov-emission for an electron\n"
        "wavelength: {:.0f}nm - {:.0f}nm".format(
            wavelength_start*1e-9,
            wavelength_end*1e-9))
    cbar = plt.colorbar(im,format=LogFormatterMathtext())
    cbar.ax.set_ylabel(r'dE/dz / eV m$^{-1}$')
    plt.savefig(path)
