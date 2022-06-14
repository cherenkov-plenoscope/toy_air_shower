import toy_air_shower as tas

wavelength_start = 250e-9  # m
wavelength_end = 700e-9  # m
primary_energy = 1e9 * tas.UNIT_CHARGE  # J


RUN_cherenkov_photons = []
RUN_particles = []
num_events = 10
for i in range(num_events):
    particles, cherenkov_photons = tas.simulate_gamma_ray_air_shower(
        random_seed=i,
        primary_energy=primary_energy,
        wavelength_start=wavelength_start,
        wavelength_end=wavelength_end,
    )

    RUN_particles.append(particles)
    RUN_cherenkov_photons.append(cherenkov_photons)

    num_electrons = 0
    num_gammas = 0
    for i in range(len(particles)):
        if particles[i]["type"] == "electron":
            num_electrons += 1
        elif particles[i]["type"] == "gamma":
            num_gammas += 1

    print(
        "Energy of gamma-ray: {:.1f}GeV".format(
            1e-9 * primary_energy / tas.UNIT_CHARGE
        )
    )
    print(
        "Num electrons: {:d}, num gammas: {:d}".format(
            num_electrons, num_gammas
        )
    )
    print(
        "Num Cherenkov-photons emitted: {:d}".format(
            cherenkov_photons.shape[0]
        )
    )
    print(
        "Primary interaction altitude: {:0.1f}km".format(
            particles[0]["end_altitude"] * 1e-3
        )
    )
