# Sebastian A. Mueller
# A simple simulation of electro-magnetic-air-showers (one-dimensional)
# Goal: Simulate the Cherenkov-photon-emission.
import numpy as np

"""
Adopting thoughts from:
    J. Matthews
    "A Heitler model of extensive air showers"
    Astroparticle Physics 22 (2005) 387–397
"""

UNIT_CHARGE = 1.602e-19 # C

SPEED_OF_LIGHT = 299792458 # m/s

PLANCK_ACTION = 6.626e-34 # Js

ELECTRON_MASS = 9.10938356e-31 # kg

ELECTRON_REST_ENERGY = ELECTRON_MASS*SPEED_OF_LIGHT**2.

CRITICAL_ENERGY = 86e6*UNIT_CHARGE # J
# At this energy, ionization takes over Bremsstrahlung
# http://pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html

PERMABILITY_AIR = 4.*np.pi*1e-7 * 1.00000037# H/m
# B. D. Cullity and C. D. Graham (2008),
# Introduction to Magnetic Materials, 2nd edition, 568 pp., p.16

RADIATION_LENGTH_AIR_E = 36.62 # g/cm^2
# Depth to be traversed before Energy is reduced to 1/e.
# http://pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html

RADIATION_LENGTH_AIR_G = 9/7*RADIATION_LENGTH_AIR_E # g/cm^2

# dE/dx = E*e^{-x/X0}
# 1/2 E = E*e^{-x/X0}
# ln(1/2) = -x/X0
# -ln(1/2)*X0 = x

HALF_DEPTH_AIR_E = RADIATION_LENGTH_AIR_E/np.log(2.) # g/cm^2
HALF_DEPTH_AIR_G = RADIATION_LENGTH_AIR_G/np.log(2.) # g/cm^2
# Depth to be traversed before Energy is reduced to 1/2.

INTERACTION_RATE_PER_UNIT_DEPTH_ELECTRON = 1./HALF_DEPTH_AIR_E # cm^2/g
INTERACTION_RATE_PER_UNIT_DEPTH_GAMMA = 1./HALF_DEPTH_AIR_G # cm^2/g

DEPTH_SEA_LEVEL_AIR = 1013.25 # g/cm^2

DENSITY_LENGTH_AIR = 8435. # m
# Rise in altitude by which atmospheric density is reduced by 1/e.

REFRACTION_AIR_0CELSIUS_1ATM = 1.00027357 # 1


def dE_over_dz(q, beta, n, mu, wavelength_start, wavelength_end):
    # Frank-Tamm Formula
    # ------------------
    #   q: particle electric charge
    #   beta: particle velocity in units of speed of light (v/c)
    #   n: refractive index of traversed medium
    #   mu: electromagnetic permability of traversed medium
    #   wavelength_start: cherenkov wavelength range
    #   wavelength_end
    # Return
    # ------
    #   Energy balance (loss) of particle per unit length traveled in medium
    omega_start = 2.*np.pi*SPEED_OF_LIGHT/wavelength_start
    omega_end = 2.*np.pi*SPEED_OF_LIGHT/wavelength_end
    return (
        q**2./(4.*np.pi)*mu*(1. - 1./(beta**2.*n**2.))*
        .5*(omega_end**2. - omega_start**2.)
    )


def refraction_in_air(altitude):
    # refractive index of atmosphere vs. altitude
    return 1. + (REFRACTION_AIR_0CELSIUS_1ATM - 1.)*np.exp(
        -altitude/DENSITY_LENGTH_AIR)


def altitude_to_depth(altitude):
    return DEPTH_SEA_LEVEL_AIR*np.exp(-altitude/DENSITY_LENGTH_AIR)


def depth_to_altitude(depth):
    return -np.log(depth/DEPTH_SEA_LEVEL_AIR)*DENSITY_LENGTH_AIR


def expovariate(rate):
    return -np.log(np.random.uniform())/rate


def lorentz_factor(kinetic_energy, rest_energy):
    return 1. + kinetic_energy/rest_energy


def natural_velocity(kinetic_energy, rest_energy):
    gamma = lorentz_factor(
        kinetic_energy=kinetic_energy,
        rest_energy=rest_energy)
    return np.sqrt(1. - 1./gamma**2)


def make_cherenkov_spectrum(wvl_range):
    spectrum = 1./wvl_range**2
    spectrum /= np.sum(spectrum)
    return spectrum


def gamma_ray_first_interaction(primary_energy):
    first_interaction_depth = expovariate(INTERACTION_RATE_PER_UNIT_DEPTH_GAMMA)
    first_interaction_altitude = depth_to_altitude(first_interaction_depth)

    particles = [
        {
            "type": "gamma",
            "start_energy": primary_energy,
            "energy": 0.,
            "start_altitude": 9e99,
            "end_altitude": first_interaction_altitude,
            "mother": -1
        },
        {
            "type": "electron",
            "start_energy": primary_energy/2.,
            "energy": primary_energy/2.,
            "start_altitude": first_interaction_altitude,
            "end_altitude": None,
            "mother": 0,
        },
        {
            "type": "electron",
            "start_energy": primary_energy/2.,
            "energy": primary_energy/2.,
            "start_altitude": first_interaction_altitude,
            "end_altitude": None,
            "mother": 0,
        }
    ]
    todo = [1, 2]
    return particles, todo


def draw_wavelength(
    cherenkov_spectrum_cdf,
    wavelength_range
):
    # CDF is cummulative_distribution_function
    wavelength = np.interp(
        x=np.random.uniform(),
        xp=cherenkov_spectrum_cdf,
        fp=wavelength_range)
    return wavelength


def heitler_correction_spliting(num_bremsstrahlungen_per_half_depth=5):
    N = num_bremsstrahlungen_per_half_depth
    n = (1./2.)**(1./N)*N
    energy_fraction_electron = n/N
    energy_fraction_gamma_ray = (1 - n/N)
    return energy_fraction_electron, energy_fraction_gamma_ray


IDX_MOTHER = 0
IDX_ALTITUDE = 1
IDX_CHERENKOV_ANGLE = 2
IDX_WAVELENGTH = 3


def simulate_gamma_ray_air_shower(
    random_seed,
    primary_energy,
    wavelength_start,
    wavelength_end,
    bremsstrahlung_correction_factor=1
):
    np.random.seed(random_seed)

    particles, todo = gamma_ray_first_interaction(primary_energy=primary_energy)

    wavelength_range = np.linspace(wavelength_start, wavelength_end, 1024)
    cherenkov_spectrum = make_cherenkov_spectrum(wavelength_range)
    cherenkov_spectrum_cdf = np.cumsum(cherenkov_spectrum)

    cherenkov_photons_altitude = []
    cherenkov_photons_wvl = []
    cherenkov_photons_theta = []
    cherenkov_photons_mother = []

    assert(np.modf(bremsstrahlung_correction_factor)[0] == 0)
    energy_fraction_electron, energy_fraction_gamma_ray = heitler_correction_spliting(
        bremsstrahlung_correction_factor)

    while True:
        if len(todo) == 0:
            break

        if particles[todo[0]]['type'] == 'gamma':
            depth_until_next_interaction = expovariate(
                INTERACTION_RATE_PER_UNIT_DEPTH_GAMMA)
            current_depth = altitude_to_depth(particles[todo[0]]['start_altitude'])
            next_interaction_depth = current_depth + depth_until_next_interaction
            next_interaction_altitude = depth_to_altitude(next_interaction_depth)
            assert(next_interaction_altitude < particles[todo[0]]['start_altitude'])

            if particles[todo[0]]['energy'] < CRITICAL_ENERGY:
                particles[todo[0]]['end_altitude'] = next_interaction_altitude
                del todo[0]
            else:
                # pair-production
                next_gen_energy = .5*particles[todo[0]]["energy"]
                # add two new electrons
                particles.append(
                    {
                        "type": "electron",
                        "start_energy": next_gen_energy,
                        "energy": next_gen_energy,
                        "start_altitude": next_interaction_altitude,
                        "mother": todo[0],
                    }
                )
                todo.append(len(particles) - 1)
                particles.append(
                    {
                        "type": "electron",
                        "start_energy": next_gen_energy,
                        "energy": next_gen_energy,
                        "start_altitude": next_interaction_altitude,
                        "mother": todo[0],
                    }
                )
                todo.append(len(particles) - 1)
                # destroy gamma-ray
                particles[todo[0]]['end_altitude'] = next_interaction_altitude
                particles[todo[0]]['energy'] = 0.
                del todo[0]

        elif particles[todo[0]]['type'] == 'electron':
            depth_until_next_interaction = (
                expovariate(INTERACTION_RATE_PER_UNIT_DEPTH_ELECTRON)/
                bremsstrahlung_correction_factor)
            current_depth = altitude_to_depth(particles[todo[0]]['start_altitude'])
            next_interaction_depth = current_depth + depth_until_next_interaction
            next_interaction_altitude = depth_to_altitude(next_interaction_depth)
            assert(next_interaction_altitude < particles[todo[0]]['start_altitude'])

            # Cherenkov emission
            # ------------------
            current_altitude = particles[todo[0]]['start_altitude']
            em = False
            while True:
                current_beta = natural_velocity(
                    kinetic_energy=particles[todo[0]]['energy'],
                    rest_energy=ELECTRON_REST_ENERGY)

                current_n = refraction_in_air(current_altitude)

                if current_beta > 1./current_n:
                    energy_loss_per_unit_length = - dE_over_dz(
                        q=UNIT_CHARGE,
                        beta=current_beta,
                        n=current_n,
                        mu=PERMABILITY_AIR,
                        wavelength_start=wavelength_start,
                        wavelength_end=wavelength_end)

                    wavelength = draw_wavelength(
                        cherenkov_spectrum_cdf=cherenkov_spectrum_cdf,
                        wavelength_range=wavelength_range)

                    energy_of_cherenkov_photon = (
                        PLANCK_ACTION*SPEED_OF_LIGHT/wavelength)

                    emmission_rate_per_unit_length = (
                        energy_loss_per_unit_length/energy_of_cherenkov_photon)
                    distance_until_next_emission = expovariate(
                        emmission_rate_per_unit_length)

                    assert(distance_until_next_emission >= 0)

                    if distance_until_next_emission > 1e2:
                        distance_until_next_emission = 1e2

                    current_altitude -= distance_until_next_emission
                    particles[todo[0]]['energy'] -= energy_of_cherenkov_photon
                    cherenkov_photons_altitude.append(current_altitude)
                    cherenkov_photons_wvl.append(wavelength)
                    cherenkov_photons_theta.append(
                        np.arccos(1.0/(current_n*current_beta)))
                    cherenkov_photons_mother.append(todo[0])
                else:
                    current_altitude -= 1e1

                if current_altitude < next_interaction_altitude:
                    break

                if particles[todo[0]]['energy'] < CRITICAL_ENERGY:
                    break

            if particles[todo[0]]['energy'] < CRITICAL_ENERGY:
                particles[todo[0]]['end_altitude'] = next_interaction_altitude
                del todo[0]
            else:
                # bremsstrahlung
                # create one new electron and one new gamma-ray
                particles.append(
                    {
                        "type": "electron",
                        "start_energy": energy_fraction_electron*particles[todo[0]]["energy"],
                        "energy": energy_fraction_electron*particles[todo[0]]["energy"],
                        "start_altitude": next_interaction_altitude,
                        "mother": todo[0],
                    }
                )
                todo.append(len(particles) - 1)
                particles.append(
                    {
                        "type": "gamma",
                        "start_energy": energy_fraction_gamma_ray*particles[todo[0]]["energy"],
                        "energy": energy_fraction_gamma_ray*particles[todo[0]]["energy"],
                        "start_altitude": next_interaction_altitude,
                        "mother": todo[0],
                    }
                )
                todo.append(len(particles) - 1)

                # destroy old electron
                particles[todo[0]]['end_altitude'] = next_interaction_altitude
                particles[todo[0]]['energy'] = 0.
                del todo[0]

    cherenkov_photons = np.zeros(shape=(len(cherenkov_photons_mother), 4))
    cherenkov_photons[:, IDX_MOTHER] = cherenkov_photons_mother
    cherenkov_photons[:, IDX_ALTITUDE] = cherenkov_photons_altitude
    cherenkov_photons[:, IDX_CHERENKOV_ANGLE] = cherenkov_photons_theta
    cherenkov_photons[:, IDX_WAVELENGTH] = cherenkov_photons_wvl

    return particles, cherenkov_photons
