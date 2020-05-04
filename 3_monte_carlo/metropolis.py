# Adapted by Aram Harrow from course material for [Computational Methods in Many-Body Physics](https://github.com/jhauschild/lecture_comp_methods) by Prof. Frank Pollmann, Prof. Michael Knap and Johannes Hauschild.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from numba import jit

@jit
def prepare_system(Lx, Ly):
    """Initialize the system."""
    system = np.zeros((Lx,Ly))
    for i in range(Lx):
        for j in range(Ly):
            system[i,j] = 1 - 2 * np.random.randint(2)
    return system

@jit(nopython=True)
def site_energy(system, i, j, Lx, Ly):
    """Energy function of spins connected to site (i, j)."""
    return -1. * system[i, j] * (system[np.mod(i - 1, Lx), j] + system[np.mod(i + 1, Lx), j] +
                                 system[i, np.mod(j - 1, Ly)] + system[i, np.mod(j + 1, Ly)])

@jit(nopython=True)
def total_energy(system):
    (Lx,Ly) = system.shape
    E = 0
    for i in range(Lx):
        for j in range(Ly):
            E += site_energy(system, i, j, Lx, Ly) / 2.
    return E

@jit(nopython=True)
def metropolis_loop(system, T, N_sweeps, N_eq, N_flips):
    """ Main loop doing the Metropolis algorithm."""
    E = total_energy(system)
    (Lx, Ly) = system.shape
    E_list = []
    for step in range(N_sweeps + N_eq):
        i = np.random.randint(0, Lx)
        j = np.random.randint(0, Ly)

        dE = -2. * site_energy(system, i, j, Lx, Ly)
        if dE <= 0.:
            system[i, j] *= -1
            E += dE
        elif np.exp(-1. / T * dE) > np.random.rand():
            system[i, j] *= -1
            E += dE

        if step >= N_eq and np.mod(step, N_flips) == 0:
            # measurement
            E_list.append(E)

    return np.array(E_list)


if __name__ == "__main__":
    """ Scan through some temperatures """
    # Set parameters here
    L = 20  # Linear system size
    N_sweeps = 10000  # Number of steps for the measurements
    N_eq = 5000  # Number of equilibration steps before the measurements start
    N_flips = 10  # Number of steps between measurements
    N_bins = 20  # Number of bins use for the error analysis

    T_range = np.arange(1.5, 3.1, 0.1)

    C_list = []
    system = prepare_system(L, L)
    for T in T_range:
        C_list_bin = []
        for k in range(N_bins):
            Es = metropolis_loop(system, T, N_sweeps, N_eq, N_flips)

            mean_E = np.mean(Es)
            mean_E2 = np.mean(Es**2)

            C_list_bin.append(1. / T**2. / L**2. * (mean_E2 - mean_E**2))
        C_list.append([np.mean(C_list_bin), np.std(C_list_bin) / np.sqrt(N_bins)])

        print(T, mean_E, C_list[-1])

    # Plot the results
    C_list = np.array(C_list)
    plt.errorbar(T_range, C_list[:, 0], C_list[:, 1])
    Tc = 2. / np.log(1. + np.sqrt(2))
    print(Tc)
    plt.axvline(Tc, color='r', linestyle='--')
    plt.xlabel('$T$')
    plt.ylabel('$c$')
    plt.show()
