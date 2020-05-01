import sys
import argparse
import h5py
from tqdm import tqdm
import jax.numpy as np
import numpy as onp

from jax import random, jit, vmap
from jax.ops import index, index_update
from typing import Tuple, Callable

import hamiltonians

parser = argparse.ArgumentParser(description="Generate MC samples for Ising model")

parser.add_argument(
    "--n_x",
    type=int,
    help="Size of first axis of the grid",
    default=8
)

parser.add_argument(
    "--n_y",
    type=int,
    help="Size of second axis of the grid",
    default=8
)

parser.add_argument(
    "--beta",
    type=float,
    help="inverse of temperature times Boltzmann constant",
    default=0.25
)

parser.add_argument(
    "--n_beta_samples",
    type=int,
    help="number of beta values to generate data",
    default=None
)

parser.add_argument(
    "--beta_min",
    type=float,
    help="minimum beta value to sample",
    default=None
)

parser.add_argument(
    "--beta_max",
    type=float,
    help="maximum beta value to sample",
    default=None
)

parser.add_argument(
    "--random_seed",
    type=int,
    help="Seed for random processes",
    default=11
)

parser.add_argument(
    "--model",
    choices={"ISING1", "ISING2"},
    help="Simulate first or second order Ising model",
    default="ISING1"
)


parser.add_argument(
    "--n_samples",
    type=int,
    help="Number of MC samples to generate",
    default=3200000
)

parser.add_argument(
    "--burn_in",
    type=int,
    help="Number of samples to discard in MC process",
    default=110000
)

parser.add_argument(
    "--filename",
    type=str,
    help="Path and filename to write generated samples",
    default='training_data.hdf5'
)


def create_grid(n_x: int, n_y: int, random_seed: int) -> np.DeviceArray:
    """Create a grid randomly filled with -1 and +1

    :param n_x: grid's first dimension
    :param n_y: grids's second dimension
    :param random_seed: seed for random functions
    :return: grid of size (n_x, n_y)
    """
    key = random.PRNGKey(random_seed)
    return random.randint(key, (n_x, n_y), 0, 2)*2 - 1


def flip_spin(grid: np.DeviceArray, n_x: int, n_y: int, x_subkey: np.DeviceArray,
              y_subkey: np.DeviceArray) -> np.DeviceArray:
    """Flip the spin of a single element on the grid

    :param grid: grid with spins
    :param n_x: grid's first dimension
    :param n_y: grids's second dimension
    :param xflip_subkey: subkey for random x coordinate in flip
    :param yflip_subkey: subkey for random y coordinate in flip
    :return: grid with one flipped spin
    """

    x = random.randint(x_subkey, (1, ), 0, n_x)
    y = random.randint(y_subkey, (1, ), 0, n_y)
    mask = index_update(np.ones_like(grid), index[x, y], -1)
    flipped_grid = grid*mask
    return flipped_grid


def metropolis(grid_curr: np.DeviceArray, H_curr: float, H: Callable[[np.DeviceArray], np.float32],
               C: float, xflip_subkey: np.DeviceArray, yflip_subkey: np.DeviceArray,
               metropolis_subkey: np.DeviceArray) -> Tuple[np.DeviceArray, float]:
    """Metropolis update rule when flipping a single spin

    :param grid_curr: current grid
    :param H_curr: current hamiltonian
    :param H: function to calculate Hamiltonian
    :param C: helper to calculate transition probability
    :param xflip_subkey: subkey for random x coordinate in flip
    :param yflip_subkey: subkey for random y coordinate in flip
    :param metropolis_subkey: subkey for random alpha in metropolis update
    :return: updated grid and Hamiltonian using Metropolis update rule
    """

    n_x, n_y = grid_curr.shape

    grid_cand = jit(flip_spin, static_argnums=(1, 2))(grid_curr, n_x, n_y,
                                                      xflip_subkey, yflip_subkey)
    H_cand = H(grid_cand)
    dH = H_cand - H_curr

    alpha = random.uniform(metropolis_subkey)
    if dH <= 0 or alpha < C**dH:
        H_curr = H_cand
        grid_curr = grid_cand

    return grid_curr, H_curr


def metropolis_chain(grid_curr: np.DeviceArray, beta: float,
                     H: Callable[[np.DeviceArray], np.float32],
                     n_iter: int, burn_in: int, random_seed: int) -> np.DeviceArray:
    """Sample chain using Metropolis algorithm

    :param grid_curr: initial grid configuration
    :param beta: inverse of Boltzmann's constant times the temperature
    :param H: function to calculate Hamiltonian
    :param n_iter: number of elements in chain
    :param burn_in: number of samples to discard in the begining of MC process
    :param random_seed: seed for random functions
    :return: chain of sampled states
    """

    C = np.exp(-beta)
    H_curr = H(grid_curr)
    n_x, n_y = grid_curr.shape
    grids = onp.zeros((n_iter, n_x, n_y))

    key = random.PRNGKey(random_seed)

    for i in tqdm(range(burn_in)):
        key, *i_subkeys = random.split(key, num=4)
        grid_curr, H_curr = metropolis(grid_curr, H_curr, H, C, *i_subkeys)

    for i in tqdm(range(n_iter)):
        key, *i_subkeys = random.split(key, num=4)
        grid_curr, H_curr = metropolis(grid_curr, H_curr, H, C, *i_subkeys)
        grids[i] = np.asarray(grid_curr, dtype=onp.int8)

    return grids


def single_beta_file(beta: float, n_x: int, n_y: int, model: str,
                     n_samples: int, burn_in: int, filename: str, random_seed: int) -> None:
    """Generate hdf5 file with generated samples

    :param beta: inverse of Boltzmann's constant times the temperature
    :param n_x: grid's first dimension size
    :param n_y: grid's second dimension size
    :param model: type of model to simulate
    :param n_samples: number of samples to generate
    :param burn_in: number of samples to discard in the begining of MC process
    :param filename: path and filename to write generated samples
    """

    if model == "ISING1":
        H = hamiltonians.H_ising_1
    elif model == "ISING2":
        H = hamiltonians.H_ising_2

    grid_init = jit(create_grid, static_argnums=(0, 1))(n_x, n_y, random_seed)
    grids = metropolis_chain(grid_init, beta, H, n_iter=n_samples,
                             burn_in=burn_in, random_seed=random_seed)

    print("Calculating grid energies")
    energies = vmap(H)(grids)
    avg_E = np.average(energies)
    print('Generation of MC data is complete')

    with h5py.File(filename, "w") as f:
        f.create_dataset("ising_grids", data=grids, chunks=True)
        f.create_dataset("true_energies", data=energies, chunks=True)
        f.create_dataset("beta", data=np.array([beta]))
        f.create_dataset("avg_E", data=np.array([avg_E]))
    sys.stdout.flush()


def multiple_betas_file(beta_min: float, beta_max: float, n_beta_samples: int,
                        n_x: int, n_y: int, model: str, n_samples: int,
                        burn_in: int, filename: str, random_seed: int) -> None:
    """Generate hdf5 file for chains with multiple beta values

    :param beta_min: minimum inverse of Boltzmann's constant times the temperature
    :param beta_max: maximum inverse of Boltzmann's constant times the temperature
    :param n_x: grid's first dimension size
    :param n_y: grid's second dimension size
    :param model: type of model to simulate
    :param n_samples: number of samples to generate
    :param burn_in: number of samples to discard in the begining of MC process
    :param filename: path and filename to write generated samples
    """

    if model == "ISING1":
        H = hamiltonians.H_ising_1
    elif model == "ISING2":
        H = hamiltonians.H_ising_2

    beta_values = np.linspace(beta_min, beta_max, n_beta_samples)
    grids_list = []
    energies_list = []
    betas_list = []

    for beta in tqdm(beta_values):

        grid_init = jit(create_grid, static_argnums=(0, 1))(n_x, n_y, random_seed)
        grids = metropolis_chain(grid_init, beta, H, n_iter=n_samples,
                                 burn_in=burn_in, random_seed=random_seed)

        grids_list.append(grids)

        energies = vmap(H)(grids)
        energies_list.append(energies)
        betas_list.append(np.array([beta]*len(energies)))

    total_grids = np.concatenate(grids_list)
    total_energies = np.concatenate(energies_list)
    total_betas = np.concatenate(betas_list)

    with h5py.File(filename, "w") as f:
        f.create_dataset("ising_grids", data=total_grids, chunks=True)
        f.create_dataset("true_energies", data=total_energies, chunks=True)
        f.create_dataset("beta", data=total_betas)
    sys.stdout.flush()


if __name__ == "__main__":

    args = parser.parse_args()

    if args.n_beta_samples is not None:
        multiple_betas_file(
            beta_min=args.beta_min, beta_max=args.beta_max, n_beta_samples=args.n_beta_samples,
            n_x=args.n_x, n_y=args.n_y, model=args.model, n_samples=args.n_samples,
            burn_in=args.burn_in, filename=args.filename, random_seed=args.random_seed
        )

    else:
        single_beta_file(
            beta=args.beta, n_x=args.n_x, n_y=args.n_y, model=args.model,
            n_samples=args.n_samples, burn_in=args.burn_in,
            filename=args.filename, random_seed=args.random_seed
        )
