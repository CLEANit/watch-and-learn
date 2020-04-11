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
    "-T",
    "--temperature",
    type=float,
    help="Temperature of the system",
    default=4.0
)

parser.add_argument(
    "--kb",
    type=float,
    help="Value for Boltzmann's constant",
    default=1.0
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


def create_grid(n_x: int, n_y: int, random_seed: int) -> np.array:
    """Create a grid randomly filled with -1 and +1

    :param n_x: grid's first dimension
    :param n_y: grids's second dimension
    :param random_seed: seed for random functions
    :return: grid of size (n_x, n_y)
    """
    key = random.PRNGKey(random_seed)
    return random.randint(key, (n_x, n_y), 0, 2)*2 - 1


def flip_spin(grid: np.array, n_x: int, n_y: int, random_seed: int) -> np.array:
    """Flip the spin of a single element on the grid

    :param grid: grid with spins
    :param n_x: grid's first dimension
    :param n_y: grids's second dimension
    :param random_seed: seed for random functions
    :return: grid with one flipped spin
    """

    key = random.PRNGKey(random_seed)
    x = random.randint(key, (1, ), 0, n_x)
    y = random.randint(key, (1, ), 0, n_y)
    mask = index_update(np.ones_like(grid), index[x, y], -1)
    flipped_grid = grid*mask
    return flipped_grid


def metropolis(grid_curr: np.array, H_curr: float,
               H: Callable[[np.array], np.float32],
               C: float, random_seed: int) -> Tuple[np.array, float]:
    """Metropolis update rule when flipping a single spin

    :param grid_curr: current grid
    :param H_curr: current hamiltonian
    :param H: function to calculate Hamiltonian
    :param C: helper to calculate transition probability
    :param random_seed: seed for random functions
    :return: updated grid and Hamiltonian using Metropolis update rule
    """

    n_x, n_y = grid_curr.shape
    grid_cand = jit(flip_spin, static_argnums=(1, 2))(grid_curr, n_x, n_y, random_seed)
    H_cand = H(grid_cand)
    dH = H_cand - H_curr

    key = random.PRNGKey(11)
    alpha = random.uniform(key)
    if dH <= 0 or alpha < C**dH:
        H_curr = H_cand
        grid_curr = grid_cand

    return grid_curr, H_curr


def metropolis_chain(grid_curr: np.array, beta: float, H: Callable[[np.array], np.float32],
                     n_iter: int, burn_in: int, random_seed: int) -> np.array:
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

    print(f"\nGenerating {burn_in} burn-in samples")
    for i in tqdm(range(burn_in)):
        grid_curr, H_curr = metropolis(grid_curr, H_curr, H, C, random_seed)

    print(f"\nGenerating {n_iter} MC samples")
    for i in tqdm(range(n_iter)):
        grid_curr, H_curr = metropolis(grid_curr, H_curr, H, C, random_seed)
        grids[i] = np.asarray(grid_curr, dtype=onp.int8)

    return grids


def h5gen(beta: float, n_x: int, n_y: int, model: str,
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


if __name__ == "__main__":

    args = parser.parse_args()

    beta = 1/(args.temperature*args.kb)

    h5gen(beta=beta, n_x=args.n_x, n_y=args.n_y, model=args.model,
          n_samples=args.n_samples, burn_in=args.burn_in,
          filename=args.filename, random_seed=args.random_seed)
