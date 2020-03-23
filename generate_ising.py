import sys
import h5py
from tqdm import tqdm
import jax.numpy as np

from jax import random, jit
from jax.ops import index, index_update
from typing import Iterator, Tuple, List

N = 8
T = 4.0
kb = 1.0
beta = 1/(kb*T)
Q = 2
flatten_pattern = "SPIRAL"


def create_grid(n_x: int, n_y: int) -> np.array:
    """Create a grid randomly filled with -1 and +1

    :param n_x: grid's first dimension
    :type n_x: int
    :param n_y: grids's second dimension
    :type n_y: int
    :return: grid of size (n_x, n_y)
    :rtype: np.array
    """
    key = random.PRNGKey(11)
    return random.randint(key, (n_x, n_y), 0, 2)*2 - 1


@jit
def flip_spin(grid: np.array) -> np.array:
    """Flip the spin of a single element on the grid

    :param grid: grid with spins
    :type grid: np.array
    :return: grid with one flipped spin
    :rtype: np.array
    """
    n_x, n_y = grid.shape
    key = random.PRNGKey(11)
    x = random.randint(key, (1, ), 0, n_x)
    y = random.randint(key, (1, ), 0, n_y)
    mask = index_update(np.ones_like(grid), index[x, y], -1)
    flipped_grid = grid*mask
    return flipped_grid


def H2(arr):
    """Calculates a 2nd-nearest-neighbor hamiltonian for the ising model"""
    # shift one to the right elements
    x = np.roll(arr, 1, axis=1)
    # shift elements one down
    y = np.roll(arr, 1, axis=0)
    # multiply original with transformations and sum each arr
    w = np.roll(x, 1, axis=0)
    # w and z are diaganol up and diaganol down
    z = np.roll(x, -1, axis=0)
    x = np.sum(np.multiply(arr, x))
    y = np.sum(np.multiply(arr, y))
    z = np.sum(np.multiply(arr, z))
    w = np.sum(np.multiply(arr, w))
    return -float(x+y) - float(z+w)/2


def H(arr, model):
    """Calculates a hamiltonian for ising (1st or 2nd nearest) or potts models"""
    if model == "ISING2":
        return H2(arr)
    # shift one to the right elements
    x = np.roll(arr, 1, axis=1)
    # shift elements one down
    y = np.roll(arr, 1, axis=0)
    # multiply original with transformations and sum each arr

    if model == "ISING1":
        x = np.sum(np.multiply(arr, x))  # Ising
        y = np.sum(np.multiply(arr, y))

    return -float(x+y)


def metropolis(grid_curr: np.array, H_curr: float, C: float, model: str) -> Tuple[np.array, float]:
    """Metropolis update rule when flipping a single spin

    :param grid_curr: current grid
    :type grid_curr: np.array
    :param H_curr: current hamiltonian
    :type H_curr: float
    :param C: quantity to calculate transition probability
    :type C: float
    :param model: model being simulated
    :type model: str
    :return: updated grid and Hamiltonian using metropolis algorithm
    :rtype: Tuple[np.array, float]
    """

    grid_cand = flip_spin(grid_curr)
    H_cand = H(grid_cand, model)
    dH = H_cand - H_curr

    key = random.PRNGKey(11)
    alpha = random.uniform(key)
    if dH <= 0 or alpha < C**dH:
        H_curr = H_cand
        grid_curr = grid_cand

    return grid_curr, H_curr


def metropolis_chain(grid_curr: np.array, beta: float,
                     model: str, n_iter=0, burn_in=0) -> Iterator[np.array]:
    """Sample chain using Metropolis algorithm

    :param grid_curr: initial grid configuration
    :type grid_curr: np.array
    :param beta: inverse of Boltzmann's constant times the temperature
    :type beta: float
    :param model: model being simulated
    :type model: str
    :param n_iter: number of elements in chain, defaults to 0
    :type n_iter: int, optional
    :param burn_in: number of burn-in steps to help convergence, defaults to 0
    :type burn_in: int, optional
    :yield: chain of sampled states
    :rtype: Iterator[np.array]
    """

    C = np.exp(-beta)
    H_curr = H(grid_curr, model)

    for i in range(burn_in):
        grid_curr, H_curr = metropolis(grid_curr, H_curr, C, model)

    for i in range(n_iter):
        grid_curr, H_curr = metropolis(grid_curr, H_curr, C, model)
        yield grid_curr


def snake(arr: np.array) -> List[float]:
    """Flatten 2D array using a snake pattern

    :param arr: array to be flattened
    :type arr: np.array
    :return: flattened array
    :rtype: List[float]
    """
    snake_ = []
    n = 1
    for a in arr:
        snake_ += a[::n]
        n *= -1
    return snake_


def spiral(arr: np.array) -> List[float]:
    """Flatten 2D array using a spiral pattern

    :param arr: array to be flattened
    :type arr: np.array
    :return: flattened array
    :rtype: List[float]
    """

    i = j = len(arr)//2
    k = -1
    spiral_ = [arr[i][j]]
    for s in range(1, len(arr)):
        for sy in range(s):
            i += k
            spiral_ += [arr[i][j]]
        for sx in range(s):
            j += k
            spiral_ += [arr[i][j]]
        k *= -1
    for sx in range(len(arr)-1):
        i += k
        spiral_ += [arr[i][j]]
    return spiral_


def flatten(grid: np.array, flatten_pattern: str) -> List[float]:
    """Flatten 2D grid in a certain pattern

    :param grid: grid to be flattened
    :type grid: np.array
    :param flatten_pattern: used pattern when flattening grid
    :type flatten_patter: str
    :return: flattened grid
    :rtype: List[float]
    """

    if flatten_pattern == "SPIRAL":
        return spiral(grid)
    else:
        return snake(grid)


def h5gen(model="ISING1", div=32, runs=1, batch_size=1e5, filename='training_data.hdf5'):
    assert runs*(div//runs) == div
    div_multiplier = batch_size
    length = div_multiplier*div

    f = h5py.File(filename, "w")
    for i in tqdm(range(range(div))):
        if i % (div // runs) == 0:
            grid_init = jit(create_grid, static_argnums=(0, 1))(N, N)
            grids = metropolis_chain(grid_init, beta, model,
                                     n_iter=div_multiplier*div//runs, burn_in=110000)

        batch = [next(grids) for i in range(batch_size)]

        out = [[[n] for n in flatten(grid, flatten_pattern)] for grid in batch]

        in_ = np.asarray([arr[:-1] for arr in out], dtype=np.int8)
        out = np.asarray([arr[1:] for arr in out], dtype=np.int8)
        if i == 0:
            f.create_dataset("Inputs", data=in_, chunks=True, maxshape=(length, N**2-1, 1))
            f.create_dataset("Labels", data=out, chunks=True, maxshape=(length, N**2-1, 1))
            f.close()
            f = h5py.File(filename, "a")
        else:
            f["Inputs"].resize((f["Inputs"].shape[0] + length // div), axis=0)
            f["Inputs"][-length // div:] = in_
            f["Labels"].resize((f["Labels"].shape[0] + length // div), axis=0)
            f["Labels"][-length // div:] = out
        print((i+1)/div*100, end="% ")
        sys.stdout.flush()
    f.close()
    print('Generation of MC data is complete')
    f.close()
