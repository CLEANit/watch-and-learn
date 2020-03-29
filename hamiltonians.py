import jax.numpy as np
from jax import jit


@jit
def H_ising_2(grid: np.array) -> np.float32:
    """Calculates Hamiltonian for an Ising model with second-order neigbors

    :param grid: grid with spins
    :type grid: np.array
    :return: value of Hamiltonian
    :rtype: np.float32
    """
    x = np.roll(grid, 1, axis=1)
    y = np.roll(grid, 1, axis=0)

    # Diagonal shifts
    w = np.roll(x, 1, axis=0)
    z = np.roll(x, -1, axis=0)

    x = np.sum(np.multiply(grid, x))
    y = np.sum(np.multiply(grid, y))
    z = np.sum(np.multiply(grid, z))
    w = np.sum(np.multiply(grid, w))
    return -(x+y).astype(np.float32) - (z+w).astype(np.float32)/2


@jit
def H_ising_1(grid: np.array) -> np.float32:
    """Calculates Hamiltonian for an Ising model with first-order neighbors

    :param grid: grid with spins
    :type grid: np.array
    :return: value of Hamiltonian
    :rtype: np.float32
    """
    x = np.roll(grid, 1, axis=1)
    y = np.roll(grid, 1, axis=0)
    x = np.sum(np.multiply(grid, x))  # Ising
    y = np.sum(np.multiply(grid, y))
    return -(x+y).astype(np.float32)
