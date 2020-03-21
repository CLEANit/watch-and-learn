import sys

import numpy as np
import h5py


MODEL = "ISING1"
FLATTEN = "SPIRAL"
N = 8
T = 4.0
KB = 1.0
Q = 3 if MODEL == "POTTS1" else 2


def domain(n, q):
    """Creates a random Ising or potts grid as specified,
    Potts grids are an integer representing which angle and need to be
    formatted for the rnn but not the hamiltonians"""
    if MODEL == "POTTS1":
        return [[np.random.randint(q) for a in range(n)]for b in range(n)]
    else:
        return [[np.random.randint(0, 2)*2-1 for a in range(n)]for b in range(n)]


def to_list(arr):
    """makes each entry in the array a Q wide array (all zeroes
    except the index of the original value)
    Ex: [[1,2]] becomes [[[0,1,0],[0,0,1]]]"""
    return [[rndarr(Q, i) for i in a] for a in arr]


def rndarr(Q, i):
    """Creates a zero array of size Q and sets index i to 1"""
    arr = [0]*Q
    arr[i] = 1
    return arr


def augment(arr):
    """Spin flip where a random spin in the grid is changed
    (Potts is still an integer representation)"""
    x, y = np.shape(arr)
    arr = [x.copy() for x in arr]
    x_ = np.random.randint(0, x)
    y_ = np.random.randint(0, y)
    if MODEL == "POTTS1":
        arr[x_][y_] += np.random.randint(1, Q)
        arr[x_][y_] %= Q
    else:
        arr[x_][y_] *= -1
    return arr


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


def H(arr):
    """Calculates a hamiltonian for ising (1st or 2nd nearest) or potts models"""
    if MODEL == "ISING2":
        return H2(arr)
    # shift one to the right elements
    x = np.roll(arr, 1, axis=1)
    # shift elements one down
    y = np.roll(arr, 1, axis=0)
    # multiply original with transformations and sum each arr

    if MODEL == "ISING1":
        x = np.sum(np.multiply(arr, x))  # Ising
        y = np.sum(np.multiply(arr, y))
    elif MODEL == "POTTS1":
        x = np.sum(np.cos(np.subtract(arr, x)*2*np.pi/Q))  # Potts
        y = np.sum(np.cos(np.subtract(arr, y)*2*np.pi/Q))
    return -float(x+y)


def metropolis(x, y, T, iterate=2500, start=0):
    """Runs a monte carlo simulation and yields every grid after 'start' iterations have occured"""
    C = np.e**(-1.0/(KB*T))
    arr = arr = domain(x, Q)
    Hi = H(arr)
    for i in range(iterate):
        arr2 = augment(arr)
        Hf = H(arr2)
        dH = Hf-Hi
        n = np.random.random()
        if dH <= 0 or n < C**dH:
            Hi = Hf
            arr = arr2
        if i >= start:
            if MODEL == "POTTS1":
                yield to_list(arr)
            else:
                yield arr


def snake(arr):
    """flattens a 2d array into a 1d in a snake pattern """
    snake_ = []
    n = 1
    for a in arr:
        snake_ += a[::n]
        n *= -1
    return snake_


def spiral(arr, growing=True):
    """flattens 2d array into a 1d via a spiral pattern"""
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


def flatten(arr):
    if FLATTEN == "SPIRAL":
        return spiral(arr)
    else:
        return snake(arr)


def h5gen(div=32, runs=1, filename='training_data.hdf5'):
    """generate samples for an NxN ising model at a specific T, throw away some configurations"""
    assert runs*(div//runs) == div
    div_multiplier = 100000
    length = div_multiplier*div
    f = h5py.File(filename, "w")
    for i in range(div):
        if i % (div//runs) == 0:
            print("||", end="")
            arrs = metropolis(N, N, T, iterate=div_multiplier*div//runs + 110000, start=110000)
        batch = [next(arrs) for x in range(length//div)]
        if MODEL == "POTTS1":
            out = [flatten(arr) for arr in batch]
        else:
            out = [[[n] for n in flatten(arr)] for arr in batch]
        in_ = np.asarray([arr[:-1] for arr in out], dtype=np.int8)
        out = np.asarray([arr[1:] for arr in out], dtype=np.int8)
        if i == 0:
            if MODEL == "POTTS1":
                print(in_.shape)
                f.create_dataset("Inputs", data=in_, chunks=True, maxshape=(length, N**2-1, Q))
                f.create_dataset("Labels", data=out, chunks=True, maxshape=(length, N**2-1, Q))
            else:
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


if __name__ == '__main__':
    DSIZE = 128
    RUNS = 1
    h5gen(div=DSIZE, runs=RUNS)
