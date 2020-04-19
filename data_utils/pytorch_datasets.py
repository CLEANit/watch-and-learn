import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ProbabilityDataset(Dataset):
    """
        Class to load dataset from hdf5 file into PyTorch's Dataset class
    """

    def __init__(self, filepath, data_key):
        self.filepath = filepath
        self.data = self.get_data(filepath, data_key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        flattened_sample = self.flatten(sample)

        X = flattened_sample[:-1]
        y = (flattened_sample[1:] == 1).float()

        return X, y

    def get_data(self, filepath: str, data_key: str) -> np.array:
        """Loads hdf5 file and return numpy array

        :param filepath: path to hdf5 file
        :param data_key: key of the dataset inside the hdf5 file
        :raises KeyError: raised when the provided data_key is not found in the file
        :return: array with dataset
        """
        with h5py.File(filepath, 'r') as h5_file:
            try:
                data = h5_file[data_key][()]
            except KeyError:
                datasets = list(h5_file.keys())
                raise KeyError(f"Dataset not found. File only has the datasets: {datasets}")

        return data

    def flatten(self, arr: np.array) -> torch.Tensor:
        """Flattens 2D array using snake pattern

        :param arr: 2D array
        :return: flattened array
        """
        snake_ = []
        k = 1
        for i, row in enumerate(arr):
            snake_ += list(row[::k])
            k *= -1
        return torch.tensor(snake_).unsqueeze(-1)



class EnergyDataset(ProbabilityDataset):

    def __init__(self, filepath, grids_data_key, energy_data_key):
        super(EnergyDataset, self).__init__(filepath, grids_data_key)

        self.energy_data = self.get_data(filepath, energy_data_key)

    def __getitem__(self, index):
        
        sample = self.data[index]
        flattened_sample = self.flatten(sample)

        X = flattened_sample[:-1]
        y = self.energy_data[index]

        return X, y
