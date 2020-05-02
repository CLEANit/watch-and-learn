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
        flattened_sample = self.flatten_rows(sample)

        X = flattened_sample[:-1].float()
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

    def flatten_rows(self, arr: np.array) -> torch.Tensor:
        """Flattens 2D array using snake pattern through rows

        :param arr: 2D array
        :return: flattened array
        """
        snake_ = []
        k = 1
        for row in arr:
            snake_ += list(row[::k])
            k *= -1
        return torch.tensor(snake_).unsqueeze(-1)

    def flatten_cols(self, arr: np.array) -> torch.Tensor:
        """Flattens 2D array using snake pattern through columns

        :param arr: 2D array
        :return: flattened array
        """
        snake_ = []
        k = 1
        for i in range(arr.shape[1]):
            snake_ += list(arr[:, i][::k])
            k *= -1
        return torch.tensor(snake_).unsqueeze(-1)


class EnergyDataset(ProbabilityDataset):

    def __init__(self, filepath, grids_data_key='ising_grids', energy_data_key='true_energies'):
        super(EnergyDataset, self).__init__(filepath, grids_data_key)

        self.energy_data = torch.tensor(self.get_data(filepath, energy_data_key))

    def __getitem__(self, index):

        sample = self.data[index]
        flattened_sample = self.flatten_rows(sample)

        X = flattened_sample[:-1].float()
        y = self.energy_data[index].unsqueeze(-1).float()

        return X, y


class EnergyDataset2D(EnergyDataset):

    def __getitem__(self, index):

        sample = self.data[index]

        X_rows = self.flatten_rows(sample)[:-1].float()
        X_cols = self.flatten_cols(sample)[:-1].float()
        y = self.energy_data[index].unsqueeze(-1).float()

        return X_rows, X_cols, y


class ProbabilityDataset2D(ProbabilityDataset):

    def __getitem__(self, index):

        sample = self.data[index]

        flattened_rows = self.flatten_rows(sample)
        flattened_cols = self.flatten_cols(sample)

        X_rows = flattened_rows[:-1].float()
        X_cols = flattened_cols[:-1].float()

        y_rows = (flattened_rows[1:] == 1).float()
        y_cols = (flattened_cols[1:] == 1).float()
        y = torch.stack([y_rows, y_cols])

        return X_rows, X_cols, y
