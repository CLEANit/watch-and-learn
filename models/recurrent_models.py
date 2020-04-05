
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_utils.pytorch_datasets import IsingDataset


class EnergyGRU(pl.LightningModule):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int,
                 train_datapath: str, val_datapath: str, test_datapath: str):
        super(EnergyGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.train_datapath = train_datapath
        self.val_datapath = val_datapath
        self.test_datapath = test_datapath

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.MSELoss()
        return {'val_loss': loss_fn(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.MSELoss()
        return {'test_loss': loss_fn(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

    def train_dataloader(self):
        params = {'batch_size': 64,
                  'shuffle': True,
                  'num_workers': 4}

        ising_dataset = IsingDataset(filepath=self.train_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, **params)

    def val_dataloader(self):

        ising_dataset = IsingDataset(filepath=self.val_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=1024)

    def test_dataloader(self):

        ising_dataset = IsingDataset(filepath=self.test_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=1024)
