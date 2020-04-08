
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_utils.pytorch_datasets import IsingDataset


class EnergyGRU(pl.LightningModule):

    def __init__(self, hparams):
        super(EnergyGRU, self).__init__()
        self.gru = nn.GRU(hparams.input_size, hparams.hidden_size,
                          num_layers=hparams.num_layers, batch_first=True)
        self.linear = nn.Linear(hparams.hidden_size, hparams.output_size)
        self.train_datapath = hparams.train_datapath
        self.val_datapath = hparams.val_datapath
        self.test_datapath = hparams.test_datapath
        self.lr = hparams.lr
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
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
        loss_fn = nn.BCEWithLogitsLoss()
        return {'test_loss': loss_fn(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):

        ising_dataset = IsingDataset(filepath=self.train_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):

        ising_dataset = IsingDataset(filepath=self.val_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size)

    def test_dataloader(self):

        ising_dataset = IsingDataset(filepath=self.test_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size)
