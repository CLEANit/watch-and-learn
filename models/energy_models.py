
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_utils.pytorch_datasets import EnergyDataset, ProbabilityDataset


class EnergyRNN(pl.LightningModule):

    def __init__(self, hparams):
        super(EnergyRNN, self).__init__()
        self.hparams = hparams
        if hparams.lstm:
            self.rnn = nn.LSTM(hparams.input_size, hparams.hidden_size,
                               num_layers=hparams.num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(hparams.input_size, hparams.hidden_size,
                              num_layers=hparams.num_layers, batch_first=True)
        self.linear = nn.Linear(hparams.hidden_size, hparams.output_size)
        self.train_datapath = hparams.train_datapath
        self.main_val_datapath = hparams.main_val_datapath
        self.val_2_datapath = hparams.val_2_datapath
        self.val_3_datapath = hparams.val_3_datapath
        self.test_datapath = hparams.test_datapath
        self.lr = hparams.lr
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.beta = ProbabilityDataset(filepath=self.train_datapath, data_key='beta').data

    def forward(self, x):
        x_inp = x
        x, _ = self.rnn(x_inp)
        x = self.linear(x)
        x = self.calculate_energy(x_inp, x)

        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits, y)
        self.logger.experiment.add_scalar('train_loss', loss, self.trainer.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x, y = batch
        logits = self(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        main_avg_loss = torch.stack([x['val_loss'] for x in outputs[0]]).mean()
        second_avg_loss = torch.stack([x['val_loss'] for x in outputs[1]]).mean()
        third_avg_loss = torch.stack([x['val_loss'] for x in outputs[2]]).mean()

        self.logger.experiment.add_scalar('Validation/main_val_loss',
                                          main_avg_loss, self.trainer.global_step)
        self.logger.experiment.add_scalar('Validation/second_val_loss',
                                          second_avg_loss, self.trainer.global_step)
        self.logger.experiment.add_scalar('Validation/third_val_loss',
                                          third_avg_loss, self.trainer.global_step)
        return {'avg_val_loss': main_avg_loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss_fn = nn.MSELoss()
        return {'test_loss': loss_fn(logits, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('test_loss', avg_loss, self.trainer.global_step)
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):

        train_dataset = EnergyDataset(filepath=self.train_datapath)

        return DataLoader(train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):

        main_dataset = EnergyDataset(filepath=self.main_val_datapath)
        main_dataloader = DataLoader(main_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers)

        second_dataset = EnergyDataset(filepath=self.val_2_datapath,)
        second_dataloader = DataLoader(second_dataset, batch_size=self.batch_size,
                                       num_workers=self.num_workers)

        third_dataset = EnergyDataset(filepath=self.val_3_datapath)
        third_dataloader = DataLoader(third_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers)

        return [main_dataloader, second_dataloader, third_dataloader]

    def test_dataloader(self):

        ising_dataset = EnergyDataset(filepath=self.test_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def calculate_probability(self, x_inp, x):

        down_state_mask = (x_inp == -1.)
        up_state_mask = (x_inp == 1.)

        true_probs = torch.zeros_like(x)
        true_probs += (x*up_state_mask)
        true_probs += (down_state_mask.int() - x*down_state_mask)
        probs = torch.prod(true_probs, axis=1)

        return torch.clamp(probs, 1e-40)

    def calculate_energy(self, x_inp, x):

        prob = self.calculate_probability(x_inp, x)
        H = (-1/torch.tensor(self.beta))*(torch.log(prob))

        return H


class EnergyAttentionRNN(EnergyRNN):

    def __init__(self, hparams):
        super(EnergyAttentionRNN, self).__init__(hparams)

        self.attention = nn.MultiheadAttention(hparams.hidden_size, hparams.n_heads)

    def forward(self, x):
        x_inp = x
        x, _ = self.rnn(x)
        x = x.permute(1, 0, 2)
        x, attn_weights = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.linear(x)
        x = self.calculate_energy(x_inp, x)

        return x, attn_weights

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits, _ = self(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits, y)
        self.logger.experiment.add_scalar('train_loss', loss, self.trainer.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x, y = batch
        logits, _ = self(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        logits, _ = self(x)
        loss_fn = nn.MSELoss()
        return {'test_loss': loss_fn(logits, y)}
