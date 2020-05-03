
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_utils.pytorch_datasets import ProbabilityDataset, ProbabilityDataset2D


class ProbabilityRNN(pl.LightningModule):

    def __init__(self, hparams):
        super(ProbabilityRNN, self).__init__()
        self.hparams = hparams
        if hparams.lstm:
            self.rnn = nn.LSTM(hparams.input_size, hparams.hidden_size,
                               num_layers=hparams.num_layers,
                               bidirectional=hparams.bidirectional,
                               batch_first=True)
        else:
            self.rnn = nn.GRU(hparams.input_size, hparams.hidden_size,
                              num_layers=hparams.num_layers,
                              bidirectional=hparams.bidirectional,
                              batch_first=True)
        if hparams.bidirectional:
            self.linear = nn.Linear(2*hparams.hidden_size, hparams.output_size)
        else:
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
        self.avg_E = ProbabilityDataset(filepath=self.train_datapath, data_key='avg_E').data

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        self.logger.experiment.add_scalar('train_loss', loss, self.trainer.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x, y = batch
        logits = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
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
        loss_fn = nn.BCEWithLogitsLoss()
        return {'test_loss': loss_fn(logits, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('test_loss', avg_loss, self.trainer.global_step)
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):

        ising_dataset = ProbabilityDataset(filepath=self.train_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):

        main_dataset = ProbabilityDataset(filepath=self.main_val_datapath, data_key='ising_grids')
        main_dataloader = DataLoader(main_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers)

        second_dataset = ProbabilityDataset(filepath=self.val_2_datapath, data_key='ising_grids')
        second_dataloader = DataLoader(second_dataset, batch_size=self.batch_size,
                                       num_workers=self.num_workers)

        third_dataset = ProbabilityDataset(filepath=self.val_3_datapath, data_key='ising_grids')
        third_dataloader = DataLoader(third_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers)

        return [main_dataloader, second_dataloader, third_dataloader]

    def test_dataloader(self):

        ising_dataset = ProbabilityDataset(filepath=self.test_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def calculate_probability(self, x, y):

        down_state_mask = (x == -1.)
        up_state_mask = (x == 1.)

        true_probs = torch.zeros_like(y)
        true_probs += (y*up_state_mask)
        true_probs += (down_state_mask.int() - y*down_state_mask)
        probs = torch.prod(true_probs, axis=1)

        return torch.clamp(probs, 1e-40)

    def predict_energy(self, x):

        y_hat = torch.sigmoid(self(x))
        prob = self.calculate_probability(x, y_hat)
        H = (-1/torch.tensor(self.beta))*(torch.log(prob))

        return H


class ProbabilityAttentionRNN(ProbabilityRNN):

    def __init__(self, hparams):
        super(ProbabilityAttentionRNN, self).__init__(hparams)

        if hparams.bidirectional:
            self.attention = nn.MultiheadAttention(2*hparams.hidden_size, hparams.n_heads)

        else:
            self.attention = nn.MultiheadAttention(hparams.hidden_size, hparams.n_heads)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.permute(1, 0, 2)
        x, attn_weights = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.linear(x)

        return x, attn_weights

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits, _ = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        self.logger.experiment.add_scalar('train_loss', loss, self.trainer.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x, y = batch
        logits, _ = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):
        x, y = batch
        logits, _ = self(x)
        loss_fn = nn.BCEWithLogitsLoss()
        return {'test_loss': loss_fn(logits, y)}

    def predict_energy(self, x):

        logits, _ = self(x)
        y_hat = torch.sigmoid(logits)
        prob = self.calculate_probability(x, y_hat)
        H = (-1/torch.tensor(self.beta))*(torch.log(prob))

        return H


class ProbabilityRNN2D(ProbabilityRNN):

    def __init__(self, hparams):
        super(ProbabilityRNN2D, self).__init__(hparams)

        self.__dict__.pop('rnn', None)
        self.__dict__.pop('linear', None)

        if hparams.lstm:
            self.rnn_rows = nn.LSTM(hparams.input_size, hparams.hidden_size,
                                    num_layers=hparams.num_layers,
                                    bidirectional=hparams.bidirectional,
                                    batch_first=True)
            self.rnn_cols = nn.LSTM(hparams.input_size, hparams.hidden_size,
                                    num_layers=hparams.num_layers,
                                    bidirectional=hparams.bidirectional,
                                    batch_first=True)
        else:
            self.rnn_rows = nn.GRU(hparams.input_size, hparams.hidden_size,
                                   num_layers=hparams.num_layers,
                                   bidirectional=hparams.bidirectional,
                                   batch_first=True)
            self.rnn_cols = nn.GRU(hparams.input_size, hparams.hidden_size,
                                   num_layers=hparams.num_layers,
                                   bidirectional=hparams.bidirectional,
                                   batch_first=True)
        if hparams.bidirectional:
            self.linear_rows = nn.Linear(2*hparams.hidden_size, hparams.output_size)
            self.linear_cols = nn.Linear(2*hparams.hidden_size, hparams.output_size)
        else:
            self.linear_rows = nn.Linear(hparams.hidden_size, hparams.output_size)
            self.linear_cols = nn.Linear(hparams.hidden_size, hparams.output_size)

    def forward(self, x_rows_in, x_cols_in):

        x_rows, _ = self.rnn_rows(x_rows_in)
        x_cols, _ = self.rnn_cols(x_cols_in)

        x_rows = self.linear_rows(x_rows)
        x_cols = self.linear_cols(x_cols)

        x = torch.stack([x_rows, x_cols], dim=1)

        return x

    def training_step(self, batch, batch_nb):
        x_rows, x_cols, y = batch
        logits = self(x_rows, x_cols)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        self.logger.experiment.add_scalar('train_loss', loss, self.trainer.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x_rows, x_cols, y = batch
        logits = self(x_rows, x_cols)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):
        x_rows, x_cols, y = batch
        logits = self(x_rows, x_cols)
        loss_fn = nn.BCEWithLogitsLoss()
        return {'test_loss': loss_fn(logits, y)}

    def train_dataloader(self):

        train_dataset = ProbabilityDataset2D(filepath=self.train_datapath, data_key='ising_grids')

        return DataLoader(train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):

        main_dataset = ProbabilityDataset2D(filepath=self.main_val_datapath, data_key='ising_grids')
        main_dataloader = DataLoader(main_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers)

        second_dataset = ProbabilityDataset2D(filepath=self.val_2_datapath, data_key='ising_grids')
        second_dataloader = DataLoader(second_dataset, batch_size=self.batch_size,
                                       num_workers=self.num_workers)

        third_dataset = ProbabilityDataset2D(filepath=self.val_3_datapath, data_key='ising_grids')
        third_dataloader = DataLoader(third_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers)

        return [main_dataloader, second_dataloader, third_dataloader]

    def test_dataloader(self):

        ising_dataset = ProbabilityDataset2D(filepath=self.test_datapath, data_key='ising_grids')

        return DataLoader(ising_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_energy(self, x_rows, x_cols):

        logits = self(x_rows, x_cols)
        y_hat = torch.sigmoid(logits)
        prob_rows = self.calculate_probability(x_rows, y_hat[:, 0, :])
        prob_cols = self.calculate_probability(x_cols, y_hat[:, 1, :])
        H_rows = (-1/torch.tensor(self.beta))*(torch.log(prob_rows))
        H_cols = (-1/torch.tensor(self.beta))*(torch.log(prob_cols))

        return (H_rows + H_cols)/2


class ProbabilityAttentionRNN2D(ProbabilityRNN2D):

    def __init__(self, hparams):
        super(ProbabilityAttentionRNN2D, self).__init__(hparams)

        if hparams.bidirectional:
            self.attention_rows = nn.MultiheadAttention(2*hparams.hidden_size, hparams.n_heads)
            self.attention_cols = nn.MultiheadAttention(2*hparams.hidden_size, hparams.n_heads)
        else:
            self.attention_rows = nn.MultiheadAttention(hparams.hidden_size, hparams.n_heads)
            self.attention_cols = nn.MultiheadAttention(hparams.hidden_size, hparams.n_heads)

    def forward(self, x_rows_in, x_cols_in):

        x_rows, _ = self.rnn_rows(x_rows_in)
        x_cols, _ = self.rnn_cols(x_cols_in)
        x_rows = x_rows.permute(1, 0, 2)
        x_cols = x_cols.permute(1, 0, 2)

        x_rows, attn_rows = self.attention_rows(x_rows, x_rows, x_rows)
        x_cols, attn_cols = self.attention_cols(x_cols, x_cols, x_cols)

        x_rows = x_rows.permute(1, 0, 2)
        x_cols = x_cols.permute(1, 0, 2)

        x_rows = self.linear_rows(x_rows)
        x_cols = self.linear_cols(x_cols)

        x = torch.stack([x_rows, x_cols], dim=1)

        return x, (attn_rows, attn_cols)

    def training_step(self, batch, batch_nb):
        x_rows, x_cols, y = batch
        logits, _ = self(x_rows, x_cols)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        self.logger.experiment.add_scalar('train_loss', loss, self.trainer.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb, dataloader_nb):
        x_rows, x_cols, y = batch
        logits, _ = self(x_rows, x_cols)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):
        x_rows, x_cols, y = batch
        logits, _ = self(x_rows, x_cols)
        loss_fn = nn.BCEWithLogitsLoss()
        return {'test_loss': loss_fn(logits, y)}

    def predict_energy(self, x_rows, x_cols):

        logits, _ = self(x_rows, x_cols)
        y_hat = torch.sigmoid(logits)
        prob_rows = self.calculate_probability(x_rows, y_hat[:, 0, :])
        prob_cols = self.calculate_probability(x_cols, y_hat[:, 1, :])
        H_rows = (-1/torch.tensor(self.beta))*(torch.log(prob_rows))
        H_cols = (-1/torch.tensor(self.beta))*(torch.log(prob_cols))

        return (H_rows + H_cols)/2
