import os
import pytorch_lightning as pl
from models.probability_models import ProbabilityAttentionRNN2D
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    '--num_workers',
    default=8,
)

parser.add_argument(
    '--train_datapath',
    default='./data/train_data.hdf5'
)

parser.add_argument(
    '--main_val_datapath',
    default='./data/val_data_beta_dot25.hdf5'
)

parser.add_argument(
   'val_2_datapath',
   default='./data/val_data_beta_dot125.hdf5'
)

parser.add_argument(
   '--val_3_datapath',
   default='./data/val_data_beta_dot5.hdf5'
)

parser.add_argument(
    'test_datapath',
    default='./data/test_data.hdf5'
)

parser.add_argument(
    '--lstm',
    default=False
)

parser.add_argument(
    '--bidirectional',
    default=False
)

parser.add_argument(
    '--n_heads',
    default=1
)

parser.add_argument(
    '--lr',
    default=0.0001
)

parser.add_argument(
    '--batch_size',
    default=512
)

parser.add_argument(
    '--input_size',
    default=1
)

parser.add_argument(
    '--output_size',
    default=1
)

parser.add_argument(
    '--num_layers',
    default=1
)

parser.add_argument(
    '--hidden_size',
    default=512
)


hparams = parser.parse_args()

checkpoint_callback = pl.callbacks.ModelCheckpoint(
          filepath=os.path.join(
              os.getcwd(),
              'checkpoints/probability/rnn_attn/2D_GRU_1L_1H/{epoch}-{avg_val_loss:.3f}'),
          monitor='avg_val_loss',
          mode='min',
          save_top_k=1,
          period=1
      )

early_stopping = pl.callbacks.EarlyStopping(
    monitor='avg_val_loss',
    patience=3
)

logger = pl.loggers.TensorBoardLogger(
    "probability_logs",
    name="rnn_attn",
    version="2D_GRU_1L_1H"
)

trainer = pl.Trainer(
    gpus=1,
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    check_val_every_n_epoch=0.5,
    max_epochs=20,
    val_percent_check=0.01
)

model = ProbabilityAttentionRNN2D(hparams)

trainer.fit(model)
