import argparse

import pytorch_lightning as pl

from models.recurrent_models import EnergyGRU

parser = argparse.ArgumentParser(description='Set hyperparameters for model and training process')

parser.add_argument(
    '--input_size',
    type=int,
    help="number of dims for input",
    default=1
)

parser.add_argument(
    '--output_size',
    type=int,
    help="number of dims for output",
    default=1
)

parser.add_argument(
    '--hidden_size',
    type=int,
    help="hidden size for the RNN hidden state",
    default=512
)

parser.add_argument(
    '--num_layers',
    type=int,
    help="number of layers in the RNN",
    default=2
)

parser.add_argument(
    '--train_datapath',
    type=str,
    help="location of training data",
    default='./data/train_data.hdf5'
)

parser.add_argument(
    '--val_datapath',
    type=str,
    help="location of validation data",
    default='./data/val_data.hdf5'
)

parser.add_argument(
    '--test_datapath',
    type=str,
    help="location of test data",
    default='./data/test_data.hdf5'
)

parser.add_argument(
    '--lr',
    type=float,
    help="learning rate for Adam optimizer",
    default=0.0001
)

parser.add_argument(
    '--num_workers',
    type=int,
    help="number of workers to use in DataLoaders",
    default=4
)

parser.add_argument(
    '--batch_size',
    type=int,
    help="batch size for train, validation and test",
    default=1024
)

parser.add_argument(
    '--early_stopping',
    type=bool,
    help="use early stopping in training",
    default=True
)

parser.add_argument(
    '--gpu',
    choices=[0, 1],
    help="GPUs available for training",
    default=0
)

parser.add_argument(
    '--max_epochs',
    type=int,
    help="maximum number of epochs to run training",
    default=40
)

parser.add_argument(
    '--val_check_interval',
    type=float,
    help="fraction of epoch to perform validation check",
    default=0.25
)


if __name__ == "__main__":

    args = parser.parse_args()

    if args.gpu == 1:

        trainer = pl.Trainer(
            early_stop_callback=args.early_stopping,
            gpus=1,
            max_epochs=args.max_epochs,
            val_check_interval=args.val_check_interval
        )

    else:
        trainer = pl.Trainer(
            early_stop_callback=args.early_stopping,
            max_epochs=args.max_epochs,
            val_check_interval=args.val_check_interval
        )

    model = EnergyGRU(args)
    trainer.fit(model)

    trainer.test()
