import os
import sys
import argparse
import torch
import lightning as L
from model import BERTClassifier
from dataset import IDBMDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision("medium")  # to make lightning happy

parser = argparse.ArgumentParser()
parser.add_argument("--INPUT_SIZE", help=" : Input dimension size", default=784)
parser.add_argument("--NUM_CLASSES", help=" : # of class", default=2)
parser.add_argument("--LEARNING_RATE", help=" : Learning rate", default=5e-5)
parser.add_argument("--BATCH_SIZE", type=int, help=" : Batch size", default=16)
parser.add_argument("--NUM_EPOCHS", type=int, help=" : # of Epoch", default=10)
parser.add_argument("--DROPOUT_RATE", type=float, help=" : Dropout rate", default=0.2)
parser.add_argument(
    "--DATA_DIR", help=" : Path of .csv file", default="../dataset/IMDB.csv"
)
parser.add_argument("--NUM_WORKERS", help=" : # of dataloader workers", default=20)
parser.add_argument("--ACCELERATOR", help=" : gpu accelerator", default="gpu")
parser.add_argument("--DEVICES", help=" : Device number", default=[0])
parser.add_argument("--PRECISION", help=" : Precision setting", default=16)
parser.add_argument(
    "--PRETRAIN_MODEL",
    help=" : str, Pretrained model name",
    default="bert-base-uncased",
)
parser.add_argument(
    "--DEFAULT",
    action="store_true",
    help=" : Default(True), --DEFAULT is config from config file",
)
parser.add_argument(
    "--TEST",
    action="store_true",
    help=" : Default(True), If you want to see the test result, enter --TEST",
)
args = parser.parse_args()

if __name__ == "__main__":
    argv = sys.argv

    if args.DEFAULT:
        logger = TensorBoardLogger("tb_logs", name="imdb_bert_v0")
        model = BERTClassifier(
            config=config.MODEL_CONFIG,
            pretrain_name=config.PRETRAIN_MODEL,
            num_classes=config.NUM_CLASSES,
            learning_rate=config.LEARNING_RATE,
        )
        dm = IDBMDataModule(
            pretrain_model_name=config.PRETRAIN_MODEL,
            data_dir=config.DATA_DIR,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )
        trainer = L.Trainer(
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=1,
            max_epochs=config.NUM_EPOCHS,
            precision=config.PRECISION,
            callbacks=[MyPrintingCallback(), EarlyStopping(monitor="train_f1_score")],
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
    else:
        logger = TensorBoardLogger("tb_logs", name="imdb_bert_v0")

        model = BERTClassifier(
            config=args.MODEL_CONFIG,
            pretrain_name=args.PRETRAIN_MODEL,
            num_classes=args.NUM_CLASSES,
            learning_rate=args.LEARNING_RATE,
        )
        model = torch.compile(model)

        dm = IDBMDataModule(
            pretrain_model_name=args.PRETRAIN_MODEL,
            data_dir=args.DATA_DIR,
            batch_size=args.BATCH_SIZE,
            num_workers=args.NUM_WORKERS,
        )

        trainer = L.Trainer(
            logger=logger,
            accelerator=args.ACCELERATOR,
            devices=args.DEVICES,
            min_epochs=1,
            max_epochs=args.NUM_EPOCHS,
            precision=args.PRECISION,
            callbacks=[MyPrintingCallback(), EarlyStopping(monitor="train_f1_score")],
        )
        trainer.fit(model, dm)

        if args.TEST:
            trainer.test(model, dm)
