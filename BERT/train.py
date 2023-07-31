import os
import torch
import lightning as L
from model import BERTClassifier
from dataset import IDBMDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision("medium")  # to make lightning happy

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="imdb_bert_v0")
    model = BERTClassifier(
        config=config.MODEL_CONFIG,
        dropout_rate=config.DROPOUT_RATE,
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
