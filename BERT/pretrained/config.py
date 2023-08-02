from transformers import BertConfig

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 2
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
NUM_EPOCHS = 10
DROPOUT_RATE = 0.2

# Dataset
DATA_DIR = "../dataset/IMDB.csv"
NUM_WORKERS = 20

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

# Model
PRETRAIN_MODEL = "bert-base-uncased"
MODEL_CONFIG = BertConfig.from_pretrained(PRETRAIN_MODEL)
