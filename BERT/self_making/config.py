from transformers import BertConfig

# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 1000
DROPOUT_RATE = 0.2

# Dataset
DATA_DIR = "./dataset/IMDB.csv"
NUM_WORKERS = 0

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

# Model
PRETRAIN_MODEL = "bert-base-uncased"
MODEL_CONFIG = BertConfig.from_pretrained(PRETRAIN_MODEL)
