import torch
from torch.utils.data import DataLoader, random_split, Dataset
import lightning as L
import pandas as pd
from transformers import BertTokenizer


class CustomIDBMDataset(Dataset):
    def __init__(self, data_file, pretrain_model_name):
        self.data = pd.read_csv(data_file)
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentiment = 1 if self.data.loc[idx, "sentiment"] == "positive" else 0
        return {
            "review": self.bert_tokenizer(
                self.data.loc[idx, "review"],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ),
            "sentiment": torch.tensor(sentiment),
        }


class IDBMDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, pretrain_model_name):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrain_model_name = pretrain_model_name

    def prepare_data(self):
        # single gpu
        # datasets.IMDB(root=self.data_dir)
        pass

    def setup(self, stage):
        # Load data from CSV file
        custom_dataset = CustomIDBMDataset(self.data_dir, self.pretrain_model_name)

        # Define sizes of train, valid, and test sets
        num_data = len(custom_dataset)
        num_train = int(num_data * 0.64)
        num_valid = int(num_data * 0.16)
        num_test = num_data - num_train - num_valid

        # Split the data into train, valid, and test sets
        train_data, valid_data, test_data = random_split(
            custom_dataset,
            [num_train, num_valid, num_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Create the train, valid, and test datasets
        self.train_ds = train_data
        self.valid_ds = valid_data
        self.test_ds = test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def valid_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )