from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

class TranslationDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ModelTrainer:
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        return TranslationDataset(data)

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_total_limit=2,
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model()

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        trainer = Trainer(
            model=self.model,
            eval_dataset=dataset,
        )

        return trainer.evaluate() 