

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
from typing import List, Tuple, Optional
import os

class StateTransitionDataset(Dataset):
    def __init__(self, data: List[Tuple], transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, sp = self.data[idx][:3]
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        sp = torch.tensor(sp, dtype=torch.float32)
        if self.transform:
            s, a, sp = self.transform((s, a, sp))
        return {"state": s, "action": a, "next_state": sp}

class TransitionDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        fpath = os.path.abspath(__file__)
        dpath = os.path.abspath(os.path.join(fpath, "../../.."))
        with open(os.path.join(dpath, self.data_path), "rb") as f:
            data = pickle.load(f)
        split = int(0.9 * len(data))
        self.train_data = StateTransitionDataset(data[:split])
        self.val_data = StateTransitionDataset(data[split:])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)