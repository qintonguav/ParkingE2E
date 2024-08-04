import random

import numpy as np
import pytorch_lightning as pl
import torch

from dataset_interface.dataset_interface import get_parking_data
from utils.config import Configuration

from torch.utils.data import DataLoader


class ParkingDataloaderModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: str):
        ParkingDataModule = get_parking_data(data_mode="real_scene")
        
        self.train_loader = DataLoader(dataset=ParkingDataModule(config=self.cfg, is_train=1),
                                       batch_size=self.cfg.batch_size,
                                       shuffle=True,
                                       num_workers=self.cfg.num_workers,
                                       pin_memory=True,
                                       worker_init_fn=self.seed_worker,
                                       drop_last=True)
        self.val_loader = DataLoader(dataset=ParkingDataModule(config=self.cfg, is_train=0),
                                     batch_size=self.cfg.batch_size,
                                     shuffle=False,
                                     num_workers=self.cfg.num_workers,
                                     pin_memory=True,
                                     worker_init_fn=self.seed_worker,
                                     drop_last=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
