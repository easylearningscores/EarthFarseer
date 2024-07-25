import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class NS2DDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.from_numpy(np.load(data_path))
        self.data.unsqueeze_(2)  
        self.transform = transform
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = self.data[idx][:10]
        output_frames = self.data[idx][10:]
        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root, num_workers):
    train_dataset = NS2DDataset(data_path=data_root + 'ns_V1e-4_train.npy', transform=None)
    test_dataset = NS2DDataset(data_path=data_root + 'ns_V1e-4_test.npy', transform=None)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std

