from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

class weatherDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with h5py.File(data_path, 'r') as f:
            self.data_uv_g = f['uv_g'][:]  
            self.data_uv_g = torch.from_numpy(self.data_uv_g).to(torch.float32)
            self.data_uv_g = self.data_uv_g.permute(0, 3, 1, 2).unsqueeze_(2) 
            
            self.data_uv_k = f['uv_k'][:]  
            self.data_uv_k = torch.from_numpy(self.data_uv_k).to(torch.float32)
            self.data_uv_k = self.data_uv_k.permute(0, 3, 1, 2).unsqueeze_(2) 
            self.data_uv_gk = torch.cat([self.data_uv_g, self.data_uv_k], dim=2)
            self.transform = transform
            self.mean = 0
            self.std = 1
    
    def __len__(self):
        return len(self.data_uv_gk)

    def __getitem__(self, idx):
        input_frames = self.data_uv_gk[idx][:5]
        output_frames = self.data_uv_gk[idx][5:]
        input_frames = (input_frames - self.mean) / self.std
        output_frames = (output_frames - self.mean) / self.std
        return input_frames, output_frames
        

def load_data(batch_size, val_batch_size, data_root, num_workers):
    dataset = weatherDataset(data_path=data_root+'kg_all_20_mask_latmean.h5', transform=None)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)
    mean, std = 0, 1

    return dataloader_train, dataloader_validation, dataloader_test, mean, std
    


if __name__ == '__main__':
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=1, 
                                                                                    val_batch_size=1, 
                                                                                    data_root='/data/workspace/yancheng/MM/earthfarseer/data/',
                                                                                    num_workers=8)
    for input_frames, output_frames in iter(dataloader_train):
        print(input_frames.shape, output_frames.shape)