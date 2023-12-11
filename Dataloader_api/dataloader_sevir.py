import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

class VilDataset(Dataset):
    def __init__(self, train=True, root='./data', transform=None):
        super().__init__()
        if train:
            npy = ['SEVIR_IR069_STORMEVENTS_2018_0101_0630.npy', 'SEVIR_IR069_STORMEVENTS_2018_0701_1231.npy']
        else:
            npy = ['SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.npy']

        data = []
        for file in npy:
            data.append(np.load(f'{root}/{file}'))

        self.data = np.concatenate(data)
        #N, L, H, W = self.data.shape
        # self.data = self.data.reshape([N  L, H, W])
        self.transform = transform
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index].reshape(20, 1, 128, 128)
        if self.transform:
            img = self.transform(img)

        input_img = img[:10]
        output_img = img[10:]
        input_img = img[:10]
        output_img = img[10:]
        input_img = torch.from_numpy(input_img)
        output_img = torch.from_numpy(output_img)
        input_img = input_img.contiguous().float()
        output_img = output_img.contiguous().float()
        return input_img, output_img


def load_data(batch_size, val_batch_size,
              data_root, num_workers):
    train_set = VilDataset(train=True, root='./data', transform=None)
    test_set = VilDataset(train=True, root='./data', transform=None)

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   num_workers=num_workers)
    dataloader_validation = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=num_workers)
    mean, std = 0, 1

    return dataloader_train, dataloader_validation, dataloader_test, mean, std


if __name__ == '__main__':
    dataset = VilDataset(root='/root/Model_Phy/data')
    input_img, output_img = dataset[1]
    # Assuming `input_img` is a NumPy array of shape (10, 64, 64, 1)
    fig, axes = plt.subplots(nrows=1, ncols=10)

    for i in range(10):
        axes[i].imshow(input_img[i, :, :, 0], cmap=None)
        axes[i].axis('off')

    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=10)

    for i in range(10):
        axes[i].imshow(output_img[i, :, :, 0], cmap=None)
        axes[i].axis('off')

    plt.show()