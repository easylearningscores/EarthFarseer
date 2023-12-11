import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# class TaxiBJDataset(torch.utils.data.Dataset):
#     def __init__(self, input_data, output_data):
#         self.input_data = input_data
#         self.output_data = output_data
#         self.mean = 0
#         self.std = 1
        
#     def __len__(self):
#         return len(self.input_data)
    
#     def __getitem__(self, idx):
#         input_sample = self.input_data[idx]
#         output_sample = self.output_data[idx]
#         input_sample = torch.from_numpy(input_sample).to(torch.float32)
#         output_sample = torch.from_numpy(output_sample).to(torch.float32)
#         return input_sample, output_sample

class TaxiBJDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.mean = np.mean(input_data)
        self.std = np.std(input_data)
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        
        # normalize the input and output samples
        #input_sample = (input_sample - self.mean) / self.std
        #output_sample = (output_sample - self.mean) / self.std
        
        input_sample = torch.from_numpy(input_sample).to(torch.float32)
        output_sample = torch.from_numpy(output_sample).to(torch.float32)
        return input_sample[:,:,::4,::4], output_sample
    

def load_data(batch_size, val_batch_size, data_root, num_workers):

    train_input_data = np.load(data_root + 'TaxiBJ/train_input_data_reshaped.npy')
    train_output_data = np.load(data_root + 'TaxiBJ/train_output_data_reshaped.npy')
    # train_input_data = train_input_data.reshape(3555, 12, 2, 128, 128)
    # train_output_data = train_output_data.reshape(3555, 12, 2, 128, 128)


    test_input_data = np.load(data_root + 'TaxiBJ/test_input_data_reshaped.npy')
    test_output_data = np.load(data_root + 'TaxiBJ/test_output_data_reshaped.npy')
    # test_input_data = test_input_data.reshape(445, 12, 2, 128, 128)
    # test_output_data = test_output_data.reshape(445, 12, 2, 128, 128)
    

    val_input_data = np.load(data_root+'TaxiBJ/val_input_data_reshaped.npy')
    val_output_data = np.load(data_root+'TaxiBJ/val_output_data_reshaped.npy')
    # val_input_data = val_input_data.reshape(444, 12, 2, 128, 128)
    # val_output_data = val_output_data.reshape(444, 12, 2, 128, 128)

    train_set = TaxiBJDataset(train_input_data, train_output_data)
    test_set = TaxiBJDataset(test_input_data, test_output_data)
    val_set = TaxiBJDataset(val_input_data, val_output_data)

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_validation = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1

    return dataloader_train, dataloader_validation, dataloader_test, mean, std

if __name__ == '__main__':
    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(batch_size=1, 
                                                                                    val_batch_size=1, 
                                                                                    data_root='/data/workspace/yancheng/MM/earthfarseer/data/',
                                                                                    num_workers=8)
    for input_frames, output_frames in iter(dataloader_train):
        print(input_frames.shape, output_frames.shape)