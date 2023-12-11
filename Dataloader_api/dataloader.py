from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_sevir import load_data as load_sevir
from .dataloader_ns2d import load_data as load_ns2d
from .dataloader_taxibj_12_12 import load_data as load_taxibj_12_12
from .dataloader_weather import load_data as load_weather

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'sevir':
        return load_sevir(batch_size, val_batch_size, data_root, num_workers)  
    elif dataname == 'ns2d':
        return load_ns2d(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'taxibj12-12':
        return load_taxibj_12_12(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'weather':
        return load_weather(batch_size, val_batch_size, data_root, num_workers)
