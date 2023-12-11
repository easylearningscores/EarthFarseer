import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results_super_res', type=str)
    parser.add_argument('--ex_name', default='results_super_res', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/data/workspace/yancheng/MM/earthfarseer/data/')
    parser.add_argument('--dataname', default='taxibj12-12', choices=['mmnist', 'taxibj', 'caltech', 'sevir', 'ns2d', 'taxibj12-12', 'weather'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[12, 2, 32, 32], type=int,nargs='*') #  [10, 2, 128, 128] for weather  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=128, type=int)
    parser.add_argument('--N_S', default=2, type=int)
    parser.add_argument('--N_T', default=4, type=int)
    parser.add_argument('--groups', default=2, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)