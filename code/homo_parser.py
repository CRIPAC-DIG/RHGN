import argparse
import torch
import fitlog
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default='homo_G')
    parser.add_argument('--data_dir', type=str, default='../data/sample/')
    parser.add_argument('--label', type=str, default='gender', choices=['age', 'gender'])
    parser.add_argument('--model', type=str, default='HGCN')
    parser.add_argument('--gpu', type=int, default=3, choices=[0, 1, 2, 3, 4, 5, 6, 7])

    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_hid', type=int, default=32)
    parser.add_argument('--n_inp', type=int, default=200)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--max_lr', type=float, default=1e-2)

    # fitlog.set_log_dir("logs/")  # 设定日志存储的目录
    args = parser.parse_args()
    # fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
    return args
