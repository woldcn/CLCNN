import data_loader
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse

from log import Log
from loss import loss_func
from other_model import Xnn
from CLCNN import CLCNN
from data_loader import get_data
from train import train
from predict import predict
from torchvision import models
import datetime
from plt import plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=10)  # 8
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--lr', default=1e-05)  # 0.00001:acc-95.52,epoch-36
    parser.add_argument('--train_path', default='aug/train')
    parser.add_argument('--test_path', default='aug/test')

    # parser.add_argument('--is_other_model', default=False)
    parser.add_argument('--train_or_test', default='train')
    parser.add_argument('--model_name', default='CLCNN')
    parser.add_argument('--log_path', default='log/log.txt')

    # parser.add_argument('--pretrain_model', default=models.vgg16(pretrained=True))

    parser.add_argument('--useCL', default=True)
    parser.add_argument('--rand_seed', default=42)
    parser.add_argument('--T', default=0.8, help='contrastive loss hyperameter')

    args = parser.parse_args()

    # fix rand seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    # load data
    train_loader, test_loader, train_len, test_len = get_data(args.train_path, args.test_path, args.batch_size)

    # 其他对比模型
    # if args.is_other_model==True:
    # 	model = Xnn(args.pretrain_model).to(args.device)
    # # CLCNN
    # else:
    # 	model = CLCNN().to(args.device)
    model = CLCNN().to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # lr = 0.001   ?weight_decay = 0.01

    # print parameters
    # parameters = vars(args)
    # parameters.pop('pretrain_model')
    # parameters.pop('is_other_model')
    # parameters.pop('useCL')

    # init Log Class, for print and save log
    log = Log(args)

    # train
    if args.train_or_test == 'train':
        train(model, loss_func, optimizer, train_loader, test_loader, args.epochs, args.device, args.useCL, args.T, log)
    # predict
    else:
        predict(model, test_loader, args.device, args.saved_model_path)

    # plt
    # train_loader, plt_loader, train_len, test_len = get_data('aug/test', args.test_path, 10)
    # plt(model, train_loader, args.saved_model_path, 'cuda')
