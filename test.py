from args import getArgs
from models.Vgg16 import Vgg16
from utils.log import Log
from utils.dataloader import get_data
from utils.init_model import init_model
import torch
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from utils.loss import get_loss
import torch.nn as nn

def main(model_name):
    # init Log class
    args = getArgs(model_name)
    log = Log(args)

    # 1. load data
    train_loader, test_loader = get_data(args, log)

    # 2. load model, optimizer, criterion
    torch.manual_seed(args.rand_seed)  # fix rand_seed
    model = Vgg16(args.checkpoint).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # criterion = get_loss(args.loss_name)
    criterion = nn.CrossEntropyLoss()


    # 3. train, valid, test
    max_test_acc = 0
    max_acc_epoch = 0
    for epoch in range(args.epochs):
        start_time = datetime.now()
        train_loss = test_loss = 0.0

        # 3.1 train
        num_correct = 0
        num_examples = 0
        for batch in train_loader:
            model.train()
            inputs, targets = batch
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            # calculate train acc
            # outputs = result[0] if isinstance(result, tuple) else result
            correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        train_loss /= len(train_loader.dataset)
        train_acc = num_correct / num_examples

        # 3.2 test
        num_correct = 0
        num_examples = 0
        for batch in test_loader:
            model.eval()
            inputs, targets = batch
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data.item()
            # calculate test acc
            correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        test_loss /= len(test_loader.dataset)
        test_acc = num_correct / num_examples

        end_time = datetime.now()
        cost_time = (end_time - start_time).seconds
        log.print(
            'Epoch: {}, train acc: {:.4f}, train loss: {:.4f}, test acc: {:.4f}, test loss: {:.4f}, cost: {:d} m {:d} s'.format(
                epoch, train_acc, train_loss, test_acc, test_loss, cost_time // 60, cost_time % 60))

        # save model
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_acc_epoch = epoch
        #     torch.save(model, args.save)
    log.print('=====================max_test_pcc: {:.4f}, at epoch: {}\n\n'.format(max_test_acc, max_acc_epoch))
    log.save()

if __name__ == '__main__':
    model_name = 'Vgg16'        # ['CLCNN', 'Vgg16']
    main(model_name)