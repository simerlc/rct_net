import importlib
import logging

from sklearn.manifold import TSNE
import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import models as models
from ScanObjectNN import ScanObjectNN
from utils import progress_bar, IOStream
from data import ModelNet40
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


# 混淆矩阵函数
def confusion(y_label, y_pred):
    con_mat = confusion_matrix(y_label.astype(str), y_pred.astype(str))
    # print(con_mat)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '10', '11', '12', '13', '14']
    # classes.sort()
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Pred')
    plt.ylabel('True')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.show()



def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, default='cot_scan_bg2', help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointMLP', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=15, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    if args.msg is None:
        message = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    else:
        message = "-"+args.msg
    args.checkpoint = 'checkpoints/' + args.model + message

    print('==> Preparing data..')
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
    #                          batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Model
    print('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes)
    criterion = cal_loss
    net = net.to(device)

    print(net)

    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    test_out = validate(net, test_loader, criterion, device)
    print(f"Vanilla out: {test_out}")


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    confusion(test_true, test_pred)  # 绘制混淆矩阵
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()