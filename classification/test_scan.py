"""
python test.py --model pointMLP --msg 20220209053148-404
"""
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
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
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

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, default='cot_scan_PB T50 RS', help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='rct-net', help='model name [default: rct-net]')
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
    net.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    total = 0
    test_true = []  # 存储所有真实标签
    test_pred = []  # 存储所有预测标签
    time_cost = datetime.datetime.now()  # 记录开始时间

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (data, label) in enumerate(testloader):
            # 将数据和标签移动到指定设备（CPU/GPU）
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)  # 调整数据维度
            logits = net(data)  # 前向传播计算logits
            loss = criterion(logits, label)  # 计算损失
            test_loss += loss.item()  # 累加损失

            preds = logits.max(dim=1)[1]  # 获取预测标签
            test_true.append(label.cpu().numpy())  # 真实标签移动到CPU并存储
            test_pred.append(preds.detach().cpu().numpy())  # 预测标签移动到CPU并存储

            total += label.size(0)  # 累加总样本数
            correct += preds.eq(label).sum().item()  # 累加正确预测数

            # 打印进度条信息
            progress_bar(
                batch_idx, len(testloader),
                '损失: %.3f | 准确率: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1),
                    100. * correct / total,
                    correct, total
                )
            )

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())  # 计算耗时
    test_true = np.concatenate(test_true)  # 将所有标签拼接为一个数组
    test_pred = np.concatenate(test_pred)  # 将所有预测拼接为一个数组

    # 生成分类报告，包括每个类别的准确率
    class_report = metrics.classification_report(
        test_true, test_pred, output_dict=True, zero_division=0
    )

    # 从分类报告中提取每个类别的准确率（即 recall）
    per_class_accuracy = {
        f'类别_{int(k)}': round(v['recall'] * 100, 3)
        for k, v in class_report.items() if k.isdigit()
    }

    return {
        "损失": float("%.3f" % (test_loss / (batch_idx + 1))),  # 平均损失
        "总体准确率": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),  # 总体准确率
        "平均准确率": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),  # 平均准确率
        "每类准确率": per_class_accuracy,  # 每个类别的准确率
        "耗时(秒)": time_cost  # 总耗时
    }



if __name__ == '__main__':
    main()
