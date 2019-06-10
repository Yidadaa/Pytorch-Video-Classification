import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
import pandas
import json
from sklearn.metrics import accuracy_score

from model import CNNEncoder, RNNDecoder
from dataloader import Dataset
import config

def train_on_epocchs(train_loader:torch.utils.data.DataLoader, test_loader:torch.utils.data.DataLoader):
    # 配置训练时环境
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # 实例化计算图模型
    cnn_encoder = CNNEncoder().to(device)
    rnn_decoder = RNNDecoder().to(device)
    
    # 多GPU训练
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print('使用{}个GPU训练'.format(device_count))
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

    # 提取网络参数，准备进行训练
    model_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

    # 设定优化器
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)

    # 训练时数据
    info = {
        'train_losses': [],
        'train_scores': [],
        'test_losses': [],
        'test_scores': []
    }

    model = [cnn_encoder, rnn_decoder]

    # 开始训练
    for ep in range(config.epoches):
        train_losses, train_scores = train(model, train_loader, optimizer, ep, device)
        test_loss, test_score = validation(model, test_loader, optimizer, ep, device)

        # 保存信息
        info['train_losses'].append(train_losses)
        info['train_scores'].append(train_scores)
        info['test_losses'].append(test_loss)
        info['test_scores'].append(test_score)

    with open('./train_info.json', 'w') as f:
        json.dump(info, f)

    print('训练结束')

def load_data_list(file_path):
    return pandas.read_csv(file_path).to_numpy()

def train(model:[nn.Module], dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, epoch, device):
    [cnn_encoder, rnn_decoder] = model
    cnn_encoder.train()
    rnn_decoder.train()

    train_losses = []
    train_scores = []
    for i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # 初始化优化器参数
        optimizer.zero_grad()
        # 执行前向传播
        y_ = rnn_decoder(cnn_encoder(X))
        # 计算loss
        loss = F.cross_entropy(y_, y)
        # 反向传播梯度
        loss.backward()
        optimizer.step()

        # 保存loss等信息
        train_losses.append(loss)
        # TODO: 计算准确率

        if (i + 1) % config.log_interval == 0:
            print('Traing')

    return train_losses, train_scores

def validation(model:[nn.Module], test_loader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, ep:int, device:int):
    [cnn_encoder, rnn_decoder] = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    y_gd = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_ = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(y_, y, reduction='sum')
            test_loss += loss.item()

            y_ = y.max(1, keepdim=True)[1]

            y_gd.append(y)
            y_pred.append(y_)

    test_loss /= len(test_loader)

    # 计算正确率
    y_gd = torch.stack(y_gd, dim=0)
    y_pred = torch.stack(y_pred, dim=0)

    test_acc = accuracy_score(
        y_gd.cpu().data.squeeze.numpy(),
        y_pred.cpu().data.squeeze.numpy()
    )

    print('\nTest set %d samples, avg loss: %0.4f, acc: %0.2f\n' % (len(y_gd), test_loss, test_acc * 100))

    return test_loss, test_acc

if __name__ == "__main__":
    train_data = pandas.read_csv('./data/train.csv')
    test_data = pandas.read_csv('./data/test.csv')
    train_loader = DataLoader(Dataset(train_data.to_numpy()), **config.train_dataset_params)
    test_loader = DataLoader(Dataset(test_data.to_numpy()), **config.train_dataset_params)
    train_on_epocchs(train_loader, test_loader)