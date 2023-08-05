import math
import torch
from config import device
import torch.nn as nn
from config import config
from utils import same_seed
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def training(train_loader, valid_loader, model):
    same_seed(config['seed'])
    model = model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])
    # 切换模型为训练模式
    model.train()
    n_epochs = config['n_epochs']
    best_loss = math.inf
    for epoch in range(n_epochs):
        train_loss = []
        early_stop_count = 0
        train_pbar = tqdm(train_loader, leave=True, position=0)
        for input, target in train_pbar:
            optimizer.zero_grad()
            target = torch.tensor(target).type(torch.LongTensor)
            if device == 'cuda':
                input = input.cuda()
                target = target.cuda()
            c0, c1, c2, c3,c4 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            loss /= 5
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        train_loss_mean = sum(train_loss) / len(train_loss)
        # 切换模型为预测模型
        model.eval()
        val_loss = []
        # 不记录模型梯度信息
        with torch.no_grad():
            for input, target in valid_loader:
                target = torch.tensor(target).type(torch.LongTensor)
                if device == 'cuda':
                    input = input.cuda()
                    target = target.cuda()

                c0, c1, c2, c3, c4 = model(input)
                loss = criterion(c0, target[:, 0]) + \
                       criterion(c1, target[:, 1]) + \
                       criterion(c2, target[:, 2]) + \
                       criterion(c3, target[:, 3]) + \
                       criterion(c4, target[:, 4])
                loss /= 5
                val_loss.append(loss.item())
        valid_loss_mean = sum(val_loss)/len(val_loss)
        print(f'[{epoch+1}/{n_epochs}]:train_loss:{round(train_loss_mean,3)},valid_loss:{round(valid_loss_mean,3)}')
        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            torch.save(model.state_dict(), config['save_path'])
            print('saving model with loss {}'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count == config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
    return best_loss