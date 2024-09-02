import os
import re
import torch
import model_atten
import utility.dataset
import matplotlib.pyplot as plt
import numpy as np
import utility.config

from tqdm import tqdm

args = utility.dataset.args

model = model_atten.WideAndDeep()
model = model.to(utility.config.device)


criterion = torch.nn.BCELoss()

# 两个部分的优化方式不同
params = model.named_parameters()
deep_params = []

for name, param in params:
    deep_params.append(param)

deep_optimizer = torch.optim.Adagrad(deep_params, lr=args.lr, weight_decay=args.weight_decay)

fold: str = re.findall('[0-9]', args.dataset)[0]
path: str = 'model_atten'


def ensure_dir(ensure_path: str) -> None:
    if not os.path.exists(ensure_path):
        os.makedirs(ensure_path)


loss_list = []


def train() -> None:

    model.train()
    for i in range(args.epoch):
        data_loader = utility.dataset.get_dataloader()
        print('load data completed.')
        bar = tqdm(data_loader, total=len(data_loader), ascii=True, desc='train')
        for idx, (datas, labels) in enumerate(bar):
            # forward
            outputs = model(datas)
            labels = labels.to(utility.config.device).float()
            outputs = outputs.view(-1).float()
            # loss_wide = criterion(outputs, labels)

            # wide_optimizer.zero_grad()
            # loss_wide.backward(retain_graph=True)
            # wide_optimizer.step()

            # outputs = model(datas)
            # labels = labels.to(utility.config.device).float()
            # outputs = outputs.view(-1).float()
            loss_deep = criterion(outputs, labels)

            deep_optimizer.zero_grad()
            loss_deep.backward()
            deep_optimizer.step()

            loss_list.append(loss_deep.item())

            bar.set_description("epoch:{} idx:{} loss:{:.3f}".format(i, idx, np.mean(loss_list)))
            if not (idx % 100):
                ensure_dir('./' + path)
                torch.save(model.state_dict(), './' + path + '/model_' + fold + '.pth')
                # torch.save(wide_optimizer.state_dict(), './' + path + '/wide_optimizer_' + fold + '.pth')
                torch.save(deep_optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')
        # torch.save(wide_optimizer.state_dict(), './' + path + '/wide_optimizer_' + fold + '.pth')
        torch.save(deep_optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')
        torch.save(model.state_dict(), './' + path + '/model_' + fold + '.pth')


if __name__ == '__main__':
    # 如果继续训练就加载模型
    if args.continue_training:
        model.load_state_dict(torch.load('./' + path + '/model_' + fold + '.pth'))
        # wide_optimizer.load_state_dict(torch.load('./' + path + '/wide_optimizer_' + fold + '.pth'))
        deep_optimizer.load_state_dict(torch.load('./' + path + '/deep_optimizer_' + fold + '.pth'))
        print('load model')

    train()
    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
