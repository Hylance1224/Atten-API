import os
import re
import torch
import model_atten
import utility.dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utility.parse_args import arg_parse
args = arg_parse()


model = model_atten.WideAndDeep()
model = model.to(args.device)
criterion = torch.nn.BCELoss()
params = model.named_parameters()
optimizer = torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)

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
            outputs = model(datas)
            labels = labels.to(args.device).float()
            outputs = outputs.view(-1).float()
            loss_deep = criterion(outputs, labels)

            optimizer.zero_grad()
            loss_deep.backward()
            optimizer.step()

            loss_list.append(loss_deep.item())

            bar.set_description("epoch:{} idx:{} loss:{:.3f}".format(i, idx, np.mean(loss_list)))
            if not (idx % 100):
                ensure_dir('./' + path)
                torch.save(model.state_dict(), './' + path + '/model_' + fold + '.pth')
                torch.save(optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')
        torch.save(optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')
        torch.save(model.state_dict(), './' + path + '/model_' + fold + '.pth')


if __name__ == '__main__':
    if args.continue_training:
        model.load_state_dict(torch.load('./' + path + '/model_' + fold + '.pth'))
        optimizer.load_state_dict(torch.load('./' + path + '/deep_optimizer_' + fold + '.pth'))
        print('load model')

    train()
    plt.figure(figsize=(50, 8))
    plt.plot(range(len(loss_list)), loss_list)
    plt.show()
