import os.path

import torch
import gc
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from net import vgg16, vgg16_bn
from models.yolov1 import Yolov1
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset
import time

from visualize import Visualizer
import numpy as np

# 设置device，取决于GPU是否能用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练的image文件夹路径。通过索引文件找到对应train 和 test的image
file_root = '/Users/ingram14/Git_Project/Code4Fun/YOLO/Yolov1/data/dataset'

# 超参数的设置
learning_rate = 1e-5
num_epochs = 40
batch_size = 16
use_resnet = False
use_vgg = False
use_pretrained = False

if use_resnet:
    net = resnet50()
elif use_vgg:
    net = vgg16_bn()
else:
    net = Yolov1()  # 这个网络不太好收敛

print(net)

# 这里是导入预训练模型，从web url中导入
print('load pre-trined model')
if use_pretrained:
    if use_resnet:
        if os.path.exists("models/resnet50-0676ba61.pth"):
            new_state_dict = torch.load("models/resnet50-0676ba61.pth")
        else:
            resnet = models.resnet50(pretrained=True)  # 在这一步请求了web，在这里下载了模型
            new_state_dict = resnet.state_dict()
            dd = net.state_dict()
            for k in new_state_dict.keys():
                print(k)
                if k in dd.keys() and not k.startswith('fc'):
                    dd[k] = new_state_dict[k]
            net.load_state_dict(dd)
    else:
        vgg = models.vgg16_bn(pretrained=True)
        new_state_dict = vgg.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            print(k)
            if k in dd.keys() and k.startswith('features'):
                print('yes')
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
# else:
#     net.load_state_dict(torch.load('best.pth'))  # 导入之前训练好的最好的模型

# try:
#     print('cuda', torch.cuda.current_device(), torch.cuda.device_count())
# finally:
#     pass


# yolo损失函数的计算，四个参数分别是S（格子大小）、B（BBox数量）、每个BBox存的属性，0.5指的是背景的系数
criterion = yoloLoss(7, 2, 5, 0.5, device)

net.to(device)

net.train()

# different learning rate，不同学习率的做法，不是直接net.parameters()
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr': learning_rate}]


# optimizer
# optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)


## 通过Dataloader类划分batch进行迭代
# origin list_file = ['voc2012.txt', 'voc2007.txt']
train_dataset = yoloDataset(root=file_root, list_file=['voc2007.txt'], train=True,
                            transform=[transforms.ToTensor()])
# num_workers在GPU上会引起爆炸，大于等于2的时候
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = yoloDataset(root=file_root, list_file='voc2007test.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))


num_iter = 0
# vis = Visualizer(env='xiong')
best_test_loss = np.inf
t0 = time.time()

# logfile = open('log.txt', 'w')

# 开始训练迭代
with open('log.txt', 'w') as logfile:
    for epoch in range(num_epochs):
        net.train()
        if epoch == 20:
            learning_rate = 0.0001
        if epoch == 30:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        for i, (images, target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)

            images, target = images.to(device), target.to(device)

            pred = net(images)
            loss = criterion(pred, target)
            total_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            if (i + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f, time: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data.item(), total_loss / (i + 1), t1 - t0))
                num_iter += 1

            t0 = t1

        # validation
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                images = Variable(images)
                target = Variable(target)

                images, target = images.to(device), target.to(device)

                pred = net(images)
                loss = criterion(pred, target)
                validation_loss += loss.data.item()

        validation_loss /= len(test_loader)
        print('Epoch [%d/%d], Validation Loss: %.4f'
              % (epoch + 1, num_epochs, validation_loss))
        # vis.plot_train_val(loss_val=validation_loss)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), 'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()
        torch.save(net.state_dict(), 'yolo.pth')
