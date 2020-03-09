import os
import tensorly as tl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from ptflops import get_model_complexity_info
import time


# 定义全局变量
batchSize = 5
nEpochs = 10
numPrint = 100
# 定义Summary_Writer

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集 (训练集和测试集)
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True,
                                                 num_workers=0, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False,
                                                num_workers=0, pin_memory=True)
# 定义神经网络

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(

                    nn.Linear(512, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5, inplace=False),
                    nn.Linear(in_features=4096, out_features=4096, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5, inplace=False),
                    nn.Linear(4096, 10),
        )


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')
# 使用测试数据测试网络
def Accuracy(net):
    start = time.time()
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / total))
    end = time.time()
    print("Execution Time: ", end - start)
    return 100.0 * correct / total

# 训练函数
def train():
    best_acc=10
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降
    iter = 0
    num = 1
    # 训练网络
    for epoch in range(nEpochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            iter = iter + 1
            # 取数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
            # 将梯度置零
            optimizer.zero_grad()
            # 训练
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()   # 反向传播

            optimizer.step()  # 优化
            # 统计数据
            running_loss += loss.item()
            if i % numPrint == 99:    # 每 batchsize * numPrint 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss / (batchSize*numPrint)))
                running_loss = 0.0
                # writer.add_scalar('accuracy', Accuracy(), num + 1)
                num = num + 1
        acc = Accuracy()
        if acc > best_acc:
            torch.save(net, './vgg16_model.pkl')
            print('model saved')
            best_acc = acc

        else:
            print('model is not better now')
        print('best_acc now is ',best_acc)

if __name__ == '__main__':

    tl.set_backend('pytorch')
    model_path = './vgg16_model.pkl'
    if os.path.exists(model_path):
        net = torch.load(model_path, map_location='cpu')
        print('model exists,model loaded!\n')
    else:
        print('model path not exist,creat new one\n')
        net = VGG16()
    print(net)
    flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=True)

    print('{:<30}  {:<10}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<10}'.format('Number of parameters: ', params))
    # train()

    acc = Accuracy(net)

    # train()
    # writer.close()
    # print('Training Finished')

