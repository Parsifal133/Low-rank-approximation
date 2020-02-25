import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
import time


# 定义全局变量

modelPath = './decomposed_model.pkl'
# modelPath = './model.pkl'
batchSize = 5
nEpochs = 10
numPrint = 100
# 定义Summary_Writer
writer = SummaryWriter('./Result')   # 数据存放在这个文件夹
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




# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=1),  # 修改了这个地方，不知道为什么就对了
#             # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2), )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 1 * 1, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 10), )
#
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 1 * 1)
#         x = self.classifier(x)
#         # return F.log_softmax(inputs, dim=3)
#         return x
# 定义Alexnet网路结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 修改了这个地方，不知道为什么就对了
            # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10), )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        # return F.log_softmax(inputs, dim=3)
        return x

# 使用测试数据测试网络
def Accuracy(net):
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
            writer.add_scalar('loss', loss.item(), iter)
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
            torch.save(net, './model.pkl')
            print('model saved')
            best_acc = acc

        else:
            print('model is not better now')
        print('best_acc now is ',best_acc)

if __name__ == '__main__':
    # 如果模型存在，加载模型
    if os.path.exists(modelPath):
        print('model exits')
        net = torch.load(modelPath, map_location='cpu')
        print('model loaded')
    else:

        print('model not exits')
        net=AlexNet()
        print(net)
    # print('Training Started')
    start = time.time()
    acc = Accuracy(net)
    end = time.time()
    print("Execution Time: ", end - start)
    # train()
    # writer.close()
    # print('Training Finished')

