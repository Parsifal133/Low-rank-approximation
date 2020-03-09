import matplotlib.pyplot as plt
import math
import random

import numpy as np

from model_train import Accuracy
from original_tucker import Tucker_decompose,Tucker_for_linear
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import time

def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')

def main():

    pop_size = 1  # 种群数量
    chromosome_length = 25# 染色体长度
    compress_rate = 0.1
    iter = 1
    pc = 1 # 杂交概率
    pm = 0.5 # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    pop = init_population(pop_size, chromosome_length)
    for i in range(iter):
        obj_value = calc_obj_value(pop, chromosome_length,compress_rate)
        print(obj_value)
        avg_fit = calc_avg_value(obj_value, pop_size)
        fit_value = calc_fit_value(obj_value)
        best_individual, best_fit = find_best(pop, fit_value)
        avg_gap = best_fit - avg_fit

        results.append([best_individual, best_fit])
        pop, obj_value=selection(pop, fit_value,obj_value)
        crossover(pop, avg_gap, pc, obj_value, best_fit, avg_fit)
        mutation(pop, avg_gap, pm, obj_value, best_fit, avg_fit)
    print('best result as follw :\n', results)

    max_value = results[0][1]
    max_index = 0
    for index in range(len(results)):
        if results[index][1] >= max_value:
            max_value = results[index][1]
            max_index = index
    print('max fit value is :', max_value)
    print('max fit value index  is:', max_index)

    best_rank = compress_rate_convert(results[max_index][0], chromosome_length, compress_rate)

    model = Tucker_decompose(best_rank)

    model = Tucker_for_linear(model, decompose_rate=0.1)


def calc_avg_value(obj_value,pop_size):
    total_value=0
    for value in obj_value:
        total_value+=value
    avg_fit=total_value/pop_size
    return  avg_fit
def init_population(pop_size, chromosome_length):
    # 形如[[0,1,..0,1],[0,1,..0,1]...]
    pop = [[random.random() for i in range(chromosome_length)] for j in range(pop_size)]
    return pop





def calc_obj_value(pop, chromosome_length,compress_rate):
    modelPath = './vgg16_model.pkl'

    if os.path.exists(modelPath):
        print('model exits')
        net = torch.load(modelPath, map_location='cpu')
        print('model loaded')
    else:
        print('model not exits')
    obj_value=[]
    for i in range(len(pop)):
        ranks=compress_rate_convert(pop[i],chromosome_length,compress_rate)
        net=Tucker_decompose(net,ranks)
        temp=Accuracy(net)
        obj_value.append(temp)
    return obj_value


def compress_rate_convert(gene,chromosome_length,compress_rate):
    channel_list = [64, 64, 64, 64,
                    128, 128, 128, 128,
                    256, 256, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512,
                    512, 512, 512, 512, 512]
    conv_channel_sum=7936
    print('present gene is :',gene)
    compress_channel=[]
    for i in range(chromosome_length):

        temp=math.ceil(gene[i]*channel_list[i])
        compress_channel.append(temp)
    print('compress channel is :',compress_channel)
    compress_sum=0
    for i in range(chromosome_length):
        compress_sum+=compress_channel[i]
    Factor=(conv_channel_sum*compress_rate)/compress_sum

    conver_pop=[]
    for i in range(chromosome_length):
        temp2=gene[i]*Factor
        conver_pop.append(temp2)
    real_compress=0
    ranks=[]
    for i in range(chromosome_length):
        real_compress+=math.ceil(conver_pop[i]*channel_list[i])
        temp3=math.ceil(conver_pop[i]*channel_list[i])
        ranks.append(temp3)
    print('real compress num is :',7936-real_compress)
    print('real ranks list is :',ranks)
    for i in range(chromosome_length):
        if ranks[i]>=channel_list[i]:
            ranks[i]=channel_list[i]-1

    return ranks


# 淘汰
def calc_fit_value(obj_value):
    fit_value = []

    c_min = 10
    for value in obj_value:
        if value > c_min:
            temp = value
        else:
            temp = 0.
        fit_value.append(temp)
    # fit_value保存的是活下来的值
    return fit_value


# 找出最优解和最优解的基因编码
def find_best(pop, fit_value):
    # 用来存最优基因编码
    best_individual =pop[0]
    # 先假设第一个基因的适应度最好
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    # best_fit是值
    # best_individual是基因序列
    return best_individual, best_fit


# 计算累计概率
def cum_sum(fit_value):
    # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
    # 这个地方遇坑，局部变量如果赋值给引用变量，在函数周期结束后，引用变量也将失去这个值
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = (sum(temp[:i + 1]))


# 轮赌法选择
def selection(pop,fit_value,obj_value):
    p_fit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    # 归一化，使概率总和为1
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit)
    cum_sum(p_fit_value)
    pop_len = len(pop)

    ms = sorted([random.random() for i in range(pop_len)])
    fitin = 0
    newin = 0
    newpop = pop[:]
    newobj=obj_value[:]
    # 复制pop的内容并指向新的内存地址.

    while newin < pop_len:
        # 如果这个概率大于随机出来的那个概率，就选这个
        if (ms[newin] < p_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newobj[newin] = obj_value[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1

    pop = newpop[:]
    obj_value=newobj[:]
    return pop,obj_value


# 杂交
def crossover(pop,avg_gap,pc,obj_value,best_fit,avg_fit):
    # 一定概率杂交，主要是杂交种群种相邻的两个个体
    pop_len = len(pop)
    for i in range(pop_len - 1):
        f_ = max(obj_value[i], obj_value[i + 1])
        if f_ > avg_fit:
            pc = np.multiply(pc, (best_fit - f_) / avg_gap)
        # 随机看看达到杂交概率没
        if (random.random() < pc):
            # 随机选取杂交点，然后交换数组
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1[:]
            pop[i + 1] = temp2[:]


# 基因突变
def mutation(pop,avg_gap,pm,obj_value,best_fit,avg_fit):
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if obj_value[i] >= avg_fit:
            pm = np.multiply(pm, (best_fit - obj_value[i]) / avg_gap)
        for j in range(py):
            if (random.random() < pm):
                pop[i][j] = random.random()


if __name__ == '__main__':
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
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
    main()
