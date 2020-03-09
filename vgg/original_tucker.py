import argparse
import tensorly as tl
import os
import torch
import torch.nn as nn

from ori_decompositions import tucker_decomposition_conv_layer, tucker_for_first_linear_layer, \
    tucker_decomposition_linear_layer,tucker_for_first_conv_layer

batchSize=5


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


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')


def Tucker_decompose(model,ranks):

    tl.set_backend('pytorch')
    print('tucker decompose start\n')
    N = len(model.features._modules.keys())
    print('N is :', N)
    conv_flag = 1
    conv_count=2
    for i, key in enumerate(model.features._modules.keys()):
        print('present layer is :', model.features._modules[key])
        if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = model.features._modules[key]
            if conv_flag != 0:
                decomposed = tucker_for_first_conv_layer(conv_layer,ranks)
                model.features._modules[key] = decomposed
                print('conv layer decompose end')
                conv_flag=0
            else:
                decomposed = tucker_decomposition_conv_layer(conv_layer,ranks,conv_count)
                model.features._modules[key] = decomposed
                print('conv layer decompose end')
                conv_count+=2
    torch.save(model, 'decomposed_model.pkl')
    print('tucker decompose end,model saved!')
    # print(model)
    return model
def Tucker_for_linear(model,decompose_rate):
    tl.set_backend('pytorch')
    linear_flag = 1
    for i, key in enumerate(model.classifier._modules.keys()):

        if isinstance(model.classifier._modules[key], torch.nn.modules.linear.Linear):
            if linear_flag != 0:
                print('first linear layer')
                linear_layer = model.classifier._modules[key]
                decomposed = tucker_for_first_linear_layer(linear_layer, decompose_rate)
                model.classifier._modules[key] = decomposed
                print('linear layer decompose end\n')
                linear_flag = 0
            else:
                linear_layer = model.classifier._modules[key]
                decomposed = tucker_decomposition_linear_layer(linear_layer, decompose_rate)
                model.classifier._modules[key] = decomposed
                print('linear layer decompose end\n')

    torch.save(model, 'decomposed_model.pkl')
    print('tucker decompose end,model saved!\n')
    print(model)
    return model

if __name__ == '__main__':

    tl.set_backend('pytorch')
    decompose_rate = 0.1 # decompose rate for conv/FC layers
    model_path='./vgg16_model.pkl'
    if os.path.exists(model_path):
        model=torch.load(model_path,map_location='cpu')
        print('model exists,model loaded!\n')
    else:
        print('model path not exist,creat new one\n')
        model = VGG16()
    print(model)


    model.eval()
    model.cpu()
    channel_list=[64,64,64,64,
                  128,128,128,128,
                  256,256,256,256,256,256,
                  512,512,512,512,512,512,
                  512,512,512,512,512]
    ranks=      [51, 7, 32,20,
                 78, 40, 110, 24,
                 194,64,108,37,206,110,
                 301,265,123,198,26,145
                 ,321,65,95,75,123]

    model=Tucker_decompose(model,ranks)
    model=Tucker_for_linear(model,decompose_rate)
    print(model)


