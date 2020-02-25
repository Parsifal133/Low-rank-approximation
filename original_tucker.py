import argparse
import tensorly as tl
import os
import torch
import torch.nn as nn

from ori_decompositions import tucker_decomposition_conv_layer, tucker_for_first_linear_layer, \
    tucker_decomposition_linear_layer,tucker_for_first_conv_layer

batchSize=5

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
                        help="Use cp decomposition. uses tucker by default")
    parser.add_argument("--model_path", type=str, default='./AlexNet.pkl')
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=True)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    tl.set_backend('pytorch')
    decompose_rate = 0.5  # decompose rate for conv/FC layers
    model_path='./model.pkl'
    if os.path.exists(model_path):
        model=torch.load(model_path,map_location='cpu')
        print('model exists,model loaded!\n')
    else:
        print('model path not exist,creat new one\n')
        model = AlexNet()
    print(model)


    model.eval()
    model.cpu()



    ##########          conv layer decompose part            ################

    print('tucker decompose start\n')
    N = len(model.features._modules.keys())
    print('N is :',N)
    conv_flag=1
    for i, key in enumerate(model.features._modules.keys()):


        print('present layer is :',model.features._modules[key])
        if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = model.features._modules[key]
            if conv_flag!=0:
                decomposed = tucker_for_first_conv_layer(conv_layer, decompose_rate)

                model.features._modules[key] = decomposed
                print('conv layer decompose end')
            else:
                decomposed = tucker_decomposition_conv_layer(conv_layer, decompose_rate)

                model.features._modules[key] = decomposed
                print('conv layer decompose end')

    # ##########          FC layer decompose part            ################
    # linear_flag=1
    # for i, key in enumerate(model.classifier._modules.keys()):
    #
    #     if isinstance(model.classifier._modules[key], torch.nn.modules.linear.Linear):
    #         if linear_flag!=0:
    #             print('first')
    #             linear_layer = model.classifier._modules[key]
    #             decomposed = tucker_for_first_linear_layer(linear_layer, decompose_rate)
    #             model.classifier._modules[key] = decomposed
    #             print('linear layer decompose end\n')
    #             linear_flag = 0
    #         else:
    #             linear_layer = model.classifier._modules[key]
    #
    #             decomposed = tucker_decomposition_linear_layer(linear_layer, decompose_rate)
    #             model.classifier._modules[key] = decomposed
    #             print('linear layer decompose end\n')

    torch.save(model, 'decomposed_model.pkl')
    print('tucker decompose end,model saved!')
    print(model)

