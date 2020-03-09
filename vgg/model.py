import torchvision.models as models

vgg16 = models.vgg16_bn(pretrained=False)

print(vgg16)