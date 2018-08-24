import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg19 = models.vgg16(pretrained=True).features


    def forward(self, X):

        features = []
        for name, layer in self.vgg19._modules.items():
            X = layer(X)
            if name in self.select:
                features.append(X)
        return features


class Vgg19_mask(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_mask, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg19 = models.vgg16(pretrained=False).features

    def forward(self, X):
        features = []
        for name, layer in self.vgg19._modules.items():
            if isinstance(layer, torch.nn.Conv2d):
                X = torch.nn.AvgPool2d(kernel_size=(3,3), stride=(1,1),padding=(1,1))(X)
            elif isinstance(layer, torch.nn.MaxPool2d):
                X = torch.nn.AvgPool2d(kernel_size=2, stride=2,padding=0)(X)
            else:
                X = X

            if name in self.select:
                features.append(X)
        return features












