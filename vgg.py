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





