import torch.nn as nn
import torch
from grl import GradientReversal
import torchbnn as bnn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 23, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
        self.gradients = None

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        self.out1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        self.out2 = self.layer1(self.out1)
        self.out3 = self.layer2(self.out2)
        self.out4 = self.layer3(self.out3)
        
        h = self.out4.register_hook(self.activations_hook)

        self.out5 = self.layer4(self.out4)
        self.out6 = self.avgpool(self.out5)
        return [self.out6, self.out4]
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, layer, x):
        return layer(x)[1]
    


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=512, out_features=100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=2),
                nn.LogSoftmax(dim=1),
            )
        
        
    def forward(self, feature, alpha):        
        grad_reverse = GradientReversal(alpha)
        flat_feat = feature.view(feature.shape[0], -1)
        rev_feat = grad_reverse(flat_feat)
        
        out = self.model(rev_feat)  
        
        return out
 