import torch.nn as nn
import torch.nn.functional as F
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        return self.sigmoid(out)
    
    
def domain_discrepancy(out1, out2):
    diff = out1 - out2
    loss = torch.mean(torch.abs(diff))
    return loss


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
        
        self.const_att = ChannelAttention(512)
        
        self.tar_ca_last = [1.]
        self.src_ca_last = [1.]
        self.weight_d = 0.3
        self.ema_alpha = 0.999
                
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    
    def forward(self, x, target=None, step=None):
        out1_s = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out2_s = self.layer1(out1_s)
        out3_s = self.layer2(out2_s)
        out4_s = self.layer3(out3_s)
        out5_s = self.layer4(out4_s)
        out6_s = self.avgpool(out5_s)
        out7_s = out6_s.view(out6_s.size(0), -1)
        
        if target is not None:
            out1_t = self.maxpool(self.relu(self.bn1(self.conv1(target))))
            out2_t = self.layer1(out1_t)
            out3_t = self.layer2(out2_t)
            out4_t = self.layer3(out3_t)
            out5_t = self.layer4(out4_t)
            out6_t = self.avgpool(out5_t)

            source_att = self.const_att(out6_s)
            target_att = self.const_att(out6_t)
            
            ema_alpha = min(1 - 1 / (step+1), self.ema_alpha)

            source_feat_att = torch.mul(out6_s, source_att)
            target_feat_att = torch.mul(out6_t, target_att)
            
            mean_tar_ca = self.tar_ca_last[0] * ema_alpha + (1. - ema_alpha) * torch.mean(target_feat_att, 0)
            self.tar_ca_last[0] = mean_tar_ca.detach()
            
            mean_src_ca = self.src_ca_last[0] * ema_alpha + (1. - ema_alpha) * torch.mean(source_feat_att, 0)
            self.src_ca_last[0] = mean_src_ca.detach()

            d_const_loss = self.weight_d * domain_discrepancy(mean_src_ca, mean_tar_ca)
            
            return out7_s, d_const_loss
        else:
            return out7_s, None



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
        x = self.classifier(x)
        return x
