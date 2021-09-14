import torch.nn as nn
from utils.cg_utils import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': './pretrain/resnet18-5c106cde.pth',
    'resnet34': './pretrain/resnet34-333f7ec4.pth',
    'resnet50': './pretrain/resnet50-19c8e357.pth',
    'resnet101': './pretrain/resnet101-5d3b4d8f.pth',
    'resnet152': './pretrain/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #print(out.shape, identity.shape) # 8, 16
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

###############################
# CIFAR
###############################
# 기본 scalable neural network
class ResNet_CIFAR(nn.Module):
    """
    For CIFAR ResNet
    """

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, align="CONV"):
        super(ResNet_CIFAR, self).__init__()
        print("num_class: ", num_classes)
        self.inplanes = 16
        self.align = align
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        #   self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        num_channels = [16, 32, 64] # 16, 32, 64
        self.layer1 = self._make_layer(block, num_channels[0], layers[0])
        self.layer2 = self._make_layer(block, num_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], layers[2], stride=2)

        print("CONV for aligning")
        self.branch1 = ScalaNet(
            channel_in=num_channels[0]*block.expansion,
            channel_out=num_channels[2]*block.expansion,
            size=8
        )
        self.branch2 = ScalaNet(
            channel_in=num_channels[1] * block.expansion,
            channel_out=num_channels[2] * block.expansion,
            size=4
        )

        self.branch3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(num_channels[2] * block.expansion, num_classes)
        self.fc2 = nn.Linear(num_channels[2] * block.expansion, num_classes)
        self.fc3 = nn.Linear(num_channels[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        feature_list.append(x) # fea1

        x = self.layer2(x)
        feature_list.append(x) # fea2

        x = self.layer3(x)
        feature_list.append(x) # fea3


        out1_feature = self.branch1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.branch2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.branch3(feature_list[2]).view(x.size(0), -1)
        
        # teacher_feature = out3_feature.detach()
        # feature_loss = ((teacher_feature - out2_feature)**2 + (teacher_feature - out1_feature)**2).sum()
        feature_loss=0

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)

        return [out1, out2, out3], feature_loss


###############################
# IMAGENET
###############################



#imaganet
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False, align="CONV"):
        super(ResNet, self).__init__()
        print("num_class: ", num_classes)
        self.inplanes = 64
        self.align = align
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        print("CONV for aligning")
        self.scala1 = ScalaNet(
            channel_in=64*block.expansion,
            channel_out=512*block.expansion,
            size=3
        )
        self.scala2 = ScalaNet(
            channel_in=128 * block.expansion,
            channel_out=512 * block.expansion,
            size=3
        )
        self.scala3 = ScalaNet(
            channel_in=256 * block.expansion,
            channel_out=512 * block.expansion,
            size=3
        )
        self.scala4 = nn.AvgPool2d(4, 4)

        self.attention1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=2, in_channels=64* block.expansion, out_channels=64* block.expansion),
            nn.BatchNorm2d(64* block.expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=64* block.expansion, out_channels=64* block.expansion),
            nn.BatchNorm2d(64* block.expansion),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=2, in_channels=128* block.expansion, out_channels=128* block.expansion),
            nn.BatchNorm2d(128* block.expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=128* block.expansion, out_channels=128* block.expansion),
            nn.BatchNorm2d(128* block.expansion),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, stride=2, in_channels=256* block.expansion, out_channels=256* block.expansion),
            nn.BatchNorm2d(256* block.expansion),
            nn.ReLU(),
            nn.ConvTranspose2d(kernel_size=4, padding=1, stride=2, in_channels=256* block.expansion, out_channels=256* block.expansion),
            nn.BatchNorm2d(256* block.expansion),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        fea1 = self.attention1(x)
        fea1 = fea1 * x
        feature_list.append(fea1)

        x = self.layer2(x)

        fea2 = self.attention2(x)
        fea2 = fea2 * x
        feature_list.append(fea2)

        x = self.layer3(x)

        fea3 = self.attention3(x)
        fea3 = fea3 * x
        feature_list.append(fea3)


        x = self.layer4(x)
        feature_list.append(x)

        out1_feature = self.scala1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.scala2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.scala3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)

        teacher_feature = out4_feature.detach()
        feature_loss = ((teacher_feature - out3_feature)**2 + (teacher_feature - out2_feature)**2 +\
                        (teacher_feature - out1_feature)**2).sum()

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], feature_loss

def resnet20(pretrained=False, pruned=True, num_classes=10, dataset='cifar10',  **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'cifar10':
        if pruned: 
            model = ResNet_CIFAR(MLBottleneck, [3, 3, 3],num_classes=10, **kwargs)
        else:
            model = ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=10, **kwargs)#Bottleneck
            #model = ResNet_CIFAR(MLBottleneck, [9, 9, 9], num_classes=10, **kwargs)


    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet56(pretrained=False, pruned=True, num_classes=10, dataset='cifar10',  **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'cifar10':
        if pruned: 
            model = ResNet_CIFAR(MLBottleneck, [9, 9, 9],num_classes=10, **kwargs)
        else:
            model = ResNet_CIFAR(BasicBlock, [9,9,9], num_classes=10, **kwargs)#Bottleneck
            #model = ResNet_CIFAR(MLBottleneck, [9, 9, 9], num_classes=10, **kwargs)
    elif dataset == 'cifar100':
        if pruned:
            model = ResNet_CIFAR(MLBottleneck, [9, 9, 9],num_classes=100, **kwargs)
        else:
            model = ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=100, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18(pretrained=False, pruned=True, dataset='imagenet', **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        if pruned:
            model = ResNet(MLBottleneck, [2, 2, 2, 2], num_classes=1000, **kwargs)
        else:
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, **kwargs)

    return model

def resnet50(pretrained=False,  pruned=True, dataset ='imagenet', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset =='imagenet':
        if pruned:
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, **kwargs)
        else:
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, **kwargs)

    return model
