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


def conv3x3_modify(in_planes, out_planes, stride=1, groups=1, relu=False):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups = groups, bias=False, ),#  groups = in_planes,
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y
class MLModule(nn.Module):
    """
    pre conv
    :
    like ghostnet
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, padding=1, stride=1, relu=True):
        super(MLModule, self).__init__()
        self.oup = oup
        self.r = relu
        self.relu = nn.ReLU(inplace=True)
        ratio = 2
        init_channels = math.ceil(oup / ratio) 
        new_channels = init_channels*(ratio-1) 
        #new_channels = int(new_channels) ; init_channels = int(init_channels)
        
        #self.mask_layer1 = MaskUnit('reduction')
        self.conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)  if relu else nn.Sequential() ,
        )

        self.more_feature_conv = nn.Sequential(
            #nn.Conv2d(init_channels, new_channels, kernel_size=5, stride=stride, groups=init_channels, padding=2, bias=False),
            MLConv2d(init_channels, new_channels, kernel_size=5, stride=stride, groups=init_channels, padding=2, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
            )           

    def forward(self, x):
        _,_,h,w = x.shape
        # 1x1 --> 3x3 --> reuse, 5x5
        x1 = self.conv(x)
        x2 = self.more_feature_conv(x1)        

        out = torch.cat([x1, x2], dim=1)

        return out[:, :self.oup, :, :]

class MLBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=2):
        super(MLBottleneck, self).__init__()

        self.conv = nn.Sequential(
            MLModule(inplanes, planes, ratio= ratio, relu=True),
            conv3x3_modify(planes, planes, stride, relu=True, groups= planes) if stride>1 else nn.Sequential(),
            #SELayer(planes) if use_se else nn.Sequential(),,
            MLModule(planes, planes * self.expansion, ratio= ratio, relu=False),

        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=None):
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


"""branch 바껴있으니 확인하기"""
# def BranchNet2(channel_in, channel_out, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=2, padding=1, groups=channel_in, bias=False),
#         nn.BatchNorm2d(channel_in),
#         nn.ReLU(),
#         nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=1, groups=channel_in, bias=False),
#         nn.BatchNorm2d(channel_out),
#         nn.ReLU(),
#         nn.AdaptiveAvgPool2d((1, 1))
#         )

def BranchNet(channel_in, channel_out):
    return nn.Sequential(
        MLModule(channel_in, channel_in, stride=1, kernel_size=1, relu=True),
        nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=3, groups=1),
        nn.BatchNorm2d(channel_in),
        nn.ReLU(),
        MLModule(channel_in, channel_out, stride=1, kernel_size=1, relu=True),
        nn.AdaptiveAvgPool2d((1, 1))
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
        self.branch1 = BranchNet(
            channel_in=num_channels[0]*block.expansion,
            channel_out=num_channels[2]*block.expansion,
        )
        self.branch2 = BranchNet(
            channel_in=num_channels[1] * block.expansion,
            channel_out=num_channels[2] * block.expansion,
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
        
        feature_loss=0

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)

        return [out1, out2, out3], feature_loss


###############################
# IMAGENET
###############################
class ResNet(nn.Module):
    """
    ImageNet ResNet
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, align="CONV"):
        super(ResNet, self).__init__()
        print("num_class: ", num_classes)
        ratio = 1.2
        self.inplanes = int(ratio*64)
        print(self.inplanes, 'first conv channels')
        self.align = align

        self.conv1 = nn.Conv2d(3, int(ratio*64), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(int(ratio*64))
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(ratio*64), layers[0])
        self.layer2 = self._make_layer(block, int(ratio*128), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(ratio*256), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(ratio*512), layers[3], stride=2)

        print("CONV for aligning")
        self.branch1 = BranchNet(
            channel_in=int(ratio*64)*block.expansion,
            channel_out=int(ratio*512)*block.expansion,
        )
        self.branch2 = BranchNet(
            channel_in=int(ratio*128) * block.expansion,
            channel_out=int(ratio*512) * block.expansion,
        )
        self.branch3 = BranchNet(
            channel_in=int(ratio*256) * block.expansion,
            channel_out=int(ratio*512) * block.expansion,
        )
        self.branch4 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(int(ratio*512) * block.expansion, num_classes)
        self.fc2 = nn.Linear(int(ratio*512) * block.expansion, num_classes)
        self.fc3 = nn.Linear(int(ratio*512) * block.expansion, num_classes)
        self.fc4 = nn.Linear(int(ratio*512) * block.expansion, num_classes)

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
        x = self.maxpool(x)


        x = self.layer1(x)
        feature_list.append(x) # fea1

        x = self.layer2(x)
        feature_list.append(x) # fea2

        x = self.layer3(x)
        feature_list.append(x) # fea3


        x = self.layer4(x)
        feature_list.append(x)



        out1_feature = self.branch1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.branch2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.branch3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.branch4(feature_list[3]).view(x.size(0), -1)

        # teacher_feature = out4_feature.detach()
        # feature_loss = ((teacher_feature - out3_feature)**2 + (teacher_feature - out2_feature)**2 +\
        #                 (teacher_feature - out1_feature)**2)#.sum()
        # feature_loss = 0
        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out1, out2, out3, out4]#, feature_loss

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
            model = ResNet(MLBottleneck, [3, 4, 6, 3], num_classes=1000, **kwargs)
        else:
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, **kwargs)

    return model

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])