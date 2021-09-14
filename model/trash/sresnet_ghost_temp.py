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


def conv3x3_modify(in_planes, out_planes, stride=1, relu=False):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups = in_planes, bias=False),# 
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

# identity도 손봐야할거 같은데,,
class SEMaskLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEMaskLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            MaskUnit()
            #nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)#.view(b, c, 1, 1)
        return x * y.expand_as(x)

# 근데, kernel size==1일 때는, padding좀 신경써야할듯 한데,,(high 3-1 로 할까..?)
# mask 구할때, abs할지말지...,  max, mean, sum중에 하나할지 말지..
class MLModule1(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, padding=1, stride=1, relu=True):
        super(MLModule1, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        #self.mask_layer = SEMaskLayer(inp)
        self.first_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.fixed_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=stride, padding=1, groups=init_channels, bias=False),# groups=init_channels,
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),# if relu else nn.Sequential(),

            nn.Conv2d(new_channels, new_channels, kernel_size, stride=stride,  bias=False),#groups=init_channels
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        
    def forward(self, x):
        #x = self.mask_layer(x)
        x1 = self.first_conv(x)
        x2 = self.fixed_conv(x1)
        out = torch.cat([x1,x2], dim=1)
        
        return out[:,:self.oup,:,:]

class MLModule2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, padding=1, stride=1, relu=True):
        super(MLModule2, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.mask_layer = SEMaskLayer(inp)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.high_level_conv = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, kernel_size=3, stride=2, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(init_channels, new_channels, kernel_size, stride=stride, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
            
        )
        
    def forward(self, x):
        #x = self.mask_layer(x)
        x1 = self.low_level_conv(x)
        x2 = self.high_level_conv(x1)
        out = torch.cat([x1,x2], dim=1)[:,:self.oup,:,:]

        return out

class MLBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MLBottleneck, self).__init__()

        self.conv = nn.Sequential(
            MLModule1(inplanes, planes, relu=True),
            conv3x3_modify(planes, planes, stride, relu=False ),# if stride>1 else nn.Sequential(),
            MLModule2(planes, planes * self.expansion, relu=False),

        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        
        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        #out = self.conv3(out)
        #out = self.bn3(out)
        out = self.conv(x)

        if self.downsample is not None:
            identity = self.downsample(x)
            #out = out + identity

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


def BranchNet(channel_in, channel_out):
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(channel_in),
        nn.ReLU(),
        nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
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
        self.inplanes = 64
        self.align = align
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #   self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        print("CONV for aligning")
        self.branch1 = BranchNet(
            channel_in=64*block.expansion,
            channel_out=512*block.expansion,
        )
        self.branch2 = BranchNet(
            channel_in=128 * block.expansion,
            channel_out=512 * block.expansion,
        )
        self.branch3 = BranchNet(
            channel_in=256 * block.expansion,
            channel_out=512 * block.expansion,
        )
        self.branch4 = nn.AdaptiveAvgPool2d((1, 1))

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
        #                  (teacher_feature - out1_feature)**2).sum()

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out1, out2, out3, out4]#, feature_loss


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
        self.branch1 = BranchNet_original(
            channel_in=64*block.expansion,
            channel_out=512*block.expansion,
            size=8, #8,
        )
        self.branch2 = BranchNet_original(
            channel_in=128 * block.expansion,
            channel_out=512 * block.expansion,
            size= 4, #4,
        )
        self.branch3 = BranchNet_original(
            channel_in=256 * block.expansion,
            channel_out=512 * block.expansion,
            size= 2, #2
        )
        #self.branch4 = nn.AvgPool2d(self.dict_[2], self.dict_[2])
        self.branch4 = nn.AdaptiveAvgPool2d((1, 1))

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
                #HTConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
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
        #x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)


        x = self.layer1(x)
        fea1 = self.attention1(x)
        fea1 = fea1 * x
        feature_list.append(fea1) # fea1

        x = self.layer2(x)

        fea2 = self.attention2(x)
        fea2 = fea2 * x
        feature_list.append(fea2) # fea2

        x = self.layer3(x)

        fea3 = self.attention3(x)
        fea3 = fea3 * x
        feature_list.append(fea3) # fea3


        x = self.layer4(x)
        feature_list.append(x)



        out1_feature = self.branch1(feature_list[0]).view(x.size(0), -1)
        out2_feature = self.branch2(feature_list[1]).view(x.size(0), -1)
        out3_feature = self.branch3(feature_list[2]).view(x.size(0), -1)
        out4_feature = self.branch4(feature_list[3]).view(x.size(0), -1)

        teacher_feature = out4_feature.detach()
        feature_loss = ((teacher_feature - out3_feature)**2 + (teacher_feature - out2_feature)**2 +\
                        (teacher_feature - out1_feature)**2)#.sum()

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out1, out2, out3, out4], feature_loss


def resnet18(pretrained=False, pruned=True, num_classes=10, dataset='cifar10', patch_size=None,  **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if dataset == 'imagenet':
        assert False, 'not implement'
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, patch_size = patch_size, **kwargs)
    elif dataset == 'cifar10':
        if pruned: 
            model = ResNet_CIFAR(MLBottleneck, [2, 2, 2, 2],num_classes=10, **kwargs)
        else:
            model = ResNet_CIFAR(Bottleneck, [2, 2, 2, 2], num_classes=10, **kwargs)#Bottleneck
    elif dataset == 'cifar100':
        if pruned:
            model = ResNet_CIFAR_PRUNE(MLBasicBlock, [2, 2, 2, 2],num_classes=100, **kwargs)
        else:
            model = ResNet_CIFAR(BasicBlock, [2, 2, 2, 2], num_classes=100, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model