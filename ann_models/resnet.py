'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, mode='normal'):
        super(PreActBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.spike1 = nn.ReLU()
        self.spike2 = nn.ReLU()
        self.mode = mode

        if (stride != 1 or inplanes != planes) and self.mode != 'none':
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(nn.AvgPool2d(stride),
                                            conv1x1(inplanes, planes),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.spike2(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, width_mult=1, in_c=3,  num_classes=10):
        super(ResNet, self).__init__()

        self.inplanes = 16 * width_mult
        # print(self.inplanes)
        dim = 16 * width_mult
        self.conv1 = nn.Conv2d(in_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block, self.inplanes, num_blocks[0], mode='normal')
        self.layer2 = self._make_layer(block, self.inplanes * 2, num_blocks[1], stride=2, mode='normal')
        self.layer3 = self._make_layer(block, self.inplanes * 2, num_blocks[2], stride=2, mode='normal')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(self.inplanes, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, mode='normal'):
        layers = []
        layers.append(block(self.inplanes, planes, stride, mode=mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, mode=mode))
        return nn.Sequential(*layers)
    def forward(self, x, return_inter=False):
        mid_features = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if return_inter:
            mid_features.append(out)
        out = self.layer2(out)
        if return_inter:
            mid_features.append(out)
        out = self.layer3(out)
        if return_inter:
            mid_features.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if return_inter:
            return out, mid_features
        else:
            return out

    def foward_snn_feature(self,feature):
        out = self.avgpool(feature)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet19(num_classes=10,width_mult=8):
    # return _resnet('resnet19', PreActBasicBlock, [3, 3, 2], pretrained, progress, **kwargs)
    return ResNet(PreActBasicBlock, [3, 3, 2],num_classes=num_classes,width_mult=width_mult)

def resnet20(num_classes=10,width_mult=1):
    # return _resnet('resnet19', PreActBasicBlock, [3, 3, 2], pretrained, progress, **kwargs)
    return ResNet(PreActBasicBlock, [3, 3, 3],num_classes=num_classes,width_mult=width_mult)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = resnet20()
    print(net)
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()


