import random
from snn_models.layers import *
# from layers import *
from torch.nn import functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, mode='normal', is_last=False):
        super(PreActBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.conv2 = SeqToANNContainer(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.spike1 = LIFSpike(dspike=dspike, gama=temp)
        if is_last:
            print("ReLU")
            self.spike2 = nn.ReLU()
        else:
            print("LIF Neuron")
            self.spike2 = LIFSpike(dspike=dspike, gama=temp)
        self.mode = mode

        if (stride != 1 or inplanes != planes) and self.mode != 'none':
            # Projection also with pre-activation according to paper.
            self.downsample = nn.Sequential(tdLayer(nn.AvgPool2d(stride)),
                                            SeqToANNContainer(conv1x1(inplanes, planes)),
                                            norm_layer(planes))

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x

        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.spike2(out)


def reduce_tensor_dimension(tensor):
    """
    将输入张量的维度从 [128, T, 128, 32, 32] 转换为 [128, T-1, 128, 32, 32]。
    通过对 T 维度的最后一个通道与其他通道按位与 (bitwise AND) 操作实现。

    Args:
        tensor (torch.Tensor): 输入张量，形状为 (128, T, 128, 32, 32)

    Returns:
        torch.Tensor: 经过操作后的张量，形状为 (128, T-1, 128, 32, 32)
    """
    # 检查输入张量的维度
    if tensor.dim() != 5:
        raise ValueError("输入张量必须是 5 维的，形状为 [128, T, 128, 32, 32]")
    
    batch_size, T, channels, height, width = tensor.shape

    if T < 2:
        raise ValueError("T 维度大小必须大于等于 2，才能进行按位与操作")

    # 初始化一个新张量，用于存储结果
    result = torch.empty((batch_size, T-1, channels, height, width), dtype=tensor.dtype, device=tensor.device)

    # 按位与操作：将最后一个通道与其他通道进行 && 操作
    for i in range(T-1):
        result[:, i, :, :, :] = tensor[:, i, :, :, :] * tensor[:, -1, :, :, :]  # 按位与
    
    return result




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, width_mult=1, T=4, in_c=3, mode='normal',
                 use_dspike=False, gamma=1):
        super(ResNet, self).__init__()

        global norm_layer
        norm_layer = tdBatchNorm
        global dspike
        dspike = use_dspike
        global temp
        temp = gamma

        self.inplanes = 16 * width_mult

        self.conv1 = nn.Conv2d(in_c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = tdBatchNorm(self.inplanes)
        self.spike = LIFSpike(dspike=dspike, gama=temp)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], mode=mode)
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], stride=2, mode=mode)
        self.layer3 = self._make_layer(block, self.inplanes * 2, layers[2], stride=2, mode=mode, is_last_layer=True)
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1 = nn.Linear(self.inplanes, num_classes)
        self.T = T
        self.add_dim = lambda x: add_dimention(x, self.T)

    def _make_layer(self, block, planes, blocks, stride=1, mode='normal', is_last_layer=False):
        layers = []
        layers.append(block(self.inplanes, planes, stride, mode=mode))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            is_last_block = is_last_layer and (i == blocks - 1)
            layers.append(block(self.inplanes, planes, mode=mode, is_last=is_last_block))
        return nn.Sequential(*layers)
            

        # return nn.Sequential(*layers)

    def _forward_impl(self, x, return_inter=False):
        mid_features = []
        
        x = self.conv1(x)
        if len(x.shape) == 4:
            x = self.add_dim(x)
        x = self.bn(x)
        x = self.spike(x)
        x = self.layer1(x)
        # x = reduce_tensor_dimension(x)
        if return_inter:
            mid_features.append(x)
        x = self.layer2(x)
        # x = reduce_tensor_dimension(x)
        if return_inter:
            mid_features.append(x)
        x = self.layer3(x)
        # x = reduce_tensor_dimension(x)
        if return_inter:
            mid_features.append(x)
        # x = self.last_relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        x = x.mean(1).squeeze(1)
        y = self.fc1(x)
        if return_inter:
            return y, mid_features
        else:
            return y

    def forward(self, x, return_inter=False):

        return self._forward_impl(x, return_inter=return_inter)



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = {}
        model.load_state_dict(state_dict)
    return model

# def _sew_resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = SEWResNet19(block, layers, **kwargs)
#     if pretrained:
#         state_dict = {}
#         model.load_state_dict(state_dict)
#     return model

# def _resnet_last_w_relu(arch, block, layers, pretrained, progress, **kwargs):
#     model = LastWoLIFSpikeResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = {}
#         model.load_state_dict(state_dict)
#     return model

# BasicBlock
# PreActBasicBlock
def resnet19(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet19', PreActBasicBlock, [3, 3, 2], pretrained, progress, **kwargs)


def resnet20(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet20', PreActBasicBlock, [3, 3, 3], pretrained, progress, **kwargs)


# def sew_resnet19(pretrained=False, progress=True, **kwargs):
#     return _sew_resnet('resnet19', SEWBasicBlock, [3, 3, 2], pretrained, progress, **kwargs)
def count_parameters_in_m():
    # 创建模型实例
    resnet = resnet20(num_classes=10, width_mult=1, T=1)

    # 统计所有参数数量
    total_params = sum(p.numel() for p in resnet.parameters())

    # 转换为百万 (M)
    total_params_in_m = total_params / 1e6
    print(f"Total parameters: {total_params_in_m:.2f}M")


# count_parameters_in_m()
    
#     x = torch.randn(1, 3, 32, 32)
#     y = sew_resnet(x)
#     print(y.shape)
# test()