# import torch
# import torch.nn as nn
# from spikingjelly.clock_driven import layer
# # from spikingjelly.cext import neuron as cext_neuron
# from spikingjelly.activation_based import neuron, functional, surrogate, layer


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, connect_f=None):
#         super(BasicBlock, self).__init__()
#         self.connect_f = connect_f
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

#         self.conv1 = layer.SeqToANNContainer(
#             conv3x3(inplanes, planes, stride),
#             norm_layer(planes)
#         )
#         self.sn1 = neuron.IFNode(step_mode='m')

#         self.conv2 = layer.SeqToANNContainer(
#             conv3x3(planes, planes),
#             norm_layer(planes)
#         )
#         self.downsample = downsample
#         self.stride = stride
#         self.sn2 = neuron.IFNode(step_mode='m')

#     def forward(self, x):
#         identity = x

#         out = self.sn1(self.conv1(x))

#         out = self.sn2(self.conv2(out))

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         if self.connect_f == 'ADD':
#             out += identity
#         elif self.connect_f == 'AND':
#             out *= identity
#         elif self.connect_f == 'IAND':
#             out = identity * (1. - out)
#         else:
#             raise NotImplementedError(self.connect_f)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None, connect_f=None):
#         super(Bottleneck, self).__init__()
#         self.connect_f = connect_f
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         self.conv1 = layer.SeqToANNContainer(
#             conv1x1(inplanes, width),
#             norm_layer(width)
#         )
#         self.sn1 = neuron.IFNode(step_mode='m')

#         self.conv2 = layer.SeqToANNContainer(
#             conv3x3(width, width, stride, groups, dilation),
#             norm_layer(width)
#         )
#         self.sn2 = neuron.IFNode(step_mode='m')

#         self.conv3 = layer.SeqToANNContainer(
#             conv1x1(width, planes * self.expansion),
#             norm_layer(planes * self.expansion)
#         )
#         self.downsample = downsample
#         self.stride = stride
#         self.sn3 = neuron.IFNode(step_mode='m')

#     def forward(self, x):
#         identity = x

#         out = self.sn1(self.conv1(x))

#         out = self.sn2(self.conv2(out))

#         out = self.sn3(self.conv3(out))

#         if self.downsample is not None:
#             identity = self.downsample(x)

        
#         if self.connect_f == 'ADD':
#             out += identity
#         elif self.connect_f == 'AND':
#             out *= identity
#         elif self.connect_f == 'IAND':
#             out = identity * (1. - out)
#         else:
#             raise NotImplementedError(self.connect_f)

#         return out
# def zero_init_blocks(net: nn.Module, connect_f: str):
#     for m in net.modules():
#         if isinstance(m, Bottleneck):
#             nn.init.constant_(m.conv3.module[1].weight, 0)
#             if connect_f == 'AND':
#                 nn.init.constant_(m.conv3.module[1].bias, 1)
#         elif isinstance(m, BasicBlock):
#             nn.init.constant_(m.conv2.module[1].weight, 0)
#             if connect_f == 'AND':
#                 nn.init.constant_(m.conv2.module[1].bias, 1)


# class SEWResNet(nn.Module):

#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None, T=4, connect_f=None):
#         super(SEWResNet, self).__init__()
#         self.T = T
#         self.connect_f = connect_f

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         # self.bn1 = norm_layer(self.inplanes)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # self.layer1 = self._make_layer(block, 64, layers[0])
#         # self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)


#         self.sn1 = neuron.IFNode(step_mode='m')
#         self.maxpool = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

#         self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0], connect_f=connect_f)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1], connect_f=connect_f)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2], connect_f=connect_f)
#         self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         # self.fc2 = nn.Linear(256, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         if zero_init_residual:
#             zero_init_blocks(self, connect_f)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 layer.SeqToANNContainer(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     norm_layer(planes * block.expansion),
#                 ),
#                 neuron.IFNode(step_mode='m')
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, connect_f))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, connect_f=connect_f))

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x,return_mid=False):
#         if return_mid:
#             midd_feat = []
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x.unsqueeze_(0)
#             x = x.repeat(self.T, 1, 1, 1, 1)
#             x = self.sn1(x)
#             x = self.maxpool(x)
#             x = self.layer1(x)
#             midd_feat.append(x)
#             x = self.layer2(x)
#             midd_feat.append(x)
#             x = self.layer3(x)
#             midd_feat.append(x)
#             x = self.layer4(x)
#             midd_feat.append(x)

#             x = self.avgpool(x)
#             x = torch.flatten(x, 2)
        
#             return self.fc(x.mean(dim=0)),midd_feat
#         else:
#             x = self.conv1(x)
#             x = self.bn1(x)
#             x.unsqueeze_(0)
#             x = x.repeat(self.T, 1, 1, 1, 1)
#             x = self.sn1(x)
#             x = self.maxpool(x)
#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)
#             x = self.avgpool(x)
#             x = torch.flatten(x, 2)
#             return self.fc(x.mean(dim=0))

#     def forward(self, x,return_mid=False):
#         return self._forward_impl(x,return_mid)


# def _sew_resnet(block, layers, **kwargs):
#     model = SEWResNet(block, layers, **kwargs)
#     return model


# def sew_resnet18(**kwargs):
#     return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

# # def sew_resnet18(**kwargs):
# #     return _sew_resnet(BasicBlock, [3 ,3, 2], **kwargs)


# def sew_resnet34(**kwargs):
#     return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def sew_resnet50(**kwargs):
#     return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def sew_resnet101(**kwargs):
#     return _sew_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


# def sew_resnet152(**kwargs):
#     return _sew_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


# def get_sewresnet(arch='18',**kwargs):
#     if arch == '18':
#         return sew_resnet18(**kwargs)
#     elif arch == '34':
#         return sew_resnet34(**kwargs)
#     elif arch == '50':
#         return sew_resnet50(**kwargs) 
#     elif arch == '101':
#         return sew_resnet101(**kwargs) 
#     elif arch == '152':
#         return sew_resnet152(**kwargs) 

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer,surrogate
from spikingjelly.activation_based import neuron as cext_neuron

__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101',
           'sew_resnet152']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class InteractiveMerge(nn.Module):
    """
    对输入脉冲张量的 t 维度进行全局交互融合，减少一个时间步。
    
    参数:
        t (int): 输入时间步数量。
    
    输入:
        spike_feature (torch.Tensor): 二值脉冲张量，形状为 (t, b, c, h, w)。
        is_shuffled (bool): 是否对时间步进行随机打乱，默认为 True。
    
    返回:
        torch.Tensor: 融合后的张量，形状为 (t, b, c, h, w)。
    """
    def __init__(self):
        super(InteractiveMerge, self).__init__()

    def forward(self, spike_feature, is_traning=True):
        if spike_feature.ndim != 5:
            raise ValueError("输入张量必须是 5 维的 (t, b, c, h, w)。")
        
        # t, b, c, h, w = spike_feature.shape
        # if t == 1:
            # return spike_feature
        
        # total_spikes = spike_feature.sum(dim=0, keepdim=True)  # 形状 (1, b, c, h, w)
            
        # output = total_spikes.repeat(t-1, 1, 1, 1, 1) # 形状 (t-1, b, c, h, w)
            
        return spike_feature


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())

    def forward(self, x, return_inter=False):
        identity = x
        outs = []

        out = self.conv1(x)
        # outs.append(out.mean(dim=0))
        out = self.sn1(out)

        out = self.conv2(out)
        # outs.append(out.mean(dim=0))
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        if return_inter:
            return out, out.mean(dim=0)
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(Bottleneck, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())

    def forward(self, x, return_inter=False):
        identity = x
        outs = []

        out = self.conv1(x)
        # outs.append(out.mean(dim=0))
        out = self.sn1(out)

        out = self.conv2(out)
        # outs.append(out.mean(dim=0))
        out = self.sn2(out)

        out = self.conv3(out)
        # outs.append(out.mean(dim=0))
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        
        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        if return_inter:
            return out, out.mean(dim=0)
        else:
            return out

def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv3.module[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv2.module[1].bias, 1)


class ShortDownSample(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=1, pool_size=1,stride=1):
        super(ShortDownSample, self).__init__()
        
        # 卷积层 (Conv)
        self.conv =  layer.SeqToANNContainer(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
        
        # 批归一化层 (BN)
        self.bn = layer.SeqToANNContainer( nn.BatchNorm2d(out_channels))
        
        # ReLU 激活函数 (作为 LIF)
        self.relu = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())
        
        # 最大池化层 (MaxPool)
        self.maxpool = layer.SeqToANNContainer( nn.MaxPool2d(kernel_size=pool_size, stride=stride))
        
        # 假设有多个输入需要拼接 (CONCAT)，这里预留一个简单的拼接逻辑
        # 实际使用时，需要根据具体任务定义多个输入分支
        self.concat_dim = 1  # 拼接维度（通道维度）

    def forward(self, x):
        # 卷积层
        conv_out = self.conv(x)
        
        # 批归一化
        bn_out = self.bn(conv_out)
        
        # ReLU 激活 (作为 LIF)
        relu_out = self.relu(bn_out)
        
        # 最大池化
        pool_out = self.maxpool(relu_out)
        
        return pool_out

# def enchance_w_noise(x, num_noise=5):
#     """
#     将输入张量 x 与指定数量的高斯噪声张量拼接，噪声使用 x 的均值和方差生成。

#     参数：
#         x (torch.Tensor): 输入张量，例如 logits，形状为 (batch_size, features)。
#         num_noise (int): 要生成并拼接的高斯噪声张量的数量。

#     返回：
#         torch.Tensor: 拼接后的张量。
#     """
#     # 计算 x 的均值和标准差（沿着 batch 维度）
#     hidden_mean = x.mean(dim=0)  # 形状: (features,)
#     hidden_std = x.std(dim=0)    # 形状: (features,)，标准差 = 方差的平方根
#     print(hidden_mean.shape)
#     # 生成高斯噪声，使用 x 的均值和标准差
#     noise = torch.normal(mean=hidden_mean, std=hidden_std, 
#                          size=(num_noise, *hidden_mean.shape), 
#                          device=x.device, dtype=x.dtype)  # 形状: (num_noise, 200)

#     # 将原始张量 x 与噪声张量拼接
#     result = torch.cat((x, noise), dim=0)

#     return result

def enchance_w_noise(x, num_noise=15):
    """
    将输入张量 x 与指定数量的高斯噪声张量拼接，噪声使用 x 的均值和方差生成。

    参数：
        x (torch.Tensor): 输入张量，例如 logits，形状为 (batch_size, 200)。
        num_noise (int): 要生成并拼接的高斯噪声张量的数量。

    返回：
        torch.Tensor: 拼接后的张量，形状为 (batch_size + num_noise, 200)。
    """
    x = x.unsqueeze(0)
    
    hidden_mean = x.mean(dim=0)  # 形状: (200,)
    
    noise = torch.randn(num_noise, *hidden_mean.shape, device=x.device, dtype=x.dtype)  # 形状: (num_noise, 200)
    
    result = torch.cat((x, noise), dim=0)
    # print(result.shape)
    return result

class SEWResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4,connect_f=None, indices=(3,),interact_t=True):
        super(SEWResNet, self).__init__()
        self.T = T
        # self.eval_T = eval_T
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.indices = indices
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)


        self.sn1 = cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())
        self.maxpool = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)
        # self.interact_t = interact_t
        if interact_t:
            self.time_interaction = InteractiveMerge()
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                cext_neuron.IFNode(step_mode='m',surrogate_function=surrogate.ATan())
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, return_inter=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        x = self.maxpool(x)
        if return_inter:
            intermeidate = []
            x = self.layer1(x)
            intermeidate.append(x)
            x = self.time_interaction(x)
            x = self.layer2(x)
            intermeidate.append(x)
            x = self.time_interaction(x)
            x = self.layer3(x)
            intermeidate.append(x)
            x = self.time_interaction(x)
            x = self.layer4(x)
            intermeidate.append(x)
            x = self.time_interaction(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 2)
            logits =  self.fc(x.mean(dim=0)) 

            # if self.training:
            #     return enchance_w_noise(logits).mean(0),intermeidate
            # else:
            #     return logits,intermeidate
            return logits,intermeidate
        else:
            x = self.layer1(x)
            x = self.time_interaction(x)
            x = self.layer2(x)
            x = self.time_interaction(x)
            x = self.layer3(x)
            x = self.time_interaction(x)
            x = self.layer4(x)
            x = self.time_interaction(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 2)
            
            logits =  self.fc(x.mean(dim=0))
            # if self.training:
            #     return enchance_w_noise(logits).mean(0)
            # else:
            #     return logits
            return logits
            

    def forward(self, x, return_inter=False):
        return self._forward_impl(x, return_inter)


def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return _sew_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)

def get_sewresnet(arch='18',**kwargs):
    if arch == '18':
        return sew_resnet18(**kwargs)
    elif arch == '34':
        return sew_resnet34(**kwargs)
    elif arch == '50':
        return sew_resnet50(**kwargs) 
    elif arch == '101':
        return sew_resnet101(**kwargs) 
    elif arch == '152':
        return sew_resnet152(**kwargs) 