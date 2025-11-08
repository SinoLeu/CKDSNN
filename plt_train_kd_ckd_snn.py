import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from torchvision import models, transforms
import torch
# from timm import create_model
from torchvision import transforms, datasets
import lightning as L
import timm
# import os
from torch.optim.lr_scheduler import StepLR
from spikingjelly.activation_based import neuron, functional, surrogate, layer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
from torchmetrics import Accuracy

from torchvision import transforms
import torchvision.models as models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.pl_data_cifar_loader import PlCAMDataModule,PlDataModule
# from models.s_model import get_sewresnet
from collections import OrderedDict
import timm
import logging
# from utils.utils import soft_kd_loss,mmd_loss,getForwardCAM,compute_kl_divergence,freeze_model_parameters
from utils.utils import mmd_loss,getForwardCAM,compute_kl_divergence,freeze_model_parameters,soft_loss_smooth,logits_external_loss,logits_internal_loss
# from spikingjelly.clock_driven.neuron import LIFNode
from spikingjelly.activation_based import neuron
from models.mpsa import MultiStageFeatureModule
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch
import torchvision.ops as ops

logging.getLogger().setLevel(logging.ERROR)
import os
os.environ['CURL_CA_BUNDLE'] = ''

from ann_models.resnet import resnet19,resnet20
# from snn_models.resnet_models import resnet20 as snn_resnet20
from snn_models.resnet_models import resnet19 as snn_resnet19


def remove_module_from_state_dict(state_dict):
    """
    遍历模型的 state_dict，将所有包含 'module' 的部分去掉。

    参数:
        state_dict (OrderedDict): 模型的 state_dict。

    返回:
        OrderedDict: 修改后的 state_dict。
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict



class CombineNet(nn.Module):
    """Combined network with ResNet backbone and MultiStageFeatureModule."""
    def __init__(self, model, stage_module):
        super().__init__()
        self.model = model
        self.da_module = stage_module

    def forward(self, x):
        # [stage1,stage2,stage3,stage4] = base_model.forward_intermediates
        # _, feat = self.model.forward_intermediates(x)
        y1 = self.model(x)
        # print(feat[1].shape,feat[2].shape,feat[3].shape)
        # formatted_feat = feat[1:]
        # out_tea = self.da_module(formatted_feat)
        # out_stu = self.da_module(spike_features)
        # return y1,out_tea,out_stu
        return y1


def calculate_accuracy(outputs, targets):
    """
    计算给定输出和目标的准确率。
    
    Args:
        outputs (torch.Tensor): 模型的输出，通常是 logits 或概率，形状为 (batch_size, num_classes)
        targets (torch.Tensor): 真实标签，形状为 (batch_size,)
    
    Returns:
        tuple: (correct_count, total_count)
            - correct_count (int): 正确预测的数量
            - total_count (int): 样本总数
    """
    _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
    total = targets.size(0)                    # 样本总数
    correct = predicted.eq(targets.data).cpu().sum().item()  # 正确预测的数量
    
    return correct / total

def load_combinenet_model(args):
    # net = resnet19(num_classes=args.num_classes,width_mult=args.width_mult)
    net = resnet19(num_classes=args.num_classes,width_mult=args.width_mult)
    # net = timm.create_model(args.tea_arch, pretrained=True, num_classes=args.num_classes, verify=False)
    # args_dim_dict = {
    #     'swin_small_patch4_window7_224':[192,384,768],
    #     'swin_base_patch4_window7_224':[256,512,1024],
    # }
    # channel_dim_list = args_dim_dict[args.tea_arch]
    # da_module = MultiStageFeatureModule(nb_class=args.num_classes,channel_dim_list=channel_dim_list)
    # net = CombineNet(model=backbone, stage_module=da_module)
    state_dict = torch.load(args.pre_trained_tea_path, map_location=torch.device('cpu'))
    
    net.load_state_dict(state_dict['state_dict'])
    freeze_model_parameters(net)
    return net

def sample_roi_from_prob(images, activate_map, labels, num_rois=5, roi_size=(32, 32)):
    """
    Args:
        images: Tensor of shape (b, c, h, w)
        activate_map: Tensor of shape (b, 1, h, w)
        labels: Tensor of shape (b,) or (b, num_classes)
        num_rois: number of ROIs to sample per image
        roi_size: size of each ROI (height, width)

    Returns:
        roi_images: Tensor of shape (b*num_rois, c, roi_h, roi_w)
        roi_labels: Tensor of shape (b*num_rois,) or (b*num_rois, num_classes)
    """
    b, _, h, w = activate_map.shape
    device = activate_map.device

    # Flatten and normalize activation map
    prob = activate_map.view(b, -1)
    prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-8)

    # Sample pixel positions
    sampler = torch.distributions.Categorical(prob)
    sampled_indices = sampler.sample(sample_shape=(num_rois,)).T  # (b, num_rois)

    # Convert to coordinates
    ys = sampled_indices // w
    xs = sampled_indices % w

    # Create boxes [x1, y1, x2, y2]
    roi_size_half_h = roi_size[0] // 2
    roi_size_half_w = roi_size[1] // 2

    x1 = (xs - roi_size_half_w).clamp(0, w)
    y1 = (ys - roi_size_half_h).clamp(0, h)
    x2 = (xs + roi_size_half_w).clamp(0, w)
    y2 = (ys + roi_size_half_h).clamp(0, h)

    rois = torch.stack([x1, y1, x2, y2], dim=-1).float()  # (b, num_rois, 4)

    # Add batch indices for roi_align
    batch_indices = torch.arange(b, device=device).repeat_interleave(num_rois).view(-1, 1)
    rois = rois.view(-1, 4)
    rois = torch.cat([batch_indices, rois], dim=1)  # (b*num_rois, 5)

    # Extract ROI images
    roi_images = ops.roi_align(images, rois, output_size=roi_size)

    # Replicate labels
    if labels.dim() == 1:
        # For class indices: (b,) -> (b*num_rois,)
        roi_labels = labels.repeat_interleave(num_rois)
    elif labels.dim() == 2:
        # For one-hot or multi-label: (b, num_classes) -> (b*num_rois, num_classes)
        roi_labels = labels.unsqueeze(1).repeat(1, num_rois, 1).view(-1, labels.size(1))
    else:
        raise ValueError("Labels must be 1D or 2D")

    return roi_images, roi_labels



def compute_cam(feature_maps, class_weights, predicted_classes):
    """
    Args:
        feature_maps: Tensor of shape (B, D, H, W)
        class_weights: Tensor of shape (C, D) -> 来自 linear 层权重
        predicted_classes: Tensor of shape (B,)
        size_upsample: tuple (H_up, W_up)

    Returns:
        cams: Tensor of shape (B, H_up, W_up)
    """
    B, D, H, W = feature_maps.shape
    device = feature_maps.device

    # Step 1: 获取每个样本对应的类别的权重 (B, D)
    weights = class_weights[predicted_classes]  # shape: (B, D)

    # Step 2: 加权求和：(B, D, H, W) * (B, D, 1, 1) -> (B, H, W)
    cams = (feature_maps * weights.view(B, D, 1, 1)).sum(dim=1)  # shape: (B, H, W)

    # Step 3: ReLU 激活
    cam = torch.relu(cams)
        # 激活值归一化处理
    cam = cam - cam.min()  # 确保非负
    cam_img = cam / (cam.max() + 1e-3)  # 归一化到 [0, 1]
    # Step 4: 归一化到 [0, 1]
    # cam_mins = cams.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
    # cam_maxs = cams.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
    # cams = (cams - cam_mins) / (cam_maxs - cam_mins + 1e-8)

    # Step 5: 上采样到指定尺寸 (使用 interpolate)
    # cams = cams.unsqueeze(1)  # shape: (B, 1, H, W)
    # cams = F.interpolate(cams, size=size_upsample, mode='bilinear', align_corners=False)
    # cams = cams.squeeze(1)  # shape: (B, H_up, W_up)

    return cam_img


def compute_entropy(logits, is_temporal=False):
    """
    计算输出 logits 的熵。
    
    参数：
        logits: Tensor，形状为 [batch_size, num_classes]（非时序）或 [batch_size, timesteps, num_classes]（时序）。
        is_temporal: bool，是否为时序数据（SNN）。
    
    返回：
        entropy: 标量，平均熵值。
    """
    # 转换为概率分布
    prob = torch.softmax(logits, dim=-1)  # 沿类别维度 softmax
    
    # 避免 log(0)，添加小常数
    prob = prob + 1e-10
    
    # 计算熵：-sum(p * log(p))
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)  # 形状为 [batch_size] 或 [batch_size, timesteps]
    
    if is_temporal:
        # 时序数据：对时间步和批量取平均
        entropy = entropy.mean(dim=[0, 1])  # 平均熵，标量
    else:
        # 非时序数据：对批量取平均
        entropy = entropy.mean(dim=0)  # 平均熵，标量
    
    return entropy.item()


class LitModel(pl.LightningModule):
    def __init__(self,args=None):
        super().__init__()
        
        self.save_hyperparameters()
        self.max_epoch = args.max_epochs
        # self.learning_rate = args.learning_rate
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) 
        self.noise_weight = args.noise_weight
        # self.top_k = args.top_k
        self.kd_weight = args.kd_weight
        # self.bkd_criterion = BKDLoss(teacher_channels=512, student_channels=512, lambda_mgd=0.15, alpha_mgd=7e-4, use_clip=False)
        
        # self.lif =  neuron.LIFNode(step_mode='m')
        # self.feature_extractor = create_qk_former(args)
        # if args.name == 'resformer':
        #     self.feature_extractor = create_resformer(args)
        # elif args.name == 'qkformer':
        #     self.feature_extractor = create_qk_former(args)
        # sew_resnet19(num_classes=10, width_mult=8, connect_f='ADD',T=1) connect_f='ADD'
        # self.feature_extractor = snn_resnet19(num_classes=args.num_classes, width_mult=args.width_mult,T=args.T)
        self.feature_extractor = snn_resnet19(num_classes=args.num_classes, width_mult=args.width_mult,T=args.T)
        print(self.feature_extractor)
        self.teacher = load_combinenet_model(args)
        # self.teacher = resnet19(num_classes=args.num_classes,width_mult=args.width_mult)
        ### load teacher model checkpoint
        
        self.crop_size = args.input_size
        self.hyper_cam = args.hyper_cam
        
        
    # will be used during inference
    def forward(self, x,return_inter=True):
        x = self.feature_extractor(x,return_inter=return_inter)
        return x
    
    def training_step(self, batch):
        # batch = batchs['image']
        # gt = batchs['label']
        batch, gt = batch[0], batch[1]
        # grad_cam = batchs['gradcam']
        
        out,mid_out_s = self.forward(batch,return_inter=True)
        mid_spike = mid_out_s[-1].permute(1, 0, 2, 3, 4)
        # print(mid_spike)
        stu_predicted_class = out.argmax(dim=1)
        stu_class_weights = self.feature_extractor.fc1.weight.data
        spike_activate_map = compute_cam(mid_spike.mean(0), stu_class_weights, stu_predicted_class)

        # spike_activate_map = getForwardCAM(mid_spike).unsqueeze(1)
        
        # layer3_out_fire_rate = F.interpolate(spike_activate_map, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
        out_t,feat = self.teacher(batch,return_inter=True)
        # bkd_loss = self.bkd_criterion(mid_spike.mean(0),feat[-1])
        feature_map = feat[-1]
        predicted_class = out_t.argmax(dim=1)
        class_weights = self.teacher.linear.weight.data  # shape: (num_classes, D)  7e-4*bkd_loss 
        grad_cam = compute_cam(feature_map, class_weights, predicted_class)
        # print(spike_activate_map)
        l3 = compute_kl_divergence(spike_activate_map, grad_cam, Tau=2.0)
        #  + 14.5*l3
        #  + soft_loss_smooth(out,out_t,noise_weight=self.noise_weight) + 0.01*l3
        # + 0.01*l3 + soft_loss_smooth(out,out_t,noise_weight=self.noise_weight)
        # + 0.001*l3+ 0.01*l3 + soft_loss_smooth(out,out_t,noise_weight=self.noise_weight)
        # 0.01*
        noise_loss,stu_noise_ent =  soft_loss_smooth(out,out_t,noise_weight=self.noise_weight,temperature=2.0)
        loss =  self.criterion(out, gt)  + l3 + noise_loss
        acc = calculate_accuracy(out, gt)
        ## compute the self.teacher.average entropy
        teacher_entropy = compute_entropy(out_t/2, is_temporal=False)
        student_entropy = compute_entropy(out/2, is_temporal=False)
        self.log("train/teacher_entropy", teacher_entropy)
        self.log("train/student_entropy", stu_noise_ent)
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        # functional.reset_net(self.feature_extractor)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch,return_inter=False)
        loss = self.criterion(out, gt)
        self.log("val/loss", loss)
        acc = calculate_accuracy(out, gt)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)
        ## strand snn 
        # functional.reset_net(self.feature_extractor)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch,return_inter=False)
        loss = self.criterion(out, gt)
        ## strand snn 
        # functional.reset_net(self.feature_extractor)
        return {"loss": loss, "outputs": out, "gt": gt}
    
    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
        gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
        self.log("test/loss", loss)
        acc = calculate_accuracy(output, gts)
        self.log("test/acc", acc)
        
        self.test_gts = gts
        self.test_output = output
        ## strand snn 
        # functional.reset_net(self.feature_extractor)

        
    def configure_optimizers(self):
        # Warmup 参数
        # # 创建调度器
        optimizer = torch.optim.SGD(
            # nn.ModuleList([self.feature_extractor,self.bkd_criterion]).parameters(),
            # nn.ModuleList([self.feature_extractor,self.conv1,self.conv2,self.conv3]).parameters(), lr=self.learning_rate, 
            self.feature_extractor.parameters(), 
            lr=self.learning_rate,  momentum=0.9,
            #  lr=1e-4,
            weight_decay=1e-4
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     patience=3, 
        #     factor=0.5, 
        #     verbose=True
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=self.max_epoch)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }



def parse_args():
    import argparse
    import yaml
    from config.config import parse_args_yml
    def print_yaml_content(file_path):
        with open(file_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
            print(yaml.dump(yaml_content, default_flow_style=False))
        

    parser = argparse.ArgumentParser(description='Parse YAML config and optionally print content')
    parser.add_argument('--print-yaml', action='store_true', help='Print raw YAML content')
    parser.add_argument('--config', default='config/ckd/plt_train_kd_ckd_snn.yml', help='Path to YAML config file')
    # args = parse_args_yml('config/nabrids/plt_our_kdtrain_snn_nabrids.yml')
    # plt_erkdtrain_snn_nabrids.yml
    args = parser.parse_args()
    if args.print_yaml:
        print_yaml_content(args.config)
    config_args = parse_args_yml(args.config)
    print(vars(config_args))  # Print parsed args
    return config_args

def main():
    args = parse_args()
    logger_name = args.name_space
    logger = CSVLogger(args.checkpoint_dir, name=logger_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",           # 监控验证集准确率
        mode="max",                  # 追踪最大值
        save_top_k=1,                # 保存最佳模型
        verbose=True,                # 输出日志
        filename="best_model"        # 文件名
    )

    dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, data_type=args.data_type)
    # dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size)
    # set fc layer of model with exact class number of current dataset
    model = LitModel(args=args)
    
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, devices=1, accelerator="gpu",callbacks=[checkpoint_callback], precision=args.mixed, gradient_clip_val=0)
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()

## nohup python3 plt_train_kd_ckd_snn.py --config config/cifar100/plt_train_kd_ckd_snn.yml &
## python3 