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
    # freeze_model_parameters(net)
    return net

def compute_cam_with_grad(feature_maps,gradients):
    """
    Compute Grad-CAM using backpropagated gradients.
    
    Args:
        feature_maps: Tensor of shape (B, D, H, W)
        class_weights: Tensor of shape (C, D) 
        predicted_classes: Tensor of shape (B,)
        gradients: Tensor of shape (B, D, H, W) -> ∂y_pred/∂feature_maps

    Returns:
        cam_img: Tensor of shape (B, H, W), values normalized to [0, 1]
    """
    B, D, H, W = feature_maps.shape

    # Step 1: Global average pooling of gradients to get channel importance weights
    # (B, D, H, W) -> (B, D)
    weights = torch.nn.functional.adaptive_avg_pool2d(gradients, (1, 1)).view(B, D)

    # Step 2: Weighted sum over channels
    # (B, D, H, W) * (B, D, 1, 1) -> (B, H, W)
    cams = (feature_maps * weights.view(B, D, 1, 1)).sum(dim=1)  # (B, H, W)

    # Step 3: ReLU + normalize to [0, 1]
    cams = torch.relu(cams)
    cams = cams - cams.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # per-sample min
    cams = cams / (cams.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

    return cams


# compute cam for last feature map
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
    
    return cam_img

import torch

def forward_with_gradients(
    model,
    inputs,
    target_classes=None,
    # return_feature_map=True,
    feature_layer_index=-1
):
    """
    Compute gradients of target class scores w.r.t. the last (or specified) feature map
    for Grad-CAM.

    Args:
        model (nn.Module): Model that supports return_inter=True to output intermediate features.
        inputs (Tensor): Input batch of shape (B, C, H_in, W_in).
        target_classes (Tensor, optional): Long tensor of shape (B,). 
            If None, uses argmax of model output (i.e., predicted classes).
        return_feature_map (bool): If True, also return the detached feature map.
        feature_layer_index (int): Index into the list of intermediate features (default: -1 for last).

    Returns:
        gradients (Tensor): Shape (B, D, H, W), gradients w.r.t. feature map.
        feature_map (Tensor, optional): Shape (B, D, H, W), detached feature map.
    """
    model.eval()  # Ensure batchnorm/dropout behave deterministically
    with torch.enable_grad():
        model.train()  # 或 eval()，但不能有 no_grad
        outputs, features = model(inputs, return_inter=True)
        feature_map = features[feature_layer_index]  # (B, D, H, W)
        # print("feature_map.requires_grad:", feature_map.requires_grad)
        # print("feature_map has grad_fn:", feature_map.grad_fn is not None)
        if target_classes is None:
            target_classes = outputs.argmax(dim=1)  # (B,)

        # Gather target class logits
        batch_size = outputs.size(0)
        target_scores = outputs[torch.arange(batch_size), target_classes]  # (B,)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=feature_map,
            retain_graph=True,
            create_graph=False        
        )[0]

    return grads, outputs, feature_map.detach()

    

class LitModel(pl.LightningModule):
    def __init__(self,args=None):
        super().__init__()
        
        self.save_hyperparameters()
        self.max_epoch = args.max_epochs
        # self.learning_rate = args.learning_rate
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.noise_weight = args.noise_weight
        self.kd_weight = args.kd_weight
        
        self.feature_extractor = snn_resnet19(num_classes=args.num_classes, width_mult=args.width_mult,T=args.T)
        print(self.feature_extractor)
        self.teacher = load_combinenet_model(args)
        
        self.crop_size = args.input_size
        self.hyper_cam = args.hyper_cam
        
        
    # will be used during inference
    def forward(self, x,return_inter=True):
        x = self.feature_extractor(x,return_inter=return_inter)
        return x
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]

        out,mid_out_s = self.forward(batch,return_inter=True)
        mid_spike = mid_out_s[-1].permute(1, 0, 2, 3, 4)
        # from utils.utils import getForwardCAM
        spike_activate_map = getForwardCAM(mid_spike)

        

        # with no grad cam / class activate by last class weights
        out_t,feat = self.teacher(batch,return_inter=True)
        feature_map = feat[-1]
        predicted_class = out_t.argmax(dim=1)
        class_weights = self.teacher.linear.weight.data  # shape: (num_classes, D)  7e-4*bkd_loss 
        grad_cam = compute_cam(feature_map, class_weights, predicted_class)
        
        ## use grad cam with gradients / class activate by backprop gradients
        # grads, out_t, feature_map = forward_with_gradients(self.teacher, batch)
        # grad_cam = compute_cam_with_grad(feature_map, grads)
        
        
        l3 = compute_kl_divergence(spike_activate_map, grad_cam, Tau=2.0)
        
        noise_loss =  soft_loss_smooth(out,out_t,noise_weight=self.noise_weight,temperature=2.0)
        loss =  self.criterion(out, gt)  + l3 + noise_loss
        acc = calculate_accuracy(out, gt)

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
