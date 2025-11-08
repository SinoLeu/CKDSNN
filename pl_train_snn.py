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
from utils.pl_data_loader import PlCAMDataModule
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
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils.pl_data_loader import PlDataModule
# from models.s_model import get_sewresnet
from collections import OrderedDict
import timm
import logging
logging.getLogger().setLevel(logging.ERROR)  # 只显示ERROR及以上级别的日志


def create_resformer(args):
    from models.spikingresformer import spikingresformer_ti, spikingresformer_s, spikingresformer_m, spikingresformer_l
    arch_dict = {
        'ti':spikingresformer_ti, 's':spikingresformer_s, 'm':spikingresformer_m, 'l':spikingresformer_l
    }
    model = arch_dict[args.arch](T=args.T)
    state_dict = torch.load(f'snn_checkpoint/spikingresformer_{args.arch}.pth', map_location=torch.device('cpu'))
    state_dict = remove_module_from_state_dict(state_dict['model'])
    model.load_state_dict(state_dict,strict=False)
    
    model.classifier = nn.Linear(
        model.classifier.in_features, args.num_classes
    )  # set fc layer of model with exact class number of current dataset
    return model

def create_resnet(args):
    from models.s_model import get_sewresnet
    model = get_sewresnet(arch=args.arch,connect_f='ADD')
    state_dict = torch.load(f'snn_checkpoint/sew_resnet-50.pth', map_location=torch.device('cpu'))
    state_dict = remove_module_from_state_dict(state_dict['model'])
    model.load_state_dict(state_dict,strict=False)
    model.fc = nn.Linear(
        model.fc.in_features, args.num_classes
    )  # set fc layer of model with exact class number of current dataset
    
    return model

def create_qk_former(args):
    from models.qkformer import QKFormer_10_384,QKFormer_10_512,QKFormer_10_768
    arch_dict = {
        'ti':QKFormer_10_384,'s':QKFormer_10_512,'m':QKFormer_10_768
    }
    
    name_dict = {
        'ti':'10-384-224', 's':'10-512-224', 'm':'10-768-224'
    }
    model = arch_dict[args.arch](T=args.T)
    state_dict = torch.load(f'snn_checkpoint/HST-{name_dict[args.arch]}.pth', map_location=torch.device('cpu'))
    state_dict = remove_module_from_state_dict(state_dict['model'])
    model.load_state_dict(state_dict,strict=False)
    model.head = nn.Linear(
        model.head.in_features, args.num_classes
    )  # set fc layer of model with exact class number of current dataset
    
    return model

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



class LitModel(pl.LightningModule):
    def __init__(self,args=None):
        super().__init__()
        ## num_classes, learning_rate=1e-4,
        # self.feature_extractor = model
        self.save_hyperparameters()
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) 
        ##
        if args.name == 'resformer':
            self.feature_extractor = create_resformer(args)
        elif args.name == 'qkformer':
            self.feature_extractor = create_qk_former(args)
        elif args.name == 'resnet':
            self.feature_extractor = create_resnet(args)
        # self.feature_extractor = create_resformer(args)
        
        
    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    
    def training_step(self, batch):
        # batch, gt = batch[0], batch[1]
        # batch = batchs['image']
        # gt = batchs['label']
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)
        ## strand snn 
        functional.reset_net(self.feature_extractor)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        acc = calculate_accuracy(out, gt)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)
        ## strand snn 
        functional.reset_net(self.feature_extractor)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)
        ## strand snn 
        functional.reset_net(self.feature_extractor)
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
        functional.reset_net(self.feature_extractor)
        
    def configure_optimizers(self):
        # Warmup 参数
        # # 创建调度器
        optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=self.learning_rate, 
            #  lr=1e-4,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=3, 
            factor=0.5, 
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'  
        }


def parse_args():
    from config.config import parse_args_yml
    # args = parse_args_yml('config/aircraft/plt_train_snn_aircraft.yml')
    # args = parse_args_yml('config/dogs/plt_train_snn_dogs.yml')
    # args = parse_args_yml('config/cars196/plt_train_snn_cars.yml')
    args = parse_args_yml('config/cubs200/plt_train_snn_cub.yml')
    # args = parse_args_yml('config/nabrids/plt_train_snn_nabrids.yml')
    print(vars(args))  # Print parsed args
    return args
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

    dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size)
    # dm = PlCAMDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, crop_size=args.input_size,cam_path=args.cam_path)
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

## python3 pl_train_snn.py