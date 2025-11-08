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

import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
from torchmetrics import Accuracy

from torchvision import transforms
# from torchvision.datasets import StanfordCars
# from torchvision.datasets.utils import download_url
import torchvision.models as models
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
# from utils.pl_data_loader import PlDataModule
from utils.pl_data_cifar_loader import PlDataModule
from utils.label_smoothing import LabelSmoothingLoss
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler, autocast
import timm

import os

os.environ['CURL_CA_BUNDLE'] = ''

from ann_models.resnet import resnet19,resnet20

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
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

def calculate_accuracy_with_soft_targets(outputs, targets):
    """
    计算带有软标签的准确率
    
    参数:
    - outputs: 模型预测输出 [batch_size, num_classes]
    - targets: Mixup后的软标签 [batch_size, num_classes]
    """
    # 对于预测，选择概率最高的类别
    _, predicted = outputs.max(1)
    
    # 对于软标签，也选择概率最高的类别
    _, target_max = targets.max(1)
    
    # 计算准确率
    correct = predicted.eq(target_max).sum().item()
    total = targets.size(0)
    
    return correct / total
## fine-tune resnet
class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        arch = args.arch
        # self.args = args
        print(f"..... train {arch} ...")
        # self.feature_extractor = model
        self.save_hyperparameters()
        self.max_epoch = args.max_epochs
        # self.learning_rate = args.learning_rate 0.05/256 * 128
        self.learning_rate =  args.learning_rate 
        self.num_classes = args.num_classes
        self.criterion = nn.CrossEntropyLoss() 
        # 0.05
        #  timm.create_model(arch, pretrained=True, num_classes=args.num_classes, verify=False)
        # self.feature_extractor = resnet19(num_classes=args.num_classes,width_mult=args.width_mult)
        self.feature_extractor = resnet20(num_classes=args.num_classes,width_mult=args.width_mult)
        print(self.feature_extractor)
        # self.fc_loss = FocalLoss()
        
    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        # if self.mixup_fn is not None:
        #     batch, gt = self.mixup_fn(batch, gt)
        
        out = self.forward(batch)
        # print(out.shape)
        loss = self.criterion(out, gt)

        acc = calculate_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # criterion = torch.nn.CrossEntropyLoss()
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        

        acc = calculate_accuracy(out, gt)
        self.log('val/acc', acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)
        
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
    
    def configure_optimizers(self):
        # Warmup 参数
        # # 创建调度器
        optimizer = torch.optim.SGD(
            self.feature_extractor.parameters(), 
            lr=self.learning_rate, 
            momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=self.max_epoch)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }

# # 设置 TensorBoardLogger
def parse_args():
    from config.config import parse_args_yml
    # args = parse_args_yml('config/aircraft/plt_train_ann_aircraft.yml')
    args = parse_args_yml('config/cifar100/resnet20.yml')
    # args = parse_args_yml('config/dogs/plt_train_ann_dogs.yml')
    # args = parse_args_yml('config/nabrids/plt_train_ann_nabrids.yml')
    # args = parse_args_yml('config/food_101/plt_train_ann_foods.yml')
    # args = parse_args_yml('config/cubs200/plt_train_ann_cub.yml')
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

    dm = PlDataModule(batch_size=args.batch_size, train_dir=args.train_dir, test_dir=args.test_dir, input_size=args.input_size, data_type=args.data_type)
    # set fc layer of model with exact class number of current dataset
    model = LitModel(args=args)
    
    # Lightning
    if args.is_distributed:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, accelerator="gpu",callbacks=[checkpoint_callback],strategy="ddp",precision=args.mixed)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs,
                             devices=1,  # 限制为单 GPU
                             accelerator="gpu",callbacks=[checkpoint_callback],
                             precision=args.mixed,gradient_clip_val=0
                             )
    trainer.fit(model, dm)
    print("end....")

if __name__ == "__main__":
    main()