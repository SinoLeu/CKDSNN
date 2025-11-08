# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# if Dataset == 'CIFAR10':
#     train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
#     test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
# elif Dataset =='CIFAR100':
#     train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
#     test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
# from timm import create_model
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from utils.save_cam import GradCamDataset_Load
import h5py

class PlCAMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir: str = './', test_dir: str = './', 
                 input_size: int = 256, crop_size: int = 256, num_works: int = 14,cam_path=None, data_type='CIFAR10'):
        super().__init__()
        self.data_type = data_type
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.cam_path = cam_path
        
        # self.num_classes = num_class
        self.num_works = num_works

        # Augmentation policy for training set
        self.transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # self.norm_transform = transforms.Compose([
        #     # transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.save_hyperparameters()  # 保存超参数以便复现

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        with h5py.File(self.cam_path, "r") as f:
            images = f["images"][:]
            labels = f["labels"][:]
            gradcam_results = f["gradcam_results"][:]
        # self.train = datasets.ImageFolder(root=self.train_dir, transform=self.augmentation)
        self.train = GradCamDataset_Load(images, labels, gradcam_results, transform=self.transform_train)
        # self.test = datasets.ImageFolder(root=self.test_dir, transform=self.transform)
        if self.data_type == 'CIFAR10':
            self.test = datasets.CIFAR10(root=self.test_dir, train=False, download=True, transform=self.transform_test)
        elif self.data_type == 'CIFAR100':
            self.test = datasets.CIFAR100(root=self.test_dir, train=False, download=True, transform=self.transform_test)

    def train_dataloader(self):
        # 动态检查是否为分布式训练
        if self.trainer.accelerator == "gpu" and self.trainer.num_devices > 1 and self.trainer.strategy == "ddp":
            sampler = DistributedSampler(
                self.train,
                num_replicas=self.trainer.num_devices,  # GPU 数量
                rank=self.trainer.global_rank,          # 当前进程的 rank
                shuffle=True                            # 保持随机性
            )
            shuffle = True  # sampler 已处理随机性
        else:
            sampler = None
            shuffle = True
        train_loader = DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_works,sampler=sampler
        )
        
        return train_loader

    def val_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_works
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_works
        )

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# 自定义 Dataset
class CIFAR5M_Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        初始化 Dataset
        :param file_path: .npz 文件路径
        :param transform: 图像数据的变换操作
        """
        # 加载 .npz 文件
        data = np.load(file_path)
        self.images = data['X']  # 图像数据
        self.labels = data['Y']  # 标签数据
        
        
        # 转换为 Tensor
        # self.images = torch.from_numpy(self.images).float()  # 转为 float32 Tensor
        self.labels = torch.from_numpy(self.labels).long()  # 转为 long Tensor
        
        self.transform = transform  # 图像变换（如归一化、随机裁剪等）

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        """
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image.astype('uint8'))  # 确保数据类型是 uint8
        # print(f"image shape: {image.shape}, label shape: {label.shape}")
        # 如果有 transform，对图像进行变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

class PlDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir: str = './', test_dir: str = './',
                 input_size: int = 32, num_works: int = 14, data_type='CIFAR10'):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_size = input_size
        # self.num_classes = num_class
        self.num_works = num_works
        self.data_type = data_type
        # Augmentation policy for training set
        # self.augmentation = transforms.Compose([
        #     transforms.Resize((input_size, input_size)),
        #     transforms.RandomCrop(crop_size, padding=8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        # self.transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪到 32x32，外加 4 像素的填充
            transforms.RandomHorizontalFlip(),    # 随机水平翻转
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  # 应用 AutoAugment 策略（专为 CIFAR-10 设计）
            transforms.ToTensor(),                # 转换为张量
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 数据归一化
            transforms.RandomErasing(scale=(0.02, 0.2)),  # Cutout 的实现，随机遮挡部分区域
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.save_hyperparameters()  # 保存超参数以便复现

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.data_type == 'CIFAR10':
            # self.train = CIFAR5M_Dataset(file_path=self.train_dir, transform=self.transform_train)
            self.train = datasets.CIFAR10(root=self.train_dir, train=True, download=True, transform=self.transform_train)
            self.test = datasets.CIFAR10(root=self.test_dir, train=False, download=True, transform=self.transform_test)
        elif self.data_type == 'CIFAR100':
            self.train = datasets.CIFAR100(root=self.train_dir, train=True, download=True, transform=self.transform_train)
            self.test = datasets.CIFAR100(root=self.test_dir, train=False, download=True, transform=self.transform_test)
        else:
            raise ValueError("Unsupported dataset type. Please use 'CIFAR10' or 'CIFAR100'.")
        # self.train = datasets.ImageFolder(root=self.train_dir, transform=self.augmentation)
        # train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        # self.train = datasets.CIFAR10(root='../data', train=True, download=True, transform=self.transform_train)
        # # test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        # self.test = datasets.CIFAR10(root='../data', train=False, download=True, transform=self.transform_test)
        
    def train_dataloader(self):
        # 动态检查是否为分布式训练
        if self.trainer.accelerator == "gpu" and self.trainer.num_devices > 1 and self.trainer.strategy == "ddp":
            sampler = DistributedSampler(
                self.train,
                num_replicas=self.trainer.num_devices,  # GPU 数量
                rank=self.trainer.global_rank,          # 当前进程的 rank
                shuffle=True                            # 保持随机性
            )
            shuffle = True  # sampler 已处理随机性
        else:
            sampler = None
            shuffle = True
        train_loader = DataLoader(
            self.train, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_works,sampler=sampler
        )
        
        return train_loader

    def val_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_works
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_works
        )