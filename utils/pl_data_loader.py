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
                 input_size: int = 256, crop_size: int = 256, num_works: int = 14,cam_path=None):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.cam_path = cam_path
        # self.num_classes = num_class
        self.num_works = num_works

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
            # transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size, padding=8),
            # Random horizontal flip
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
        self.train = GradCamDataset_Load(images, labels, gradcam_results, transform=self.augmentation)
        self.test = datasets.ImageFolder(root=self.test_dir, transform=self.transform)

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
    
class PlDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir: str = './', test_dir: str = './', 
                 input_size: int = 256, crop_size: int = 224, num_works: int = 14):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_size = input_size
        # self.num_classes = num_class
        self.num_works = num_works

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomCrop(crop_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.save_hyperparameters()  # 保存超参数以便复现

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train = datasets.ImageFolder(root=self.train_dir, transform=self.augmentation)
        self.test = datasets.ImageFolder(root=self.test_dir, transform=self.transform)

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