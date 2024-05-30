import os
from typing import List

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder

class ManufacturingDM(pl.LightningDataModule):
    def __init__(self, cfg, leave_out: List=None) -> None:
        super().__init__()
        self.data_dir = cfg.data.path
        self.batch_size = cfg.param.batch_size

        train_transform = T.Compose([
            T.Resize((224,224), interpolation=T.InterpolationMode.NEAREST_EXACT),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(360, interpolation=T.InterpolationMode.NEAREST_EXACT),
            T.ToTensor(),
            # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        val_transform = T.Compose([
            T.Resize((224,224), interpolation=T.InterpolationMode.NEAREST_EXACT),
            T.ToTensor(),
            # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.train_set = ImageFolder(os.path.join(cfg.data.path, 'train'), transform=train_transform)
        self.test_set = ImageFolder(os.path.join(cfg.data.path, 'test'), transform=val_transform)

        self.cfg = cfg
        self.num_classes = self.train_set.classes.__len__()

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            pass
            
        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )
    
def main():
    pass

if __name__ == '__main__':
    main()