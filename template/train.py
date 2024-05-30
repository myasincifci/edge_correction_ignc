import os
import hydra
import pytorch_lightning as L
import torch
import torch.nn as nn
from data_modules.manufacturing_dm import ManufacturingDM
from model import Regressor
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms as T
import wandb

@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    print(os.getcwd())
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")

    logger = True
    if cfg.logging:
        logger = WandbLogger(
            project=cfg.logger.project,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )

    L.seed_everything(42, workers=True)

    # Data
    data_module = ManufacturingDM(cfg)

    # Model
    model = Regressor(cfg=cfg, c_to_i=data_module.train_set.class_to_idx)

    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=5,

        accelerator="gpu",
        devices=2,
        strategy='ddp'
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )


if __name__ == "__main__":
    main()
