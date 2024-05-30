import numpy as np
import pytorch_lightning as L
import torch
from torch import nn
from torch import nn
from torch.nn import functional as F
import timm

class Regressor(L.LightningModule):
    def __init__(self, cfg, c_to_i):
        super().__init__()
        self.backbone, out_dim = get_backbone(cfg.backbone.type)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64), #128
            nn.LeakyReLU(),
            nn.Linear(64, 32), #64
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        self.lr = cfg.param.lr
        self.c_to_i = c_to_i

    def idx_to_class(self, idcs: torch.Tensor):
        reverse_dict = {v: float(k) for k, v in self.c_to_i.items()}
        _idcs = idcs.cpu().numpy()

        mapped = np.vectorize(reverse_dict.get)(_idcs)
        return torch.from_numpy(mapped).to(torch.float32).to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = F.sigmoid(x)

        x = 50.0*x - 25.0

        return x

    def training_step(self, batch, batch_idx):
        x, t = batch

        y = self(x).squeeze()
        t_ = self.idx_to_class(t)

        t_.requires_grad_(True)


        loss = F.mse_loss(y, t_)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch

        y = self(x).squeeze()
        t_ = self.idx_to_class(t)

        t_.requires_grad_(True)

        loss = F.mse_loss(y, t_)
        error = y - t_

        self.log("val/loss", loss, prog_bar=True)
        self.log('val/mean_abs_error', torch.abs(error).mean())
        self.log('val/mean_std', error.std())

        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def get_backbone(type: str):
    bb = None
    
    match type:
        case 'swin':
            bb = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            bb.head.fc = nn.Identity()
            out_dim = 768
        case _:
            raise Exception('Invalid backbone type')
        
    return bb, out_dim