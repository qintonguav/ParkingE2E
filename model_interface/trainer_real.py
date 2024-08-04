import pytorch_lightning as pl
import torch

from loss.traj_point_loss import TokenTrajPointLoss, TrajPointLoss
from model_interface.model.parking_model_real import ParkingModelReal
from utils.config import Configuration
from utils.metrics import CustomizedMetric


class ParkingTrainingModuleReal(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(ParkingTrainingModuleReal, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.traj_point_loss_func = self.get_loss_function()
        
        self.parking_model = ParkingModelReal(self.cfg)


    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)

        train_loss = self.traj_point_loss_func(pred_traj_point, batch)        

        loss_dict.update({"train_loss": train_loss})

        self.log_dict(loss_dict)

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)

        val_loss = self.traj_point_loss_func(pred_traj_point, batch)

        val_loss_dict.update({"val_loss": val_loss})

        customized_metric = CustomizedMetric(self.cfg, pred_traj_point, batch)
        val_loss_dict.update(customized_metric.calculate_distance(pred_traj_point, batch))

        self.log_dict(val_loss_dict)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.learning_rate,
                                     weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_loss_function(self):
        traj_point_loss_func = None
        if self.cfg.decoder_method == "transformer":
            traj_point_loss_func = TokenTrajPointLoss(self.cfg)
        elif self.cfg.decoder_method == "gru":
            traj_point_loss_func = TrajPointLoss(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        return traj_point_loss_func