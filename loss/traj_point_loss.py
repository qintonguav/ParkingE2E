from torch import nn

from utils.config import Configuration


class TokenTrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TokenTrajPointLoss, self).__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.PAD_token)

    def forward(self, pred, data):
        pred = pred[:, :-1,:]
        pred_traj_point = pred.reshape(-1, pred.shape[-1])
        gt_traj_point_token = data['gt_traj_point_token'][:, 1:-1].reshape(-1).cuda()

        traj_point_loss = self.ce_loss(pred_traj_point, gt_traj_point_token)
        return traj_point_loss


class TrajPointLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(TrajPointLoss, self).__init__()
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, data):
        gt = data['gt_traj_point'].view(-1, self.cfg.autoregressive_points, 2)
        traj_point_loss = self.mse_loss(pred, gt)
        return traj_point_loss