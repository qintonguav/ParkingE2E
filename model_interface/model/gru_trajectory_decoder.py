import torch
from torch import nn

from utils.config import Configuration


class GRUTrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super(GRUTrajectoryDecoder, self).__init__()

        self.cfg = cfg
        self.predict_num = self.cfg.autoregressive_points
        self.hidden_size = 3000

        # waypoints prediction
        self.join = nn.Sequential(
            nn.Linear(int(self.cfg.final_dim[0] * self.cfg.final_dim[1]), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.hidden_size),
            nn.ReLU(inplace=True),
        )

        self.predict_item_number = self.cfg.item_number
        
        self.decoder = nn.GRUCell(input_size=self.predict_item_number, hidden_size=self.hidden_size)

        self.output = nn.Linear(self.hidden_size, self.cfg.item_number)
        
        self.init_weights()


    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, data=None):
        bs = z.shape[0]
        # flatten z
        z = torch.flatten(z, 1)
        z = self.join(z)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(bs, self.predict_item_number), dtype=z.dtype).cuda()
        
        # autoregressive generation of output waypoints
        for _ in range(self.predict_num):
            x_in = x
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx[:, :] + x
            output_wp.append(x[:, :])

        pred_wp = torch.stack(output_wp, dim=1)
        return pred_wp
    