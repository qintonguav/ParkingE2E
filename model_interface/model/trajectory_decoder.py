import torch
from torch import nn
from timm.models.layers import trunc_normal_

from utils.config import Configuration


class TrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.embedding = nn.Embedding(self.cfg.token_nums + self.cfg.append_token, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        item_cnt = self.cfg.autoregressive_points

        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.item_number*item_cnt + 2, self.cfg.tf_de_dim) * .02)

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums + self.cfg.append_token)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def create_mask(self, tgt):
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (tgt == self.PAD_token)

        return tgt_mask, tgt_padding_mask

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask):
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        pred_traj_points = self.tf_decoder(tgt=tgt_embedding,
                                        memory=encoder_out,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)
        pred_traj_points = pred_traj_points.transpose(0, 1)
        return pred_traj_points

    def forward(self, encoder_out, tgt):
        tgt = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)

        pred_traj_points = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)
        return pred_traj_points
    
    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding_num = self.cfg.item_number * self.cfg.autoregressive_points + 2 - length
        
        offset = 1
        if padding_num > 0:
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to('cuda')
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + self.pos_embed

        pred_traj_points = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)[:, length - offset, :]

        pred_traj_points = torch.softmax(pred_traj_points, dim=-1)
        pred_traj_points = pred_traj_points.argmax(dim=-1).view(-1, 1)
        return pred_traj_points