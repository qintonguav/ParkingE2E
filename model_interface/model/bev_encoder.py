import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet18

from utils.config import Configuration


class BevEncoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        trunk = resnet18(weights=None, zero_init_residual=True)

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.max_pool = trunk.maxpool

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

    def forward(self, x, flatten=True):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if flatten:
            x = torch.flatten(x, 2)
        return x


class BevQuery(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.query_en_dim, nhead=self.cfg.query_en_heads, batch_first=True, dropout=self.cfg.query_en_dropout)
        self.tf_query = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.query_en_layers)

        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.query_en_bev_length, self.cfg.query_en_dim) * .02)


        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, tgt_feature, img_feature):
        assert tgt_feature.shape == img_feature.shape
        batch_size, channel, h, w = tgt_feature.shape

        tgt_feature = tgt_feature.view(batch_size, channel, -1)
        img_feature = img_feature.view(batch_size, channel, -1)
        tgt_feature = tgt_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]
        img_feature = img_feature.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]

        tgt_feature = tgt_feature + self.pos_embed
        img_feature = img_feature + self.pos_embed

        bev_feature = self.tf_query(tgt_feature, memory=img_feature)
        bev_feature = bev_feature.permute(0, 2, 1)

        bev_feature = bev_feature.view(batch_size, channel, h, w)
        return bev_feature


# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y)
#         return x * y


# class BEVTfEncoder(nn.Module):
#     def __init__(self, cfg: Configuration, input_dim):
#         super().__init__()
#         self.cfg = cfg

#         tf_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=self.cfg.tf_en_heads)
#         self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=self.cfg.tf_en_layers)

#         self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.tf_en_bev_length, input_dim) * .02)
#         self.pos_drop = nn.Dropout(self.cfg.tf_en_dropout)

#         self.init_weights()

#     def init_weights(self):
#         for name, p in self.named_parameters():
#             if 'pos_embed' in name:
#                 continue
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         trunc_normal_(self.pos_embed, std=.02)

#     def forward(self, bev_feature, mode):
#         bev_feature = bev_feature
#         if mode == "train":
#             bev_feature = self.pos_drop(bev_feature)
#         bev_feature = bev_feature.transpose(0, 1)
#         bev_feature = self.tf_encoder(bev_feature)
#         bev_feature = bev_feature.transpose(0, 1)
#         return bev_feature
