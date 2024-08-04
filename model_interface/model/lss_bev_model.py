from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
from torch import nn
from utils.bev_utils import DeepLabHead, UpsamplingConcat, VoxelsSumming, calculate_birds_eye_view_parameters
from utils.config import Configuration


class LssBevModel(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        bev_res, bev_start_pos, bev_dim = calculate_birds_eye_view_parameters(self.cfg.bev_x_bound,
                                                                              self.cfg.bev_y_bound,
                                                                              self.cfg.bev_z_bound)
        self.bev_res = nn.Parameter(bev_res, requires_grad=False)
        self.bev_start_pos = nn.Parameter(bev_start_pos, requires_grad=False)
        self.bev_dim = nn.Parameter(bev_dim, requires_grad=False)

        self.down_sample = self.cfg.bev_down_sample

        self.frustum = self.create_frustum()
        self.depth_channel, _, _, _ = self.frustum.shape
        self.cam_encoder = CamEncoder(self.cfg, self.depth_channel)

    def create_frustum(self):
        h, w = self.cfg.final_dim
        down_sample_h, down_sample_w = h // self.down_sample, w // self.down_sample

        depth_grid = torch.arange(*self.cfg.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, down_sample_h, down_sample_w)
        depth_slice = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, down_sample_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, down_sample_w).expand(depth_slice, down_sample_h, down_sample_w)
        y_grid = torch.linspace(0, h - 1, down_sample_h, dtype=torch.float)
        y_grid = y_grid.view(1, down_sample_h, 1).expand(depth_slice, down_sample_h, down_sample_w)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinsics, extrinsics):
        extrinsics = torch.inverse(extrinsics).cuda()
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        b, n, _ = translation.shape

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        combine_transform = rotation.matmul(torch.inverse(intrinsics)).cuda()
        points = combine_transform.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(b, n, 1, 1, 1, 3)

        return points

    def encoder_forward(self, images):
        b, n, c, h, w = images.shape
        images = images.view(b * n, c, h, w)
        x, depth = self.cam_encoder(images)

        depth_prob = None
        if self.cfg.use_depth_distribution:
            depth_prob = depth.softmax(dim=1)
            x = depth_prob.unsqueeze(1) * x.unsqueeze(2)
        else:
            x = x.unsqueeze(2).repeat(1, 1, self.depth_channel, 1, 1)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth_prob

    def proj_bev_feature(self, geom, image_feature):
        batch, n, d, h, w, c = image_feature.shape
        output = torch.zeros((batch, c, self.bev_dim[0], self.bev_dim[1]),
                             dtype=torch.float, device=image_feature.device)
        N = n * d * h * w
        for b in range(batch):
            image_feature_b = image_feature[b]
            geom_b = geom[b]

            x_b = image_feature_b.reshape(N, c)

            geom_b = ((geom_b - (self.bev_start_pos - self.bev_res / 2.0)) / self.bev_res)
            geom_b = geom_b.view(N, 3).long()

            mask = ((geom_b[:, 0] >= 0) & (geom_b[:, 0] < self.bev_dim[0])
                    & (geom_b[:, 1] >= 0) & (geom_b[:, 1] < self.bev_dim[1])
                    & (geom_b[:, 2] >= 0) & (geom_b[:, 2] < self.bev_dim[2]))
            x_b = x_b[mask]
            geom_b = geom_b[mask]

            ranks = ((geom_b[:, 0] * (self.bev_dim[1] * self.bev_dim[2])
                     + geom_b[:, 1] * self.bev_dim[2]) + geom_b[:, 2])
            sorts = ranks.argsort()
            x_b, geom_b, ranks = x_b[sorts], geom_b[sorts], ranks[sorts]

            x_b, geom_b = VoxelsSumming.apply(x_b, geom_b, ranks)

            bev_feature = torch.zeros((self.bev_dim[2], self.bev_dim[0], self.bev_dim[1], c),
                                      device=image_feature_b.device)
            bev_feature[geom_b[:, 2], geom_b[:, 0], geom_b[:, 1]] = x_b
            tmp_bev_feature = bev_feature.permute((0, 3, 1, 2)).squeeze(0)
            output[b] = tmp_bev_feature

        return output

    def calc_bev_feature(self, images, intrinsics, extrinsics):
        geom = self.get_geometry(intrinsics, extrinsics)
        x, pred_depth = self.encoder_forward(images)
        bev_feature = self.proj_bev_feature(geom, x)
        return bev_feature, pred_depth

    def forward(self, images, intrinsics, extrinsics):
        bev_feature, pred_depth = self.calc_bev_feature(images, intrinsics, extrinsics)
        return bev_feature.squeeze(1), pred_depth


class CamEncoder(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.D = D
        self.C = cfg.bev_encoder_in_channel
        self.use_depth_distribution = cfg.use_depth_distribution
        self.downsample = cfg.bev_down_sample
        self.version = cfg.backbone.split('-')[1]

        self.backbone = EfficientNet.from_pretrained(cfg.backbone)
        self.delete_unused_layers()
        if self.version == 'b4':
            self.reduction_channel = [0, 24, 32, 56, 160, 448]
        elif self.version == 'b0':
            self.reduction_channel = [0, 16, 24, 40, 112, 320]
        else:
            raise NotImplementedError
        self.upsampling_out_channel = [0, 48, 64, 128, 512]

        index = np.log2(self.downsample).astype(np.int)

        if self.use_depth_distribution:
            self.depth_layer_1 = DeepLabHead(self.reduction_channel[index + 1],
                                             self.reduction_channel[index + 1],
                                             hidden_channel=64)
            self.depth_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index],
                                                  self.D)

        self.feature_layer_1 = DeepLabHead(self.reduction_channel[index + 1],
                                           self.reduction_channel[index + 1],
                                           hidden_channel=64)
        self.feature_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index],
                                                self.C)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features_depth(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        index = np.log2(self.downsample).astype(np.int)
        input_1 = endpoints['reduction_{}'.format(index + 1)]
        input_2 = endpoints['reduction_{}'.format(index)]

        feature = self.feature_layer_1(input_1)
        feature = self.feature_layer_2(feature, input_2)

        if self.use_depth_distribution:
            depth = self.depth_layer_1(input_1)
            depth = self.depth_layer_2(depth, input_2)
        else:
            depth = None

        return feature, depth

    def forward(self, x):
        feature, depth = self.get_features_depth(x)
        
        return feature, depth

