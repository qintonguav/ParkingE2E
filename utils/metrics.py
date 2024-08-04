
import numpy as np
import torch

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryDistance, detokenize_traj_point


class CustomizedMetric:
    def __init__(self, cfg: Configuration, pred_traj_point, batch) -> None:
        self.cfg = cfg
        self.BOS_token = self.cfg.token_nums
        self.distance_dict = self.calculate_distance(pred_traj_point, batch)

    def calculate_distance(self, pred_traj_point, batch):
        distance_dict = {}

        if self.cfg.decoder_method == "transformer":
            prediction_points = self.get_predict_points(pred_traj_point)
            gt_points = self.get_gt_points(batch['gt_traj_point_token'])
            prediction_points = prediction_points.reshape(self.cfg.batch_size, -1, self.cfg.item_number)
            gt_points = gt_points.reshape(self.cfg.batch_size, -1, self.cfg.item_number)
            valid_mask = ((gt_points < self.BOS_token) & (prediction_points < self.BOS_token)).all(dim=-1)
            prediction_points_np = []
            gt_points_np = []
            for index in range(self.cfg.batch_size):
                prediction_points_np.append(self.get_valid_np_points(prediction_points[index], valid_mask[index]))
                gt_points_np.append(self.get_valid_np_points(gt_points[index], valid_mask[index]))
        elif self.cfg.decoder_method == "gru":
            prediction_points_np = np.array(pred_traj_point.view(-1, 2).cpu())
            gt_points_np = np.array(batch['gt_traj_point'].view(-1, 2).cpu())
            prediction_points_np = prediction_points_np.reshape(self.cfg.batch_size, -1, self.cfg.item_number)
            gt_points_np = gt_points_np.reshape(self.cfg.batch_size, -1, self.cfg.item_number)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")

        l2_list, haus_list, fourier_difference = [], [], []
        for index in range(self.cfg.batch_size):
            distance_obj = TrajectoryDistance(prediction_points_np[index], gt_points_np[index])
            if distance_obj.get_len() < 1:
                continue
            l2_list.append(distance_obj.get_l2_distance())
            if distance_obj.get_len() > 1:
                haus_list.append(distance_obj.get_haus_distance())
                fourier_difference.append(distance_obj.get_fourier_difference())
        if len(l2_list) > 0:
            distance_dict.update({"L2_distance": np.mean(l2_list)})
        if len(haus_list) > 0:
            distance_dict.update({"hausdorff_distance": np.mean(haus_list)})
        if len(fourier_difference) > 0:
            distance_dict.update({"fourier_difference": np.mean(fourier_difference)})
        return distance_dict

    def get_valid_np_points(self, torch_points, valid_mask):
        torch_points_valid = torch_points[valid_mask]
        torch_points_valid_detoken = detokenize_traj_point(torch_points_valid, 
                                                           token_nums=self.cfg.token_nums, 
                                                           item_num=self.cfg.item_number, 
                                                           xy_max=self.cfg.xy_max)
        torch_points_valid_detoken = torch_points_valid_detoken[:, :2]
        np_points_valid_detoken = np.array(torch_points_valid_detoken.cpu())
        return np_points_valid_detoken

    def get_predict_points(self, pred_traj_point):
        prediction = torch.softmax(pred_traj_point, dim=-1)
        prediction = prediction[:, :-2, :]
        prediction = prediction.argmax(dim=-1).view(-1, self.cfg.item_number)
        return prediction

    def get_gt_points(self, batch_gt_points):
        return batch_gt_points[:, 1:-2].reshape(-1).view(-1, self.cfg.item_number)