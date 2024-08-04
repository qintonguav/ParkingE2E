import datetime
import os
from dataclasses import dataclass
from typing import List

import torch
import yaml
from loguru import logger


@dataclass
class Configuration:
    data_mode: str
    num_gpus: int
    cuda_device_index: str
    data_dir: str
    log_root_dir: str
    checkpoint_root_dir: str
    log_every_n_steps: int
    check_val_every_n_epoch: int

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_workers: int

    training_dir: str
    validation_dir: str
    autoregressive_points: int
    item_number: int
    token_nums: int
    xy_max: float
    process_dim: List[int]

    use_fuzzy_target: bool
    bev_encoder_in_channel: int

    bev_x_bound: List[float]
    bev_y_bound: List[float]
    bev_z_bound: List[float]
    d_bound: List[float]
    final_dim: List[int]
    bev_down_sample: int
    backbone: str

    tf_de_dim: int
    tf_de_heads: int
    tf_de_layers: int
    tf_de_dropout: float

    append_token: int
    traj_downsample_stride: int

    add_noise_to_target: bool
    target_noise_threshold: float

    fusion_method: str
    decoder_method: str
    query_en_dim: int
    query_en_heads: int
    query_en_layers: int
    query_en_dropout: float
    query_en_bev_length: int
    target_range: float

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_path: str = None
    config_path: str = None
    log_dir: str = None
    checkpoint_dir: str = None
    use_depth_distribution: bool = False
    tf_en_motion_length: str = None


@dataclass
class InferenceConfiguration:
    model_ckpt_path: str
    training_config: str
    predict_mode: str

    trajectory_pub_frequency: int
    cam_info_dir: str
    progress_threshold: float

    train_meta_config: Configuration = None

def get_train_config_obj(config_path: str):
    exp_name = get_exp_name()
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            config_obj = Configuration(**config_yaml)
            config_obj.config_path = config_path
            config_obj.log_dir = os.path.join(config_obj.log_root_dir, exp_name)
            config_obj.checkpoint_dir = os.path.join(config_obj.checkpoint_root_dir, exp_name)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    return config_obj


def get_exp_name():
    today = datetime.datetime.now()
    today_str = "{}_{}_{}_{}_{}_{}".format(today.year, today.month, today.day,
                                           today.hour, today.minute, today.second)
    exp_name = "exp_{}".format(today_str)
    return exp_name

def get_inference_config_obj(config_path: str):
    with open(config_path, 'r') as yaml_file:
        try:
            config_yaml = yaml.safe_load(yaml_file)
            inference_config_obj = InferenceConfiguration(**config_yaml)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", config_path)
    training_config_path = os.path.join(os.path.dirname(config_path), "{}.yaml".format(inference_config_obj.training_config))
    inference_config_obj.train_meta_config = get_train_config_obj(training_config_path)
    return inference_config_obj