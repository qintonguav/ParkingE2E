import os
import shutil
import sys

from loguru import logger


def init(config_obj):
    init_cuda(config_obj)
    init_log(config_obj)

def finish(config_obj):
    finish_log(config_obj)


def init_cuda(config_obj):
    os.environ['CUDA_VISIBLE_DEVICES'] = config_obj.cuda_device_index
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def init_log(config_obj):
    logger.remove()
    logger.add(config_obj.log_dir + '/training_log_{time}.log', enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stderr, enqueue=True)
    logger.info("Config File: {}", config_obj.config_path)

def save_code(cfg):
    def _ignore(path, content):
        return ["carla", "ckpt", "e2e_parking", "log"]
    project_root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    shutil.copytree(project_root_dir, os.path.join(cfg.checkpoint_dir, "code"), ignore=_ignore)
    # os.path.

def finish_log(config_obj):
    project_root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    source_log_path = os.path.join(project_root_dir, "log", os.path.basename(config_obj.checkpoint_dir))
    target_dir_path = os.path.join(project_root_dir, os.path.join(config_obj.checkpoint_dir, "code"), "log")
    target_log_path = os.path.join(target_dir_path, os.path.basename(config_obj.checkpoint_dir))
    os.makedirs(target_dir_path, exist_ok=True)
    shutil.copytree(source_log_path, target_log_path)