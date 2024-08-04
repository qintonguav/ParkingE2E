import os

import numpy as np
from PIL import Image
import torch
import torchvision

from utils.common import get_json_content
from utils.pose_utils import PoseFlow


def get_normalized_torch_image(image: Image):
    torch_convert = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return torch_convert(np.array(image)).unsqueeze(0)

def get_torch_intrinsics_or_extrinsics(intrinsic: np.array):
    return torch.from_numpy(intrinsic).float().unsqueeze(0)



class CameraInfoParser:
    def __init__(self, task_index, parser_dir):
        self.task_index = task_index
        self.parser_dir = parser_dir
        self.intrinsic, self.extrinsic = self.parser_info()

    def parser_info(self):
        camera_info = get_json_content(os.path.join(self.parser_dir, "camera_config_right_hand.json"))
        camera_info_right_hand = {}
        for camera_channel, cam_info in camera_info.items():
            camera_label = self.get_dir_name_from_channel(camera_channel)
            camera_info_right_hand[camera_label] = {
                "x": cam_info["extrinsics"]["x"], "y": cam_info["extrinsics"]["y"], "z": cam_info["extrinsics"]["z"],
                "roll": cam_info["extrinsics"]["roll"], "pitch": cam_info["extrinsics"]["pitch"], "yaw": cam_info["extrinsics"]["yaw"], 
                "width": cam_info["intrinsics"]["width"], "height": cam_info["intrinsics"]["height"], "fov": cam_info["intrinsics"]["fov"]
            }

        intrinsic = {}
        extrinsic = {}
        for cam_label, cam_spec in camera_info_right_hand.items():
            intrinsic[cam_label] = self.get_intrinsics(cam_spec["width"], cam_spec["height"], cam_spec["fov"])
            extrinsic[cam_label] = self.get_extrinsics(cam_spec["x"], cam_spec["y"], cam_spec["z"], cam_spec["yaw"], cam_spec["pitch"], cam_spec["roll"])
        return intrinsic, extrinsic

    def get_extrinsics(self, x_right, y_right, z_right, yaw_right, pitch_right, roll_right):
        cam2pixel_3 = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]], dtype=float)
        pose_flow_obj = PoseFlow(att_input=[yaw_right, pitch_right, roll_right], type="euler", deg_or_rad="deg")
        rotation_3 = pose_flow_obj.get_rotation_matrix()
        translation = np.array([[x_right], [y_right], [z_right]])
        cam2veh_3 = self.concat_rotaion_and_translation(rotation_3, translation).tolist()
        veh2cam = cam2pixel_3 @ np.array(np.linalg.inv(cam2veh_3))
        return veh2cam

    def get_intrinsics(self, width, height, fov):
        f = width / (2 * np.tan(np.deg2rad(fov) / 2))
        Cu = width / 2
        Cv = height / 2
        intrinsic = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=float)
        return intrinsic

    def concat_rotaion_and_translation(self, rotation, translation):
        ret = np.eye(4)
        ret[:3, :] = np.concatenate([rotation, translation], axis=1)
        return ret

    def get_dir_name_from_channel(self, channel_name):
        return "_".join(['rgb'] + channel_name.split("_")[1:]).lower()



class ProcessImage:
    def __init__(self, pil_image: Image, intrinsics, extrinsics, target_size=(256, 256)):
        self.pil_image = pil_image
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.image_width, self.image_height = self.pil_image.size
        self.target_width, self.target_height = target_size
        self.width_scale = self.target_width / self.image_width
        self.height_scale = self.target_height / self.image_height

        self.resize_img = None
        self.resize_intrinsics = None

    def resize_pil_image(self):
        self.resize_img = self.pil_image.resize((self.target_width, self.target_height), resample=Image.NEAREST)
        self.resize_intrinsics = self._update_intrinsics()
        return self.resize_img, self.resize_intrinsics

    def _update_intrinsics(self):
        updated_intrinsics = self.intrinsics.copy()
        # Adjust intrinsics scale due to resizing
        updated_intrinsics[0, 0] *= self.width_scale
        updated_intrinsics[0, 2] *= self.width_scale
        updated_intrinsics[1, 1] *= self.height_scale
        updated_intrinsics[1, 2] *= self.height_scale

        return updated_intrinsics