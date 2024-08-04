import sys
sys.path.append('./')
sys.path.append('../')
import utils.fix_libtiff

import cv2
import os

from cv_bridge import CvBridge
from fisheye_undistort.fish_eye_camera import FisheyeCamera

class FisheyeUndistort:
    def __init__(self, para_path, channel_list) -> None:
        self.fisheye_camera_obj = FisheyeCamera(para_path=para_path)
        self.max_persp = self.check_total_info(check_channel=channel_list)
        self.bridge = CvBridge()

    def get_undistorted_image(self, channel_tag, msg, downsample_rate=4):
        mapx_persp_32, mapy_persp_32 = self.max_persp[channel_tag]["x"], self.max_persp[channel_tag]["y"]
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        dst_scaramuzza = cv2.remap(cv_img, mapx_persp_32, mapy_persp_32, cv2.INTER_LINEAR)

        new_width, new_height = (int(dst_scaramuzza.shape[1] / downsample_rate), int(dst_scaramuzza.shape[0] / downsample_rate))
        resize_dst_scaramuzza = cv2.resize(dst_scaramuzza, (new_width, new_height))
        return  resize_dst_scaramuzza

    def check_total_info(self, check_channel):
        max_persp = {}
        for channel_item in check_channel:
            max_persp[channel_item] = {}
            max_persp[channel_item]["x"], max_persp[channel_item]["y"] = self.fisheye_camera_obj.check_undistort_info(channel_item)
        return max_persp



if __name__ == "__main__":
    para_path = os.path.join(os.path.dirname(__file__), "para")
    channel_list=["left", "right", "front", "back"]
    FisheyeUndistort(para_path=para_path, channel_list=channel_list)
