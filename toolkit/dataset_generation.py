import os
import sys

import yaml
sys.path.append('./')
sys.path.append('../')
import utils.fix_libtiff

import click
import cv2
import json
import os
import rosbag
import shutil
import threading
import tqdm

from cv_bridge import CvBridge
from enum import Enum
from rospy import Time

from fisheye_undistort.fish_cam_process import FisheyeUndistort
from utils.pose_utils import PoseFlow

class DrivingState(Enum):
    CANT_DET = 0
    FORWARD = 1
    BACKWARD = -1

class TaskExecutor:
    def __init__(self, task_func) -> None:
        self.task_func = task_func
        self.threads = []

    def run_tasks(self, args):
        for arg in args:
            thread = threading.Thread(target=self.task_func, args=arg)
            thread.start()
            self.threads.append(thread)

    def wait_for_completion(self):
        for thread in self.threads:
            thread.join()
            self.threads = []


class Bag2E2E:
    def __init__(self, bag_file_path, output_folder, config_file_path, start_time=0.0, duration_time=-1, pose_format= "euler_angle", sample_rate=1, image_channel_list=["left", "right", "front", "back"]):
        self.bag_tage = os.path.splitext(os.path.basename(bag_file_path))[0]
        
        self.get_topic(config_file_path)

        self.output_folder = os.path.join(output_folder, self.bag_tage)
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.image_channel_list = image_channel_list
        self.pose_format = pose_format
        self.start_time = start_time
        self.duration_time = duration_time

        self.sample_rate = sample_rate
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.bag = rosbag.Bag(bag_file_path)
        self.bridge = CvBridge()

        para_path = "./fisheye_undistort/para"
        self.undistort_obj = FisheyeUndistort(para_path=para_path, channel_list=self.image_channel_list)

        self.measurements = None
        self.candidate_parking_goal = None
        self.yaw_list = None
        self.current_time = None
        self.initial_time = None
        self.image_bank = {}
        for item in self.image_channel_list:
            self.image_bank[item] = None

        # check_topics use to accerlate iteration.
        self.check_topics = self.get_check_topics()

        self.start_flag = False
        self.save_trigger_channel = "front"
        assert self.save_trigger_channel in self.image_channel_list

        self.name_cnt, self.frequenct_cnt = None, None
        self.driving_state = DrivingState.CANT_DET
        self.segment_path = None
        self.unbelieve_cnt = 0
        self.believe_cnt = 0

        self.record_state = DrivingState.BACKWARD
        self.record_start = False
        
        self.save_img_threads = TaskExecutor(self.save_img)
        

    def get_check_topics(self):
        check_topics = [self.measurements_topic_name]
        check_topics.append(self.fisheye_left_topic)
        check_topics.append(self.fisheye_right_topic)
        check_topics.append(self.fisheye_front_topic)
        check_topics.append(self.fisheye_back_topic)
        return check_topics

    def get_topic(self, config_file_path):
        with open(config_file_path, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
        self.fisheye_front_topic = content["fisheye_front_topic"]
        self.fisheye_back_topic = content["fisheye_back_topic"]
        self.fisheye_left_topic = content["fisheye_left_topic"]
        self.fisheye_right_topic = content["fisheye_right_topic"]
        self.camera_topic = [self.fisheye_front_topic, self.fisheye_back_topic, 
                             self.fisheye_left_topic, self.fisheye_right_topic]
        self.measurements_topic_name = content["localization_topic"]

    def save_img(self, img, channel, cnt):
        channel_path = os.path.join(self.segment_path, self.get_format_from_channel(channel))
        os.makedirs(channel_path, exist_ok=True)
        img_filename = os.path.join(channel_path,"{:04d}.png".format(cnt))
        cv2.imwrite(img_filename, img)

    def get_format_from_channel(self, channel):
        if channel == "back":
            channel = "rear"
        return "rgb_{}".format(channel)

    def save_measurements(self, measurements, cnt, measurement_tag="measurements"):
        measurements_path = os.path.join(self.segment_path, measurement_tag)
        os.makedirs(measurements_path, exist_ok=True)
        measurements_filename = os.path.join(measurements_path, "{:04d}.json".format(cnt))
        with open(measurements_filename, 'w') as json_file:
            json.dump(measurements, json_file, indent=4)

    def save_parking_goals(self, parking_goals):
        assert self.pose_format == "euler_angle"
        measurements = {
            "x": parking_goals["x"],
            "y": parking_goals["y"],
            "z": parking_goals["z"],
            "yaw": parking_goals["yaw"]
        }
        self.save_measurements(measurements, 1, measurement_tag="parking_goal")

    def get_channel_name_from_topic(self, topic):
        return topic.split("/")[-2]

    def parser_compressed_image_msg(self, msg, undsitort=True, current_channel="error"):
        if undsitort:
            image = self.undistort_obj.get_undistorted_image(current_channel, msg)
        return image

    def parser_measurements_msg(self, msg, pose_format):
        if not hasattr(msg, 'pose') or not msg.pose._type == 'geometry_msgs/Pose':
            assert print("The pose is incomplete!")
        pose_obj = msg.pose
        if pose_format == "quaternion":    
            pose_ret = {
                'position':{
                    'x':pose_obj.position.x,
                    'y':pose_obj.position.y,
                    'z':pose_obj.position.z
                },
                'orientation':{
                    'x':pose_obj.orientation.x,
                    'y':pose_obj.orientation.y,
                    'z':pose_obj.orientation.z,
                    'w':pose_obj.orientation.w
                }
            }
        elif pose_format == "euler_angle":
            quad = [pose_obj.orientation.w, pose_obj.orientation.x, pose_obj.orientation.y, pose_obj.orientation.z]

            pose_flow_obj = PoseFlow(att_input=quad, type="quad", deg_or_rad="deg")
            yaw, pitch, roll = pose_flow_obj.get_euler()
            pose_ret = {
                'x':pose_obj.position.x,
                'y':pose_obj.position.y,
                'z':pose_obj.position.z,
                'roll':roll,
                'yaw':yaw,
                'pitch':pitch
            }

        return pose_ret


    def start_trigger(self):
        if self.measurements is None: 
            return False
        for channel in self.image_bank:
            if self.image_bank[channel] is None: 
                return False
        return True

    def get_current_drive_state(self, long_speed, longi_speed_threshold):
        believe = True
        driving_state = DrivingState.BACKWARD
        if abs(long_speed) < longi_speed_threshold:
            believe = False
        if long_speed > 0.:
            driving_state = DrivingState.FORWARD
        return driving_state, believe

    def update_drive_state(self, long_speed, longi_speed_threshold=0.03, unbelieve_tolerance=100, believe_tolerance=3):
        current_forward, current_believe = self.get_current_drive_state(long_speed, longi_speed_threshold)
        if not current_believe:
            self.believe_cnt = 0
            self.unbelieve_cnt += 1
        else:
            self.unbelieve_cnt = 0
            self.believe_cnt += 1

        if self.believe_cnt == believe_tolerance:
            self.driving_state = current_forward

        if self.unbelieve_cnt == unbelieve_tolerance:
            self.driving_state = DrivingState.CANT_DET
    
    def update_save_state(self, length_threshold=15):
        if not self.record_start:
            if self.driving_state == self.record_state:
                self.record_start = True
                self.segment_path = os.path.join(self.output_folder, "{}".format(str(int(self.current_time.to_sec()))))
                self.name_cnt = 0
                self.frequenct_cnt = 0
                self.yaw_list = []
                self.candidate_parking_goal = {}
        if self.record_start:
            if self.driving_state != self.record_state:
                self.record_start = False

                if not self.candidate_parking_goal:
                    return

                self.save_parking_goals(self.candidate_parking_goal)
                total_len = len(self.yaw_list)
                can_use, direction = self.judging_direction()
                os.rename(self.segment_path, "{}_{}".format(self.segment_path, direction))
                self.segment_path = "{}_{}".format(self.segment_path, direction)
                if (not can_use or total_len < length_threshold):
                    shutil.rmtree(self.segment_path)

    def judging_direction(self, threshold_min=50, threshold_max=135):
        direction = "idontknow"
        can_use = True
        for index in range(len(self.yaw_list)):
            if self.yaw_list[index] < 0:
                self.yaw_list[index] += 360
        delta_yaw = self.yaw_list[-1] - self.yaw_list[0]
        if (delta_yaw < -180):
            delta_yaw += 360
        if abs(delta_yaw) > threshold_max or abs(delta_yaw) < threshold_min:
            can_use = False
        else:
            if delta_yaw > 0:
                direction = "right"
            else:
                direction = "left"
        return can_use, direction


    def transfer(self):
        save_trigger = False
        for topic, msg, current_time in tqdm.tqdm(self.bag.read_messages(topics=self.check_topics, start_time=Time.from_sec(self.start_time))):
            self.current_time = current_time
            if self.initial_time is None:
                self.initial_time = current_time

            if not self.start_flag:
                self.start_flag = self.start_trigger()

            self.update_save_state()

            if topic in self.camera_topic:
                current_channel = self.get_channel_name_from_topic(topic)
                if current_channel not in self.image_channel_list:
                    continue
                self.image_bank[current_channel] = self.parser_compressed_image_msg(msg, undsitort=True, current_channel=current_channel)

                if self.start_flag and self.record_start and current_channel == self.save_trigger_channel:
                    self.frequenct_cnt += 1
                    if self.frequenct_cnt % self.sample_rate == 0:
                        save_trigger = True
                        self.frequenct_cnt = 0

            if topic == self.measurements_topic_name:
                self.measurements = self.parser_measurements_msg(msg, self.pose_format)
                self.update_drive_state(msg.velocity.linear.x)

            if save_trigger:
                task_args = []
                for img_channel, img in self.image_bank.items():
                    task_args.append([img, img_channel, self.name_cnt])
                self.save_img_threads.run_tasks(task_args)
                self.save_img_threads.wait_for_completion()

                self.save_measurements(self.measurements, self.name_cnt)
                self.candidate_parking_goal = self.measurements
                assert self.pose_format == "euler_angle"
                self.yaw_list.append(self.measurements["yaw"])
                save_trigger = False
                self.name_cnt += 1

            if self.duration_time != -1 and (current_time - self.initial_time).to_sec() >= self.duration_time:
                break
        self.bag.close()

@click.command()
@click.option("--bag_file_path", type=str, default="")
@click.option("--output_folder_path", type=str, default="")
@click.option("--config_file_path", type=str, default="./catkin_ws/src/core/config/params.yaml")
@click.option("--start_time", type=float, default=0.0)
@click.option("--duration_time", type=int, default=-1)
@click.option("--sampling_rate", type=int, default=1)
def main(bag_file_path, output_folder_path, config_file_path, start_time, duration_time, sampling_rate):

    transform_obj = Bag2E2E(bag_file_path, output_folder_path, config_file_path, start_time, duration_time, sample_rate=sampling_rate)
    transform_obj.transfer()

if __name__ == "__main__":
    main()
