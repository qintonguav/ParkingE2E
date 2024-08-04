import os
import rospkg
import rospy
import sys
import time

rp = rospkg.RosPack()
workspace_root = rp.get_path('undistort')
project_workspace = workspace_root[:workspace_root.find("catkin_ws")]
sys.path.append(project_workspace)

from functools import partial
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from fisheye_undistort.fish_cam_process import FisheyeUndistort
from utils.msg._LocalizationEstimate import LocalizationEstimate
from utils.pose_utils import CustomizePose, PoseFlow, pose2customize_pose

class RawDataProcessNode(FisheyeUndistort):
    def __init__(self, para_path, channel_list) -> None:
        super().__init__(para_path, channel_list)
        self.init_flag = False
        self.world2ego_mat = None

    def main(self):
        rospy.init_node("undistort", log_level=rospy.DEBUG)
        fisheye_front_topic = rospy.get_param("fisheye_front_topic")
        fisheye_back_topic = rospy.get_param("fisheye_back_topic")
        fisheye_left_topic = rospy.get_param("fisheye_left_topic")
        fisheye_right_topic = rospy.get_param("fisheye_right_topic")
        localization_topic = rospy.get_param("localization_topic")

        self.pub_dict = {}
        for channel_item in channel_list:
            self.pub_dict[channel_item] = rospy.Publisher("/driver/pinhole_vitual/{}".format(channel_item), Image, queue_size=1)
        self.ego_pose_pub = rospy.Publisher("/ego_pose".format(channel_item), PoseStamped, queue_size=1)
        time.sleep(0.5)

        rospy.Subscriber(fisheye_front_topic, CompressedImage, partial(self.callback_func, "front"), queue_size=1)
        rospy.Subscriber(fisheye_back_topic, CompressedImage, partial(self.callback_func, "back"), queue_size=1)
        rospy.Subscriber(fisheye_left_topic, CompressedImage, partial(self.callback_func, "left"), queue_size=1)
        rospy.Subscriber(fisheye_right_topic, CompressedImage, partial(self.callback_func, "right"), queue_size=1)
        rospy.Subscriber(localization_topic, LocalizationEstimate, self.callback_debias, queue_size=1)
        rospy.Subscriber("/reinit", Bool, self.callback_reinit, queue_size=1)

        rospy.spin()

    def callback_func(self, channel_tag, msg):
        dst_scaramuzza = self.get_undistorted_image(channel_tag, msg)
        img_msg = self.bridge.cv2_to_imgmsg(dst_scaramuzza, encoding="passthrough")
        pub = self.pub_dict[channel_tag]
        pub.publish(img_msg)

    def callback_debias(self, msg: LocalizationEstimate):
        if not self.init_flag:
            init_ego_pose = pose2customize_pose(msg.pose)
            self.init_flag = True
            self.world2ego_mat = init_ego_pose.get_homogeneous_transformation().get_inverse_matrix()
        ego_pose = pose2customize_pose(msg.pose)
        local_ego = ego_pose.get_pose_in_ego(self.world2ego_mat)
        predict_pose = self.get_posestamp_info(local_ego)
        self.ego_pose_pub.publish(predict_pose)

    def callback_reinit(self, msg: Bool):
        if msg.data == True:
            self.init_flag = False

    def get_posestamp_info(self, ego: CustomizePose):
        predict_pose = PoseStamped()
        pose_flow_obj = PoseFlow(att_input=[ego.yaw, ego.pitch, ego.roll], type="euler", deg_or_rad="deg")
        quad = pose_flow_obj.get_quad()
        predict_pose.pose.position = Point(x=ego.x, y=ego.y, z=ego.z)
        predict_pose.pose.orientation = Quaternion(x=quad.x, y=quad.y,z=quad.z, w=quad.w)
        return predict_pose

if __name__ == "__main__":
    para_path = os.path.join(project_workspace, "fisheye_undistort", "para")
    print(para_path)
    channel_list = ["left", "right", "front", "back"]
    undistort_node = RawDataProcessNode(para_path=para_path, channel_list=channel_list)
    undistort_node.main()