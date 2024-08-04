import sys
import copy
import rospy
import rospkg
import threading

rp = rospkg.RosPack()
workspace_root = rp.get_path('core')
sys.path.append(workspace_root[:workspace_root.find("catkin_ws")])
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from nav_msgs.msg import Path
from utils.pose_utils import HomogeneousTrans, PoseFlow


class VisualizeTrajectory:
    def __init__(self, frame_id):
        self.frame_id = frame_id

    def get_homo_mat_from_ros_pose(self, pose):
        return HomogeneousTrans(position_list=[pose.position.x, pose.position.y, pose.position.z], 
                                att_input=[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z], type="quad", deg_or_rad="deg")

    def traj_callback(self, msg: Path):
        self.ego_pose_msg_lock.acquire()
        while not self.ego_pose_msg:
            continue
        cnt = 0
        while self.ego_pose_msg:
            cnt += 1
            if cnt > 10:
                break
        ego2map_matrix = self.get_homo_mat_from_ros_pose(self.ego_pose_msg.pose).get_matrix()
        
        pub_path = copy.deepcopy(msg)
        pub_path.header.frame_id = self.frame_id
        for item in pub_path.poses:
            cur_mat = self.get_homo_mat_from_ros_pose(item.pose).get_matrix()
            ret_mat = ego2map_matrix @ cur_mat
            x, y, z = ret_mat[:3,-1].tolist()
            qw, qx, qy, qz = PoseFlow(att_input=ret_mat[:3,:3], type="rot_mat").get_quad()
            item.pose.position = Point(x=x, y=y, z=z)
            item.pose.orientation = Quaternion(x=qx, y=qy,z=qz, w=qw)
        self.predict_traj_pub.publish(pub_path)

        self.ego_pose_msg_lock.release()

    def localization_callback(self, msg: PoseStamped):
        self.ego_pose_msg = msg
        cur_pose_stamp = self.get_stamped_ego_pose(msg.pose)
        self.ego_pose_pub.publish(cur_pose_stamp)

    def get_stamped_ego_pose(self, pose):
        cur_pose_stamp = PoseStamped()
        cur_pose_stamp.header.frame_id = self.frame_id
        cur_pose_stamp.pose = pose
        return cur_pose_stamp

    def main(self):
        rospy.init_node("e2e_traj_show")
        rospy.Subscriber("/e2e_traj_pred_topic", Path, self.traj_callback, queue_size=1)
        rospy.Subscriber("/ego_pose", PoseStamped, self.localization_callback, queue_size=1)
        self.predict_traj_pub = rospy.Publisher("e2e_traj_pred_in_map_topic", Path, queue_size=1)
        self.ego_pose_pub = rospy.Publisher("ego", PoseStamped, queue_size=1) 
        self.ego_pose_msg_lock = threading.Lock()
        self.ego_pose_msg = None
        rospy.spin()


if __name__ == "__main__":
    obj = VisualizeTrajectory(frame_id="iekf_map")
    obj.main()