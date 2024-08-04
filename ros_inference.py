import argparse
import rospy

import utils.fix_libtiff
from model_interface.model_interface import get_parking_model
from utils.config import get_inference_config_obj
from utils.ros_interface import RosInterface


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    args = arg_parser.parse_args()


    rospy.init_node("e2e_traj_pred")

    ros_interface_obj = RosInterface()
    threads = ros_interface_obj.receive_info()

    inference_cfg = get_inference_config_obj(args.inference_config_path)

    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")
    parking_inference_obj = ParkingInferenceModelModule(inference_cfg, ros_interface_obj=ros_interface_obj)
    parking_inference_obj.predict(mode=inference_cfg.predict_mode)

    for thread in threads:
        thread.join()