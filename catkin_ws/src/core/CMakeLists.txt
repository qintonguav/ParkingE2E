cmake_minimum_required(VERSION 3.0.2)
project(core)

find_package(catkin REQUIRED COMPONENTS
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package()

catkin_install_python(PROGRAMS 
  scripts/e2e_traj_show.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
