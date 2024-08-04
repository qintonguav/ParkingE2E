cmake_minimum_required(VERSION 3.0.2)
project(undistort)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package()

catkin_install_python(PROGRAMS 
  scripts/undistort.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
