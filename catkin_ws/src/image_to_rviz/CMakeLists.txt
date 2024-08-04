cmake_minimum_required(VERSION 2.8.3)
project(image_to_rviz)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  dynamic_reconfigure
  roslib
  geometry_msgs
  nav_msgs
)
find_package(OpenCV  REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(compressed_image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

add_executable(image_to_rviz src/image_to_rviz.cpp)
add_dependencies(image_to_rviz ${PROJECT_NAME}_gencfg)
target_link_libraries(image_to_rviz ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${cv_bridge_LIBRARIES} ${compressed_image_transport_LIBRARIES} ${sensor_msgs_LIBRARIES})
