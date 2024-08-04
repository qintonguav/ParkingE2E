# image_to_rviz

## A tool to load the image to RVIZ  

1.1 Down load the project and complie in you catkin_ws.

1.2 Run the node. Add a marker (/image_to_rviz/visualization_marker) in your RVIZ.
```
rosrun image_to_rviz image_to_rviz_node
```
![Alt text](https://github.com/qintony/image_to_rviz/blob/master/img/img1.png)
1.3 dynamiclly adjust the position of the image.
      scale, x, y, z and yaw angle 
```
rosrun rqt_reconfigure rqt_reconfigure
```
![Alt text](https://github.com/qintony/image_to_rviz/blob/master/img/img2.png)