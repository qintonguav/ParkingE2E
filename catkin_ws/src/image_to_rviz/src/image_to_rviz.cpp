#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <ros/package.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_datatypes.h>


double sizeSq = 1080 / (108 / 2);

double vehicle_x = 0.0;
double vehicle_y = 0.0;
double vehicle_z = 0.0;
double vehicle_yaw = 0.0;


cv::Mat current_image;
bool new_image_received = false;


void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        current_image = cv_ptr->image;

        new_image_received = true;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}


void posecallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
    vehicle_x = msg->pose.position.x;
    vehicle_y = msg->pose.position.y;
    vehicle_z = msg->pose.position.z;
    tf::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
    vehicle_yaw = tf::getYaw(q) + 3.1416;
}


std::pair<double, double> rotation_2d(std::pair<double, double> coordinate, double rotation_angle)
{
    double x_trans = coordinate.first * cos(rotation_angle) - coordinate.second * sin(rotation_angle);
    double y_trans = coordinate.first * sin(rotation_angle) + coordinate.second * cos(rotation_angle);

    return std::pair<double, double>(x_trans, y_trans);
}

std::pair<double, double> scale_2d(std::pair<double, double> coordinate, double scale)
{
    double x_trans = coordinate.first * scale;
    double y_trans = coordinate.second * scale;
    return std::pair<double, double>(x_trans, y_trans);
}

std::pair<double, double> bias_2d(std::pair<double, double> coordinate, std::pair<double, double> bias)
{
    double x_trans = coordinate.first + bias.first;
    double y_trans = coordinate.second + bias.second;
    return std::pair<double, double>(x_trans, y_trans);
}


geometry_msgs::Point pix2rviz(std::pair<double, double> raw_pixel, double scale, double rotation, std::pair<double, double> bias)
{
    std::pair<double, double> scale_point = scale_2d(raw_pixel, scale);
    std::pair<double, double> rotation_point = rotation_2d(scale_point, rotation);
    std::pair<double, double> ret = bias_2d(rotation_point, std::pair<double, double>(bias.first, bias.second));

    geometry_msgs::Point point;
    point.x = ret.first;
    point.y = ret.second;
    point.z = -100; // Make sure the trajectory is on the upper side of the IPM
    return point;
}



int main(int argc, char** argv)
{
    ROS_INFO("Image to RViz node started");
    ros::init(argc, argv, "image_to_rviz");
    ros::NodeHandle n("~");
    ros::Rate r(5);  // Increased rate to 30 Hz 
    ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization", 1);
    ros::Subscriber pose_sub = n.subscribe("/ego_pose",1, posecallback);
    ros::Subscriber image_sub = n.subscribe("/driver/fisheye/avm", 1, imageCallback);

    visualization_msgs::Marker image;
    image.header.frame_id = "iekf_map";
    image.ns = "image";
    image.id = 0;
    image.action = visualization_msgs::Marker::ADD;
    image.type = visualization_msgs::Marker::TRIANGLE_LIST;
    image.scale.x = 1;
    image.scale.y = 1;
    image.scale.z = 1;

    image.pose.orientation.w = 1.0;
    image.pose.orientation.x = 0.0;
    image.pose.orientation.y = 0.0;
    image.pose.orientation.z = 0.0;

    double pix;
    geometry_msgs::Point p;
    std_msgs::ColorRGBA crgb;

    while (ros::ok())
    {
        if (new_image_received && !current_image.empty()) {
            double center_x = current_image.rows / 2.0;
            double center_y = current_image.cols / 2.0;

            double image_width = current_image.cols;
            double image_height = current_image.rows;

            int D_SAMPLE = 4;

            pix = sizeSq / current_image.rows;
            image.points.clear();
            image.colors.clear();

            for(int r = 0; r < current_image.rows;) {
                for(int c = 0; c < current_image.cols;) {
                    cv::Vec3b intensity = current_image.at<cv::Vec3b>(r, c);
                    crgb.r = intensity.val[2] / 255.0;
                    crgb.g = intensity.val[1] / 255.0;
                    crgb.b = intensity.val[0] / 255.0;
                    crgb.a = 1.0;

                    double x_unbias = r - center_x;
                    double y_unbias = c - center_y;

                    p = pix2rviz(std::pair<double, double>(x_unbias, y_unbias), pix, vehicle_yaw, 
                                 std::pair<double, double>(vehicle_x, vehicle_y));
                    image.points.push_back(p);
                    image.colors.push_back(crgb);
                    
                    p = pix2rviz(std::pair<double, double>(x_unbias + D_SAMPLE, y_unbias), pix, vehicle_yaw, 
                                 std::pair<double, double>(vehicle_x, vehicle_y));
                    image.points.push_back(p);
                    image.colors.push_back(crgb);

                    p = pix2rviz(std::pair<double, double>(x_unbias, y_unbias + D_SAMPLE), pix, vehicle_yaw, 
                                 std::pair<double, double>(vehicle_x, vehicle_y));
                    image.points.push_back(p);
                    image.colors.push_back(crgb);

                    p = pix2rviz(std::pair<double, double>(x_unbias + D_SAMPLE, y_unbias), pix, vehicle_yaw, 
                                 std::pair<double, double>(vehicle_x, vehicle_y));
                    image.points.push_back(p);
                    image.colors.push_back(crgb);

                    p = pix2rviz(std::pair<double, double>(x_unbias + D_SAMPLE, y_unbias + D_SAMPLE), pix, vehicle_yaw, 
                                 std::pair<double, double>(vehicle_x, vehicle_y));
                    image.points.push_back(p);
                    image.colors.push_back(crgb);

                    p = pix2rviz(std::pair<double, double>(x_unbias, y_unbias + D_SAMPLE), pix, vehicle_yaw, 
                                 std::pair<double, double>(vehicle_x, vehicle_y));
                    image.points.push_back(p);
                    image.colors.push_back(crgb);
                    
                    c += D_SAMPLE;
                }
                r += D_SAMPLE;
            }

            image.header.stamp = ros::Time::now();
            marker_pub.publish(image);
            new_image_received = false;
        }

        ros::spinOnce();
        r.sleep();
    }
    marker_pub.shutdown();
    pose_sub.shutdown();
    image_sub.shutdown();
    ROS_INFO("Image to RViz node stopped");

    return 0;
}