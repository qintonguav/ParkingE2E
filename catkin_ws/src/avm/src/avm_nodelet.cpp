#include <memory>
#include <queue>

#include <boost/endian/conversion.hpp>
#include <boost/function/function_fwd.hpp>
#include <nodelet/nodelet.h>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <sensor_msgs/Image.h>

#include "fisheye_avm.hpp"
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class avm_nodelet : public nodelet::Nodelet {
 public:
 private:
  virtual void onInit() override;

  struct camera {
    std::string topic;
    std::string param;
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub;
    std::queue<std::pair<int, cv::Mat>> imgs;
  };

  void images_callback(sensor_msgs::CompressedImageConstPtr msg_back,
                                    sensor_msgs::CompressedImageConstPtr msg_front,
                                    sensor_msgs::CompressedImageConstPtr msg_left,
                                    sensor_msgs::CompressedImageConstPtr msg_right);

 private:
  camera back;
  camera front;
  camera left;
  camera right;

  std::string output_topic;
  int output_width;
  int output_height;
  double output_scale;
  int interpolation_mode;
  bool logging;

  FisheyeAVM avm;

  sensor_msgs::ImagePtr msg;
  cv::Mat img;

  image_transport::Publisher pub;
  int image_num = 0;

  using Policy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CompressedImage, sensor_msgs::CompressedImage,
      sensor_msgs::CompressedImage, sensor_msgs::CompressedImage>;
  std::shared_ptr<message_filters::Synchronizer<Policy>> p_sync;
};

#define PARAM(name, var)                                     \
  do {                                                       \
    if (!nhp.getParam(name, var) && !nh.getParam(name, var)) {                          \
      NODELET_ERROR_STREAM("missing parameter '" #name "'"); \
      return;                                                \
    }                                                        \
  } while (0)

void avm_nodelet::onInit() {
  NODELET_INFO_STREAM(__PRETTY_FUNCTION__);

  auto &nhp = getMTPrivateNodeHandle();
  auto &nh = getMTNodeHandle();

  PARAM("fisheye_back_topic", back.topic);
  PARAM("fisheye_front_topic", front.topic);
  PARAM("fisheye_left_topic", left.topic);
  PARAM("fisheye_right_topic", right.topic);

  PARAM("back_param", back.param);
  PARAM("front_param", front.param);
  PARAM("left_param", left.param);
  PARAM("right_param", right.param);

  PARAM("output_topic", output_topic);
  PARAM("output_width", output_width);
  PARAM("output_height", output_height);
  PARAM("output_scale", output_scale);
  PARAM("interpolation_mode", interpolation_mode);
  PARAM("logging", logging);

  avm.open({back.param, front.param, left.param, right.param});
  avm.set_output_resolution({output_width, output_height});
  avm.set_scale(output_scale);
  avm.set_interpolation_mode(interpolation_mode);
  avm.set_logging(logging);
  avm.initialize();

  image_transport::ImageTransport it(nhp);
  pub = it.advertise(output_topic, 1);

  back.sub.subscribe(nhp, back.topic, 1);
  front.sub.subscribe(nhp, front.topic, 1);
  left.sub.subscribe(nhp, left.topic, 1);
  right.sub.subscribe(nhp, right.topic, 1);

  p_sync = std::make_shared<message_filters::Synchronizer<Policy>>(Policy(10));
  p_sync->connectInput(back.sub, front.sub, left.sub, right.sub);
  p_sync->registerCallback(&avm_nodelet::images_callback, this);
}

void avm_nodelet::images_callback(sensor_msgs::CompressedImageConstPtr msg_back,
                                  sensor_msgs::CompressedImageConstPtr msg_front,
                                  sensor_msgs::CompressedImageConstPtr msg_left,
                                  sensor_msgs::CompressedImageConstPtr msg_right)
{
  cv_bridge::CvImagePtr cv_ptr_back = cv_bridge::toCvCopy(msg_back, "bgr8");
  cv_bridge::CvImagePtr cv_ptr_front = cv_bridge::toCvCopy(msg_front, "bgr8");
  cv_bridge::CvImagePtr cv_ptr_left = cv_bridge::toCvCopy(msg_left, "bgr8");
  cv_bridge::CvImagePtr cv_ptr_right = cv_bridge::toCvCopy(msg_right, "bgr8");

  cv::Mat cv_back = cv_ptr_back->image;
  cv::Mat cv_front = cv_ptr_front->image;
  cv::Mat cv_left = cv_ptr_left->image;
  cv::Mat cv_right = cv_ptr_right->image;

  std::vector<cv::Mat> imgs = {cv_back, cv_front, cv_left, cv_right};
  msg = boost::make_shared<sensor_msgs::Image>();
  msg->header.stamp = msg_back->header.stamp;
  msg->header.seq = msg_back->header.seq;
  msg->height = output_height;
  msg->width = output_width;
  msg->encoding = "bgr8";
  msg->is_bigendian =
      (boost::endian::order::native == boost::endian::order::big);
  msg->step = msg->width * 3;
  msg->data.resize(msg->width * msg->height * 3);

  img = cv::Mat(msg->height, msg->width, CV_8UC3, msg->data.data(), msg->step);
  // calculate
  avm(imgs, img);

  // publish
  #ifndef USE_CUDA
  size_t dataSize = img.rows * img.cols * img.channels();
  std::memcpy(msg->data.data(), img.data, dataSize);
  #endif

  pub.publish(msg);


}
PLUGINLIB_EXPORT_CLASS(avm_nodelet, nodelet::Nodelet);
