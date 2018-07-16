/*
 * yolo_obstacle_detector_node.cpp
 *
 *  Created on: June 19, 2018
 *      Author: Zhang Handuo
 *   Institute: NTU, ST Corp Lab
 */


#include <darknet_ros/YoloObjectDetector.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
//#include <message_filters/sync_policies/exact_time.h>
#include <sensor_msgs/Image.h>

using namespace message_filters;

int main(int argc, char** argv) {
  ros::init(argc, argv, "cubicle_detect");
  ros::NodeHandle nh_pub;
  ros::NodeHandle nodeHandle("~");

  std::string image_left_topic, image_right_topic, image_left_info, image_right_info;

  image_left_topic = "/long/left/image_rect";

  image_right_topic = "/long/right/image_rect";

  image_left_info = "/long/left/camera_info";

  image_right_info = "/long/right/camera_info";

  darknet_ros::YoloObjectDetector detector(nodeHandle, nh_pub);

  //  cubicle_detect::Detection detection(nodeHandle);

  if(nodeHandle.getParam("image_left_topic", image_left_topic))
    ROS_INFO("Get left image topic: %s", image_left_topic.c_str());

  if(nodeHandle.getParam("image_right_topic", image_right_topic))
    ROS_INFO("Get right image topic: %s", image_right_topic.c_str());

  if(nodeHandle.getParam("image_info_left", image_left_info))
    ROS_INFO("Get left image topic: %s", image_left_info.c_str());

  if(nodeHandle.getParam("image_info_right", image_right_info))
    ROS_INFO("Get right image info topic: %s", image_right_info.c_str());

  Subscriber<sensor_msgs::Image> image1_sub(nh_pub, image_left_topic, 20);
  Subscriber<sensor_msgs::Image> image2_sub(nh_pub, image_right_topic, 20);
  Subscriber<sensor_msgs::CameraInfo> sub_info_l_(nh_pub, image_left_info, 20);
  Subscriber<sensor_msgs::CameraInfo> sub_info_r_(nh_pub, image_right_info, 20);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image,
          sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> MySyncPolicy;

  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10),
                                  image1_sub, image2_sub, sub_info_l_, sub_info_r_);

  sync.registerCallback(boost::bind(&darknet_ros::YoloObjectDetector::cameraCallback,
                                    &detector ,_1, _2, _3, _4));

  ros::spin();
  return 0;
}
