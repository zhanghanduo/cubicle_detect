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
#include <stereo_msgs/DisparityImage.h>

using namespace message_filters;

int main(int argc, char** argv) {
  ros::init(argc, argv, "cubicle_detect");
  ros::NodeHandle nh_pub;
  ros::NodeHandle nodeHandle("~");

  std::string image_left_topic, image_right_topic, image_left_info, image_right_info, disparity_topic;


  darknet_ros::YoloObjectDetector detector(nodeHandle, nh_pub);

  nodeHandle.param<std::string>("image_left_topic", image_left_topic, "/left/image_rect_color");
  nodeHandle.param<std::string>("image_right_topic", image_right_topic, "/right/image_rect_color");
  nodeHandle.param<std::string>("image_info_left", image_left_info, "/left/camera_info");
  nodeHandle.param<std::string>("image_info_right", image_right_info, "/right/camera_info");
  nodeHandle.param<std::string>("disparity_topic", disparity_topic, "/disparity/disparity_image");

  Subscriber<sensor_msgs::Image> image1_sub(nh_pub, image_left_topic, 10);
  Subscriber<sensor_msgs::Image> image2_sub(nh_pub, image_right_topic, 10);
  Subscriber<sensor_msgs::CameraInfo> sub_info_l_(nh_pub, image_left_info, 10);
  Subscriber<sensor_msgs::CameraInfo> sub_info_r_(nh_pub, image_right_info, 10);
  Subscriber<stereo_msgs::DisparityImage> disparity_sub(nh_pub, disparity_topic, 10);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image,
          sensor_msgs::Image, sensor_msgs::CameraInfo,
          sensor_msgs::CameraInfo, stereo_msgs::DisparityImage> MySyncPolicy;

  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10),
                                  image1_sub, image2_sub, sub_info_l_, sub_info_r_, disparity_sub);

  sync.registerCallback(boost::bind(&darknet_ros::YoloObjectDetector::cameraCallback,
                                    &detector ,_1, _2, _3, _4, _5));

  ros::spin();
  return 0;
}
