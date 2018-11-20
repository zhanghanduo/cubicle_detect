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

using namespace message_filters;

int main(int argc, char** argv) {
  ros::init(argc, argv, "cubicle_detect");
  ros::NodeHandle nh_pub;
  ros::NodeHandle nodeHandle("~");

  std::string image_left_topic, image_right_topic, image_left_info, image_right_info;

  image_left_topic = "/kitti_stereo/left/image_rect";
  image_right_topic = "/kitti_stereo/right/image_rect";
  image_left_info = "/kitti_stereo/left/camera_info";
  image_right_info = "/kitti_stereo/right/camera_info";
//    image_left_topic = "/wide/left/image_rect";
//    image_right_topic = "/wide/right/image_rect";
//    image_left_info = "/wide/left/camera_info";
//    image_right_info = "/wide/right/camera_info";

  darknet_ros::YoloObjectDetector detector(nodeHandle, nh_pub);

  if(nodeHandle.getParam("image_left_topic", image_left_topic))
    ROS_INFO("Get left image topic: %s", image_left_topic.c_str());

  if(nodeHandle.getParam("image_right_topic", image_right_topic))
    ROS_INFO("Get right image topic: %s", image_right_topic.c_str());

  if(nodeHandle.getParam("image_info_left", image_left_info))
    ROS_INFO("Get left camera info topic: %s", image_left_info.c_str());

  if(nodeHandle.getParam("image_info_right", image_right_info))
    ROS_INFO("Get right camera info topic: %s", image_right_info.c_str());

  sensor_msgs::CameraInfoConstPtr left_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_left_info);
  sensor_msgs::CameraInfoConstPtr right_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_right_info);

  detector.loadCameraCalibration(left_info, right_info);
//  detector.init();
  detector.DefineLUTs();

  Subscriber<sensor_msgs::Image> image1_sub(nh_pub, image_left_topic, 20);
  Subscriber<sensor_msgs::Image> image2_sub(nh_pub, image_right_topic, 20);
//  Subscriber<sensor_msgs::CameraInfo> sub_info_l_(nh_pub, image_left_info, 20);
//  Subscriber<sensor_msgs::CameraInfo> sub_info_r_(nh_pub, image_right_info, 20);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image,
          sensor_msgs::Image> MySyncPolicy;

  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10),
                                  image1_sub, image2_sub);

  sync.registerCallback(boost::bind(&darknet_ros::YoloObjectDetector::cameraCallback,
                                    &detector ,_1, _2));

  ros::spin();
  return 0;
}
