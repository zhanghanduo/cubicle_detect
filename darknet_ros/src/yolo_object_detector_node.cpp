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
#include <sensor_msgs/PointCloud2.h>

using namespace message_filters;

int main(int argc, char** argv) {
  ros::init(argc, argv, "cubicle_detect");
  ros::NodeHandle nh_pub;
  ros::NodeHandle nodeHandle("~");

  std::string image_topic, image_info, disparity_topic, pointcloud_topic;


  darknet_ros::YoloObjectDetector detector(nodeHandle, nh_pub);

  nodeHandle.param<std::string>("image_topic", image_topic, "/rgb/image_rect_color");
  nodeHandle.param<std::string>("image_info", image_info, "/rgb/camera_info");
  nodeHandle.param<std::string>("disparity_topic", disparity_topic, "/disparity/disparity_image");
  nodeHandle.param<std::string>("pointcloud_topic", pointcloud_topic, "/zed/point_cloud/cloud_registered");

  Subscriber<sensor_msgs::Image> image_sub(nh_pub, image_topic, 10);
  Subscriber<sensor_msgs::CameraInfo> sub_info_(nh_pub, image_info, 10);
//  Subscriber<stereo_msgs::DisparityImage> disparity_sub(nh_pub, disparity_topic, 10);
  Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh_pub, "/zed/point_cloud/cloud_registered", 10);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image,
          sensor_msgs::CameraInfo, sensor_msgs::PointCloud2> MySyncPolicy;

  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10),
                                  image_sub, sub_info_, cloud_sub);

  sync.registerCallback(boost::bind(&darknet_ros::YoloObjectDetector::cameraCallback,
                                    &detector ,_1, _2, _3));

  ros::spin();
  return 0;
}
