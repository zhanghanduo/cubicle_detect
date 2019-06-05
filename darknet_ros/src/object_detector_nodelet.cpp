/*
 * yolo_obstacle_detector_node.cpp
 *
 *  Created on: June 19, 2018
 *      Author: Zhang Handuo
 *   Institute: NTU, ST Corp Lab
 */
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <darknet_ros/YoloObjectDetector.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
//#include <message_filters/sync_policies/approximate_time.h>
#include <pluginlib/class_list_macros.h>
#include <message_filters/sync_policies/exact_time.h>

using namespace message_filters;
using namespace std;

namespace darknet_ros {
    class detection_nodelet : public nodelet::Nodelet {
    public:
        detection_nodelet() : Nodelet() {};

        ~detection_nodelet() {}
        virtual void onInit();

        void callback(const sensor_msgs::ImageConstPtr &image1,
                      const sensor_msgs::ImageConstPtr &image2) {
            detector.cameraCallback(image1, image2);
        }

    private:
        std::string image_left_topic, image_right_topic, image_left_info, image_right_info;
        darknet_ros::YoloObjectDetector detector;
        Subscriber<sensor_msgs::Image> image1_sub, image2_sub;
//        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
        typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
        boost::shared_ptr<ExactSync> exact_sync_;
    };

    void detection_nodelet::onInit() {
        ros::NodeHandle n_ = getMTPrivateNodeHandle();
        ros::NodeHandle nh_ = getMTNodeHandle();
        detector.readParameters(n_, nh_);

        n_.param<std::string>("image_left_topic", image_left_topic, "/left/image_rect");
        n_.param<std::string>("image_right_topic", image_right_topic, "/right/image_rect");
        n_.param<std::string>("image_info_left", image_left_info, "/left/camera_info");
        n_.param<std::string>("image_info_right", image_right_info, "/right/camera_info");

        sensor_msgs::CameraInfoConstPtr left_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_left_info);
        sensor_msgs::CameraInfoConstPtr right_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_right_info);

        detector.loadCameraCalibration(left_info, right_info);
        detector.DefineLUTs();

        image1_sub.subscribe(nh_, image_left_topic, 10);
        image2_sub.subscribe(nh_, image_right_topic, 10);

        exact_sync_.reset( new ExactSync( ExactPolicy(10),
                                          image1_sub,
                                          image2_sub ) );

        exact_sync_->registerCallback( boost::bind(
                &darknet_ros::YoloObjectDetector::cameraCallback, &detector, _1, _2 ) );

//        ros::Timer timer_ = nh_.createTimer(ros::Duration(0.1), boost::bind
//        (&darknet_ros::YoloObjectDetector::timerCallback, &detector, _1), false);

    }
}

PLUGINLIB_EXPORT_CLASS(darknet_ros::detection_nodelet, nodelet::Nodelet)