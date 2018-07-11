//
// Created by hd on 1/16/18.
//

#ifndef PROJECT_DEPTHOBJECTDETECTOR_H
#define PROJECT_DEPTHOBJECTDETECTOR_H
// C++
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include<thread>
#include <mutex>
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// ROS
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

// VisionWorks
#include <NVX/nvx.h>
#include <NVX/nvx_timer.hpp>
#include <NVX/nvx_opencv_interop.hpp>
#include <NVXIO/Application.hpp>
#include <NVXIO/ConfigParser.hpp>
#include <NVXIO/FrameSource.hpp>
#include <NVXIO/Render.hpp>
#include <NVXIO/SyncTimer.hpp>
#include <NVXIO/Utility.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Obstacle ros msgs
#include <obstacle_msgs/MapInfo.h>
#include <obstacle_msgs/obs.h>
#include <obstacle_msgs/point3.h>

#include "darknet_ros/stereo_matching.h"
#include "darknet_ros/Blob.h"
#include "darknet_ros/YoloObjectDetector.hpp"
#include "utils/timing.h"

namespace darknet_ros {

//struct u_span {
//    int u_left; int u_right; double u_d;
//    int u_pixels;
//    bool checked;
//    int ID;
//};

class YoloObjectDetector;

class Detection {
public:

    Detection();

    Detection(YoloObjectDetector* pYolo, ros::NodeHandle n);

    ~Detection();

    void getImage(cv::Mat &Frame1, cv::Mat &Frame2);

    void getParams();

    /*!
    * As a thread, keep running!
    */
    void Run();

//    bool CheckDepthUpdate();

private:

//      void ImageGrabber(const sensor_msgs::ImageConstPtr &image1, const sensor_msgs::ImageConstPtr &image2);

    void CreateMsg();

    bool read(StereoMatching::StereoMatchingParams &config);

    vx_status createMatFromImage(cv::Mat &mat, vx_image image);

    vx_image createImageFromMat(vx_context context, const cv::Mat & mat);

    void GenerateDisparityMap();

    void Initialize();

    void VisualizeResults();

    void RequestStop();

    bool Stop();

    bool isStopped();

    bool stopRequested();

    void SetFinish();

    bool isFinished();

    void RequestFinish();

    bool CheckFinish();

    YoloObjectDetector* mpYolo;

    int Height, Width, Scale, frame, disp_size, min_disparity;

    cv::Size Initial_size;
    cv::Rect roi1, roi2;

    cv::Mat M1, D1, M2, D2;
    cv::Mat R, T, R1, P1, R2, P2, Q;
    cv::Mat left_original, right_original, left_resized, right_resized, left_rectified, right_rectified;// imFull, images;
    cv::Mat map11, map12, map21, map22;
    cv::Mat disparity_map;

    vx_image vxiLeft_U8, vxiRight_U8, vxiLeft, vxiRight, vxiDisparity;
    nvxio::ContextGuard context;
    StereoMatching::StereoMatchingParams params;
    StereoMatching::ImplementationType implementationType = StereoMatching::HIGH_LEVEL_API;
    std::unique_ptr<StereoMatching> stereo;

    std::vector<double> depth;

    std::string disparity_frame;

//    ros::Time image_time;
//    std_msgs::Header image_header;

    bool isDepthNew;
    bool isReceiveImage;
    bool mbFinished_;
    bool mbFinishRequested_;
    bool mbStopped;
    bool mbStopRequested;

    std::mutex mMutexFinish;
    std::mutex mMutexStop;
    std::mutex mMutexNewImage;
    std::mutex mMutexDepth;

    Util::CPPTimer timer1;

};

}




#endif //PROJECT_DEPTHOBJECTDETECTOR_H
