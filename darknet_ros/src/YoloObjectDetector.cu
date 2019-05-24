/*
 * YoloObjectDetector.cpp
 *
 *  Created on: June 19, 2018
 *      Author: Zhang Handuo
 *   Institute: NTU, ST Corp Lab
 */


// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"
#include "utils/data.h"
#include <math.h>
#include <ros/package.h>
// Check for xServer
#include <X11/Xlib.h>
#include <algorithm>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;

#include "../../darknet/src/image_opencv.h"
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

using namespace message_filters;

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;
char **compact_detectionNames;

YoloObjectDetector::YoloObjectDetector()
    : numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0),
      use_grey(false),
      blnFirstFrame(true),
//      globalframe(0),
      is_even_crop(false),
      rem_w(0),
      rem_h(0)
{
  ROS_INFO("[ObstacleDetector] Node started.");

  // Read Cuda Info and ROS parameters from config file.
  if (!CudaInfo()) {
    ros::requestShutdown();
  }

//  nullHog.assign(36, 0.0);
//  init();

//  hog_descriptor = new Util::HOGFeatureDescriptor(8, 2, 9, 180.0);
    if(enableEvaluation_) {
        img_name = ros::package::getPath("cubicle_detect") + "/seq_1/f000.png";
        file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/f000.txt";
    }
    frame_num = 0;
}

YoloObjectDetector::~YoloObjectDetector()
{
//  finish_disparity_method();
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }

  delete ssgm;
  cv::destroyAllWindows();
//  yoloThread_.join();
  free(depth3D);
  free(x3DPosition);
  free(y3DPosition);
  free(cfg);
  free(weights);
  free(detectionNames);
  free(compact_detectionNames);
  free(data);
  free(roiBoxes_);
}

bool YoloObjectDetector::CudaInfo() {

  int deviceCount, device;
  int gpuDeviceCount = 0;
  struct cudaDeviceProp properties{};
  cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode != cudaSuccess)
    deviceCount = 0;
  /* machine with no GPUs can still report one emulation device */
  for (device = 0; device < deviceCount; ++ device){
    cudaGetDeviceProperties(&properties, device);
    if (properties.major != 9999)
      ++gpuDeviceCount;
  }
  std::cout << gpuDeviceCount << " GPU CUDA device(s) found" << std::endl;

  if (gpuDeviceCount > 0) {
    std::cout << "GPU load success!" << std::endl;
    return true;
  }
  else {
    std::cout << "GPU load fail!" << std::endl;
    return false;
  }
}

bool YoloObjectDetector::readParameters(ros::NodeHandle nh, ros::NodeHandle nh_p)
{
  nodeHandle_ = nh;
  nodeHandle_pub = nh_p;
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);
  nodeHandle_.param("image_view/eval", enableEvaluation_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(nullptr)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  nodeHandle_.param("yolo_model/compact_classes/names", compact_classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  compact_numClasses_ = compact_classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

    // Initialize deep network of darknet.
    std::string weightsPath;
    std::string configPath;
    std::string dataPath;
    std::string configModel;
    std::string weightsModel;

    // Look up table initialization
    counter = 0;

    nodeHandle_.param<int>("min_disparity", min_disparity, 7);
    nodeHandle_.param<int>("disparity_scope", disp_size, 128);
    nodeHandle_.param<bool>("use_grey", use_grey, false);
    nodeHandle_.param<bool>("enable_stereo", enableStereo, true);
    nodeHandle_.param<bool>("enable_classification", enableClassification, true);
    nodeHandle_.param<int>("scale", Scale, 1);
    nodeHandle_.param<bool>("filter_dynamic", filter_dynamic_, true);
    // Threshold of object detection.
    float thresh;
    nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

    // Path to weights file.
    nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                      std::string("yolov3-spp.weights"));
    nodeHandle_.param("weights_path", weightsPath, ros::package::getPath("cubicle_detect") + "/yolo_network_config/weights");
    weightsPath += "/" + weightsModel;
    weights = new char[weightsPath.length() + 1];
    strcpy(weights, weightsPath.c_str());

    // Path to config file.
    nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov3-spp.cfg"));
    nodeHandle_.param("config_path", configPath, ros::package::getPath("cubicle_detect") + "/yolo_network_config/cfg");
    configPath += "/" + configModel;
    cfg = new char[configPath.length() + 1];
    strcpy(cfg, configPath.c_str());

    // Path to data folder.
    dataPath = darknetFilePath_;
    dataPath += "/data";
    data = new char[dataPath.length() + 1];
    strcpy(data, dataPath.c_str());

    // Get classes.
    detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
    for (int i = 0; i < numClasses_; i++) {
        detectionNames[i] = new char[classLabels_[i].length() + 1];
        strcpy(detectionNames[i], classLabels_[i].c_str());
    }

    compact_detectionNames = (char**) realloc((void*) compact_detectionNames, (compact_numClasses_ + 1) * sizeof(char*));
    for (int i = 0; i < compact_numClasses_; i++) {
        compact_detectionNames[i] = new char[compact_classLabels_[i].length() + 1];
        strcpy(compact_detectionNames[i], compact_classLabels_[i].c_str());
    }

    // Load network.
    setupNetwork(cfg, weights, data, thresh, detectionNames, compact_detectionNames, numClasses_,
                 0, nullptr, 1, 0.5, 0, 0, 0, 0);

    std::string detectionImageTopicName;
    int detectionImageQueueSize;
    bool detectionImageLatch;
    std::string obstacleBoxesTopicName;
    int obstacleBoxesQueueSize;
    std::string disparityTopicName;
    int disparityQueueSize;
    std::string obs_disparityTopicName;
    int obs_disparityQueueSize;

    nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                      std::string("detection_image"));
    nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
    nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

    nodeHandle_.param("publishers/obstacle_boxes/topic", obstacleBoxesTopicName,
                      std::string("/obs_map"));
    nodeHandle_.param("publishers/obstacle_boxes/queue_size", obstacleBoxesQueueSize, 1);
    nodeHandle_.param("publishers/obstacle_boxes/frame_id", pub_obs_frame_id, std::string("refined_camera"));

    nodeHandle_.param("publishers/disparity_map/topic", disparityTopicName,
                      std::string("/disparity_map"));
    nodeHandle_.param("publishers/disparity_map/queue_size", disparityQueueSize, 1);

    nodeHandle_.param("publishers/obs_disparity_map/topic", obs_disparityTopicName,
                      std::string("/obs_disparity_map"));
    nodeHandle_.param("publishers/obs_disparity_map/frame_id", obs_disparityFrameId,
                      std::string("refined_camera"));
    nodeHandle_.param("publishers/obs_disparity_map/queue_size", obs_disparityQueueSize, 1);

    disparityPublisher_ = nodeHandle_pub.advertise<stereo_msgs::DisparityImage>(disparityTopicName,
                                                                                disparityQueueSize);

    obs_disparityPublisher_ = nodeHandle_pub.advertise<stereo_msgs::DisparityImage>(obs_disparityTopicName,
                                                                                    obs_disparityQueueSize);

    obstaclePublisher_ = nodeHandle_pub.advertise<obstacle_msgs::MapInfo>(
            obstacleBoxesTopicName, obstacleBoxesQueueSize);

    detectionImagePublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                            detectionImageQueueSize,
                                                                            detectionImageLatch);

    disparityColorPublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>("disparity_color", 10);
    trackingPublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>("track_image", 10);
    obstacleMaskPublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>("obstacle_image", 10);
    slopePublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>("slope_image", 10);

  return true;
}

void YoloObjectDetector:: loadCameraCalibration(const sensor_msgs::CameraInfoConstPtr &left_info,
                                               const sensor_msgs::CameraInfoConstPtr &right_info) {

  ROS_INFO_STREAM("init calibration");

  // Check if a valid calibration exists
  if (left_info->K[0] == 0.0) {
    ROS_ERROR("The camera is not calibrated");
    return;
  }

  sensor_msgs::CameraInfoPtr left_info_copy = boost::make_shared<sensor_msgs::CameraInfo>(*left_info);
  sensor_msgs::CameraInfoPtr right_info_copy = boost::make_shared<sensor_msgs::CameraInfo>(*right_info);
//  left_info_copy->header.frame_id = "stereo";
//  right_info_copy->header.frame_id = "stereo";

  // Get Stereo Camera Model from Camera Info message
  image_geometry::StereoCameraModel stereoCameraModel;
  stereoCameraModel.fromCameraInfo(left_info_copy, right_info_copy);

  // Get PinHole Camera Model from the Stereo Camera Model
  const image_geometry::PinholeCameraModel &cameraLeft = stereoCameraModel.left();
  const image_geometry::PinholeCameraModel &cameraRight = stereoCameraModel.right();

//    double data[16] = { 1, 0, 0, -left_info->P[2]/Scale, 0, 1, 0, -left_info->P[6]/Scale, 0, 0, 0, left_info->P[0], 0};
//    double data[16] = { 1, 0, 0, -322.94284058, 0, 1, 0, -232.25880432, 0, 0, 0, 922.9965, 0, 0, 0.001376324, 0};
//    Q = cv::Mat(4, 4, CV_64F, data);

  // Get rectify intrinsic Matrix (is the same for both cameras because they are rectified)
  cv::Mat projectionLeft = cv::Mat(cameraLeft.projectionMatrix());
  cv::Matx33d intrinsicLeft = projectionLeft(cv::Rect(0, 0, 3, 3));
  cv::Mat projectionRight = cv::Mat(cameraRight.projectionMatrix());
  cv::Matx33d intrinsicRight = projectionRight(cv::Rect(0, 0, 3, 3));

  u0 = left_info->P[2];
  v0 = left_info->P[6];
  focal = left_info->P[0];
  Width = left_info->width;
  Height = left_info->height;
  if(Width < Height){
      Width = left_info->height;
      Height = left_info->width;
  }

  Width /= Scale;
  Height /= Scale;
  u0 /= Scale;
  v0 /= Scale;
  focal /= Scale;

  rem_w = Width % 4;
  rem_h = Height % 4;
  ROS_WARN("remainder width: %d | remainder height: %d", rem_w, rem_h);

  Width_crp = Width - rem_w;
  Height_crp = Height - rem_h;

  if(rem_w || rem_h)
      is_even_crop = true;

  assert(intrinsicLeft == intrinsicRight);
  const cv::Matx33d &intrinsic = intrinsicLeft;

  // Save the baseline
  stereo_baseline_ = stereoCameraModel.baseline();
  ROS_INFO_STREAM("baseline: " << stereo_baseline_);
  assert(stereo_baseline_ > 0);

  int ii;
  x3DPosition = static_cast<double **>(calloc(Width, sizeof(double *)));
  for(ii = 0; ii < Width; ii++)
    x3DPosition[ii] = static_cast<double *>(calloc(disp_size + 1, sizeof(double)));

  y3DPosition = static_cast<double **>(calloc(Height, sizeof(double *)));
  for(ii = 0; ii < Height; ii++)
    y3DPosition[ii] = static_cast<double *>(calloc(disp_size + 1, sizeof(double)));

  depth3D = static_cast<double *>(calloc(disp_size + 1, sizeof(double)));

//  ObstacleDetector.Initiate(left_info_copy->header.frame_id, disp_size, stereo_baseline_, u0, v0, focal, Width, Height, Scale, min_disparity);
  ObstacleDetector.Initiate(disp_size, stereo_baseline_, u0, v0, focal, Width_crp, Height_crp, Scale, min_disparity);


//  // get the Region Of Interests (If the images are already rectified but invalid pixels appear)
//  left_roi_ = cameraLeft.rawRoi();
//  right_roi_ = cameraRight.rawRoi();
}

cv::Mat YoloObjectDetector::getDepth(cv::Mat &leftFrame, cv::Mat &rightFrame) {

//    float elapsed_time_ms;
    cv::Mat disparity_SGBM(leftFrame.size(), CV_8UC1);
//    cv::Mat disparity_SGM(leftFrame.size(), CV_8UC1);

//    sgm::StereoSGM ssgm(leftFrame.cols, leftFrame.rows, disp_size, 8, 8, sgm::EXECUTE_INOUT_HOST2HOST);
//    ssgm.execute(leftFrame.data, rightFrame.data, (void**)&disparity_SGBM.data);
//    demoTime_ = what_time_is_it_now();

//    sgm::StereoSGM ssgm(leftFrame.cols, leftFrame.rows, disp_size, 8, 8, sgm::EXECUTE_INOUT_HOST2HOST);
	ssgm->execute(leftFrame.data, rightFrame.data, disparity_SGBM.data);

//    disparity_SGBM = compute_disparity_method(leftFrame, rightFrame, &elapsed_time_ms);

//    for (int r=0; r<disparity_SGBM.rows;r++){
//        for (int c=0; c<disparity_SGBM.cols;c++){
//            int dispAtPoint = (int)disparity_SGBM.at<uchar>(r,c);
////            int dispAtPoint2 = (int)disparity_SGM.at<uchar>(r,c);
//            if (dispAtPoint>disp_size)
//                disparity_SGBM.at<uchar>(r,c) = 0;
////            else if (dispAtPoint<min_disparity)
////                disparity_SGBM.at<uchar>(r,c) = 0;
////            if (dispAtPoint2>disp_size)
////                disparity_SGM.at<uchar>(r,c) = 0;
////            else if (dispAtPoint2<min_disparity)
////                disparity_SGM.at<uchar>(r,c) = 0;
//        }
//    }

//    fps_ = 1./(what_time_is_it_now() - demoTime_);

//    cv::imshow("sgm",disparity_SGM);
//    cv::imshow("sgbm",disparity_SGBM);

    return disparity_SGBM;
}

void YoloObjectDetector::DefineLUTs() {

  ROS_WARN("u0: %f | v0: %f | focal: %f | base: %f | width: %d | Height: %d", u0, v0, focal, stereo_baseline_, Width_crp, Height_crp);

    for (int r=0; r<Width_crp; r++) {
        x3DPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            x3DPosition[r][c]=(r-u0)*stereo_baseline_/c;
//        std::cout<<xDirectionPosition[r][c]<<std::endl;
        }
    }

    for (int r=0; r<Height_crp; r++) {
//    for (int r=300; r<301; r++) {
        y3DPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            y3DPosition[r][c]=(v0-r)*stereo_baseline_/c;
//      std::cout<<r<<", "<<c<<": "<<yDirectionPosition[r][c]<<"; ";//std::endl;
        }
    }

    depth3D[0] =0;
    for( int i = 1; i < disp_size+1; ++i){
        depth3D[i]=focal*stereo_baseline_/i; //Y*dx/B
//      std::cout<<"i: "<<i<<", "<<depthTable[i]<<"; \n";
    }

}

void YoloObjectDetector::timerCallback(const ros::TimerEvent& event ) {
//    while(ros::ok()) {
        ROS_WARN("haha");

/*        if (viewImage_) {
            if (!color_out.empty())
                cv::imshow("Detection and Tracking", color_out);

            if ((enableStereo) && (!disparityFrame.empty())) {
                // To better visualize the result, apply a colormap to the computed disparity
                double min, max;
                minMaxIdx(disparityFrame, &min, &max);
//            std::cout << "disp min " << min << std::endl << "disp max " << max << std::endl;
                cv::Mat cm_disp, scaledDisparityMap;
                convertScaleAbs(disparityFrame, scaledDisparityMap, 3.1);
                applyColorMap(scaledDisparityMap, cm_disp, cv::COLORMAP_JET);
                cv::imshow("Disparity", cm_disp);
//            cv::imshow("ObsDisparity", ObsDisparity * 255 / disp_size);
            }
            cv::waitKey(waitKeyDelay_);
        }*/
//        std::chrono::milliseconds dura(2);
//        std::this_thread::sleep_for(dura);
//    }
}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &image1,
                                        const sensor_msgs::ImageConstPtr &image2){
//    ROS_WARN("[ObstacleDetector] Stereo images received.");

    // std::cout<<"Debug starting cameraCallBack"<<std::endl;
    cv_bridge::CvImageConstPtr cam_image1, cam_image2, cv_rgb;

    try {
        cam_image1 = cv_bridge::toCvShare(image1, sensor_msgs::image_encodings::MONO8);
        cam_image2 = cv_bridge::toCvShare(image2, sensor_msgs::image_encodings::MONO8);

        if(use_grey) {
            cv_rgb = cam_image1;
        }
        else {
            cv_rgb = cv_bridge::toCvShare(image1, sensor_msgs::image_encodings::BGR8);
        }
//        image_time_ = image1->header.stamp;
        image_time_ = ros::Time::now();
        imageHeader_ = image1->header;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (cam_image1) {

        // std::cout<<"Debug inside cameraCallBack scaling image height "<<frameHeight_<<std::endl;
        {
            boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
            origLeft = cam_image1->image.clone();//cv::Mat(cam_image1->image, left_roi_);
            origRight = cam_image2->image.clone();//cv::Mat(cam_image2->image, right_roi_);
//            cv::cvtColor(origLeft, camImageOrig, cv::COLOR_GRAY2BGR);
            camImageOrig = cv_rgb->image.clone();//cv::Mat(cv_rgb->image.clone(), left_roi_);
        }
        {
            boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
            imageStatus_ = true;
        }

        // std::cout<<"Debug inside cameraCallBack starting image resize"<<std::endl;
        cv::Mat left_resized, right_resized, camImageResized;

        if(Scale != 1) {
            cv::resize(origLeft, left_resized, cv::Size(Width, Height));
            cv::resize(origRight, right_resized, cv::Size(Width, Height));
            cv::resize(camImageOrig, camImageResized, cv::Size(Width, Height));
        }else{
            left_resized = origLeft;
            right_resized = origRight;
            camImageResized = camImageOrig;
        }

//         std::cout<<"Debug inside cameraCallBack starting image padding"<<std::endl;

//        cv::Mat left_widthAdj, right_widthAdj, camImageWidthAdj;

        if (is_even_crop) {
//            std::cout<<left_resized.size<<", "<<camImageResized.size<<", "<<right_resized.size<<std::endl;
//            std::cout<<left_resized.size<<", "<<camImageResized.size<<", "<<right_resized.size<<std::endl;
            left_rectified = left_resized(cv::Rect(0, 0, Width_crp, Height_crp)).clone();
            right_rectified = right_resized(cv::Rect(0, 0, Width_crp, Height_crp)).clone();
            camImageCopy_ = camImageResized(cv::Rect(0, 0, Width_crp, Height_crp)).clone();
        } else {
            left_rectified = left_resized;
            right_rectified = right_resized;
            camImageCopy_ = camImageResized;
        }

        if (notInitiated)
            yolo();

        Process();
    }
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage, const ros::Publisher& publisher_)
{
  if (publisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
//  cvImage.header.stamp = ros::Time::now();
  cvImage.header.stamp = image_time_;
  cvImage.header.frame_id = "camera";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  publisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void *YoloObjectDetector::detectInThread()
{
  double classi_time_ = what_time_is_it_now();
//  globalframe++;
//  running_ = 1;
  float nms = .45;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_.data;
  network_predict(*net_, X);

  image display = buff_;
  int nboxes = 0;

  detection *dets = get_network_boxes(net_, display.w, display.h, demoThresh_, demoHier_, nullptr, 1, &nboxes, 1);

  do_nms_sort(dets, nboxes, l.classes, nms);

  // 1 means output classes, 0 means ignored
  draw_detections_less(display, dets, nboxes, demoThresh_, demoNames_, compactDemoNames_, demoAlphabet_, 8, 0);

  // Or else if you want to detect every specific class, use following function instead:
//  draw_detections_v3(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, l.classes, 1);

//  if ( (enableConsoleOutput_)&&(globalframe%20==1) ) {
////    printf("\033[2J");
////    printf("\033[1;1H");
////    printf("\nFPS:%.1f\n",fps_);
////    printf("Objects:\n\n");
//      printf("FPS:%.1f\n", fps_);
//  }

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;

//  ROS_WARN("nboxes: %d\n", nboxes);

  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;
        // define 2D bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.02 && BoundingBox_height > 0.02) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
//          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];

          // Combine similar classes into one class
          if(j == 0)
              roiBoxes_[count].Class = 0;   // Person
          else if((j == 2) || (j == 5) || (j == 7))
              roiBoxes_[count].Class = 1;   // Vehicle
          else if((j == 1) || (j == 3))
              roiBoxes_[count].Class = 2;   // Bicycle
          else if(j == 9)
              roiBoxes_[count].Class = 3;   // traffic light
          else if(j == 11)
              roiBoxes_[count].Class = 6;   // stop sign
          else if(j == 12)
              roiBoxes_[count].Class = 4;   // parking meter
          else if(j == 13)
              roiBoxes_[count].Class = 5;   // bench
          else if((j < 24) && (j > 13))
              roiBoxes_[count].Class = 7;   // animals
          else
              roiBoxes_[count].Class = 8;   // others
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  roiBoxes_[0].num = count;

  free_detections(dets, nboxes);
//  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
//  running_ = 0;

  classi_fps_ = 1./(what_time_is_it_now() - classi_time_);

  return nullptr;
}

void *YoloObjectDetector::stereoInThread()
{
    double stereo_time_ = what_time_is_it_now();
    disparityFrame = getDepth(left_rectified, right_rectified);
    stereo_fps_ = 1./(what_time_is_it_now() - stereo_time_);

//    cv::imshow("left_rectified", buff_cv_l_);
//    cv::imshow("right_rectified",  buff_cv_r_);
//    cv::waitKey(1);

//    output = buff_cv_l_[(buffIndex_ + 2) % 3].clone();

        disparity_info.header.stamp = image_time_;
        disparity_info.header.frame_id = obs_disparityFrameId;
        cv_bridge::CvImage out_msg;
        out_msg.header.frame_id = obs_disparityFrameId;
        out_msg.header.stamp = image_time_;
        out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
        out_msg.image = disparityFrame;
        disparity_info.image = *out_msg.toImageMsg();

        disparity_info.f = focal;
        disparity_info.T = stereo_baseline_;
        disparity_info.min_disparity = min_disparity;
        disparity_info.max_disparity = disp_size; //128

    if(counter > 2){
        disparityPublisher_.publish(disparity_info);
//        ObstacleDetector.ExecuteDetection(disparityFrame[(buffIndex_ + 2) % 3], output);
    }

    counter ++;



//        ObstacleDetector.ExecuteDetection(disparityFrame[(buffIndex_ + 2) % 3], output);
//    }
    return nullptr;
}

void *YoloObjectDetector::fetchInThread()
{
//  IplImage* ROS_img = getIplImage();
//  ipl_into_image(ROS_img, buff_);//[buffIndex_]);

  buff_ = mat_to_image(camImageCopy_);

//  {
//    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
//    buffId_ = actionId_;//[buffIndex_] = actionId_;
//  }
  if(!use_grey)
    rgbgr_image(buff_);//[buffIndex_]);

  letterbox_image_into(buff_, net_->w, net_->h, buffLetter_);//[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);

  return nullptr;
}

void *YoloObjectDetector::displayInThread()
{
//  show_image_cv(buff_, "YOLO V3", ipl_);//[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
  show_image_cv(buff_, "YOLO V3");
  int c = cv::waitKey(waitKeyDelay_);
  if (c != -1) c = c%256;
  if (c == 27) {
      demoDone_ = 1;
      return nullptr;
  } else if (c == 82) {
      demoThresh_ += .02;
  } else if (c == 84) {
      demoThresh_ -= .02;
      if(demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
      demoHier_ += .02;
  } else if (c == 81) {
      demoHier_ -= .02;
      if(demoHier_ <= .0) demoHier_ = .0;
  }
  return nullptr;
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, char **less_names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  compactDemoNames_ = less_names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
//  printf("YOLO V3\n");
  net_ = load_network_custom(cfgfile, weightfile, 0, 1);
  fuse_conv_batchnorm(*net_);
//  calculate_binary_weights(*net_);
}

void YoloObjectDetector:: yolo()
{
//  srand(2222222);

//  int i;
//  demoTotal_ = sizeNetwork(net_);
//  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
//  for (i = 0; i < demoFrame_; ++i){
//      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
//  }
//  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  ssgm = new sgm::StereoSGM(left_rectified.cols, left_rectified.rows, disp_size, 8, 8, sgm::EXECUTE_INOUT_HOST2HOST);

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  buff_ = mat_to_image(camImageCopy_);

  buffLetter_ = letterbox_image(buff_, net_->w, net_->h);

  disparityFrame = cv::Mat(Height_crp, Width_crp, CV_8UC1, cv::Scalar(0));

  buff_cv_l_ = left_rectified.clone();//camImageCopy_.clone();

  buff_cv_r_ = right_rectified.clone();

//  ipl_cv = cv::Mat(cvSize(buff_.w, buff_.h), CV_8U, buff_.c);

    if(viewImage_) {

//        cvNamedWindow("Initial roadmap", CV_WINDOW_AUTOSIZE);
//        cvMoveWindow("Initial roadmap", 650, 350);
//
//        cvNamedWindow("Refined roadmap", CV_WINDOW_AUTOSIZE);
//        cvMoveWindow("Refined roadmap", 900, 350);

        cvNamedWindow("Slope Map", CV_WINDOW_AUTOSIZE);
        cvMoveWindow("Slope Map", 1460, 0);

        cvNamedWindow("Detection and Tracking", CV_WINDOW_NORMAL);
        cvMoveWindow("Detection and Tracking", 0, 0);
        cvResizeWindow("Detection and Tracking", 720, 453);

        cvNamedWindow("Obstacle Mask", CV_WINDOW_NORMAL);
        cvMoveWindow("Obstacle Mask", 700, 0);
        cvResizeWindow("Obstacle Mask", 720, 453);

        if(enableStereo) {
            cvNamedWindow("Disparity", CV_WINDOW_NORMAL);
            cvMoveWindow("Disparity", 0, 500);
            cvResizeWindow("Disparity", 720, 453);
        }
//        cv::startWindowThread();
    }

//  int count = 0;

//  if (!demoPrefix_ && viewImage_) {
//    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
//    if (fullScreen_) {
//      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
//    } else {
//      cvMoveWindow("YOLO V3", 0, 0);
//      cvResizeWindow("YOLO V3", 640, 480);
//    }
//  }

//  demoTime_ = what_time_is_it_now();
  notInitiated = false;

}

cv::Mat YoloObjectDetector::occlutionMap(cv::Rect_<int> bbox, size_t kk, bool FNcheck) {

    cv::Mat maskImg(camImageCopy_.size(), CV_8UC1, cv::Scalar::all(1));
//    cv::Mat mask;
//    maskImg(bbox).copyTo(mask);

    for (size_t ii = 0; ii < detListCurrFrame.size(); ii++) {
        if (ii == kk)
            continue;

        auto xminii = static_cast<int>(detListCurrFrame.at(ii).bb_left);
        auto yminii = static_cast<int>(detListCurrFrame.at(ii).bb_top);
        auto widthii = static_cast<int>(detListCurrFrame.at(ii).bb_right) - xminii;
        auto heightii = static_cast<int>(detListCurrFrame.at(ii).bb_bottom) - yminii;
        cv::Rect_<int> bboxii = cv::Rect_<int>(xminii, yminii, widthii, heightii);
//            std::cout<<"( "<<xminii<<", "<<yminii<<", "<<widthii<<", "<<heightii<<") ";

        cv::Rect intersection = bbox & bboxii;

        if (intersection.area() > 0) {
            if (FNcheck) {
                maskImg(intersection).setTo(cv::Scalar::all(0));
            } else {
                if (detListCurrFrame.at(ii).det_conf > detListCurrFrame.at(kk).det_conf) {
                    maskImg(intersection).setTo(cv::Scalar::all(0));
                }
            }
        }


    }

    cv::Mat maskroi;
    maskImg(bbox).copyTo(maskroi);

    return maskroi;
}

void YoloObjectDetector::calculateHistogram(Blob &currentDet, cv::Mat hsv, cv::Mat mask, int widthSeperate, int heightSeperate) {
    // Quantize the hue to 30 levels and the saturation to 32 levels
    std::vector<cv::Mat> inputHSV, inputMask;
    int widthii = hsv.cols / widthSeperate;
    int heightii = hsv.rows / heightSeperate;
    int countImages = 0;

//    std::cout<<widthii<<", "<<heightii<<std::endl;
//    cv::imshow("mask",mask*255);
//    cv::waitKey(0);

    for (int ii = 0; ii < widthSeperate; ii++) {
        int xminii = ii * widthii;
        for (int jj = 0; jj < heightSeperate; jj++) {
            int yminii = jj * heightii;
//            std::cout<<"("<<xminii<<", "<<yminii<<", "<<widthii<<", "<<heightii<<")"<<std::endl;
            cv::Rect roi = cv::Rect_<int>(xminii, yminii, widthii, heightii);
            cv::Mat imgROI, maskROI;
            mask(roi).copyTo(maskROI);
            hsv(roi).copyTo(imgROI);
            inputHSV.push_back(imgROI);
            inputMask.push_back(maskROI);
            countImages++;
        }
    }

//    int hbins = 30, sbins = 32;
    int histSize[] = {15, 16}; //{hbins, sbins};{hbins, sbins};
    float hranges[] = {0, 180}; //hue varies from 0 to 179, see cvtColor
    float sranges[] = {0, 256}; //saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
    const float *ranges[] = {hranges, sranges};
    std::vector<cv::MatND> hist;
    std::vector<bool> occluded;
    std::vector<float> occlusionLevel;
    int channels[] = {0, 1}; //we compute the histogram from the 0-th and 1-st channels

//    std::cout<<"calculating histograms"<<std::endl;
    for (size_t ii = 0; ii < inputHSV.size(); ii++) {

        double sumNonOccludded = cv::sum(inputMask.at(ii))[0];
        double sumTotal = inputMask.at(ii).rows * inputMask.at(ii).cols;
//        std::cout<<"sumNonOccludded: "<<sumNonOccludded<<", sumTotal: "<<sumTotal<<", ratio: "<<sumNonOccludded / sumTotal<<std::endl;
        occluded.push_back(sumNonOccludded / sumTotal <= 0.5);

        cv::MatND histROI;
        cv::calcHist(&inputHSV.at(ii), 1, channels, cv::Mat(), histROI, 2, histSize, ranges, true, false);
        cv::normalize(histROI, histROI, 1, 0, 2, -1, cv::Mat());
        hist.push_back(histROI);
    }

    int occludedROIs = 0;
    for (size_t ii = 0; ii < occluded.size(); ii++) {
        if (occluded.at(ii))
            occludedROIs++;
    }
    occlusionLevel.push_back(occludedROIs / ((float) occluded.size()));

    std::vector<std::vector<cv::MatND> > histList;
    std::vector<std::vector<bool> > occludedList;
    histList.push_back(hist);
    occludedList.push_back(occluded);

    currentDet.occluded = occludedList;
    currentDet.hist = histList;
    currentDet.overallOcclusion = occlusionLevel;

}

void YoloObjectDetector::calculateLPBH(Blob &currentDet, cv::Mat rgb, int grid_x, int grid_y) {

    cv::Mat src;
    cvtColor(rgb, src, cv::COLOR_BGR2GRAY);

    int radius = 1;
    int neighbors = 8;
    // allocate memory for result
    cv::Mat dst = cv::Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
    dst.setTo(0);
    for (int n = 0; n < neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // iterate through your data
        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1 * src.at<uchar>(i + fy, j + fx) + w2 * src.at<uchar>(i + fy, j + cx) +
                                             w3 * src.at<uchar>(i + cy, j + fx) + w4 * src.at<uchar>(i + cy, j + cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) ||
                                                        (std::abs(t - src.at<uchar>(i, j)) <
                                                         std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }

//    int grid_x = 3, grid_y=6;
    int numPatterns = static_cast<int>(std::pow(2.0, static_cast<double>(neighbors)));
    // calculate LBP patch size
    int width = dst.cols / grid_x;
    int height = dst.rows / grid_y;
    // allocate memory for the spatial histogram
    cv::Mat result = cv::Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for (int i = 0; i < grid_y; i++) {
        for (int j = 0; j < grid_x; j++) {
            cv::Mat src_cell = cv::Mat(dst, cv::Range(i * height, (i + 1) * height),
                                       cv::Range(j * width, (j + 1) * width));
            cv::Mat src_cell_vis = cv::Mat(src, cv::Range(i * height, (i + 1) * height),
                                           cv::Range(j * width, (j + 1) * width));
//            std::cout<<src.size()<<","<<src_cell_vis.size()<<std::endl;
//            cv::imshow("src_cell_vis",src_cell_vis);
//            cv::waitKey(0);
            cv::Mat src_cell2 = cv::Mat_<float>(src_cell);
            cv::Mat cell_hist;//= histc(src_cell, 0, (numPatterns-1), true);
            int maxVal = numPatterns - 1;
            int minVal = 0;
            // Establish the number of bins.
            int histSize = maxVal - minVal + 1;
            // Set the ranges.
            float range[] = {static_cast<float>(minVal), static_cast<float>(maxVal + 1)};
            const float *histRange = {range};
            // calc histogram
            calcHist(&src_cell2, 1, 0, cv::Mat(), cell_hist, 1, &histSize, &histRange, true, false);
            // normalize
            cell_hist /= (int) src_cell2.total();

            // copy to the result matrix
            cv::Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }

    result.copyTo(currentDet.lpbHist);

//    double min;
//    double max;
//    cv::minMaxIdx(dst, &min, &max);
//    cv::Mat adjMap;
//    dst.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
//
//    cv::imshow("src", src);
//    cv::imshow("census", adjMap);
//    cv::waitKey(0);
}

void *YoloObjectDetector::trackInThread() {
    // Publish image.
//    image copy = copy_image(buff_);
//    constrain_image(copy);
    cv::Mat output_label = image_to_mat(buff_);
    if (output_label.channels() == 3) cv::cvtColor(output_label, output_label, cv::COLOR_RGB2BGR);
    else if (output_label.channels() == 4) cv::cvtColor(output_label, output_label, cv::COLOR_RGBA2BGR);

    if (!publishDetectionImage(output_label, detectionImagePublisher_)) {
        ROS_DEBUG("Detection image has not been broadcasted.");
    }

    free(buff_.data);
//    cv::Mat clrImage = camImageCopy_;
    detListCurrFrame.clear();
    bboxResize = cv::Size(180,60);

//    std::cout<<"defore loop"<<std::endl;

    // Publish bounding boxes and detection result.
    int num = roiBoxes_[0].num;
    if (num > 0 ) {
//    if (num > 0 && num <= 100) {
        for (int i = 0; i < num; i++) {
            for (int j = 0; j < compact_numClasses_ - 1; j++) {
                if (roiBoxes_[i].Class == j) {
                    rosBoxes_[j].push_back(roiBoxes_[i]);
                    rosBoxCounter_[j]++;
                }
            }
        }

//    std_msgs::Int8 msg;
//    msg.data = static_cast<signed char>(num);
//    objectPublisher_.publish(msg);

        for (int i = 0; i < compact_numClasses_ - 1; i++) {
            if (rosBoxCounter_[i] > 0) {
                for (int j = 0; j < rosBoxCounter_[i]; j++) {
                    inputDetection input_det;
                    input_det.bb_left = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * Width_crp;
                    input_det.bb_top = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * Height_crp;
                    input_det.bb_right = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * Width_crp;
                    input_det.bb_bottom = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * Height_crp;
                    input_det.objCLass = compact_classLabels_[i];
                    input_det.det_conf = rosBoxes_[i][j].prob;
                    detListCurrFrame.push_back(input_det);
                }
            }
        }

//        std::cout<<"detListCurrFrame: "<<detListCurrFrame.size()<<std::endl;

        for (size_t ii = 0; ii < detListCurrFrame.size(); ii++) {

            auto center_c_ = static_cast<int>((detListCurrFrame.at(ii).bb_left + detListCurrFrame.at(ii).bb_right) /
                                              2.f);     //2D column
            auto center_r_ = static_cast<int>((detListCurrFrame.at(ii).bb_top + detListCurrFrame.at(ii).bb_bottom) /
                                              2.f);    //2D row

            auto xmin = detListCurrFrame.at(ii).bb_left;
            auto ymin = detListCurrFrame.at(ii).bb_top;
            auto xmax = detListCurrFrame.at(ii).bb_right;
            auto ymax = detListCurrFrame.at(ii).bb_bottom;

//            std::cout << "xmin: " << xmin << ", ymin: " <<ymin<<", xmax: " <<xmax<<", ymax: "<< ymax << std::endl;

            if (ymax >= (float) Height_crp) ymax = Height_crp - 1.0f;
            if (xmax >= (float) Width_crp) xmax = Width_crp - 1.0f;
            if (ymin < 0.f) ymin = 0.f;
            if (xmin < 0.f) xmin = 0.f;
            int median_kernel = static_cast<int>(std::min(xmax - xmin, ymax - ymin) / 2);

//        if(compact_classLabels_[i] != "traffic light") {
            int dis = 0;
            if (enableStereo)
                dis = static_cast<int>(Util::median_mat(disparityFrame, center_c_, center_r_,
                                                        median_kernel));  // find 3x3 median

            auto rect = cv::Rect_<int>(static_cast<int>(xmin),
                                       static_cast<int>(ymin),
                                       static_cast<int>(xmax - xmin),
                                       static_cast<int>(ymax - ymin));

            cv::Mat hsvBBox, mask, clrBBox, resizeClrBBox, resizeMask;
            camImageCopy_(rect).copyTo(clrBBox);
            mask = occlutionMap(rect, ii, false);

            std::vector<cv::Point3f> cent_2d, cent_3d;
            Blob outputObs(xmin, ymin, xmax - xmin, ymax - ymin);
            outputObs.category = detListCurrFrame.at(ii).objCLass;
            outputObs.probability = detListCurrFrame.at(ii).det_conf;

            if (outputObs.category != "others" && rect.area() > 300) {
                if (outputObs.category == "person"){
                    cellsX = 3;
                    cellsY = 4;//6;
                    bboxResize = cv::Size(60,180);
                }
                cv::resize(clrBBox,resizeClrBBox,bboxResize);//bboxResize
                cv::resize(mask,resizeMask,bboxResize);//bboxResize
                cv::cvtColor(resizeClrBBox, hsvBBox, CV_BGR2HSV);
                calculateHistogram(outputObs, hsvBBox, resizeMask, cellsX, cellsY);
                calculateLPBH(outputObs, resizeClrBBox, cellsX, cellsY);

                if (enableStereo) {
                    if (dis > min_disparity) { //(3600-200)*(dis-12)/(128-12)
                        outputObs.position_3d[0] = x3DPosition[center_c_][dis];
                        outputObs.position_3d[1] = y3DPosition[center_r_][dis];
                        outputObs.position_3d[2] = depth3D[dis];
                        double xmin_3d, xmax_3d, ymin_3d, ymax_3d;
                        xmin_3d = x3DPosition[static_cast<int>(xmin)][dis];
                        xmax_3d = x3DPosition[static_cast<int>(xmax)][dis];
                        ymin_3d = y3DPosition[static_cast<int>(ymin)][dis];
                        ymax_3d = y3DPosition[static_cast<int>(ymax)][dis];
                        outputObs.diameter = abs(static_cast<int>(xmax_3d - xmin_3d));
                        outputObs.height = abs(static_cast<int>(ymax_3d - ymin_3d));
                        outputObs.disparity = dis;
                        currentFrameBlobs.push_back(outputObs);

                    }
                } else {
                    if (rect.area() > 400)
                        currentFrameBlobs.push_back(outputObs);
                }
            }
        }
//    std::cout<<"currentFrameBlobs: "<<currentFrameBlobs.size()<<std::endl;
    }

    Tracking();

    for (int i = 0; i < compact_numClasses_; i++) {
        rosBoxes_[i].clear();
        rosBoxCounter_[i] = 0;
    }
    return nullptr;
}

void YoloObjectDetector::matchCurrentFrameBlobsToExistingBlobs() {

    int simHeight = (int) blobs.size();
    int simWidth = (int) currentFrameBlobs.size();
    cv::Mat localizeDisSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(1.0));
    cv::Mat appDisSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(1.0));
//    cv::Mat relativeDisSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(1.0));
    cv::Mat overallDisSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(1.0));

//     std::cout<<"Debug matchCurrentDetsToTracks 1"<<std::endl;

    for (int c = 0; c < simWidth; c++) {
        Blob &currDet = currentFrameBlobs[c];

        for (int r = 0; r < simHeight; r++) {
            Blob &currentTrackInList = blobs[r];
            if (currentTrackInList.blnStillBeingTracked) {
                if (currDet.category == currentTrackInList.category) {

                    double appSimilarity = 0.0;
                    int countHist = 0;

                    for (size_t kk = 0; kk < currentTrackInList.hist.size(); kk++) {

                        for (size_t ll = 0; ll < currentTrackInList.hist.at(kk).size(); ll++) {
                            if (!currentTrackInList.occluded.at(kk).at(ll) && !currDet.occluded.back().at(ll)) {
                                appSimilarity += cv::compareHist(currDet.hist.back().at(ll),
                                                                 currentTrackInList.hist.at(kk).at(ll),
                                                                 CV_COMP_CORREL);//HISTCMP_CHISQR_ALT, CV_COMP_CORREL
                                countHist++;
//                            std::cout<<"inside loop"<<std::endl;
                            }
                        }
                    }

                    if (countHist > 0)
                        appSimilarity /= countHist;

                    appDisSimilarity.at<double>(r, c) = 1.0 - appSimilarity;//1.0 - appSimilarity;

//                    cv::Rect predictedBB = cv::Rect(currDet.boundingRects.back().x, currDet.boundingRects.back().y, static_cast<int>(currentTrackInList.predictedWidth), static_cast<int>(currentTrackInList.predictedHeight));//currentTrackInList.preditcRect;

                    cv::Rect predictedBB;
                    int normWidth = currentTrackInList.boundingRects.back().width;
                    double distanceToEachOther;

                    if (currentTrackInList.counter > 4) {
                        predictedBB = currentTrackInList.preditcRect;
                        distanceToEachOther =
                                cv::norm(currentTrackInList.predictedNextPositionf - currDet.centerPositions.back()) /
                                normWidth;
                    } else {
                        predictedBB = currentTrackInList.boundingRects.back();
                        distanceToEachOther =
                                cv::norm(currentTrackInList.centerPositions.back() - currDet.centerPositions.back()) /
                                normWidth;
                    }
//                    cv::Rect predictedBB = currentTrackInList.preditcRect;
                    cv::Rect intersection = predictedBB & currDet.boundingRects.back();
                    cv::Rect unio = predictedBB | currDet.boundingRects.back();
                    double iou = (double) intersection.area() / unio.area();
                    localizeDisSimilarity.at<double>(r, c) = 1.0 - iou;


                    if (distanceToEachOther > 1.0)
                        distanceToEachOther = 1.0;

                    double lbphDist = 1.0 - compareHist(currDet.lpbHist, currentTrackInList.lpbHist, CV_COMP_CORREL); //HISTCMP_CHISQR_ALT, CV_COMP_CORREL
//                    overallDisSimilarity.at<double>(r, c) = lbphDist;
//                    overallDisSimilarity.at<double>(r, c) = appDisSimilarity.at<double>(r, c);

                    overallDisSimilarity.at<double>(r, c) = (appDisSimilarity.at<double>(r, c)+localizeDisSimilarity.at<double>(r,c)+lbphDist+distanceToEachOther)/4.0;

                }
            }
        }
    }


    std::vector<std::vector<double> > costMatrix;
//    cv::Mat overallDisSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(maxDisSim));

    for (int r = 0; r < simHeight; r++) {
        std::vector<double> costForEachTrack;
        for (int c = 0; c < simWidth; c++) {
            costForEachTrack.push_back(overallDisSimilarity.at<double>(r, c));
        }
        costMatrix.push_back(costForEachTrack);
//        std::cout<<std::endl;
    }

//    std::cout<<"costMatrix: "<<costMatrix.size()<<", "<<costMatrix[0].size()<<"; simHeight: "<<simHeight<<", simWidth: "<<simWidth<<std::endl;
//    std::cout<<"MAtching dets to tracks"<<std::endl;
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    double hungarianCost = HungAlgo.Solve(costMatrix, assignment);

    for (int trackID = 0; trackID < costMatrix.size(); trackID++) {
//        std::cout << trackID << "," << assignment[trackID] << "\t";
        if (assignment[trackID] > -1) {
            Blob &currentFrmDet = currentFrameBlobs.at(static_cast<unsigned long>(assignment[trackID]));
            double disSimilarityVal = overallDisSimilarity.at<double>(trackID, assignment[trackID]);
//            std::cout << trackID << "," << assignment[trackID] <<", "<<disSimilarityVal<< "\t";
            if ((!blobs[trackID].trackedInCurrentFrame)
                && disSimilarityVal < 0.75) { //TODO: define varying threshold for each sequence

                currentFrmDet.trackedInCurrentFrame = true;
                addBlobToExistingBlobs(currentFrmDet, blobs, trackID, true);

            }
        }
    }

}

void YoloObjectDetector::addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex, bool isDet) {

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
    existingBlobs[intIndex].boundingRects.push_back(currentFrameBlob.boundingRects.back());
    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].probability = currentFrameBlob.probability;
    existingBlobs[intIndex].disparity = currentFrameBlob.disparity;
    existingBlobs[intIndex].position_3d = currentFrameBlob.position_3d;
//    existingBlobs[intIndex].category = currentFrameBlob.category;
    existingBlobs[intIndex].height = currentFrameBlob.height;
    existingBlobs[intIndex].diameter = currentFrameBlob.diameter;

    if (existingBlobs[intIndex].hist.size() >= 3) {

        if(existingBlobs[intIndex].overallOcclusion.at(0)<=existingBlobs[intIndex].overallOcclusion.at(2)){
            existingBlobs[intIndex].overallOcclusion.erase(existingBlobs[intIndex].overallOcclusion.begin());
            existingBlobs[intIndex].hist.erase(existingBlobs[intIndex].hist.begin());
            existingBlobs[intIndex].occluded.erase(existingBlobs[intIndex].occluded.begin());
        } else {
            existingBlobs[intIndex].overallOcclusion.pop_back();
            existingBlobs[intIndex].hist.pop_back();
            existingBlobs[intIndex].occluded.pop_back();
        }
    }

    existingBlobs[intIndex].hist.push_back(currentFrameBlob.hist.back());
    existingBlobs[intIndex].occluded.push_back(currentFrameBlob.occluded.back());
    existingBlobs[intIndex].overallOcclusion.push_back(currentFrameBlob.overallOcclusion.back());

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
    existingBlobs[intIndex].blnAlreadyTrackedInThisFrame = true;
    existingBlobs[intIndex].counter ++;
    existingBlobs[intIndex].intNumOfConsecutiveFramesWithoutAMatch =0;

    if (isDet) {
        existingBlobs[intIndex].numOfConsecutiveFramesWithoutDetAsso = 0;
        existingBlobs[intIndex].lpbHist = currentFrameBlob.lpbHist;
    } else {
        existingBlobs[intIndex].numOfConsecutiveFramesWithoutDetAsso++;
    }

    //update motion model
//    existingBlobs[intIndex].meas.at<float>(0) = currentFrameBlob.meas.at<float>(0);
//    existingBlobs[intIndex].meas.at<float>(1) = currentFrameBlob.meas.at<float>(1);
//    existingBlobs[intIndex].meas.at<float>(2) = currentFrameBlob.meas.at<float>(2);
//    existingBlobs[intIndex].meas.at<float>(3) = currentFrameBlob.meas.at<float>(3);
//    existingBlobs[intIndex].kf.correct(existingBlobs[intIndex].meas); // Kalman Correction
//    existingBlobs[intIndex].UpdateAUKF(true);

}

void YoloObjectDetector::addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
    currentFrameBlob.blnStillBeingTracked = true;
    currentFrameBlob.blnAlreadyTrackedInThisFrame = true;

    existingBlobs.push_back(currentFrameBlob);
}

void YoloObjectDetector::trackingFNs() {

//    cv::Mat outputBP = camImageCopy_.clone();

//    std::vector<cv::Scalar> colors;
//    cv::RNG rng(0);
//    for(int i=0; i < trackList.size(); i++)
//        colors.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));


    /*tracking Result*/
//    std::cout<<"-------------Tracking Results-------------" <<std::endl;
//    for (long int i = 0; i < trackList.size(); i++) {
//        if (trackList[i].currentMatchFoundOrNewTrack) {
//            cv::rectangle(output, trackList[i].boundingRects.back(), colors.at(i), 2);
//            int rectMinX = trackList[i].boundingRects.back().x;
//            int rectMinY = trackList[i].boundingRects.back().y;
//            cv::rectangle(output, cv::Rect(rectMinX, rectMinY, 40, 20), colors.at(i), CV_FILLED);
//
//            std::ostringstream str;
//            str << i ;//<<", "<<trackList[i].probability;
//            cv::putText(output, str.str(), cv::Point(rectMinX, rectMinY+16) , CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255,255));
//        }
//    }


    int simHeight = (int) blobs.size();
    int simWidth = (int) currentFrameBlobs.size();
    cv::Mat localizeDisSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(1.0));
//    std::cout << "debug 1" << std::endl;

    for (int r = 0; r < simHeight; r++) {
        Blob &existingTrack = blobs[r];
        bool isNotOverlap = true;
        if (existingTrack.blnStillBeingTracked && !existingTrack.trackedInCurrentFrame) {
//        if (existingTrack.stillTracked && existingTrack.numOfConsecutiveFramesWithoutAMatch == 1 &&
//            existingTrack.counter > 4) {
//            existingTrack.preditcRect = getPredictedTrackLocation(static_cast<unsigned long>(r));

//            cv::rectangle(output, existingTrack.preditcRect, cv::Scalar(0,0,0), 2);
//            int rectMinX = existingTrack.preditcRect.x;
//            int rectMinY = existingTrack.preditcRect.y;
//            cv::rectangle(output, cv::Rect(rectMinX, rectMinY, 40, 20), cv::Scalar(0,0,0), CV_FILLED);
//
//            std::ostringstream str;
//            str << r ;//<<", "<<trackList[i].probability;
//            cv::putText(output, str.str(), cv::Point(rectMinX, rectMinY+16) , CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255,255));

            for (int c = 0; c < simWidth; c++) {
                Blob &currDet = currentFrameBlobs[c];
                if (!currDet.trackedInCurrentFrame) {
                    double iou = (double) (existingTrack.preditcRect & currDet.boundingRects.back()).area() /
                                 (existingTrack.preditcRect | currDet.boundingRects.back()).area();
                    localizeDisSimilarity.at<double>(r, c) = 1.0 - iou;

                    if (iou > 0.5) {
                        isNotOverlap = false;
                    }
                }
            }

            if (isNotOverlap && existingTrack.intNumOfConsecutiveFramesWithoutAMatch == 0 && existingTrack.counter > 4 &&
                existingTrack.numOfConsecutiveFramesWithoutDetAsso < 4) {
                cv::Rect trackLastBBox = existingTrack.boundingRects.back();
                cv::Rect imgLeftExit = cv::Rect_<int>(0, 0, 2 * trackLastBBox.width, Height_crp);
                cv::Rect imgRightExit = cv::Rect_<int>((Width_crp - trackLastBBox.width), 0,
                                                       trackLastBBox.width, Height_crp);
                double intersectionLeft = ((double) (imgLeftExit & trackLastBBox).area()) / trackLastBBox.area();
                double intersectionRight = ((double) (imgRightExit & trackLastBBox).area()) / trackLastBBox.area();

                if (intersectionLeft != 1.0 && intersectionRight != 1.0) {

                    float width = existingTrack.boundingRects.back().width;//existingTrack.preditcRect.width;//existingTrack.predictedWidth;//predictedBBox.width;
                    float height = existingTrack.boundingRects.back().height;//existingTrack.preditcRect.height;//existingTrack.predictedHeight;//
                    float xmin = existingTrack.preditcRect.x;//- width/ 2;
                    float ymin = existingTrack.preditcRect.y;// - height/2;

                    Blob tmpTrack(xmin, ymin, width, height);
                    tmpTrack.category = existingTrack.category;
                    tmpTrack.probability = existingTrack.probability;

                    if (xmin < 0.0) {
                        width += xmin;
                        xmin = 0.0;
                    }
                    if (ymin < 0.0) {
                        height += ymin;
                        ymin = 0.0;
                    }

                    float width_del = xmin + width - float(Width_crp);
                    float height_del = ymin + height - float(Height_crp);
                    if (width_del > 0.0)
                        width -= width_del;
                    if (height_del > 0.0)
                        height -= height_del;

                    if (width <= 0.0 || height <= 0.0 || xmin >= Width_crp || ymin >= Height_crp ||
                        width > Width_crp || height > Height_crp) {
                        existingTrack.blnStillBeingTracked = false;
                    } else {

                        cv::Rect hsvBBox = cv::Rect_<int>(static_cast<int>(xmin), static_cast<int>(ymin),
                                                          static_cast<int>(width), static_cast<int>(height));

                        cv::Mat hsvBBoxROI, mask, clrBBoxROI, resizeClrBBox, resizeMask;
                        camImageCopy_(hsvBBox).copyTo(clrBBoxROI);
                        mask = occlutionMap(hsvBBox, detListCurrFrame.size(), true);

                        if (tmpTrack.category == "person"){
                            cellsX = 3;
                            cellsY = 4;//6;
                            bboxResize = cv::Size(60,180);
                        }

                        cv::resize(clrBBoxROI, resizeClrBBox, bboxResize);
                        cv::resize(mask, resizeMask, bboxResize);
                        cv::cvtColor(resizeClrBBox, hsvBBoxROI, CV_BGR2HSV);

                        calculateHistogram(tmpTrack, hsvBBoxROI, resizeMask, cellsX, cellsY);

                        if (tmpTrack.overallOcclusion.back() < 0.8) {
                            double appSimilarity = 0.0;
                            int countHist = 0;

                            for (size_t kk = 0; kk < existingTrack.hist.size(); kk++) {
                                for (size_t ll = 0; ll < existingTrack.hist.at(kk).size(); ll++) {
                                    if (!existingTrack.occluded.at(kk).at(ll) && !tmpTrack.occluded.back().at(ll)) {
                                        appSimilarity += cv::compareHist(tmpTrack.hist.back().at(ll),
                                                                         existingTrack.hist.at(kk).at(ll),
                                                                         CV_COMP_CORREL);
                                        countHist++;
                                    }
                                }
                            }

                            if (countHist > 0)
                                appSimilarity /= countHist;

//                            std::cout<<r<<": "<<appSimilarity<<std::endl;

                            if (appSimilarity > 0.6) {

                                if (enableStereo) {
                                    auto center_c_ = static_cast<int>((xmin+width)/2);     //2D column
                                    auto center_r_ = static_cast<int>((ymin+height)/2);    //2D row

                                    int median_kernel = static_cast<int>(std::min(width, height) / 2);
                                    int dis = static_cast<int>(Util::median_mat(disparityFrame, center_c_, center_r_,
                                                                                median_kernel));
                                    if (dis > min_disparity && hsvBBox.area() > 300) { //(3600-200)*(dis-12)/(128-12)
                                        tmpTrack.position_3d[0] = x3DPosition[center_c_][dis];
                                        tmpTrack.position_3d[1] = y3DPosition[center_r_][dis];
                                        tmpTrack.position_3d[2] = depth3D[dis];
                                        double xmin_3d, xmax_3d, ymin_3d, ymax_3d;
                                        xmin_3d = x3DPosition[static_cast<int>(xmin)][dis];
                                        xmax_3d = x3DPosition[static_cast<int>(xmin+width)][dis];
                                        ymin_3d = y3DPosition[static_cast<int>(ymin)][dis];
                                        ymax_3d = y3DPosition[static_cast<int>(ymin+height)][dis];
                                        tmpTrack.diameter = abs(static_cast<int>(xmax_3d - xmin_3d));
                                        tmpTrack.height = abs(static_cast<int>(ymax_3d - ymin_3d));
                                        tmpTrack.disparity = dis;
                                        matchedFNs.push_back(tmpTrack);
                                        matchedTrackID.push_back(r);
                                    }
                                } else {
                                    if (hsvBBox.area() > 400) {
                                        matchedFNs.push_back(tmpTrack);
                                        matchedTrackID.push_back(r);
                                    }
                                }

                            }
                        }
                    }
                }
            }

        }
    }

//    cv::imshow("debug123", output);
//    cv::waitKey(waitKeyDelay_);
//    std::cout << "debug 2" << std::endl;

    std::vector<std::vector<double> > costMatrix;
    for (int r = 0; r < simHeight; r++) {
        std::vector<double> costForEachTrack;
        for (int c = 0; c < simWidth; c++) {
            costForEachTrack.push_back(localizeDisSimilarity.at<double>(r, c));
        }
        costMatrix.push_back(costForEachTrack);
    }

//    std::cout << "debug 3" << std::endl;
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    double hungarianCost = HungAlgo.Solve(costMatrix, assignment);

    for (int trackID = 0; trackID < costMatrix.size(); trackID++) {
        if (assignment[trackID] > -1) {
            Blob &currentFrmDet = currentFrameBlobs.at(static_cast<unsigned long>(assignment[trackID]));
            double disSimilarityVal = localizeDisSimilarity.at<double>(trackID, assignment[trackID]);
            Blob &existingTrack = blobs[trackID];
            if ((!existingTrack.trackedInCurrentFrame) && disSimilarityVal < 0.5) {
                matchedFrmIDTrackID.push_back(static_cast<unsigned long>(trackID));
                matchedFrmID.push_back(static_cast<unsigned long>(assignment[trackID]));
            }
        }
    }

//    std::cout << "debug 4" << std::endl;
//    cv::Mat intermediary = camImageCopy_.clone();

    for (unsigned long ii = 0; ii < matchedFrmID.size(); ii++) {
        int &indexID = reinterpret_cast<int &>(matchedFrmIDTrackID.at(ii));
        Blob &existingTrack = blobs[matchedFrmIDTrackID.at(ii)];
        if (!existingTrack.trackedInCurrentFrame) {
            Blob &tmpTrack = currentFrameBlobs[matchedFrmID.at(ii)];
            tmpTrack.trackedInCurrentFrame = true;
            addBlobToExistingBlobs(tmpTrack, blobs, indexID, true);
        }
    }

//    for (long int i = 0; i < trackList.size(); i++) {
//        if (trackList[i].currentMatchFoundOrNewTrack) {
//            cv::rectangle(intermediary, trackList[i].boundingRects.back(), cv::Scalar(0,0,0), 2);
//            int rectMinX = trackList[i].boundingRects.back().x;
//            int rectMinY = trackList[i].boundingRects.back().y;
//            cv::rectangle(intermediary, cv::Rect(rectMinX, rectMinY, 40, 20), cv::Scalar(0,0,0), CV_FILLED);
//
//            std::ostringstream str;
//            str << i;//<<", "<<trackList[i].probability;
//            cv::putText(intermediary, str.str(), cv::Point(rectMinX, rectMinY + 16), CV_FONT_HERSHEY_PLAIN, 1,
//                        CV_RGB(255, 255, 255));
//        }
//    }

//    cv::imshow("inter2",intermediary);
//    std::cout << "debug 5" << std::endl;

    for (unsigned long ii = 0; ii < matchedFNs.size(); ii++) {
        int &indexID = reinterpret_cast<int &>(matchedTrackID.at(ii));
        Blob &existingTrack = blobs[matchedTrackID.at(ii)];
        if (!existingTrack.trackedInCurrentFrame) {
            Blob &tmpTrack = matchedFNs[ii];
            tmpTrack.probability = existingTrack.probability;
            addBlobToExistingBlobs(tmpTrack, blobs, indexID, false);
        }
    }

//    for (long int i = 0; i < trackList.size(); i++) {
//        if (trackList[i].currentMatchFoundOrNewTrack) {
//            cv::rectangle(intermediary, trackList[i].boundingRects.back(), cv::Scalar(0,255,0), 2);
//            int rectMinX = trackList[i].boundingRects.back().x;
//            int rectMinY = trackList[i].boundingRects.back().y;
//            cv::rectangle(intermediary, cv::Rect(rectMinX, rectMinY, 40, 20), cv::Scalar(0,255,0), CV_FILLED);
//
//            std::ostringstream str;
//            str << i;//<<", "<<trackList[i].probability;
//            cv::putText(intermediary, str.str(), cv::Point(rectMinX, rectMinY + 16), CV_FONT_HERSHEY_PLAIN, 1,
//                        CV_RGB(255, 255, 255));
//        }
//    }

//    cv::imshow("inter1",intermediary);
}

void YoloObjectDetector::addNewTracks() {

//    std::cout << "debug 6" << std::endl;

    for (int c = 0; c < currentFrameBlobs.size(); c++) {
        Blob &currentFrmDet = currentFrameBlobs.at(c);
        if (!currentFrmDet.trackedInCurrentFrame) {
            matchedTrackID.push_back(blobs.size());
            addNewBlob(currentFrmDet, blobs);
        }
    }
}

void YoloObjectDetector::updateUnmatchedTracks() {

    cv::Rect imgFrame = cv::Rect_<int>(0, 0, Width_crp, Height_crp);
    int objectIDinTrack = 0;
    for (auto &existingTrack : blobs) {

        if (!existingTrack.blnCurrentMatchFoundOrNewBlob) {
//            double intersectionWithFrame = ((double)(imgFrame & existingTrack.preditcRect).area()) / existingTrack.preditcRect.area();
            double intersectionWithFrame = ((double) (imgFrame & existingTrack.boundingRects.back()).area()) /
                                           existingTrack.boundingRects.back().area();
            if (intersectionWithFrame < 0.6) //0.3
                existingTrack.blnStillBeingTracked = false;
            existingTrack.intNumOfConsecutiveFramesWithoutAMatch++;
        }
//        std::cout<<"ID: "<<objectIDinTrack<<",  Unmatched Frames: "<<existingTrack.numOfConsecutiveFramesWithoutAMatch<<", still tracked: "<<existingTrack.stillTracked<<std::endl;
        if (existingTrack.intNumOfConsecutiveFramesWithoutAMatch >= trackLife) {
            existingTrack.blnStillBeingTracked = false;
        }
        objectIDinTrack++;

    }
}

void YoloObjectDetector::Tracking (){

    if (blnFirstFrame) {
        prvImageTime = image_time_;
        if (!currentFrameBlobs.empty()){
            blnFirstFrame = false;
            for (auto &currentFrameBlob : currentFrameBlobs){
                currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
                currentFrameBlob.trackedInCurrentFrame = true;
                currentFrameBlob.blnStillBeingTracked = true;
                blobs.push_back(currentFrameBlob);
            }
        }
    } else {
        for (auto &existingBlob : blobs) {
            existingBlob.blnCurrentMatchFoundOrNewBlob = false;
            existingBlob.blnAlreadyTrackedInThisFrame = false;

            existingBlob.predictNextPosition();
            existingBlob.predictWidthHeight();
            int xmin = static_cast<int>(existingBlob.predictedNextPosition.x - existingBlob.predictedWidth/2);
            int ymin = static_cast<int>(existingBlob.predictedNextPosition.y - existingBlob.predictedHeight/2);
            existingBlob.preditcRect = cv::Rect(xmin, ymin, static_cast<int>(existingBlob.predictedWidth), static_cast<int>(existingBlob.predictedHeight));
            // >>>> Matrix A
//            double timeDiff = (image_time_-prvImageTime).toNSec() * 1e-9;
//            std::cout<< timeDiff <<std::endl;
//            prvImageTime = image_time_;
//            auto dT = static_cast<float>(0.04 + (0.04 * existingBlob.intNumOfConsecutiveFramesWithoutAMatch));
//            existingBlob.kf.transitionMatrix.at<float>(2) = dT;//dT;
//            existingBlob.kf.transitionMatrix.at<float>(9) = dT;//dT;
            // <<<< Matrix A
//            existingBlob.state = existingBlob.kf.predict();
//            existingBlob.preditcRect = existingBlob.GetRectPrediction();
        }

//            std::cout<<"blob prediction finished"<<std::endl;

        if (!currentFrameBlobs.empty()){
            matchCurrentFrameBlobsToExistingBlobs();
            trackingFNs();
            addNewTracks();
            updateUnmatchedTracks();
        } else {
            for (auto &existingBlob : blobs) {
                if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
                    existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
//                    existingBlob.UpdateAUKF(false);
                }
                if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= trackLife) {
                    existingBlob.blnStillBeingTracked = false;
                    //blobs.erase(blobs.begin() + i);
                }
            }
        }
//            std::cout<<"blob association finished"<<std::endl;
    }

//    if (blnFirstFrame) {
//        blnFirstFrame = false;
//        for (auto &currentFrameBlob : currentFrameBlobs)
//            blobs.push_back(currentFrameBlob);
//    } else
//        matchCurrentFrameBlobsToExistingBlobs();

    currentFrameBlobs.clear();
    matchedTrackID.clear();
    matchedFNs.clear();
    matchedFrmID.clear();
    matchedFrmIDTrackID.clear();
}

void YoloObjectDetector::CreateMsg(){
//    updateOutput = true;
    cv::Mat output1;
//    if (enableStereo)
//        output1 = disparityFrame.clone();

    output = camImageCopy_.clone();// buff_cv_l_.clone();
    if(output.type() == CV_8UC1)
        cv::cvtColor(output, color_out, CV_GRAY2RGB);
    else
        color_out = output;

    std::vector<cv::Scalar> colors;
    cv::RNG rng(0);
    for(int i=0; i < blobs.size(); i++)
        colors.emplace_back(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

    for (long int i = 0; i < blobs.size(); i++) {
//            if (blobs[i].blnStillBeingTracked == true) {
        std::ostringstream str_;
        if (blobs[i].blnCurrentMatchFoundOrNewBlob) {
            cv::rectangle(color_out, blobs[i].boundingRects.back(), colors.at(i), 2);
            int rectMinX = blobs[i].boundingRects.back().x;
            int rectMinY = blobs[i].boundingRects.back().y;
            cv::rectangle(color_out, cv::Rect(rectMinX, rectMinY, blobs[i].boundingRects.back().width, 20), colors.at(i), CV_FILLED);
            int distance = static_cast<int>(sqrt(pow(blobs[i].position_3d[2],2)+pow(blobs[i].position_3d[0],2)));
            str_ << distance <<"m, "<<i;//<<"; "<<blobs[i].disparity;
//            str_ << i;
            cv::putText(color_out, str_.str(), cv::Point(rectMinX, rectMinY+16) , CV_FONT_HERSHEY_PLAIN, 0.8, CV_RGB(255,255,255));

//            cv::Rect predRect;
//            predRect.width = static_cast<int>(blobs[i].t_lastRectResult.width);//static_cast<int>(blobs[i].state.at<float>(4));
//            predRect.height = static_cast<int>(blobs[i].t_lastRectResult.height);//static_cast<int>(blobs[i].state.at<float>(5));
//            predRect.x = static_cast<int>(blobs[i].t_lastRectResult.x);//static_cast<int>(blobs[i].state.at<float>(0) - predRect.width / 2);
//            predRect.y = static_cast<int>(blobs[i].t_lastRectResult.y);//static_cast<int>(blobs[i].state.at<float>(1) - predRect.height / 2);
//
//            cv::rectangle(color_out, blobs[i].preditcRect, CV_RGB(255,255,255), 2);
//            cv::rectangle(output1, blobs[i].currentBoundingRect, cv::Scalar( 255, 255, 255 ), 2);
        }

    }

    cv::Mat cm_disp;

    if (enableStereo) {
        // To better visualize the result, apply a colormap to the computed disparity
        double min, max;
        minMaxIdx(disparityFrame, &min, &max);
//            std::cout << "disp min " << min << std::endl << "disp max " << max << std::endl;
        cv::Mat scaledDisparityMap;
        convertScaleAbs(disparityFrame, scaledDisparityMap, 3.1);
        applyColorMap(scaledDisparityMap, cm_disp, cv::COLORMAP_JET);
    }

    if(viewImage_) {
        cv::imshow("Detection and Tracking", color_out);
        if (enableStereo) {
            cv::imshow("Disparity", cm_disp);
//            cv::imshow("ObsDisparity", ObsDisparity * 255 / disp_size);
        }
       cv::waitKey(waitKeyDelay_);
    } else {
        if (!publishDetectionImage(color_out, trackingPublisher_)) {
            ROS_DEBUG("Tracking image has not been broadcasted.");
        }

        if (!publishDetectionImage(cm_disp, disparityColorPublisher_)) {
            ROS_DEBUG("Disparity image has not been broadcasted.");
        }

        if (!publishDetectionImage(ObstacleDetector.left_rect_clr, obstacleMaskPublisher_)) {
            ROS_DEBUG("Obstacle image has not been broadcasted.");
        }

        if (!publishDetectionImage(ObstacleDetector.slope_map, slopePublisher_)) {
            ROS_DEBUG("Slope map image has not been broadcasted.");
        }
    }

    if(enableEvaluation_){
        sprintf(s, "f%03d.txt", frame_num);
        sprintf(im, "f%03d.png", frame_num);
//        file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/" + s;
//        file_name = ros::package::getPath("cubicle_detect") + "/dis_1/" + s;
//        img_name = ros::package::getPath("cubicle_detect") + "/seq_1/" + im;
        img_name = std::string("/home/ugv/seq_1/") + im;
//    file.open(file_name.c_str(), std::ios::app);
    }
    int cate = 0;
    for (unsigned long int i = 0; i < blobs.size(); i++) {
        obstacle_msgs::obs tmpObs;
        tmpObs.identityID = i;

        tmpObs.centerPos.x = static_cast<float>(blobs[i].position_3d[0]);
        tmpObs.centerPos.y = static_cast<float>(blobs[i].position_3d[1]);
        tmpObs.centerPos.z = static_cast<float>(blobs[i].position_3d[2]);
        tmpObs.diameter = static_cast<float>(blobs[i].diameter);
        tmpObs.height = static_cast<float>(blobs[i].height);
        tmpObs.counter = blobs[i].counter;
        tmpObs.classes = blobs[i].category;
        tmpObs.probability = static_cast<float>(blobs[i].probability);
        if (blobs[i].blnCurrentMatchFoundOrNewBlob) {

            if(!blobs[i].category.empty() ) {

                if(enableEvaluation_){
                    if( (blobs[i].category == "vehicle") || (blobs[i].category == "bicycle") )
                        cate = 0;
                    else if(blobs[i].category == "person")
                        cate = 1;
                }

                tmpObs.xmin = static_cast<unsigned int>(blobs[i].boundingRects.back().x);
                tmpObs.ymin = static_cast<unsigned int>(blobs[i].boundingRects.back().y);
                tmpObs.xmax = tmpObs.xmin + blobs[i].boundingRects.back().width;
                tmpObs.ymax = tmpObs.ymin + blobs[i].boundingRects.back().height;
//            tmpObs.histogram = blobs[i].obsHog;

                obstacleBoxesResults_.obsData.push_back(tmpObs);

//            ROS_WARN("center ID: %d | type: %s\nx: %f| y: %f| z: %f \n", i, tmpObs.classes,
//                    tmpObs.centerPos.x, tmpObs.centerPos.y, tmpObs.centerPos.z);

                ////*--------------Generate Evaluation files----------------------*////
                if(enableEvaluation_){
                    file << i << " " << blobs[i].boundingRects.back().x << " " << blobs[i].boundingRects.back().y << " "
                         << blobs[i].boundingRects.back().x + blobs[i].boundingRects.back().width << " " <<
                         blobs[i].boundingRects.back().y + blobs[i].boundingRects.back().height << " " << cate
                         << std::endl;
                }
            }
        }
    }
    if(enableEvaluation_){
        cv::imwrite(img_name, color_out);
//        cv::imwrite(file_name, output1*255/disp_size);
    }
}

void YoloObjectDetector::generateStaticObsDisparityMap() {
    for (auto & blob : blobs) {
        if (blob.blnCurrentMatchFoundOrNewBlob) {
            ObsDisparity(blob.boundingRects.back()).setTo(cv::Scalar::all(0));
        }
    }
}

void YoloObjectDetector::Process(){

    demoTime_ = what_time_is_it_now();
//    std::cout<<"before fetchInThread"<<std::endl;
    // 1. To get the image data
    fetchInThread();

//    std::cout<<"before detect_thread"<<std::endl;
    // 2. YOLOv3 to detect 2D bounding boxes
    if (enableClassification)
        detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
//        detectInThread();

//    std::cout<<"before stereo_thread"<<std::endl;
    // 3. Stereo matching to get disparity map
    if (enableStereo)
        stereo_thread = std::thread(&YoloObjectDetector::stereoInThread, this);
//        stereoInThread();

    if (enableClassification)
        detect_thread.join();

//    std::cout<<"before ObstacleDetector"<<std::endl;
    if (enableStereo) {
        stereo_thread.join();
        double obs_time_ = what_time_is_it_now();
        // 4. Obstacle detection according to the u-v disparity
        ObstacleDetector.ExecuteDetection(disparityFrame, left_rectified);
        obs_fps_ = 1./(what_time_is_it_now() - obs_time_);
    }

//    std::cout<<"before trackInThread"<<std::endl;

    // 5. Put bounding boxes into 3D blobs and track them.
    trackInThread();

//    std::cout<<"before ObsDisparity"<<std::endl;

    // 6. Get obstacle disparity map by filtering ground and moving objects
    ObsDisparity = cv::Mat(camImageCopy_.size(), CV_8UC1, cv::Scalar::all(0));
    if (enableClassification && enableStereo){
        ObsDisparity = ObstacleDetector.obstacleDisparityMap.clone();
        if(filter_dynamic_)
            generateStaticObsDisparityMap();

        disparity_obs.header.stamp = image_time_;
        disparity_obs.header.frame_id = obs_disparityFrameId;
        cv_bridge::CvImage out_msg;
        out_msg.header.frame_id = obs_disparityFrameId;
        out_msg.header.stamp = image_time_;
        out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
        out_msg.image = ObsDisparity;
        disparity_obs.image = *out_msg.toImageMsg();

        disparity_obs.f = focal;
        disparity_obs.T = stereo_baseline_;
        disparity_obs.min_disparity = min_disparity;
        disparity_obs.max_disparity = disp_size; //128

        obs_disparityPublisher_.publish(disparity_obs);
    }

    fps_ = 1./(what_time_is_it_now() - demoTime_);
//    demoTime_ = what_time_is_it_now();

//    if (enableClassification)
//        displayInThread();

    CreateMsg();

    obstacleBoxesResults_.header.stamp = image_time_;
    obstacleBoxesResults_.header.frame_id = pub_obs_frame_id;
    obstacleBoxesResults_.real_header.stamp = ros::Time::now();
    obstacleBoxesResults_.real_header.frame_id = pub_obs_frame_id;
    obstaclePublisher_.publish(obstacleBoxesResults_);

    obstacleBoxesResults_.obsData.clear();
    obstacleBoxesResults_.laneData.clear();

//    char name[256];
//    sprintf(name, "%s_%08d", "/home/ugv/yolo/f", frame_num);
//    save_image(buff_, name);//[(buffIndex_ + 1) % 3], name);

    if ( frame_num%30==1 ) {
        printf("FPS:%.1f, Stereo:%.1f, Obs:%.1f, Classification:%.1f\n", fps_, stereo_fps_, obs_fps_, classi_fps_);
    }

    frame_num++;
}


} /* namespace darknet_ros*/
