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
#include <ros/package.h>
// Check for xServer
#include <X11/Xlib.h>
#include <algorithm>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

using namespace message_filters;

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle nh_p)
    : nodeHandle_(nh),
      nodeHandle_pub(nh_p),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0),
      use_grey(false),
      blnFirstFrame(true),
      globalframe(0),
      isDepthNew(false)
{
  ROS_INFO("[ObstacleDetector] Node started.");

  // Read Cuda Info and ROS parameters from config file.
  if (!CudaInfo() || !readParameters()) {
    ros::requestShutdown();
  }
//  mpDetection = new Detection(this, nodeHandle_);
//  nullHog.assign(36, 0.0);
  init();

  SGM = new disparity_sgm(7, 86);
  SGM->init_disparity_method(7, 86);

//  mpDepth_gen_run = new std::thread(&Detection::Run, mpDetection);

  hog_descriptor = new Util::HOGFeatureDescriptor(8, 2, 9, 180.0);
  img_name = ros::package::getPath("cubicle_detect") + "/seq_1/f000.png";
  file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/f000.txt";
  frame_num = 0;
}

YoloObjectDetector::~YoloObjectDetector()
{
  SGM->finish_disparity_method();
    {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
  free(depthTable);
  free(xDirectionPosition);
  free(yDirectionPosition);
  free(cfg);
  free(weights);
  free(detectionNames);
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

bool YoloObjectDetector::readParameters()
{
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
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[ObstacleDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Look up table initialization
  counter = 0;

  nodeHandle_.param<int>("min_disparity", min_disparity, 12);
  nodeHandle_.param<int>("disparity_scope", disp_size, 128);
  nodeHandle_.param<bool>("use_grey", use_grey, false);
  nodeHandle_.param<int>("scale", Scale, 1);

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolo_bdd_c1_165300.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolo_bdd_c1.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
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

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, nullptr, 1, 0.5, 0, 0, 0, 0);


//  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

//    yolo();

  // Initialize publisher and subscriber.
//  std::string cameraTopicName;
//  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;
  std::string obstacleBoxesTopicName;
  int obstacleBoxesQueueSize;

  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  nodeHandle_.param("publishers/obstacle_boxes/topic", obstacleBoxesTopicName,
                    std::string("/cubicle_detection/long_map_msg"));
  nodeHandle_.param("publishers/obstacle_boxes/queue_size", obstacleBoxesQueueSize, 1);
  nodeHandle_.param("publishers/obstacle_boxes/frame_id", pub_obs_frame_id, std::string("long_obs"));

  objectPublisher_ = nodeHandle_pub.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch);

  obstaclePublisher_ = nodeHandle_pub.advertise<obstacle_msgs::MapInfo>(
          obstacleBoxesTopicName, obstacleBoxesQueueSize);

  detectionImagePublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);

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
  left_info_copy->header.frame_id = "stereo";
  right_info_copy->header.frame_id = "stereo";

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

  u0 = left_info->K[2];
  v0 = left_info->K[5];
  focal = left_info->K[0];
  Width = left_info->width;
  Height = left_info->height;

  Width /= Scale;
  Height /= Scale;

  int widthMiss = Width%4;
  int heightMiss = Height%4;

  Width = Width + widthMiss;
  Height = Height + heightMiss;

  assert(intrinsicLeft == intrinsicRight);

  const cv::Matx33d &intrinsic = intrinsicLeft;

  // Save the baseline
  stereo_baseline_ = stereoCameraModel.baseline();
  ROS_INFO_STREAM("baseline: " << stereo_baseline_);
  assert(stereo_baseline_ > 0);

    int ii;
    xDirectionPosition = static_cast<double **>(calloc(Width, sizeof(double *)));
    for(ii = 0; ii < Width; ii++)
        xDirectionPosition[ii] = static_cast<double *>(calloc(disp_size+1, sizeof(double)));

    yDirectionPosition = static_cast<double **>(calloc(Height, sizeof(double *)));
    for(ii = 0; ii < Height; ii++)
        yDirectionPosition[ii] = static_cast<double *>(calloc(disp_size+1, sizeof(double)));

    depthTable = static_cast<double *>(calloc(disp_size+1, sizeof(double)));
//
//  // get the Region Of Interests (If the images are already rectified but invalid pixels appear)
//  left_roi_ = cameraLeft.rawRoi();
//  right_roi_ = cameraRight.rawRoi();
}

cv::Mat YoloObjectDetector::getDepth(cv::Mat &leftFrame, cv::Mat &rightFrame) {

    float elapsed_time_ms;
    cv::Mat disparity_SGBM(leftFrame.size(), CV_8UC1);

    disparity_SGBM = SGM->compute_disparity_method(leftFrame, rightFrame, &elapsed_time_ms);

     cv::imshow("disparity_SGBM",disparity_SGBM);
     cv::waitKey(1);

    isDepthNew = true;
    return disparity_SGBM;
}

void YoloObjectDetector::DefineLUTs() {

  ROS_WARN("u0: %f | v0: %f | focal: %f | base: %f | width: %d", u0, v0, focal, stereo_baseline_, Width);

    for (int r=0; r<Width; r++) {
        xDirectionPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            xDirectionPosition[r][c]=(r-u0)*stereo_baseline_/c;
//        std::cout<<xDirectionPosition[r][c]<<std::endl;
        }
    }

    for (int r=0; r<Height; r++) {
//    for (int r=300; r<301; r++) {
        yDirectionPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            yDirectionPosition[r][c]=(v0-r)*stereo_baseline_/c;
//      std::cout<<r<<", "<<c<<": "<<yDirectionPosition[r][c]<<"; ";//std::endl;
        }
    }

    depthTable[0] =0;
    for( int i = 1; i < disp_size+1; ++i){
        depthTable[i]=focal*stereo_baseline_/i; //Y*dx/B
//      std::cout<<"i: "<<i<<", "<<depthTable[i]<<"; \n";
    }

}

    void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &image1,
                                            const sensor_msgs::ImageConstPtr &image2,
                                            const sensor_msgs::ImageConstPtr &image3){
//                                            const sensor_msgs::CameraInfoConstPtr& left_info,
//                                            const sensor_msgs::CameraInfoConstPtr& right_info){
        ROS_DEBUG("[ObstacleDetector] Stereo images received.");

        // std::cout<<"Debug starting cameraCallBack"<<std::endl;

        cv_bridge::CvImageConstPtr cam_image1, cam_image2, cv_rgb;

        try {
            cam_image1 = cv_bridge::toCvShare(image1, sensor_msgs::image_encodings::MONO8);
            cam_image2 = cv_bridge::toCvShare(image2, sensor_msgs::image_encodings::MONO8);

            if(use_grey) {
                cv_rgb = cam_image1;
            }
            else {
                cv_rgb = cv_bridge::toCvShare(image3, sensor_msgs::image_encodings::BGR8);
            }

            image_time_ = image1->header.stamp;
            imageHeader_ = image1->header;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if (cam_image1) {
            // std::cout<<"Debug inside cameraCallBack starting first image callback"<<std::endl;

            frameWidth_ = cam_image1->image.size().width;
            // std::cout<<"Debug inside cameraCallBack reading image width "<<frameWidth_<<std::endl;
            frameHeight_ = cam_image1->image.size().height;
            frameWidth_ = frameWidth_ / Scale;
            frameHeight_ = frameHeight_ / Scale;
            // std::cout<<"Debug inside cameraCallBack scaling image height "<<frameHeight_<<std::endl;
            {
                boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
                origLeft = cam_image1->image;//cv::Mat(cam_image1->image, left_roi_);

                origRight = cam_image2->image;//cv::Mat(cam_image2->image, right_roi_);

                camImageOrig = cv_rgb->image.clone();//cv::Mat(cv_rgb->image.clone(), left_roi_);
            }
            {
                boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
                imageStatus_ = true;
            }

            // std::cout<<"Debug inside cameraCallBack starting image resize"<<std::endl;

            cv::Mat left_resized, right_resized, camImageResized;
            cv::resize(origLeft, left_resized, cv::Size(frameWidth_, frameHeight_));
            cv::resize(origRight, right_resized, cv::Size(frameWidth_, frameHeight_));
            cv::resize(camImageOrig, camImageResized, cv::Size(frameWidth_, frameHeight_));
            // cv::resize(origLeft, left_rectified, cv::Size(frameWidth_, frameHeight_));
            // cv::resize(origRight, right_rectified, cv::Size(frameWidth_, frameHeight_));
            // cv::resize(camImageOrig, camImageCopy_, cv::Size(frameWidth_, frameHeight_));

            // std::cout<<"Debug inside cameraCallBack starting image padding"<<std::endl;

            int widthMiss = frameWidth_%4;
            int heightMiss = frameHeight_%4;
//            int widthMiss = 0;
//            int heightMiss = 0;

            cv::Mat left_widthAdj, right_widthAdj, camImageWidthAdj;

            if (widthMiss>0) {
                copyMakeBorder( left_resized, left_widthAdj, 0, 0, 0, widthMiss, cv::BORDER_CONSTANT, 0 );
                copyMakeBorder( right_resized, right_widthAdj, 0, 0, 0, widthMiss, cv::BORDER_CONSTANT, 0 );
                copyMakeBorder( camImageResized, camImageWidthAdj, 0, 0, 0, widthMiss, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            } else {
                left_widthAdj = left_resized.clone();
                right_widthAdj = right_resized.clone();
                camImageWidthAdj = camImageResized.clone();
            }

            // cv::Mat left_heightAdj, right_heightAdj, camImageHeightAdj;

            if (heightMiss>0) {
                copyMakeBorder( left_widthAdj, left_rectified, 0, heightMiss, 0, 0, cv::BORDER_CONSTANT, 0 );
                copyMakeBorder( right_widthAdj, right_rectified, 0, heightMiss, 0, 0, cv::BORDER_CONSTANT, 0 );
                copyMakeBorder( camImageWidthAdj, camImageCopy_, 0, heightMiss, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            } else {
                left_rectified = left_widthAdj.clone();
                right_rectified = right_widthAdj.clone();
                camImageCopy_ = camImageWidthAdj.clone();
            }

//      mpDetection -> getImage(left_rectified, right_rectified);
        }

        if (initiated) {
            processImage();
        } else {
            yolo();
        }

    }


    void YoloObjectDetector::processImage() {
        buffIndex_ = (buffIndex_ + 2) % 3;
        fetchInThread();
        detectInThread();
        displayInThread();
        publishInThread();
    }

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
//  cvImage.header.stamp = ros::Time::now();
  cvImage.header.stamp = image_time_;
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

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

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes, 0);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  globalframe++;
  running_ = 1;
  float nms = .45;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
  network_predict(*net_, X);

//  int size_of_array = sizeof(ss)/sizeof(ss[0]);
//
//
//  for (int i=0; i < size_of_array; i++){
//      printf("%lf\n", ss[i]);
//  }
//  printf("output array size: %d\n\n", size_of_array);

  image display = buff_[(buffIndex_ + 2) % 3];
  int nboxes = 0;

  detection *dets = get_network_boxes(net_, display.w, display.h, demoThresh_, demoHier_, nullptr, 1, &nboxes, 1);

  if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

  draw_detections_v3(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, l.classes, 0); // 1 means output classes, here I ignore

  if ( (enableConsoleOutput_)&&(globalframe%20==1) ) {
//    printf("\033[2J");
//    printf("\033[1;1H");
//    printf("\nFPS:%.1f\n",fps_);
//    printf("Objects:\n\n");
      printf("FPS:%.1f\n", fps_);
  }

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
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

//        std::cout<<x_center<<", "<<y_center<<", "<<BoundingBox_width<<", "<<BoundingBox_height<<std::endl;

        // define 2D bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.02 && BoundingBox_height > 0.02) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  roiBoxes_[0].num = count;

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return nullptr;
}

void *YoloObjectDetector::fetchInThread()
{
  IplImage* ROS_img = getIplImage();
  ipl_into_image(ROS_img, buff_[buffIndex_]);
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_[buffIndex_] = actionId_;
  }
  if(!use_grey)
    rgbgr_image(buff_[buffIndex_]);

  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);

  buff_cv_l_[(buffIndex_)] = left_rectified.clone();
  buff_cv_r_[(buffIndex_)] = right_rectified.clone();
  buff_cv_rgb_[(buffIndex_)] = camImageCopy_.clone();

  if(counter > 2) {
      disparityFrame[(buffIndex_ + 2) % 3] = getDepth(buff_cv_l_[(buffIndex_ + 2) % 3], buff_cv_r_[(buffIndex_ + 2) % 3]);
  }

  counter ++;
  return nullptr;
}

void *YoloObjectDetector::displayInThread()
{
  show_image_cv(buff_[(buffIndex_ + 2)%3], "YOLO V3", ipl_);
//  show_image_layers(get_network_image(*net_), "c");
//  visualize_network(*net_);

  // cv::imshow("disparity_map",disparityFrame); // * 256 / disp_size);
//  cv::imshow("left_rect", origLeft);
//  cv::imshow("right_rect", origRight);
//  int c = cvWaitKey(waitKeyDelay_);
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
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
//  net_ = load_network(cfgfile, weightfile, 0);
  net_ = load_network_custom(cfgfile, weightfile, 0, 1);
//  visualize_network(*net_);
//  set_batch_network(net_, 1);
  fuse_conv_batchnorm(*net_);
}

void YoloObjectDetector:: yolo()
{
//  const auto wait_duration = std::chrono::milliseconds(2000);
//  while (!getImageStatus()) {
//    printf("Waiting for image.\n");
//    if (!isNodeRunning()) {
//      return;
//    }
//    std::this_thread::sleep_for(wait_duration);
//  }
//
//  std::thread detect_thread;
//  std::thread fetch_thread;
//  std::thread depth_detect_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  IplImage* ROS_img = getIplImage();
  buff_[0] = ipl_to_image(ROS_img);
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
    disparityFrame[0] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
    disparityFrame[1] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
    disparityFrame[2] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
    buff_cv_l_[0] = camImageCopy_.clone();
    buff_cv_l_[1] = camImageCopy_.clone();
    buff_cv_l_[2] = camImageCopy_.clone();
    buff_cv_rgb_[0] = camImageCopy_.clone();
    buff_cv_rgb_[1] = camImageCopy_.clone();
    buff_cv_rgb_[2] = camImageCopy_.clone();
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
    if (fullScreen_) {
      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
      cvMoveWindow("YOLO V3", 0, 0);
      cvResizeWindow("YOLO V3", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();

  initiated = true;

//  while (!demoDone_) {
//    buffIndex_ = (buffIndex_ + 1) % 3;
//    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
//    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
//
//    if (!demoPrefix_) {
//      fps_ = 1./(what_time_is_it_now() - demoTime_);
//      demoTime_ = what_time_is_it_now();
//      if (viewImage_) {
//        displayInThread();
//      }
//      publishInThread();
//    } else {
//      char name[256];
//      sprintf(name, "%s_%08d", demoPrefix_, count);
//      save_image(buff_[(buffIndex_ + 1) % 3], name);
//    }
//
//    fetch_thread.join();
//    detect_thread.join();
//
////    if(!disparityFrame.empty()) {
////        cv::imshow("disparity_map", disparityFrame);
////        cv::waitKey(0);
////    }
//      ++count;
//    if (!isNodeRunning()) {
//      demoDone_ = true;
//    }
//  }

}

IplImage* YoloObjectDetector::getIplImage()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  auto * ROS_img = new IplImage(camImageCopy_);
  return ROS_img;
}

bool YoloObjectDetector::getImageStatus()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
//  std::cout<<std::endl<<"*******************New Frame***************************"<<std::endl<<std::endl;

  if (num > 0 && num <= 100) {

    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    cv::Mat rgbOutput = buff_cv_rgb_[(buffIndex_ + 2) % 3].clone();
    cv::Mat greyOutput = buff_cv_l_[(buffIndex_ + 2) % 3].clone();
    cv::Mat disparityMap = disparityFrame[(buffIndex_ + 2) % 3].clone();
    cv::Mat hsv;
    cv::cvtColor(rgbOutput, hsv, CV_BGR2HSV);

    std_msgs::Int8 msg;
    msg.data = static_cast<signed char>(num);
    objectPublisher_.publish(msg);

//    float *netowrk_output = get_network_output(*net_);
    layer cost_layer = net_->layers[net_->n-1];
    int l_w = cost_layer.out_w;
    int l_h = cost_layer.out_h;
    int l_c = cost_layer.out_c;
    int fil_r = (int)l_c/2;
    int fil_c = l_c-fil_r;

    cv::Mat layer_cost = cv::Mat::zeros(l_h, l_w, CV_32FC(l_c));
//    cv::Mat layer_cost_vis = cv::Mat::zeros(l_h, l_w, CV_8UC(l_c));

//    std::cout<<layer_cost.size()<<std::endl;
    for(int kk = 0; kk < l_c; ++kk){
        for(int ii = 0; ii < l_h; ++ii){
            for(int jj = 0; jj < l_w; ++jj){
                layer_cost.at<cv::Vec<float, 10> >(ii,jj)[kk]=cost_layer.output[jj+ii*l_w+l_w*l_h*kk];
//                std::cout<<cost_layer.output[ii+l_w*l_h*kk]<<"; ";
            }
        }
    }

//    double min, max;
//    cv::minMaxLoc(layer_cost, &min, &max);
//    layer_cost.convertTo(layer_cost_vis,CV_8U,255.0/(max-min),-255.0*min/(max-min));

//    std::cout<<l_w<<", "<<l_h<<", "<<l_c<<std::endl;

    cv::Mat layer_output = cv::Mat::zeros(l_h, l_w, CV_8UC3);

      for (int i = 0; i < numClasses_; i++) {
          if (rosBoxCounter_[i] > 0) {
              for (int j = 0; j < rosBoxCounter_[i]; j++) {
                  auto center_c_ = static_cast<int>(rosBoxes_[i][j].x * frameWidth_);     //2D column
                  auto center_r_ = static_cast<int>(rosBoxes_[i][j].y * frameHeight_);    //2D row

                  auto xmin = static_cast<int>((rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_);
                  auto ymin = static_cast<int>((rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_);
                  auto xmax = static_cast<int>((rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_);
                  auto ymax = static_cast<int>((rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_);

                  auto xmin_net = static_cast<int>((rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * l_w);
                  auto ymin_net = static_cast<int>((rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * l_h);
                  auto xmax_net = static_cast<int>((rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * l_w);
                  auto ymax_net = static_cast<int>((rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * l_h);

                  if ((counter>2) ) {

                      cv::Rect_<int> rect_layer = cv::Rect_<int>(xmin_net, ymin_net, xmax_net - xmin_net, ymax_net - ymin_net);
                      cv::rectangle(layer_output, rect_layer, cv::Scalar( 0, 0, 255 ), 1);
                      cv::Mat cost_roi;
                      layer_cost(rect_layer).copyTo(cost_roi);
//                      cv::Mat roi_vis[l_c];
//                      split(cost_vis_roi,roi_vis);
//                      for(int ch = 0; ch < l_c; ++ch){
//                          std::ostringstream str;
//                          str << "Layer "<<ch;
//                          cv::imshow(str.str(),roi_vis[ch]);
//                      }

//                      cv::Mat cost = cv::Mat::zeros((ymax_net-ymin_net)*fil_r, (xmax_net-xmin_net)*fil_c, CV_32FC1);
////                    std::cout<<std::endl<<"New Object"<<std::endl;
//                      for(int jj = 0; jj < l_c; ++jj){
//                          for(int ii = xmin_net*ymin_net; ii < xmax_net*ymax_net; ++ii){
//                              std::cout<<cost_layer.output[ii+l_w*l_h*jj]<<"; ";
////                            printf("%lf\n", l.output[ii+l.out_w*l.out_h*(l.out_c-1)]);
//                          }
//                      }


                      cv::Rect_<int> rect = cv::Rect_<int>(xmin, ymin, xmax - xmin, ymax - ymin);
//                      cv::Mat mask = cv::Mat::zeros(greyOutput.size(), CV_8UC1);  // type of mask is CV_8U
//                      cv::Mat roi(mask, rect);
//                      roi = cv::Scalar(255);
//                      std::vector<cv::KeyPoint> kpts;
//                      cv::Mat desc;
//                      akaze->detectAndCompute(greyOutput, mask, kpts, desc);
//
//                      // Quantize the hue to 30 levels
//                      // and the saturation to 32 levels
//                      int hbins = 30, sbins = 32;
//                      int histSize[] = {30,32};//{hbins, sbins};{hbins, sbins};
//                      // // hue varies from 0 to 179, see cvtColor
//                      float hranges[] = { 0, 180 };
//                      // // saturation varies from 0 (black-gray-white) to
//                      // // 255 (pure spectrum color)
//                      float sranges[] = { 0, 256 };
//                      const float* ranges[] = { hranges, sranges };
//                      cv::MatND hist;
//                      // we compute the histogram from the 0-th and 1-st channels
//                      int channels[] = {0, 1};
//
//                      cv::calcHist( &hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false );
//                      cv::normalize(hist, hist, 1, 0, 2, -1, cv::Mat());

                      Blob outputObs(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
                      outputObs.category = classLabels_[i];
                      outputObs.probability = rosBoxes_[i][j].prob;
                      outputObs.feature_cost = cost_roi;
//                      outputObs.kpDesc = desc;
//                      outputObs.nHist = hist;
//                      for (int i=0; i<kpts.size(); i++){
//                          float kptDis = (float) disparityMap.at<uchar>((int)kpts[i].pt.x,(int)kpts[i].pt.y);
//                          outputObs.keyPoints.push_back(cv::Point3f(kpts[i].pt.x,kpts[i].pt.y,kptDis));
//                      }

                      currentFrameBlobs.push_back(outputObs);

                  } else {
                      ROS_WARN("*********************************************************");
                  }

              }
          }
      }
      cv::imshow("layer_output",layer_output);

  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
//    std::cout << "************************************************num 0" << std::endl;
  }

  Tracking();
  CreateMsg();

  obstacleBoxesResults_.header.stamp = image_time_;
  obstacleBoxesResults_.header.frame_id = pub_obs_frame_id;
  obstacleBoxesResults_.real_header.stamp = ros::Time::now();
  obstacleBoxesResults_.real_header.frame_id = pub_obs_frame_id;
  obstaclePublisher_.publish(obstacleBoxesResults_);

  obstacleBoxesResults_.obsData.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return nullptr;
}

    void YoloObjectDetector::matchCurrentFrameBlobsToExistingBlobs() {

        int simHeight = blobs.size();
        int simWidth = currentFrameBlobs.size();
        // cv::Mat simSize(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        //cv::Mat simApp(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        //cv::Mat simKeyPts(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        // cv::Mat simPos(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
//        cv::Mat similarity(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        cv::Mat disSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        // std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 1"<<std::endl;

//        for (int c=0; c<simWidth; c++){
//            for (int r=0; r<simHeight; r++){
//                disSimilarity.at<double>(r,c) = 10000000;
//            }
//        }

        for (int c=0; c<simWidth; c++){
            Blob currBlob = currentFrameBlobs[c];
            int ch = currBlob.feature_cost.channels();
            cv::Mat currBlbCosts[ch];
            split(currBlob.feature_cost,currBlbCosts);
//            currBlob.feature_cost.copyTo(currBlbCosts);
            for (int r=0; r<simHeight; r++){
                Blob blob = blobs[r];
                if (blob.blnStillBeingTracked){
                    if (currBlob.category == blob.category){
                        cv::Mat blbCosts[ch], reSizedBlbCosts[ch];
                        split(blob.feature_cost,blbCosts);
//                        blob.feature_cost.copyTo(blbCosts);
                        for(int ii = 0; ii < ch; ++ii){
                            if(blob.currentBoundingRect.area()>currBlob.currentBoundingRect.area()){
                                cv::resize(blbCosts[ii], reSizedBlbCosts[ii], currBlbCosts[ii].size(), 0, 0, CV_INTER_AREA);
                            } else {
                                cv::resize(blbCosts[ii], reSizedBlbCosts[ii], currBlbCosts[ii].size(), 0, 0, CV_INTER_LANCZOS4);
                            }
                            disSimilarity.at<double>(r,c) += cv::norm(currBlbCosts[ii],reSizedBlbCosts[ii]);
                        }
                    }
                }
            }
        } 
        //for (int r=0; r<simHeight; r++){
            //Blob blob = blobs[r];
            //if (blob.blnStillBeingTracked){
                // blob.predictNextPosition();
                //for (int c=0; c<simWidth; c++){
                    //Blob currBlob = currentFrameBlobs[c];
                    //if (currBlob.category == blob.category){
                        //std::vector< std::vector<cv::DMatch> > matches;
                        //matcher->knnMatch(currBlob.kpDesc, blob.kpDesc, matches, 2);
                        //double kptSim = 0.0;
                        //int matchedkpts = 0;
                        //for(unsigned i = 0; i < matches.size(); i++) {
                            //if(matches[i][0].distance < 0.8 * matches[i][1].distance) {
                                //kptSim = kptSim+matches[i][0].distance;
                                //matchedkpts++;
                            //}
                        //}
                        //if (matchedkpts>1){
                            //kptSim = kptSim/matchedkpts;
                        //}

                        // double sizeSim = intersectionOverUnion(currBlob.currentBoundingRect, blob.currentBoundingRect);
                        //double appSim = 1.0-cv::compareHist(currBlob.nHist, blob.nHist, cv::HISTCMP_HELLINGER );
                        // double posSim = 1.0/(0.001+ (double)distanceBetweenPoints(currBlob.centerPositions.back(), blob.predictedNextPosition));

                        // simSize.at<double>(r,c) = sizeSim;
                        //simApp.at<double>(r,c) = appSim;
                        //simKeyPts.at<double>(r,c) = kptSim;
                        // simPos.at<double>(r,c) = posSim;
                    //}
                //}
            //}
        //}
        // std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 2"<<std::endl;
        // cv::normalize(simSize, simSize, 0, 1, cv::NORM_MINMAX);
        //cv::normalize(simApp, simApp, 0, 1, cv::NORM_MINMAX);
        //cv::normalize(simKeyPts, simKeyPts, 0, 1, cv::NORM_MINMAX);
        // cv::normalize(simPos, simPos, 0, 1, cv::NORM_MINMAX);

        // std::cout<<"Size"<<std::endl<<simSize<<std::endl;
        // std::cout<<"App"<<std::endl<<simApp<<std::endl;
        // std::cout<<"KeyPts"<<std::endl<<simKeyPts<<std::endl;
        // std::cout<<"Pos"<<std::endl<<simPos<<std::endl;

        //similarity = simKeyPts  ;//+ simApp; //simSize + simPos;
        // similarity = simKeyPts;

         std::cout<<"similarity"<<std::endl<<disSimilarity<<std::endl<<std::endl;

        double min, max;
        //cv::minMaxLoc(similarity, &min, &max);
        cv::minMaxLoc(disSimilarity, &min, &max);

        // std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 3"<<std::endl;

        for (int c=0; c<simWidth; c++){
            double minDisSimilarity = 10000000.0;
            int indexMinDisSimilarity = -1;
            Blob &currentFrameBlob = currentFrameBlobs.at(c);
            for (int r=0; r<simHeight; r++){
                double disSimilarityVal = disSimilarity.at<double>(r,c);
                if (disSimilarityVal>0){
                    if (disSimilarityVal < minDisSimilarity) {
                        minDisSimilarity = disSimilarityVal;
                        indexMinDisSimilarity = r;
                    }
                }
            }
            if (indexMinDisSimilarity != -1) {
                if ((minDisSimilarity < max) && (!blobs[indexMinDisSimilarity].blnAlreadyTrackedInThisFrame)) {
                    addBlobToExistingBlobs(currentFrameBlob, blobs, indexMinDisSimilarity);
                } else {
                    addNewBlob(currentFrameBlob, blobs);
                }
            } else {
                addNewBlob(currentFrameBlob, blobs);
            }
        }

//        for (int c=0; c<simWidth; c++){
//            double maxSimilarity = 0.0;
//            int indexMaxSimilarity = -1;
//            Blob &currentFrameBlob = currentFrameBlobs.at(c);
//            for (int r=0; r<simHeight; r++){
//                double similarityVal = similarity.at<double>(r,c);
//                if (maxSimilarity < similarityVal) {
//                    maxSimilarity = similarityVal;
//                    indexMaxSimilarity = r;
//                }
//            }
//            if (indexMaxSimilarity != -1) {
//                if ((maxSimilarity > max*0.5) && (!blobs[indexMaxSimilarity].blnAlreadyTrackedInThisFrame)) {
//                    addBlobToExistingBlobs(currentFrameBlob, blobs, indexMaxSimilarity);
//                } else {
//                    addNewBlob(currentFrameBlob, blobs);
//                }
//            } else {
//                addNewBlob(currentFrameBlob, blobs);
//            }
//        }

        // std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 4"<<std::endl;

        for (auto &existingBlob : blobs) {
            if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
                existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
            }
            if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 10) {
                existingBlob.blnStillBeingTracked = false;
//      blobs.erase(blobs.begin() + i);
            }
        }

        // std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 5"<<std::endl;

    }

    void YoloObjectDetector::addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

        existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
        existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
        existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
        existingBlobs[intIndex].probability = currentFrameBlob.probability;
//        existingBlobs[intIndex].keyPoints = currentFrameBlob.keyPoints;
//        existingBlobs[intIndex].kpDesc = currentFrameBlob.kpDesc;
//        existingBlobs[intIndex].nHist = currentFrameBlob.nHist;
        existingBlobs[intIndex].feature_cost = currentFrameBlob.feature_cost;
        existingBlobs[intIndex].blnStillBeingTracked = true;
        existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
        existingBlobs[intIndex].blnAlreadyTrackedInThisFrame = true;
        existingBlobs[intIndex].counter = currentFrameBlob.counter + 1;
        existingBlobs[intIndex].intNumOfConsecutiveFramesWithoutAMatch =0;
    }

    void YoloObjectDetector::addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

        currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
        currentFrameBlob.blnStillBeingTracked = true;
        existingBlobs.push_back(currentFrameBlob);
    }

    void YoloObjectDetector::Tracking (){

        if (blnFirstFrame) {
            if (currentFrameBlobs.size()>0){
                blnFirstFrame = false;
                for (auto &currentFrameBlob : currentFrameBlobs)
                    blobs.push_back(currentFrameBlob);
            }
        } else {
            for (auto &existingBlob : blobs) {
                existingBlob.blnCurrentMatchFoundOrNewBlob = false;
                existingBlob.blnAlreadyTrackedInThisFrame = false;
                existingBlob.predictNextPosition();
            }
            if (currentFrameBlobs.size()>0){
                matchCurrentFrameBlobsToExistingBlobs();
            } else {
                for (auto &existingBlob : blobs) {
                    if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
                        existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
                    }
                    if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 10) {
                        existingBlob.blnStillBeingTracked = false;
                        //blobs.erase(blobs.begin() + i);
                    }
                }
            }

        }

        currentFrameBlobs.clear();
    }

    void YoloObjectDetector::CreateMsg(){

        // cv::Mat output1 = disparityFrame[(buffIndex_ + 1) % 3].clone();
        cv::Mat output = buff_cv_rgb_[(buffIndex_ + 2) % 3].clone();//camImageCopy_.clone();

        // for (long int i = 0; i < currentFrameBlobs.size(); i++) {
        //   cv::rectangle(output, currentFrameBlobs[i].currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
        // }
        // currentFrameBlobs.clear();

        for (long int i = 0; i < blobs.size(); i++) {
//            if (blobs[i].blnStillBeingTracked == true) {
            if (blobs[i].blnCurrentMatchFoundOrNewBlob) {
                cv::rectangle(output, blobs[i].currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
                // cv::rectangle(output1, blobs[i].currentBoundingRect, cv::Scalar( 255, 255, 255 ), 2);
//                for(int j=0; j<blobs[i].keyPoints.size();j++){
//                     cv::circle(output,cv::Point((int)blobs[i].keyPoints[j].x,(int)blobs[i].keyPoints[j].y),2,cv::Scalar( 0, 255, 0 ));
////                    std::cout<<blobs[i].keyPoints[j]<<"- "<<cv::Point((int)blobs[i].keyPoints[j].x,(int)blobs[i].keyPoints[j].y)<<";;; ";
//                    // output.at<cv::Vec3b>((int)blobs[i].keyPoints[j].x, (int)blobs[i].keyPoints[j].y)[2]=255;//cv::Vec3b(0,0,255);
//                }
                std::ostringstream str;
                // str << blobs[i].position_3d[2] <<"m, ID="<<i<<"; "<<blobs[i].disparity;
                str << "ID="<<i<<"; ";
                cv::putText(output, str.str(), blobs[i].centerPositions.back(), CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(0,250,255));
                // cv::putText(output1, str.str(), blobs[i].centerPositions.back(), CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 250, 255));
            }
        }
        cv::imshow("debug", output);
        // cv::imshow("disparity", output1);
        cv::waitKey(0);

        frame_num ++;
        // sprintf(im, "f%03d.png", frame_num);
        // img_name = ros::package::getPath("cubicle_detect") + "/seq_1/" + im;
        // std::cout<<img_name<<std::endl;
        // cv::imwrite(img_name, output);

    }


} /* namespace darknet_ros*/
