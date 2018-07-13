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
      isReceiveDepth(false),
      blnFirstFrame(true),
      globalframe(0)
{
  ROS_INFO("[ObstacleDetector] Node started.");

  // Read Cuda Info and ROS parameters from config file.
  if (!CudaInfo() || !readParameters()) {
    ros::requestShutdown();
  }

  mpDetection = new Detection(this, nodeHandle_);

  init();

  mpDepth_gen_run = new std::thread(&Detection::Run, mpDetection);

  hog_descriptor = new Util::HOGFeatureDescriptor(8, 2, 9, 180.0);
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
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
  u0 = 0;
  disp_size = static_cast<size_t>(mpDetection->disp_size);
  Width = static_cast<size_t>(mpDetection->Width);
  Height = static_cast<size_t>(mpDetection->Height);

  int ii;
  xDirectionPosition = static_cast<double **>(calloc(Width, sizeof(double *)));
  for(ii = 0; ii < Width; ii++)
      xDirectionPosition[ii] = static_cast<double *>(calloc(disp_size+1, sizeof(double)));

  yDirectionPosition = static_cast<double **>(calloc(Height, sizeof(double *)));
  for(ii = 0; ii < Height; ii++)
      yDirectionPosition[ii] = static_cast<double *>(calloc(disp_size+1, sizeof(double)));

  depthTable = static_cast<double *>(calloc(disp_size+1, sizeof(double)));


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
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;
  std::string obstacleBoxesTopicName;
  int obstacleBoxesQueueSize;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  nodeHandle_.param("publishers/obstacle_boxes/topic", obstacleBoxesTopicName,
                    std::string("obstacle"));
  nodeHandle_.param("publishers/obstacle_boxes/queue_size", obstacleBoxesQueueSize, 1);
  nodeHandle_.param("publishers/obstacle_boxes/frame_id", pub_obs_frame_id, std::string("obstacle"));

  objectPublisher_ = nodeHandle_pub.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch);
  boundingBoxesPublisher_ = nodeHandle_pub.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);

  obstaclePublisher_ = nodeHandle_pub.advertise<obstacle_msgs::MapInfo>(
          obstacleBoxesTopicName, obstacleBoxesQueueSize);

  detectionImagePublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);
  nodeHandle_.param<bool>("use_grey", use_grey, false);
  nodeHandle_.param<int>("scale", Scale, 2);

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
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

  assert(intrinsicLeft == intrinsicRight);

  const cv::Matx33d &intrinsic = intrinsicLeft;

  // Save the baseline
  stereo_baseline_ = stereoCameraModel.baseline();
  ROS_INFO_STREAM("baseline: " << stereo_baseline_);
  assert(stereo_baseline_ > 0);

  // get the Region Of Interests (If the images are already rectified but invalid pixels appear)
  left_roi_ = cameraLeft.rawRoi();
  right_roi_ = cameraRight.rawRoi();
//    {
//        double tmp_left, tmp_right;
//        tmp_left = left_roi_.height;
//        left_roi_.height = left_roi_.width;
//        left_roi_.width = tmp_left;
//        tmp_left = right_roi_.height;
//        right_roi_.height = right_roi_.width;
//        right_roi_.width = tmp_left;
//    }

}

void YoloObjectDetector::getDepth(cv::Mat &depthFrame) {

  depthFrame.copyTo(disparityFrame);

//    disparityFrame.convertTo(doubleFrame, CV_32FC1);

  isReceiveDepth = true;
}

void YoloObjectDetector::DefineLUTs() {

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
//      std::cout<<"i: "<<uDispThresh[i]<<"; ";
    }

}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &image1,
                                        const sensor_msgs::ImageConstPtr &image2,
                                        const sensor_msgs::CameraInfoConstPtr& left_info,
                                        const sensor_msgs::CameraInfoConstPtr& right_info){
  ROS_DEBUG("[ObstacleDetector] Stereo images received.");

  cv_bridge::CvImageConstPtr cam_image1, cam_image2, cv_rgb;

  try {
    cam_image1 = cv_bridge::toCvShare(image1, sensor_msgs::image_encodings::MONO8);
    cam_image2 = cv_bridge::toCvShare(image2, sensor_msgs::image_encodings::MONO8);

    if(use_grey) {
      cv_rgb = cam_image1;
    }
    else {
      cv_rgb = cv_bridge::toCvShare(image2, sensor_msgs::image_encodings::BGR8);
    }

    imageHeader_ = image1->header;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if(u0 == 0) {
      loadCameraCalibration(left_info, right_info);
      DefineLUTs();
  }

  if (cam_image1) {
      frameWidth_ = cam_image1->image.size().width;
      frameHeight_ = cam_image1->image.size().height;
      frameWidth_ = frameWidth_ / Scale;
      frameHeight_ = frameHeight_ / Scale;
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      origLeft = cv::Mat(cam_image1->image, left_roi_);

      origRight = cv::Mat(cam_image2->image, right_roi_);
      camImageOrig = cv::Mat(cv_rgb->image.clone(), left_roi_);
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
      cv::resize(origLeft, left_rectified, cv::Size(frameWidth_, frameHeight_));
      cv::resize(origRight, right_rectified, cv::Size(frameWidth_, frameHeight_));
      cv::resize(camImageOrig, camImageCopy_, cv::Size(frameWidth_, frameHeight_));
      mpDetection -> getImage(left_rectified, right_rectified);
  }

}

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
    frameHeight_ = cam_image->image.size().height;
    frameWidth_ /= Scale;
    frameHeight_ /= Scale;
  }
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
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

  image display = buff_[(buffIndex_ + 2) % 3];
  int nboxes = 0;

  detection *dets = get_network_boxes(net_, display.w, display.h, demoThresh_, demoHier_, nullptr, 1, &nboxes, 0);
  if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
  draw_detections_v3(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, l.classes, 0); // 1 means output classes

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

        // define 2D bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.08 && BoundingBox_height > 0.08) {
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

//  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  buffLetter_[buffIndex_] = resize_image(buff_[buffIndex_], net_->w, net_->h);
  return nullptr;
}

void *YoloObjectDetector::displayInThread()
{
  show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
//  cv::imshow("disparity_map",disparityFrame * 256 / disp_size);
  int c = cvWaitKey(waitKeyDelay_);
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
//  set_batch_network(net_, 1);
  fuse_conv_batchnorm(*net_);
}

void YoloObjectDetector:: yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;
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
  buffLetter_[0] = resize_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = resize_image(buff_[1], net_->w, net_->h);
  buffLetter_[2] = resize_image(buff_[2], net_->w, net_->h);
//  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
//  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
//  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
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

  while (!demoDone_) {
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
//    depth_detect_thread = std::thread(&Detection::disparityInThread, mpDetection);
//        if(isReceiveDepth){
//
//          isReceiveDepth = false;
//        }
    if (!demoPrefix_) {
      fps_ = 1./(what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread();
      }
      publishInThread();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }

}

IplImage* YoloObjectDetector::getIplImage()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  auto * ROS_img = new IplImage(camImageCopy_);
//  IplImage* ROS_img = new IplImage(left_rectified);
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
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = static_cast<signed char>(num);
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          auto center_c_ = static_cast<int>(rosBoxes_[i][j].x * frameWidth_);     //2D column
          auto center_r_ = static_cast<int>(rosBoxes_[i][j].y * frameHeight_);    //2D row

          auto xmin = static_cast<int>((rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_);
          auto ymin = static_cast<int>((rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_);
          auto xmax = static_cast<int>((rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_);
          auto ymax = static_cast<int>((rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_);

            if ((xmin > 2) &&(ymin > 2)) {
//                auto dis = (int)disparityFrame.at<uchar>(center_r_, center_c_);
                auto dis = static_cast<int>(Util::median_mat(disparityFrame, center_c_, center_r_, 2));  // find 5x5 median
                if(dis!=0) {
                    // Hog features
                    cv::Rect_<int> rect = cv::Rect_<int>(xmin, ymin, xmax - xmin, ymax - ymin);
                    cv::Mat roi = left_rectified(rect).clone();
                    cv::resize(roi, roi, cv::Size(15, 15));
                    std::vector<float> hog_feature;
                    hog_descriptor -> computeHOG(hog_feature, roi);

                    std::vector<cv::Point3f> cent_2d, cent_3d;
                    Blob outputObs(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
//                    obstacle_msgs::obs outputObs;
                    outputObs.category = classLabels_[i];
                    outputObs.probability = rosBoxes_[i][j].prob;
                    outputObs.position_3d[0] = xDirectionPosition[center_c_][dis];
                    outputObs.position_3d[1] = yDirectionPosition[center_r_][dis];
                    outputObs.position_3d[2] = depthTable[dis];
                    double xmin_3d, xmax_3d, ymin_3d, ymax_3d;
                    xmin_3d = xDirectionPosition[xmin][dis];
                    xmax_3d = xDirectionPosition[xmax][dis];
                    ymin_3d = yDirectionPosition[ymin][dis];
                    ymax_3d = yDirectionPosition[ymax][dis];
                    outputObs.diameter = abs(static_cast<int>(xmax_3d - xmin_3d));
                    outputObs.height = abs(static_cast<int>(ymax_3d - ymin_3d));
                    outputObs.obsHog = hog_feature;
//                    obstacleBoxesResults_.obsData.push_back(outputObs);
                    currentFrameBlobs.push_back(outputObs);
                }

            }

          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);

        }
      }
    }
    if(isReceiveDepth) {

        Tracking();
        CreateMsg();

        currentFrameBlobs.clear();

        obstacleBoxesResults_.header.stamp = ros::Time::now();
        obstacleBoxesResults_.header.frame_id = pub_obs_frame_id;
        obstacleBoxesResults_.image_header = imageHeader_;
        obstacleBoxesResults_.num = obstacleBoxesResults_.obsData.size();
        obstaclePublisher_.publish(obstacleBoxesResults_);
        obstacleBoxesResults_.obsData.clear();

        isReceiveDepth = false;
    }

//    boundingBoxesResults_.header.stamp = ros::Time::now();
//    boundingBoxesResults_.header.frame_id = "detection";
//    boundingBoxesResults_.image_header = imageHeader_;
//    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return nullptr;
}

void YoloObjectDetector::matchCurrentFrameBlobsToExistingBlobs() {

    for (auto &existingBlob : blobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.blnAlreadyTrackedInThisFrame = false;

        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        dblLeastDistance = 100000.0;

        for (unsigned int j = 0; j < blobs.size(); ++ j) {

            if (blobs[j].blnStillBeingTracked) {

                int dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), blobs[j].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {

                    dblLeastDistance = dblDistance;

                    intIndexOfLeastDistance = j;
                }
            }
        }

        if ( (dblLeastDistance < (static_cast<int>(currentFrameBlob.dblCurrentDiagonalSize * 1.4)) ) &&
             (!blobs[intIndexOfLeastDistance].blnAlreadyTrackedInThisFrame)) {

            double hog_dist = cv::norm(currentFrameBlob.obsHog,blobs[intIndexOfLeastDistance].obsHog, cv::NORM_L2);
//            std::cout<<"hog_dist :"<<hog_dist<<std::endl;
            if (hog_dist<0.5) {
                addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance);
            } else {
                addNewBlob(currentFrameBlob, blobs);
            }

        } else {

            addNewBlob(currentFrameBlob, blobs);
        }
    }

    for (auto &existingBlob : blobs) {

        if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {

            existingBlob.intNumOfConsecutiveFramesWithoutAMatch ++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 10) {

            existingBlob.blnStillBeingTracked = false;
        }
    }
}

void YoloObjectDetector::addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;

    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
//    existingBlobs[intIndex].max_disparity = currentFrameBlob.max_disparity;
    existingBlobs[intIndex].obsPoints = currentFrameBlob.obsPoints;
    existingBlobs[intIndex].obsHog = currentFrameBlob.obsHog;

    existingBlobs[intIndex].blnStillBeingTracked = true;

    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs[intIndex].blnAlreadyTrackedInThisFrame = true;

    existingBlobs[intIndex].counter = currentFrameBlob.counter +1;
}

void YoloObjectDetector::addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    currentFrameBlob.blnStillBeingTracked = true;

    existingBlobs.push_back(currentFrameBlob);
}

void YoloObjectDetector::Tracking (){

    if (blnFirstFrame) {

        blnFirstFrame = false;

        for (auto &currentFrameBlob : currentFrameBlobs) {

            blobs.push_back(currentFrameBlob);
        }
    } else { matchCurrentFrameBlobsToExistingBlobs(); }
}

void YoloObjectDetector::CreateMsg(){

    for (unsigned long int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnCurrentMatchFoundOrNewBlob) {

            obstacle_msgs::obs tmpObs;

            tmpObs.identityID = i;

            tmpObs.centerPos.x = blobs[i].position_3d[0];
            tmpObs.centerPos.y = blobs[i].position_3d[1];
            tmpObs.centerPos.z = blobs[i].position_3d[2];
            tmpObs.diameter = blobs[i].diameter;
            tmpObs.height = blobs[i].height;

            tmpObs.counter = blobs[i].counter;
            tmpObs.classes = blobs[i].category;
            tmpObs.probability = blobs[i].probability;
//            tmpObs.histogram = blobs[i].obsHog;

            obstacleBoxesResults_.obsData.push_back(tmpObs);
        }
    }
}


} /* namespace darknet_ros*/
