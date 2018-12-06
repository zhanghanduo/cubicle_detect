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
      isDepthNew(false),
      is_even_crop(false)
{
  ROS_INFO("[ObstacleDetector] Node started.");

  // Read Cuda Info and ROS parameters from config file.
  if (!CudaInfo() || !readParameters()) {
    ros::requestShutdown();
  }
//  mpDetection = new Detection(this, nodeHandle_);
//  nullHog.assign(36, 0.0);
  init();

  init_disparity_method(7, 86);

//  mpDepth_gen_run = new std::thread(&Detection::Run, mpDetection);

  hog_descriptor = new Util::HOGFeatureDescriptor(8, 2, 9, 180.0);
  img_name = ros::package::getPath("cubicle_detect") + "/seq_1/f000.png";
  file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/f000.txt";
  frame_num = 0;
}

YoloObjectDetector::~YoloObjectDetector()
{
  finish_disparity_method();
    {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
  free(depth3D);
  free(x3DPosition);
  free(y3DPosition);
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
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

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
                    std::string("/obs_map"));
  nodeHandle_.param("publishers/obstacle_boxes/queue_size", obstacleBoxesQueueSize, 1);
  nodeHandle_.param("publishers/obstacle_boxes/frame_id", pub_obs_frame_id, std::string("camera_frame"));

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

  u0 = left_info->K[2];
  v0 = left_info->K[5];
  focal = left_info->K[0];
  Width = left_info->width;
  Height = left_info->height;

  Width /= Scale;
  Height /= Scale;

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

  ObstacleDetector.Initiate(left_info_copy->header.frame_id, disp_size, stereo_baseline_, u0, v0, focal, Width, Height);


//  // get the Region Of Interests (If the images are already rectified but invalid pixels appear)
//  left_roi_ = cameraLeft.rawRoi();
//  right_roi_ = cameraRight.rawRoi();
}

cv::Mat YoloObjectDetector::getDepth(cv::Mat &leftFrame, cv::Mat &rightFrame) {

    float elapsed_time_ms;
    cv::Mat disparity_SGBM(leftFrame.size(), CV_8UC1);

    disparity_SGBM = compute_disparity_method(leftFrame, rightFrame, &elapsed_time_ms);

    isDepthNew = true;
    return disparity_SGBM;
}

void YoloObjectDetector::DefineLUTs() {

  ROS_WARN("u0: %f | v0: %f | focal: %f | base: %f | width: %d | Height: %d", u0, v0, focal, stereo_baseline_, Width_crp, Height_crp);

    for (int r=0; r<Width; r++) {
        x3DPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            x3DPosition[r][c]=(r-u0)*stereo_baseline_/c;
//        std::cout<<xDirectionPosition[r][c]<<std::endl;
        }
    }

    for (int r=0; r<Height; r++) {
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

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &image1,
                                        const sensor_msgs::ImageConstPtr &image2){
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
            cv_rgb = cv_bridge::toCvShare(image1, sensor_msgs::image_encodings::BGR8);
        }
        image_time_ = image1->header.stamp;
        imageHeader_ = image1->header;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (cam_image1) {

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

        if(Scale != 1) {
            cv::resize(origLeft, left_resized, cv::Size(Width, Height));
            cv::resize(origRight, right_resized, cv::Size(Width, Height));
            cv::resize(camImageOrig, camImageResized, cv::Size(Width, Height));
        }else{
            left_resized = origLeft.clone();
            right_resized = origRight.clone();
            camImageResized = camImageOrig;
        }

        // std::cout<<"Debug inside cameraCallBack starting image padding"<<std::endl;

//        cv::Mat left_widthAdj, right_widthAdj, camImageWidthAdj;

        if (is_even_crop) {
//            copyMakeBorder( left_resized, left_widthAdj, 0, 0, 0, rem_w, cv::BORDER_CONSTANT, 0 );
//            copyMakeBorder( right_resized, right_widthAdj, 0, 0, 0, rem_w, cv::BORDER_CONSTANT, 0 );
//            copyMakeBorder( camImageResized, camImageWidthAdj, 0, 0, 0, rem_w, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
            left_rectified = left_resized(cv::Rect(0, 0, Width_crp, Height_crp)).clone();
            right_rectified = right_resized(cv::Rect(0, 0, Width_crp, Height_crp)).clone();
            camImageCopy_ = camImageResized(cv::Rect(0, 0, Width_crp, Height_crp)).clone();
        } else {
            left_rectified = left_resized.clone();
            right_rectified = right_resized.clone();
            camImageCopy_ = camImageResized.clone();
        }

        // cv::Mat left_heightAdj, right_heightAdj, camImageHeightAdj;
//
//        if (is_even_crop_h) {
//            copyMakeBorder( left_widthAdj, left_rectified, 0, rem_h, 0, 0, cv::BORDER_CONSTANT, 0 );
//            copyMakeBorder( right_widthAdj, right_rectified, 0, rem_h, 0, 0, cv::BORDER_CONSTANT, 0 );
//            copyMakeBorder( camImageWidthAdj, camImageCopy_, 0, rem_h, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
//        } else {
//            left_rectified = left_widthAdj;
//            right_rectified = right_widthAdj;
//            camImageCopy_ = camImageWidthAdj;
//        }

//        ROS_WARN("width: %d | height: %d", left_rectified.cols, left_rectified.rows);
    }
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

  if(counter > 2) {

      disparityFrame[(buffIndex_ + 2) % 3] = getDepth(buff_cv_l_[(buffIndex_ + 2) % 3], buff_cv_r_[(buffIndex_ + 2) % 3]);
  }

  counter ++;
  return nullptr;
}

void *YoloObjectDetector::displayInThread()
{
  show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
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
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  disparityFrame[0] = cv::Mat(Height_crp, Width_crp, CV_8UC1, cv::Scalar(0));
  disparityFrame[1] = cv::Mat(Height_crp, Width_crp, CV_8UC1, cv::Scalar(0));
  disparityFrame[2] = cv::Mat(Height_crp, Width_crp, CV_8UC1, cv::Scalar(0));
  buff_cv_l_[0] = camImageCopy_.clone();
  buff_cv_l_[1] = camImageCopy_.clone();
  buff_cv_l_[2] = camImageCopy_.clone();
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

//    if(!disparityFrame.empty()) {
//        cv::imshow("disparity_map", disparityFrame);
//        cv::waitKey(0);
//    }
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
        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          auto center_c_ = static_cast<int>(rosBoxes_[i][j].x * Width_crp);     //2D column
          auto center_r_ = static_cast<int>(rosBoxes_[i][j].y * Height_crp);    //2D row

          auto xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * Width_crp;
          auto ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * Height_crp;
          auto xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * Width_crp;
          auto ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * Height_crp;

//            std::cout << "xmin: " << xmin << ", ymin: " <<ymin<<", xmax: " <<xmax<<", ymax: "<< ymax << std::endl;

          if(ymax >= Height_crp)    ymax = Height_crp - 1;
          if(xmax >= Width_crp)     xmax = Width_crp - 1;
          int median_kernel = std::min(xmax - xmin, ymax - ymin);

            // if ((xmin > 2) &&(ymin > 2) && (counter>2) ) {
          if ((counter>2) ) {

              if((classLabels_[i] == "car") || (classLabels_[i] == "bus")|| (classLabels_[i] == "motor") || (classLabels_[i] == "bike")
                 || (classLabels_[i] == "truck")  || (classLabels_[i] == "rider") || (classLabels_[i] == "person")) {

                  auto dis = static_cast<int>(Util::median_mat(disparityFrame[(buffIndex_ + 1) % 3], center_c_, center_r_, median_kernel));  // find 3x3 median
                  cv::Rect_<int> rect = cv::Rect_<int>(static_cast<int>(xmin),
                                                       static_cast<int>(ymin),
                                                       static_cast<int>(xmax - xmin),
                                                       static_cast<int>(ymax - ymin));

                  if(dis>=12) {

//                    if(dis < 12){

//                        ROS_WARN("dis too small: %d", dis);
//                    }

                      std::vector<cv::Point3f> cent_2d, cent_3d;
                      Blob outputObs(static_cast<float>(xmin),
                                     static_cast<float>(ymin),
                                     static_cast<float>(xmax - xmin),
                                     static_cast<float>(ymax - ymin));
//                    obstacle_msgs::obs outputObs;
                      outputObs.category = classLabels_[i];
                      outputObs.probability = rosBoxes_[i][j].prob;
                      outputObs.position_3d[0] = x3DPosition[center_c_][dis];
                      outputObs.position_3d[1] = y3DPosition[center_r_][dis];
                      outputObs.position_3d[2] = depth3D[dis];
                      outputObs.xmin = xmin;
                      outputObs.xmax = xmax;
                      outputObs.ymin = ymin;
                      outputObs.ymax = ymax;
//                    ROS_WARN("center 3D\nx: %f| y: %f| z: %f",
//                             outputObs.position_3d[0], outputObs.position_3d[1], depthTable[dis]);

                      double xmin_3d, xmax_3d, ymin_3d, ymax_3d;
                      xmin_3d = x3DPosition[static_cast<int>(xmin)][dis];
                      xmax_3d = x3DPosition[static_cast<int>(xmax)][dis];
                      ymin_3d = y3DPosition[static_cast<int>(ymin)][dis];
                      ymax_3d = y3DPosition[static_cast<int>(ymax)][dis];
//                    ROS_WARN("min 3D\nx: %f| y: %f", xmin_3d, xmax_3d);
//                    ROS_WARN("max 3D\nx: %f| y: %f", xmax_3d, ymax_3d);
                      outputObs.diameter = abs(static_cast<int>(xmax_3d - xmin_3d));
                      outputObs.height = abs(static_cast<int>(ymax_3d - ymin_3d));
//                    outputObs.obsHog = hog_feature;
                      outputObs.disparity = dis;
//                    obstacleBoxesResults_.obsData.push_back(outputObs);
                      currentFrameBlobs.push_back(outputObs);

                  } else {
                      std::string classname = classLabels_[i];
                      ROS_WARN("class, dis: %s, %d", classname.c_str(), dis);
                  }

              }

            } else {
//              ROS_WARN("*********************************************************");
            }

        }
      }
    }

//    std::cout<<"currentFrameBlobs: "<<currentFrameBlobs.size()<<std::endl;

//    cv::Mat beforeTracking = buff_cv_l_[(buffIndex_ + 1) % 3].clone();
//    for (auto &currentFrameBlob : currentFrameBlobs) {
//      cv::rectangle(beforeTracking, currentFrameBlob.currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
//    }
//    cv::imshow("beforeTracking", beforeTracking);

        // TODO: wait until isDepth_new to be true
//      Tracking();
//      CreateMsg();
      roiBoxes_[0].num = 0;
//    boundingBoxesResults_.header.stamp = ros::Time::now();
//    boundingBoxesResults_.header.frame_id = "detection";
//    boundingBoxesResults_.image_header = imageHeader_;
//    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
//    std::cout << "************************************************num 0" << std::endl;
  }

//    std::cout << "************************************************new frame" << std::endl;
    cv::Mat beforeTracking = buff_cv_l_[(buffIndex_ + 1) % 3].clone();
//    cv::imshow("beforeTracking", beforeTracking);

  ObstacleDetector.ExecuteDetection(disparityFrame[(buffIndex_ + 1) % 3], beforeTracking);
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

    int tracksOrMatHeight = (int)blobs.size();
    int detsOrMatWidth = (int)currentFrameBlobs.size();
//    cv::Mat appDisSimilarity(tracksOrMatHeight, detsOrMatWidth, CV_64FC1, cv::Scalar(1.0));
//    cv::Mat motionDisSimilarity(tracksOrMatHeight, detsOrMatWidth, CV_64FC1, cv::Scalar(1.0));
    cv::Mat disSimilarity(tracksOrMatHeight, detsOrMatWidth, CV_64FC1, cv::Scalar(1.0));

    for (int c=0; c<detsOrMatWidth; c++) {

        Blob currBlob = currentFrameBlobs[c];

        for (int r = 0; r < tracksOrMatHeight; r++) {
            Blob blob = blobs[r];
            if (blob.blnStillBeingTracked) {
                if (currBlob.category == blob.category) {

                    cv::Rect predRect;
                    predRect.width = static_cast<int>(blob.state.at<float>(4));
                    predRect.height = static_cast<int>(blob.state.at<float>(5));
                    predRect.x = static_cast<int>(blob.state.at<float>(0) - predRect.width / 2);
                    predRect.y = static_cast<int>(blob.state.at<float>(1) - predRect.height / 2);

//                    std::cout<< r <<" predRect: "<< predRect <<", "<<c<<" detectRect: " << currBlob.currentBoundingRect <<std::endl;

                    cv::Rect intersection = predRect & currBlob.currentBoundingRect;//currBlob.boundingRects.back();
                    cv::Rect unio = predRect | currBlob.currentBoundingRect;//currBlob.boundingRects.back();
                    disSimilarity.at<double>(r,c) = 1.0 - (double)intersection.area()/unio.area();

                }
            }
        }
    }

//    double min, max;
//    cv::minMaxLoc(disSimilarity, &min, &max);
//        double thForHungarianCost = std::max(0.75,max*0.5);

    std::vector< std::vector<double> > costMatrix;
    for (int r = 0; r < tracksOrMatHeight; r++)  {
        std::vector<double> costForEachTrack;
        for (int c=0; c<detsOrMatWidth; c++) {
            costForEachTrack.push_back(disSimilarity.at<double>(r,c));
        }
        costMatrix.push_back(costForEachTrack);
    }

//        std::cout<<"costMatrix: "<<costMatrix.size()<<", "<<costMatrix[0].size()<<"; simHeight: "<<simHeight<<", simWidth: "<<simWidth<<std::endl;

    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;

    double hungarianCost = HungAlgo.Solve(costMatrix, assignment);
//        std::cout<<"hungarianCost: "<<hungarianCost<<std::endl;

    for (int trackID = 0; trackID < costMatrix.size(); trackID++){
//            std::cout << trackID << "," << assignment[trackID] << "\t";
        if (assignment[trackID]>-1) {
            Blob &currentFrameBlob = currentFrameBlobs.at(static_cast<unsigned long>(assignment[trackID]));
            double disSimValue = disSimilarity.at<double>(trackID,assignment[trackID]);
            if ( (!blobs[trackID].blnAlreadyTrackedInThisFrame) && disSimValue<0.7 ) { //(minDisSimilarity < max)
                currentFrameBlob.blnAlreadyTrackedInThisFrame = true;
                addBlobToExistingBlobs(currentFrameBlob, blobs, trackID);
            } else {
                addNewBlob(currentFrameBlob, blobs);
            }
        }
    }
//        std::cout<<std::endl;

    for (int c=0; c<detsOrMatWidth; c++){
        Blob &currentFrameBlob = currentFrameBlobs.at(c);
        if(!currentFrameBlob.blnAlreadyTrackedInThisFrame)
            addNewBlob(currentFrameBlob, blobs);
    }

    for (auto &existingBlob : blobs) {
        if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }
        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 50) {
            existingBlob.blnStillBeingTracked = false;
        }
    }

// std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 5"<<std::endl;

}

void YoloObjectDetector::addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].probability = currentFrameBlob.probability;
    existingBlobs[intIndex].disparity = currentFrameBlob.disparity;
    existingBlobs[intIndex].position_3d = currentFrameBlob.position_3d;

    existingBlobs[intIndex].xmin = currentFrameBlob.xmin;
    existingBlobs[intIndex].xmax = currentFrameBlob.xmax;
    existingBlobs[intIndex].ymin = currentFrameBlob.ymin;
    existingBlobs[intIndex].ymax = currentFrameBlob.ymax;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
    existingBlobs[intIndex].blnAlreadyTrackedInThisFrame = true;
    existingBlobs[intIndex].counter = currentFrameBlob.counter + 1;
    existingBlobs[intIndex].intNumOfConsecutiveFramesWithoutAMatch =0;

    //update motion model
    existingBlobs[intIndex].meas.at<float>(0) = currentFrameBlob.meas.at<float>(0);
    existingBlobs[intIndex].meas.at<float>(1) = currentFrameBlob.meas.at<float>(1);
    existingBlobs[intIndex].meas.at<float>(2) = currentFrameBlob.meas.at<float>(2);
    existingBlobs[intIndex].meas.at<float>(3) = currentFrameBlob.meas.at<float>(3);
    existingBlobs[intIndex].kf.correct(existingBlobs[intIndex].meas); // Kalman Correction

}

void YoloObjectDetector::addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    currentFrameBlob.blnStillBeingTracked = true;

    existingBlobs.push_back(currentFrameBlob);
}

void YoloObjectDetector::Tracking (){

    if (blnFirstFrame) {
        if (!currentFrameBlobs.empty()){
            blnFirstFrame = false;
            for (auto &currentFrameBlob : currentFrameBlobs)
                blobs.push_back(currentFrameBlob);
        }
    } else {
        for (auto &existingBlob : blobs) {
            existingBlob.blnCurrentMatchFoundOrNewBlob = false;
            existingBlob.blnAlreadyTrackedInThisFrame = false;
            // >>>> Matrix A
            auto dT = static_cast<float>(0.04 + (0.04 * existingBlob.intNumOfConsecutiveFramesWithoutAMatch));
            existingBlob.kf.transitionMatrix.at<float>(2) = dT;//dT;
            existingBlob.kf.transitionMatrix.at<float>(9) = dT;//dT;
            // <<<< Matrix A
            existingBlob.state = existingBlob.kf.predict();
        }

//            std::cout<<"blob prediction finished"<<std::endl;

        if (!currentFrameBlobs.empty()){
            matchCurrentFrameBlobsToExistingBlobs();
        } else {
            for (auto &existingBlob : blobs) {
                if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
                    existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
                }
                if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 50) {
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
}

void YoloObjectDetector::CreateMsg(){
    cv::Mat color_out;

    cv::Mat output1 = disparityFrame[(buffIndex_ + 1) % 3].clone();
    cv::Mat output = buff_cv_l_[(buffIndex_ + 1) % 3].clone();
    if(output.type() == CV_8UC1)
        cv::cvtColor(output, color_out, CV_GRAY2RGB);
    else
        color_out = output;

    std::vector<cv::Scalar> colors;
    cv::RNG rng(0);
    for(int i=0; i < blobs.size(); i++)
        colors.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));

    for (long int i = 0; i < blobs.size(); i++) {
//            if (blobs[i].blnStillBeingTracked == true) {
        if (blobs[i].blnCurrentMatchFoundOrNewBlob) {
            cv::rectangle(color_out, blobs[i].currentBoundingRect, colors.at(i), 2);
            int rectMinX = blobs[i].currentBoundingRect.x;
            int rectMinY = blobs[i].currentBoundingRect.y;
            cv::rectangle(color_out, cv::Rect(rectMinX, rectMinY, 40, 20), colors.at(i), CV_FILLED);
            std::ostringstream str;
            // str << blobs[i].position_3d[2] <<"m, ID="<<i<<"; "<<blobs[i].disparity;
            str << i;
            cv::putText(color_out, str.str(), cv::Point(rectMinX, rectMinY+16) , CV_FONT_HERSHEY_PLAIN, 1.5, CV_RGB(255,255,255));

            cv::rectangle(output1, blobs[i].currentBoundingRect, cv::Scalar( 255, 255, 255 ), 2);
        }
    }
    if(viewImage_) {
      cv::imshow("debug", color_out);
      cv::imshow("disparity", output1);
      // cv::waitKey(0);
    }
    frame_num ++;
    if(enableEvaluation_){
    sprintf(s, "f%03d.txt", frame_num);
    sprintf(im, "f%03d.png", frame_num);
    file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/" + s;
    img_name = ros::package::getPath("cubicle_detect") + "/seq_1/" + im;

    file.open(file_name.c_str(), std::ios::app);
    }

    int cate = 0;

    for (unsigned long int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnCurrentMatchFoundOrNewBlob) {

            if((blobs[i].category == "car") || (blobs[i].category == "bus")|| (blobs[i].category == "motor")
                || (blobs[i].category == "truck")  || (blobs[i].category == "rider") || (blobs[i].category == "person")
                || (blobs[i].category == "train") ) {

                if(enableEvaluation_){
                    if((blobs[i].category == "car") || (blobs[i].category == "bus")|| (blobs[i].category == "motor")
                       || (blobs[i].category == "truck")  || (blobs[i].category == "rider")
                       || (blobs[i].category == "train") )
                        cate = 0;
                    else if(blobs[i].category == "person")
                        cate = 1;
                }
                obstacle_msgs::obs tmpObs;

                tmpObs.identityID = i;

                tmpObs.centerPos.x = blobs[i].position_3d[0];
                tmpObs.centerPos.y = blobs[i].position_3d[1];
                tmpObs.centerPos.z = blobs[i].position_3d[2];
                tmpObs.diameter = blobs[i].diameter;
                tmpObs.height = blobs[i].height;
                tmpObs.xmin = blobs[i].xmin;
                tmpObs.ymin = blobs[i].ymin;
                tmpObs.xmax = blobs[i].xmax;
                tmpObs.ymax = blobs[i].ymax;

                tmpObs.counter = blobs[i].counter;
                tmpObs.classes = blobs[i].category;
                tmpObs.probability = blobs[i].probability;
//            tmpObs.histogram = blobs[i].obsHog;

                obstacleBoxesResults_.obsData.push_back(tmpObs);


//            ROS_WARN("center ID: %d | type: %s\nx: %f| y: %f| z: %f \n", i, tmpObs.classes,
//                    tmpObs.centerPos.x, tmpObs.centerPos.y, tmpObs.centerPos.z);


                ////*--------------Generate Evaluation files----------------------*////
                if(enableEvaluation_){
                    file << i << " " << blobs[i].currentBoundingRect.x << " " << blobs[i].currentBoundingRect.y << " "
                         << blobs[i].currentBoundingRect.x + blobs[i].currentBoundingRect.width << " " <<
                         blobs[i].currentBoundingRect.y + blobs[i].currentBoundingRect.height << " " << cate
                         << std::endl;
                }
            }
        }
    }
    if(enableEvaluation_){
        file.close();
        cv::imwrite(img_name, buff_cv_l_[(buffIndex_ + 1) % 3]);
    }
}


} /* namespace darknet_ros*/
