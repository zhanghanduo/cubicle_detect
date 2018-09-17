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
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
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

    std::string YoloObjectDetector::type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle nh_p)
    : nodeHandle_(nh),
      nodeHandle_pub(nh_p),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0),
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

//  DefineLUTs();

//  mpDepth_gen_run = new std::thread(&Detection::Run, mpDetection);

  hog_descriptor = new Util::HOGFeatureDescriptor(8, 2, 9, 180.0);
  img_name = ros::package::getPath("cubicle_detect") + "/seq_1/f000.png";
  file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/f000.txt";
  frame_num = 0;
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
//  free(depthTable);
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
  u0 = 0;
  counter = 0;

    nodeHandle_.param<int>("disparity_scope", disp_size, 128);
    nodeHandle_.param<int>("image_width", Width, 1280);
    nodeHandle_.param<int>("image_height", Height, 720);
    nodeHandle_.param<int>("scale", Scale, 1);

    Width /= Scale;
    Height /= Scale;

//  int ii;
//  xDirectionPosition = static_cast<double **>(calloc(Width, sizeof(double *)));
//  for(ii = 0; ii < Width; ii++)
//      xDirectionPosition[ii] = static_cast<double *>(calloc(disp_size+1, sizeof(double)));
//
//  yDirectionPosition = static_cast<double **>(calloc(Height, sizeof(double *)));
//  for(ii = 0; ii < Height; ii++)
//      yDirectionPosition[ii] = static_cast<double *>(calloc(disp_size+1, sizeof(double)));

//  depthTable = static_cast<double *>(calloc(disp_size+1, sizeof(double)));


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
                    std::string("/cubicle_detection/long_map_msg"));
  nodeHandle_.param("publishers/obstacle_boxes/queue_size", obstacleBoxesQueueSize, 1);
  nodeHandle_.param("publishers/obstacle_boxes/frame_id", pub_obs_frame_id, std::string("zed_obs"));
  nodeHandle_.param("image_view/disparity_far", disparity_far, 10);

  objectPublisher_ = nodeHandle_pub.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch);

  obstaclePublisher_ = nodeHandle_pub.advertise<obstacle_msgs::MapInfo>(
          obstacleBoxesTopicName, obstacleBoxesQueueSize);

  detectionImagePublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);

    printf("width: %d | height: %d\n", Width, Height);

}

void YoloObjectDetector:: loadCameraCalibration(const sensor_msgs::CameraInfoConstPtr &info) {

  ROS_INFO_STREAM("init calibration");

  // Check if a valid calibration exists
  if (info->K[0] == 0.0) {
    ROS_ERROR("The camera is not calibrated");
    return;
  }

  sensor_msgs::CameraInfoPtr left_info_copy = boost::make_shared<sensor_msgs::CameraInfo>(*info);
  left_info_copy->header.frame_id = "stereo";

  u0 = info->K[2];
  v0 = info->K[5];
  focal = info->K[0];
}

void YoloObjectDetector::DefineLUTs() {

  ROS_WARN("u0: %f | v0: %f | focal: %f | base: %f", u0, v0, focal, stereo_baseline_);

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

    void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& image,
                                            const sensor_msgs::CameraInfoConstPtr& info,
                                            const stereo_msgs::DisparityImageConstPtr& disparity){
//        ROS_DEBUG("[ObstacleDetector] Stereo images received.");

        cv_bridge::CvImageConstPtr cam_image1, cam_image2, cv_rgb, disp_input;

        try {
            cv_rgb = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
            disp_input = cv_bridge::toCvCopy(disparity->image, sensor_msgs::image_encodings::TYPE_32FC1);

            max_disparity = disparity->max_disparity;
            min_disparity = disparity->min_disparity;
            image_time_ = image->header.stamp;
            imageHeader_ = image->header;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if(u0 == 0) {
            stereo_baseline_ = disparity->T;
            loadCameraCalibration(info);
            DefineLUTs();
        }

        // std::cout<<"Debug inside cameraCallBack after DefineLUTs"<<std::endl;

        if (cv_rgb) {
            // std::cout<<"Debug inside cameraCallBack starting first image callback"<<std::endl;

            frameWidth_ = cv_rgb->image.size().width;
            // std::cout<<"Debug inside cameraCallBack reading image width "<<frameWidth_<<std::endl;
            frameHeight_ = cv_rgb->image.size().height;
            frameWidth_ = frameWidth_ / Scale;
            frameHeight_ = frameHeight_ / Scale;
            // std::cout<<"Debug inside cameraCallBack scaling image height "<<frameHeight_<<std::endl;
            {
                boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);

                camImageOrig = cv_rgb->image.clone();//cv::Mat(cv_rgb->image.clone(), left_roi_);

                disparity_float = disp_input->image;
                disparity_float.convertTo(disparity_image, CV_8U, 255.0 / (max_disparity - min_disparity), - min_disparity);
//                        -255.0 * min_disparity / (max_disparity - min_disparity));

            }
            {
                boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
                imageStatus_ = true;
            }

            // std::cout<<"Debug inside cameraCallBack starting image resize"<<std::endl;

            cv::resize(camImageOrig, camImageCopy_, cv::Size(frameWidth_, frameHeight_));
            cv::resize(disparity_image, disparity_resized, cv::Size(frameWidth_, frameHeight_));

//            disparity_float.convertTo(disparity_resized, CV_8UC1, 255, 0);

//            std::string ty =  type2str( disparity_resized.type() );
//            ROS_WARN("Type: %s \n", ty.c_str());

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

  rgbgr_image(buff_[buffIndex_]);

  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);

  buff_cv_l_[(buffIndex_)] = camImageCopy_;
//  buff_cv_r_[(buffIndex_)] = right_rectified.clone();

  if(counter > 2) {

      disparityFrame[(buffIndex_ + 2) % 3] = disparity_resized;
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
  disparityFrame[0] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
  disparityFrame[1] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
  disparityFrame[2] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
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
  cvImage = cv::cvarrToMat(ipl_);
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
          auto center_c_ = static_cast<int>(rosBoxes_[i][j].x * frameWidth_);     //2D column
          auto center_r_ = static_cast<int>(rosBoxes_[i][j].y * frameHeight_);    //2D row

          auto xmin = static_cast<int>((rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_);
          auto ymin = static_cast<int>((rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_);
          auto xmax = static_cast<int>((rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_);
          auto ymax = static_cast<int>((rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_);

          int median_kernel = std::min(xmax - xmin, ymax - ymin);

            // if ((xmin > 2) &&(ymin > 2) && (counter>2) ) {
          if ((counter>2) ) {

//                auto dis = (int)disparityFrame.at<uchar>(center_r_, center_c_);
                auto dis = static_cast<int>(Util::median_mat(disparityFrame[(buffIndex_ + 1) % 3], center_c_, center_r_, median_kernel));  // find 3x3 median
//                std::cout << "dis: " << dis << std::endl;
                cv::Rect_<int> rect = cv::Rect_<int>(xmin, ymin, xmax - xmin, ymax - ymin);
//                cv::Mat roi_dis = disparityFrame(rect).clone();
//                double max,min;
//                cv::minMaxLoc(roi_dis, &min, &max);

//                int dis = static_cast<int>(max);
                if(dis > disparity_far) { // disregard the far object

//                    ROS_WARN("center 2D\ncol: %d| row: %d", center_c_, center_r_);
//                    ROS_WARN("min 2D\ncol: %d| row: %d", xmin, ymin);
//                    ROS_WARN("max 2D\ncol: %d| row: %d", xmax, ymax);
                    // Hog features
                    cv::Mat roi = buff_cv_l_[(buffIndex_ + 1) % 3](rect).clone();
                    cv::resize(roi, roi, cv::Size(22, 22));
                    std::vector<float> hog_feature;
                    hog_descriptor -> computeHOG(hog_feature, roi);
//                    ROS_WARN("hog_size: %d", hog_feature.size());

                    std::vector<cv::Point3f> cent_2d, cent_3d;
                    Blob outputObs(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
//                    obstacle_msgs::obs outputObs;
                    outputObs.category = classLabels_[i];
                    outputObs.probability = rosBoxes_[i][j].prob;
                    outputObs.position_3d[0] = xDirectionPosition[center_c_][dis];
                    outputObs.position_3d[1] = yDirectionPosition[center_r_][dis];
                    outputObs.position_3d[2] = depthTable[dis];
                    outputObs.xmin = xmin;
                    outputObs.xmax = xmax;
                    outputObs.ymin = ymin;
                    outputObs.ymax = ymax;
//                    ROS_WARN("center 3D\nx: %f| y: %f| z: %f",
//                             outputObs.position_3d[0], outputObs.position_3d[1], depthTable[dis]);

                    double xmin_3d, xmax_3d, ymin_3d, ymax_3d;
                    xmin_3d = xDirectionPosition[xmin][dis];
                    xmax_3d = xDirectionPosition[xmax][dis];
                    ymin_3d = yDirectionPosition[ymin][dis];
                    ymax_3d = yDirectionPosition[ymax][dis];
//                    ROS_WARN("min 3D\nx: %f| y: %f", xmin_3d, xmax_3d);
//                    ROS_WARN("max 3D\nx: %f| y: %f", xmax_3d, ymax_3d);
                    outputObs.diameter = abs(static_cast<int>(xmax_3d - xmin_3d));
                    outputObs.height = abs(static_cast<int>(ymax_3d - ymin_3d));
                    outputObs.obsHog = hog_feature;
                    outputObs.disparity = dis;
//                    obstacleBoxesResults_.obsData.push_back(outputObs);
                    currentFrameBlobs.push_back(outputObs);

//                    ROS_WARN("cata: %s, depth: %f", outputObs.category.c_str(), depthTable[dis]);
//                    Tracking();
//                    CreateMsg();
                }
//                else {
//                  std::string classname = classLabels_[i];
//                  ROS_WARN("class, dis: %s, %d", classname.c_str(), dis);
//                }

            } else {
              ROS_WARN("*********************************************************");
            }

        }
      }
    }

    // TODO: for debugging
    cv::Mat beforeTracking = buff_cv_l_[(buffIndex_ + 1) % 3];
    for (auto &currentFrameBlob : currentFrameBlobs) {
      cv::rectangle(beforeTracking, currentFrameBlob.currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
    }
    cv::imshow("beforeTracking", beforeTracking);


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

  // TODO: wait until isDepth_new to be true
  Tracking();

  CreateMsg();

  obstacleBoxesResults_.header.stamp = image_time_;
  obstacleBoxesResults_.header.frame_id = pub_obs_frame_id;
  obstacleBoxesResults_.real_header.stamp = ros::Time::now();
  obstacleBoxesResults_.real_header.frame_id = pub_obs_frame_id;
  obstaclePublisher_.publish(obstacleBoxesResults_);

  obstacleBoxesResults_.obsData.clear();
  obstacleBoxesResults_.segsData.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return nullptr;
}

void YoloObjectDetector::matchCurrentFrameBlobsToExistingBlobs() {

//    for (auto &existingBlob : blobs) {
//
//        existingBlob.blnCurrentMatchFoundOrNewBlob = false;
//
//        existingBlob.blnAlreadyTrackedInThisFrame = false;
//
//        existingBlob.predictNextPosition();
//    }

//    std::list<Blob>::iterator blobList;
//    for(blobList = blobs.begin(); blobList != blobs.end();)

    for (auto &currentFrameBlob : currentFrameBlobs) {

      int intIndexOfLeastDistance = -1;
//        int intIndexOfLeastHogDis = -1;
      dblLeastDistance = 100000.0;
//        hogLeastDistance = 100000.0;

      for (unsigned int j = 0; j < blobs.size(); ++j) {

        if (blobs[j].blnStillBeingTracked) {

          if (currentFrameBlob.category == blobs[j].category) {
            ////*--------------HOG FEATURE----------------------*////
//                    double hogDistance = cv::norm(currentFrameBlob.obsHog, blobs[j].obsHog, cv::NORM_L2);
//
//
//                    if (hogDistance < hogLeastDistance) {
//
//                        hogLeastDistance = hogDistance;
//
//                        intIndexOfLeastHogDis = j;
//                    }
            ////*--------------HOG FEATURE----------------------*////



            ////*--------------POSITION----------------------*////
            int dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(),
                                                    blobs[j].predictedNextPosition);

            if (dblDistance < dblLeastDistance) {

              dblLeastDistance = dblDistance;

              intIndexOfLeastDistance = j;
            }


          }


        }
      }
//        std::cout << "hogdis: " << hogLeastDistance <<", category: "<< currentFrameBlob.category<< std::endl;
//
//        // TODO: hog feature to replace diagonalsize!
////*--------------HOG FEATURE----------------------*////
//      if(intIndexOfLeastHogDis != -1){
//        double hogSelf = cv::norm(currentFrameBlob.obsHog, nullHog, cv::NORM_L2);
//        std::cout << "hogSelf: " << hogSelf <<", category: "<< currentFrameBlob.category<< std::endl;
//
//        if( (hogLeastDistance < hogSelf * 0.2) && (!blobs[intIndexOfLeastHogDis].blnAlreadyTrackedInThisFrame)) {
//          addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastHogDis);
//
//        }else{
//          addNewBlob(currentFrameBlob, blobs);
//        }
//      } else{
//        addNewBlob(currentFrameBlob, blobs);
//      }
////*--------------HOG FEATURE----------------------*////


      if (intIndexOfLeastDistance != -1) {
        if ((dblLeastDistance < (static_cast<int>(currentFrameBlob.dblCurrentDiagonalSize * 1.4))) &&
            (!blobs[intIndexOfLeastDistance].blnAlreadyTrackedInThisFrame)) {

          addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance);

        } else {
          addNewBlob(currentFrameBlob, blobs);
        }
      } else {
        addNewBlob(currentFrameBlob, blobs);
      }
    }
//  for (auto &existingBlob : blobs) {
//      if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
//      existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
//    }
//    if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 100) {
//      existingBlob.blnStillBeingTracked = false;
//    }
//  }

}

void YoloObjectDetector::addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;

//    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
    existingBlobs[intIndex].disparity = currentFrameBlob.disparity;

    existingBlobs[intIndex].xmin = currentFrameBlob.xmin;
    existingBlobs[intIndex].xmax = currentFrameBlob.xmax;
    existingBlobs[intIndex].ymin = currentFrameBlob.ymin;
    existingBlobs[intIndex].ymax = currentFrameBlob.ymax;

    existingBlobs[intIndex].position_3d = currentFrameBlob.position_3d;

    existingBlobs[intIndex].obsHog = currentFrameBlob.obsHog;

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
    cv::Mat color_out;

    cv::Mat output1 = disparityFrame[(buffIndex_ + 1) % 3].clone();
    cv::Mat output = buff_cv_l_[(buffIndex_ + 1) % 3].clone();
    if(output.type() == CV_8UC1)
        cv::cvtColor(output, color_out, CV_GRAY2RGB);
    else
        color_out = output;

    for (long int i = 0; i < blobs.size(); i++) {
//            if (blobs[i].blnStillBeingTracked == true) {
        if (blobs[i].blnCurrentMatchFoundOrNewBlob) {
            cv::rectangle(color_out, blobs[i].currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
            cv::rectangle(output1, blobs[i].currentBoundingRect, cv::Scalar( 255, 255, 255 ), 2);
//            for(int j=0; j<blobs[i].obsPoints.size();j++){
//                output.at<cv::Vec3b>(blobs[i].obsPoints[j].x, blobs[i].obsPoints[j].y)[2]=255;//cv::Vec3b(0,0,255);
//            }
            std::ostringstream str;
            str << blobs[i].position_3d[2] <<"m, ID="<<i<<"; "<<blobs[i].disparity;
            cv::putText(color_out, str.str(), blobs[i].centerPositions.back(), CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(0,250,0));
            cv::putText(output1, str.str(), blobs[i].centerPositions.back(), CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(255, 250, 255));
        }
    }
    cv::imshow("debug", color_out);
    cv::imshow("disparity", output1);

//    char img2[20];

    frame_num ++;
    if(enableEvaluation_){
    sprintf(s, "f%03d.txt", frame_num);
    sprintf(im, "f%03d.png", frame_num);

//    sprintf(img2, "y%03d", frame_num);

    file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/" + s;
    img_name = ros::package::getPath("cubicle_detect") + "/seq_1/" + im;
//    img_name2 = ros::package::getPath("cubicle_detect") + "/seq_1/yolo/" + img2;
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
//        cv::imwrite(img_name2, cvImage);
//        char img_[20];
//        sprintf(img_, "/home/hd/catkin_ws/src/cubicle_detect/darknet_ros/seq_1/yolo/%s", img2);
//        save_image_png(buff_[(buffIndex_ + 1)%3], img_);
    }
}

} /* namespace darknet_ros*/
