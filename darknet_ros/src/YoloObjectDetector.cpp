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
//  init();

  std::string model_folder = "/home/ugv/catkin_ws/src/cubicle_detect/models/";//std::string(std::getenv("OPENPOSE_HOME")) + std::string("/models/");
//  const auto heatMapTypes = op::flagsToHeatMaps(false, false, false);
//  const auto heatMapScale = op::flagsToHeatMapScaleMode(op::ScaleMode::ZeroToOne);
//  op::flagsTo
//  const op::WrapperStructPose wrapperStructPose{
//      true, "-1x368", "-1x-1", 0, -1, 0, 1, (float) 0.3, op::flagsToRenderMode(-1, false), "BODY_25", true,
//      (float) 0.6, (float) 0.7, 0, model_folder, heatMapTypes, op::ScaleMode::ZeroToOne, false, (float) 0.05, -1, true};

//  std::cout<<"before WrapperStructPose"<<std::endl;
//
//  const op::WrapperStructPose wrapperStructPose{
//          true, {}, {}, {}, {}, {}, {}, {}, op::flagsToRenderMode(2, false), {}, {},
//          {}, {}, {}, model_folder, {}, {}, {}, {}, {}, true};

  const op::WrapperStructPose wrapperStructPose{true, op::Point<int>{656, 368}, op::Point<int>{-1, -1},
                                                op::ScaleMode::InputResolution, -1, 0, 1, 0.15f, op::RenderMode::Gpu,
                                                op::PoseModel::BODY_25, true, op::POSE_DEFAULT_ALPHA_KEYPOINT,
                                                op::POSE_DEFAULT_ALPHA_HEAT_MAP, 0, model_folder, {}, op::ScaleMode::ZeroToOne, false,
                                                0.05f, -1, true};
//
//  std::cout<<"WrapperStructPose"<<std::endl;
//
  opWrapper.configure(wrapperStructPose);
  opWrapper.configure(op::WrapperStructFace{});
  opWrapper.configure(op::WrapperStructHand{});
  opWrapper.configure(op::WrapperStructExtra{});
  opWrapper.configure(op::WrapperStructOutput{});
//
//  std::cout<<"before disable multithread"<<std::endl;
  opWrapper.disableMultiThreading();
    opWrapper.start();
//  std::cout<<"after disable multithread"<<std::endl;


//  std::cout<<"opWrapper start"<<std::endl;

//  SGM = new disparity_sgm(7, 86);
//  SGM->init_disparity_method(7, 86);

//  mpDepth_gen_run = new std::thread(&Detection::Run, mpDetection);

//  hog_descriptor = new Util::HOGFeatureDescriptor(8, 2, 9, 180.0);
//  img_name = ros::package::getPath("cubicle_detect") + "/seq_1/f000.png";
//  file_name = ros::package::getPath("cubicle_detect") + "/seq_1/results/f000.txt";
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

//  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
//                    std::string("found_object"));
//  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
//  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
//  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
//                    std::string("detection_image"));
//  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
//  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);
//
//  nodeHandle_.param("publishers/obstacle_boxes/topic", obstacleBoxesTopicName,
//                    std::string("/cubicle_detection/long_map_msg"));
//  nodeHandle_.param("publishers/obstacle_boxes/queue_size", obstacleBoxesQueueSize, 1);
//  nodeHandle_.param("publishers/obstacle_boxes/frame_id", pub_obs_frame_id, std::string("long_obs"));
//
//  objectPublisher_ = nodeHandle_pub.advertise<std_msgs::Int8>(objectDetectorTopicName,objectDetectorQueueSize, objectDetectorLatch);
//
//  obstaclePublisher_ = nodeHandle_pub.advertise<obstacle_msgs::MapInfo>(
//          obstacleBoxesTopicName, obstacleBoxesQueueSize);
//
//  detectionImagePublisher_ = nodeHandle_pub.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);

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

    void YoloObjectDetector::setFileName(std::string name){
        file_name = ros::package::getPath("cubicle_detect") +"/"+ name + ".txt";
        file.open(file_name);
    }

    void YoloObjectDetector::closeFile(){

        frame_num = 0;
        blnFirstFrame = true;
        initiated = false;
        blobs.clear();
        file.close();

    }

    void YoloObjectDetector::displayPose(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr){

        cv::Mat outputKP = camImageCopy_.clone();

        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
//            op::log("Body keypoints: " + datumsPtr->at(0).poseKeypoints.toString());

//            op::log("Pose ID: " + datumsPtr->at(0).poseIds.toString());

            for(int ii=0; ii<datumsPtr->at(0).poseKeypoints.getVolume();ii+=3){
                std::ostringstream str;
                cv::Point2i position;
                position.x = (int) datumsPtr->at(0).poseKeypoints.at(ii);
                position.y = (int) datumsPtr->at(0).poseKeypoints.at(ii+1);
                // str << blobs[i].position_3d[2] <<"m, ID="<<i<<"; "<<blobs[i].disparity;
//                if (ii%75==0){
//                    std::cout<<"new person"<<std::endl;
//                }
//                std::cout<<position.x<<", "<<position.y<<", "<<datumsPtr->at(0).poseKeypoints.at(ii+2)<<std::endl;
                if (datumsPtr->at(0).poseKeypoints.at(ii+2)>0.0){
//                    std::cout<<position.x<<", "<<position.y<<", "<<datumsPtr->at(0).poseKeypoints.at(ii+2)<<std::endl;
                    str << ii;//datumsPtr->at(0).poseKeypoints.at(ii+2);
                    cv::putText(outputKP, str.str(), position, CV_FONT_HERSHEY_PLAIN, 0.7, CV_RGB(0,0,255));
                }

            }
            cv::imshow("outputKP",outputKP);
            // Display image
//            float fuk = datumsPtr->at(0).poseKeypoints.at(0);
            cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
//            cv::waitKey(0);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }

    cv::MatND YoloObjectDetector::calculateHistogram(cv::Mat hsv, cv::Mat mask){
        // Quantize the hue to 30 levels and the saturation to 32 levels
        int hbins = 30, sbins = 32;
        int histSize[] = {30,32};//{hbins, sbins};{hbins, sbins};
        // // hue varies from 0 to 179, see cvtColor
        float hranges[] = { 0, 180 };
        // // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
        float sranges[] = { 0, 256 };
        const float* ranges[] = { hranges, sranges };
        cv::MatND hist;
        // we compute the histogram from the 0-th and 1-st channels
        int channels[] = {0, 1};

        cv::calcHist( &hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false );
        cv::normalize(hist, hist, 1, 0, 2, -1, cv::Mat());

        return hist;
    }

    void YoloObjectDetector::extractBodyParts(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr){

        cv::Mat outputBP = camImageCopy_.clone();
        cv::Mat hsv;
        cv::cvtColor(outputBP, hsv, CV_BGR2HSV);
        currentFrameBlobs.clear();

        if (datumsPtr != nullptr && !datumsPtr->empty()) {

            std::vector<cv::Point3f> partList;

//            std::cout<<"extracting pose points"<<std::endl;

            for(int ii=0; ii<datumsPtr->at(0).poseKeypoints.getVolume();ii+=3) {
                cv::Point3f partDescriptor;
                partDescriptor.x = datumsPtr->at(0).poseKeypoints.at(ii);
                partDescriptor.y = datumsPtr->at(0).poseKeypoints.at(ii+1);
                partDescriptor.z = datumsPtr->at(0).poseKeypoints.at(ii+2);
                partList.push_back(partDescriptor);
            }

            int noOfPersons = static_cast<int>(partList.size() / 25);
//            std::cout<<"allocating person for each detection box, #persons:"<<noOfPersons<<std::endl;

            for (size_t kk = 0; kk < detListCurrFrame.size(); kk++) {

                if (detListCurrFrame.at(kk).validDet) {

                    auto xmin = detListCurrFrame.at(kk).bb_left;
                    auto ymin = detListCurrFrame.at(kk).bb_top;
                    auto width = detListCurrFrame.at(kk).bb_width;
                    auto height = detListCurrFrame.at(kk).bb_height;

                    cv::Rect_<int> detBBox = cv::Rect_<int>(static_cast<int>(xmin), static_cast<int>(ymin),
                                                            static_cast<int>(width), static_cast<int>(height));

                    float maxCost = 0.0, maxIOU = 0.0;
                    int indexOfMaxCost = -1;

//                std::cout<<"new detection"<<std::endl;

                    for (int ii = 0; ii < noOfPersons; ii++) {
                        float costForPerson = 0.0;
                        int noOfPartsOutside = 0;// allPartsInside = true;

                        float partsXMin = 10000, partsYMin = 10000, partsXMax = 0, partsYMax = 0;

                        for (int jj = 0; jj < 25; jj++) {
                            auto ppart = partList.at(ii * 25 + jj);
                            if (ppart.z > 0.0) {

                                float xCord = ppart.x;
                                float yCord = ppart.y;

                                if (xCord >= xmin && xCord <= xmin + width &&
                                    yCord >= ymin && yCord <= ymin + height) {
                                    costForPerson += ppart.z;
                                } else {
                                    noOfPartsOutside++;
//                                std::cout<<ii<<"th person, "<<jj<<std::endl;
                                }

                                if (partsXMin > xCord)
                                    partsXMin = xCord;

                                if (partsXMax < xCord)
                                    partsXMax = xCord;

                                if (partsYMin > yCord)
                                    partsYMin = yCord;

                                if (partsYMax < yCord)
                                    partsYMax = yCord;

                            }
                        }

                        cv::Rect_<int> minBBox = cv::Rect_<int>(static_cast<int>(partsXMin),
                                                                static_cast<int>(partsYMin),
                                                                static_cast<int>(partsXMax - partsXMin),
                                                                static_cast<int>(partsYMax - partsYMin));

                        cv::Rect intersection = detBBox & minBBox;
                        cv::Rect unio = detBBox | minBBox;
                        float iou = (float) intersection.area() / unio.area();

                        costForPerson *= iou;

                        if (noOfPartsOutside > 10)
                            costForPerson = 0.0;

                        if (costForPerson > maxCost) {
                            maxCost = costForPerson;
                            indexOfMaxCost = ii;
                            maxIOU = iou;
                        }

//                    std::cout<<"bbox: "<<kk<<", pID: "<<ii<<", iou: "<<iou<<", partsOutside: "<<noOfPartsOutside<<", costFor: "<<costForPerson<<std::endl;
                    }

                    if (indexOfMaxCost != -1 && maxIOU > 0.1) {
                        detListCurrFrame.at(kk).personID = indexOfMaxCost;
                    }
                }

            }

            for (size_t kk = 0; kk < detListCurrFrame.size(); kk++) {

                inputDetections inputDetkk = detListCurrFrame.at(kk);

                if(inputDetkk.personID!=-1){

                    for (size_t ll = 0; ll < detListCurrFrame.size(); ll++) {
                        if (ll==kk)
                            continue;

                        inputDetections inputDetll = detListCurrFrame.at(ll);

                        if(inputDetkk.personID==inputDetll.personID){
                            float partsXMin = 10000, partsYMin = 10000, partsXMax =0, partsYMax =0;
                            for (int jj=0; jj<25; jj++){
                                auto ppart = partList.at(inputDetll.personID*25+jj);
                                if (ppart.z>0.0){
                                    float xCord = ppart.x;
                                    float yCord = ppart.y;

                                    if (partsXMin>xCord)
                                        partsXMin = xCord;
                                    if (partsXMax<xCord)
                                        partsXMax = xCord;

                                    if (partsYMin>yCord)
                                        partsYMin = yCord;
                                    if (partsYMax<yCord)
                                        partsYMax = yCord;
                                }
                            }

                            cv::Rect_<int> minBBox = cv::Rect_<int>(static_cast<int>(partsXMin), static_cast<int>(partsYMin),
                                                                    static_cast<int>(partsXMax - partsXMin),
                                                                    static_cast<int>(partsYMax - partsYMin));

                            cv::Rect_<int> bboxkk = cv::Rect_<int>(static_cast<int>(inputDetkk.bb_left),
                                                                   static_cast<int>(inputDetkk.bb_top),
                                                                   static_cast<int>(inputDetkk.bb_width),
                                                                   static_cast<int>(inputDetkk.bb_height));
                            cv::Rect_<int> bboxll = cv::Rect_<int>(static_cast<int>(inputDetll.bb_left),
                                                                   static_cast<int>(inputDetll.bb_top),
                                                                   static_cast<int>(inputDetll.bb_width),
                                                                   static_cast<int>(inputDetll.bb_height));

                            cv::Rect intersectionkk = bboxkk & minBBox;
                            cv::Rect uniokk = bboxkk | minBBox;
                            float ioukk = (float)intersectionkk.area() / uniokk.area();

                            cv::Rect intersectionll = bboxll & minBBox;
                            cv::Rect unioll = bboxll | minBBox;
                            float ioull = (float)intersectionll.area() / unioll.area();

                            if (ioukk>ioull) {
                                detListCurrFrame.at(ll).personID = -1;
                                detListCurrFrame.at(ll).validDet = false;
                            } else {
                                detListCurrFrame.at(kk).personID = -1;
                                detListCurrFrame.at(kk).validDet = false;
                            }
                        }
                    }
                } else {
                    if (inputDetkk.bb_height>frameHeight_*.2)
                        detListCurrFrame.at(kk).validDet = false;
                }
            }

            std::vector<cv::Scalar> colors;
            cv::RNG rng(0);
            for(int i=0; i < detListCurrFrame.size(); i++)
                colors.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));

//            std::cout<<"extracting head, body and leg"<<std::endl;

            for (size_t kk = 0; kk < detListCurrFrame.size(); kk++) {

                if (detListCurrFrame.at(kk).validDet) {

                    auto xmin = static_cast<int>(detListCurrFrame.at(kk).bb_left);
                    auto ymin = static_cast<int>(detListCurrFrame.at(kk).bb_top);
                    auto width = static_cast<int>(detListCurrFrame.at(kk).bb_width);
                    auto height = static_cast<int>(detListCurrFrame.at(kk).bb_height);
                    auto pID = detListCurrFrame.at(kk).personID;

                    cv::Rect_<int> bbox = cv::Rect_<int>(xmin, ymin, width, height);
                    cv::rectangle(outputBP, bbox, colors.at(kk), 2);

//                    Blob outputObs(bbox);
                    Blob outputObs(detListCurrFrame.at(kk).bb_left,
                                   detListCurrFrame.at(kk).bb_top,
                                   detListCurrFrame.at(kk).bb_width,
                                   detListCurrFrame.at(kk).bb_height);
                    outputObs.category = "Pedestrian";//classLabels_[i]; //mot
                    outputObs.probability = detListCurrFrame.at(kk).det_conf;

//                std::cout<<"bbox: "<<kk<<", pID: "<<pID<<std::endl;

                    if (pID != -1) {

                        cv::Mat blobMask = occlutionMap(bbox, kk);
                        cv::Mat hsvBBoxROI;
                        hsv(bbox).copyTo(hsvBBoxROI);
//                        std::ostringstream str;
//                        str << "BBox "<<kk;
//                        cv::imshow(str.str(),blobMask*255);

//                    cv::rectangle(outputBP, bbox, colors.at(kk),2);
//                    cv::Point3f nose = partList.at(25*pID);
//                    cv::Point3f neck = partList.at(25*pID+1);
                        cv::Point3f Rshoulder = partList.at(25 * pID + 2);
                        cv::Point3f Lshoulder = partList.at(25 * pID + 5);
                        cv::Point3f Midhip = partList.at(25 * pID + 8);
//                    cv::Point3f Rhip = partList.at(25*pID+9);
//                    cv::Point3f Lhip = partList.at(25*pID+12);
                        cv::Point3f Rankle = partList.at(25 * pID + 11);
                        cv::Point3f Lankle = partList.at(25 * pID + 14);
                        cv::Point3f Rear = partList.at(25 * pID + 17);
                        cv::Point3f Lear = partList.at(25 * pID + 18);

//                    std::cout<<Rshoulder<<", "<<Lshoulder<<", "<<Midhip<<", "<<Rear<<", "<<Lear<<std::endl;

                        //Extract head
                        cv::Mat maskHead(camImageCopy_.size(),CV_8UC1,cv::Scalar::all(0));
                        if (Lear.z > 0.0 && Rear.z > 0.0) {

                            cv::Point2i center = cv::Point2i((int) ((Lear.x + Rear.x) / 2),
                                                             (int) ((Lear.y + Rear.y) / 2));

                            if (bbox.contains(cv::Point((int) Lear.x, (int) Lear.y)) &&
                                bbox.contains(cv::Point((int) Rear.x, (int) Rear.y))){

                                cv::circle(maskHead, center, (int) (std::fabs(Lear.x - Rear.x) / 2), cv::Scalar::all(1),CV_FILLED);
                                cv::circle(outputBP, center, (int) (std::fabs(Lear.x - Rear.x) / 2), colors.at(kk));

                                cv::Mat maskRoiHead = maskHead(bbox).mul(blobMask);
                                outputObs.histHead = calculateHistogram(hsvBBoxROI,maskRoiHead);
                                outputObs.isHead = true;
//                                cv::imshow("maskRoiHead",maskRoiHead*255);
                            }
                        }
//                        maskHead(bbox).copyTo(maskRoiHead);

                        //extract body
                        cv::Mat maskBody(camImageCopy_.size(),CV_8UC1,cv::Scalar::all(0));
                        cv::Point2i topLeft = cv::Point2i(static_cast<int>(Rshoulder.x), static_cast<int>(Rshoulder.y));
                        if (Rshoulder.x > Lshoulder.x)
                            topLeft = cv::Point2i(static_cast<int>(Lshoulder.x), static_cast<int>(Lshoulder.y));
                        int bodyWidth = static_cast<int>(std::fabs(Rshoulder.x - Lshoulder.x));

                        if (Rshoulder.z > 0.0 && Lshoulder.z > 0.0 && Midhip.z > 0.0) {
                            int bodyHeight = static_cast<int>(std::fabs(Rshoulder.y - Midhip.y));

                            if (bbox.contains(topLeft) &&
                                bbox.contains(cv::Point(topLeft.x + bodyWidth, topLeft.y + bodyHeight))) {

                                cv::rectangle(maskBody, cv::Rect(topLeft.x, topLeft.y, bodyWidth, bodyHeight),
                                              cv::Scalar::all(1),CV_FILLED);
                                cv::rectangle(outputBP, cv::Rect(topLeft.x, topLeft.y, bodyWidth, bodyHeight),
                                              colors.at(kk));

                                cv::Mat maskRoiBody = maskBody(bbox).mul(blobMask);
                                outputObs.histBody = calculateHistogram(hsvBBoxROI,maskRoiBody);
                                outputObs.isBody = true;
//                                cv::imshow("maskRoiBody",maskRoiBody*255);
                            }

                        } else if (Rshoulder.z > 0.0 && Lshoulder.z > 0.0) {
                            int bodyHeight = static_cast<int>(std::fabs(ymin + height - Rshoulder.y));
                            if (bbox.contains(topLeft) &&
                                bbox.contains(cv::Point(topLeft.x + bodyWidth, topLeft.y + bodyHeight))) {

                                cv::rectangle(maskBody, cv::Rect(topLeft.x, topLeft.y, bodyWidth, bodyHeight),
                                              cv::Scalar::all(1),CV_FILLED);
                                cv::rectangle(outputBP, cv::Rect(topLeft.x, topLeft.y, bodyWidth, bodyHeight),
                                              colors.at(kk));

                                cv::Mat maskRoiBody = maskBody(bbox).mul(blobMask);
                                outputObs.histBody = calculateHistogram(hsvBBoxROI,maskRoiBody);
                                outputObs.isBody = true;
//                                cv::imshow("maskRoiBody",maskRoiBody*255);
                            }

                        }

                        //extract legs
                        cv::Mat maskLegs(camImageCopy_.size(),CV_8UC1,cv::Scalar::all(0));
                        if (Midhip.z > 0.0 && Rshoulder.z > 0.0 && Lshoulder.z > 0.0) {
                            int bodyHeight = static_cast<int>(std::fabs(ymin + height - Midhip.y));
                            if (Rankle.z > 0.0 && Rankle.y < ymin + height)
                                bodyHeight = static_cast<int>(std::fabs(Rankle.y - Midhip.y));
                            else if (Lankle.z > 0.0 && Lankle.y < ymin + height)
                                bodyHeight = static_cast<int>(std::fabs(Lankle.y - Midhip.y));

                            int topOff = static_cast<int>(Midhip.y);

                            if (bbox.contains(cv::Point(topLeft.x, topOff)) &&
                                bbox.contains(cv::Point(topLeft.x + bodyWidth, topOff + bodyHeight))) {

                                cv::rectangle(maskLegs, cv::Rect(topLeft.x, topOff, bodyWidth, bodyHeight),
                                              cv::Scalar::all(1),CV_FILLED);
                                cv::rectangle(outputBP, cv::Rect(topLeft.x, topOff, bodyWidth, bodyHeight),
                                              colors.at(kk));

                                cv::Mat maskRoiLegs = maskLegs(bbox).mul(blobMask);
                                outputObs.histLegs = calculateHistogram(hsvBBoxROI,maskRoiLegs);
                                outputObs.isLegs = true;
//                                cv::imshow("maskRoiLegs",maskRoiLegs*255);
                            }
                        }
                    }

                    currentFrameBlobs.push_back(outputObs);
                }

            }

        }

        cv::imshow("outputBP",outputBP);
        cv::waitKey(0);

    }

    cv::Mat YoloObjectDetector::occlutionMap(cv::Rect_<int> bbox, size_t kk) {

//        cv::Mat output = camImageCopy_.clone();

//        for (size_t kk = 0; kk < detListCurrFrame.size(); kk++) {
//            auto xmin = static_cast<int>(detListCurrFrame.at(kk).bb_left);
//            auto ymin = static_cast<int>(detListCurrFrame.at(kk).bb_top);
//            auto width = static_cast<int>(detListCurrFrame.at(kk).bb_width);
//            auto height = static_cast<int>(detListCurrFrame.at(kk).bb_height);
//            cv::Rect_<int> bbox = cv::Rect_<int>(xmin, ymin, width, height);
//
//            cv::rectangle(output, bbox, cv::Scalar( 0, 0, 255 ), 2);

            cv::Mat mask(camImageCopy_.size(),CV_8UC1,cv::Scalar::all(1));// = cv::Mat::ones(camImageCopy_.size(),CV_8UC3);

            for (size_t ii = 0; ii < detListCurrFrame.size(); ii++){
                if (ii==kk)
                    continue;

                if(detListCurrFrame.at(ii).validDet){
                    auto xminii = static_cast<int>(detListCurrFrame.at(ii).bb_left);
                    auto yminii = static_cast<int>(detListCurrFrame.at(ii).bb_top);
                    auto widthii = static_cast<int>(detListCurrFrame.at(ii).bb_width);
                    auto heightii = static_cast<int>(detListCurrFrame.at(ii).bb_height);
                    cv::Rect_<int> bboxii = cv::Rect_<int>(xminii, yminii, widthii, heightii);

                    cv::Rect intersection = bbox & bboxii;

                    if (intersection.area()>0){
                        if (bbox.y+bbox.height<yminii+heightii) {
                            mask(intersection).setTo(cv::Scalar::all(0));
//                        std::cout<<intersection<<std::endl;
                        }
                    }
                }

            }

            cv::Mat maskroi;
            mask(bbox).copyTo(maskroi);

            return maskroi;

//            cv::Mat imgCopy, imgroi, maskedimg;
//            camImageCopy_.copyTo(imgCopy);
//            maskedimg = imgCopy.mul(mask);
//            maskedimg(bbox).copyTo(imgroi);

//            std::ostringstream str;
//            str << "BBox "<<kk;
//            cv::imshow(str.str(),imgroi);
//            cv::imshow(str.str(),mask*255);

//        }

//        cv::imshow("output",output);
    }

    void YoloObjectDetector::readImage(cv::Mat img, std::string name){

        camImageCopy_ = img.clone();
        img_name = name;
        frameWidth_ = img.size().width;
        frameHeight_ = img.size().height;

//        cv::imshow("camImageCopy_", camImageCopy_);
//        cv::waitKey(0);


        auto datumProcessed = opWrapper.emplaceAndPop(camImageCopy_);
        if (datumProcessed != nullptr) {
            displayPose(datumProcessed);
        }
//        else
//            op::log("Image could not be processed.", op::Priority::High);

        processImage();
        extractBodyParts(datumProcessed);

//        if (initiated) {
//            processImage();
//        } else {
//            yolo();
//            processImage();
//        }
    }

    void YoloObjectDetector::readDetections(std::string name){

        std::cout<<name<<std::endl;
        std::ifstream infile(name);
        std::string line;
        detList.clear();
        while(std::getline(infile,line)) {
            std::stringstream   linestream(line);
            std::string         value;
            std::vector<std::string> fields;
//
            while(getline(linestream,value,',')) {
                fields.push_back(value);
//                std::cout << value << ",";
            }

            if (!fields.empty()){
                inputDetections fileLine;
                fileLine.frameNum =  std::stoi(fields.at(0));
                fileLine.objTrackID = std::stoi(fields.at(1));
                fileLine.bb_left = std::stof(fields.at(2));
                fileLine.bb_top = std::stof(fields.at(3));
                fileLine.bb_width = std::stof(fields.at(4));
                fileLine.bb_height = std::stof(fields.at(5));
                fileLine.det_conf = std::stof(fields.at(6));
                if (fileLine.bb_left<0.0)
                    fileLine.bb_left = 0.0;
                if (fileLine.bb_top<0.0)
                    fileLine.bb_top = 0.0;

                detList.push_back(fileLine);
            }
//            std::cout<<std::endl;
//            std::cout << "Line Finished" << std::endl;

        }

//        for (size_t i = 0; i < detList.size(); i++) {
//            std::cout<<detList.at(i).frameNum<<","<<detList.at(i).objTrackID<<"," <<
//            detList.at(i).bb_left<<","<<detList.at(i).bb_top<<","<<detList.at(i).bb_width<<","<<detList.at(i).bb_height<<","<<
//                    detList.at(i).det_conf<<std::endl;
//        }

        std::cout << "Detections Read" << std::endl;
    }

    void YoloObjectDetector::rearrangeDetection(int imgHeight, int imgWidth) {

//        std::cout<<"rearrangeDetection"<<std::endl;

        for (size_t i = 0; i < detList.size(); i++) {
            float width_del = detList.at(i).bb_left + detList.at(i).bb_width - float(imgWidth);
            float height_del = detList.at(i).bb_top + detList.at(i).bb_height - float(imgHeight);

            if (width_del > 0.0)
                detList.at(i).bb_width -= width_del;

            if (height_del > 0.0)
                detList.at(i).bb_height -= height_del;

        }

//        std::cout<<"medianPedHeight"<<std::endl;

//        std::vector<std::vector<float> > pedHeight;
        std::vector<float>  medianPedHeight;

        for (int i =0; i<imgHeight; i++){
            std::vector<float> heightArray;
            for (size_t kk = 0; kk < detList.size(); kk++){
                int bottomRow = static_cast<int>(detList.at(kk).bb_top + detList.at(kk).bb_height);
                if (bottomRow==i){//} && detList.at(kk).det_conf>2.5){
                    heightArray.push_back(detList.at(kk).bb_height);
                }
            }

//            std::cout<<"sorting heightarray"<<std::endl;

            if (!heightArray.empty())
                std::sort(heightArray.begin(),heightArray.end());
//            pedHeight.push_back(heightArray);

//            std::cout<<"median position calculation"<<std::endl;

            int medianPosition = -1;
//            if (heightArray.empty())
//                medianPosition = -1;
            if(heightArray.size()<5)
                medianPosition = -1;
            else if (heightArray.size()%2==0)
                medianPosition = static_cast<int>(heightArray.size() / 2);
            else if (heightArray.size()%2==1)
                medianPosition = static_cast<int>((heightArray.size()+1)/ 2);

//            std::cout<<"median array creation"<<std::endl;
            float medianHeight = 0.0;
            if (medianPosition!=-1)
                medianHeight = heightArray.at(medianPosition);

            medianPedHeight.push_back(medianHeight);
        }

//        for (int i =0; i<pedHeight.size(); i++){
//            std::cout<<std::endl;
//            for (size_t kk = 0; kk < pedHeight.at(i).size(); kk++){
//                std::cout<<pedHeight.at(i).at(kk)<<", ";
//            }
//        }

//        std::cout<<"invalidation big detections"<<std::endl;

//        for (int i =0; i<medianPedHeight.size(); i++){
//            std::cout<<i<<", "<<medianPedHeight.at(i)<<std::endl;
//        }

        float prvValue = 0.0;
        for (int i = static_cast<int>(medianPedHeight.size() - 1); i > -1; i--){
            float value = medianPedHeight.at(i);
            if (value == 0.0)
                medianPedHeight.at(i) = prvValue;
            else if (prvValue>0.0 && value > prvValue)
                medianPedHeight.at(i) = prvValue;
            else if (prvValue>0.0 && value<prvValue)
                prvValue = value;
            else if (value>0.0)
                prvValue = value;
//            std::cout<<i<<", "<<medianPedHeight.at(i)<<std::endl;
        }

        for (size_t i = 0; i < detList.size(); i++) {
            int bottomRow = static_cast<int>(detList.at(i).bb_top + detList.at(i).bb_height);
            float medianHeightRow = static_cast<float>(1.1 * medianPedHeight.at(
                    static_cast<unsigned long>(bottomRow - 1)));
            if (medianHeightRow>0.0 && detList.at(i).bb_height>medianHeightRow){
                detList.at(i).validDet = false;
//                std::cout<<"invalidated: "<<i<<std::endl;
            }
        }

//        for (int i =0; i<medianPedHeight.size(); i++){
//            std::cout<<i<<", "<<medianPedHeight.at(i)<<std::endl;
//        }
//
//        std::cout<<"removing detections within detections"<<std::endl;

//        for (size_t kk = 0; kk < detList.size(); kk++) {
//
//            if(detList.at(kk).validDet){
//                auto xmin = static_cast<int>(detList.at(kk).bb_left);
//                auto ymin = static_cast<int>(detList.at(kk).bb_top);
//                auto width = static_cast<int>(detList.at(kk).bb_width);
//                auto height = static_cast<int>(detList.at(kk).bb_height);
//                cv::Rect_<int> bbox = cv::Rect_<int>(xmin, ymin, width, height);
//
//                for (size_t ii = 0; ii < detList.size(); ii++) {
//                    if (ii == kk)
//                        continue;
//
//                    auto xminii = static_cast<int>(detList.at(ii).bb_left);
//                    auto yminii = static_cast<int>(detList.at(ii).bb_top);
//                    auto widthii = static_cast<int>(detList.at(ii).bb_width);
//                    auto heightii = static_cast<int>(detList.at(ii).bb_height);
//                    cv::Rect_<int> bboxii = cv::Rect_<int>(xminii, yminii, widthii, heightii);
//
//                    cv::Rect intersection = bbox & bboxii;
//
//                    if (intersection.area() == bbox.area() ) {
//                        if(bbox.area()<bboxii.area()){
//                            detList.at(kk).validDet = false;
//                        } else {
//                            detList.at(ii).validDet = false;
//                        }
//                    }
//
//                }
//            }
//
//        }

    }

    void YoloObjectDetector::readCurrFrameDets(float det_conf_th){

        detListCurrFrame.clear();

        for (size_t i = 0; i < detList.size(); i++) {
            if(detList.at(i).frameNum==(frame_num+1)){//} && detList.at(i).det_conf>det_conf_th){
                float width_del = detList.at(i).bb_left+detList.at(i).bb_width-float(frameWidth_);
                float height_del = detList.at(i).bb_top+detList.at(i).bb_height-float(frameHeight_);

                if (width_del>0.0)
                    detList.at(i).bb_width -= width_del;

                if (height_del>0.0)
                    detList.at(i).bb_height -= height_del;

                detListCurrFrame.push_back(detList.at(i));

//                if (detList.at(i).bb_left>=0.0 && detList.at(i).bb_left<=float(frameWidth_) && detList.at(i).bb_top>=0.0 && detList.at(i).bb_top<=float(frameHeight_))
////                    if(detList.at(i).bb_width>0.0 && detList.at(i).bb_height>0.0)
//                    if((detList.at(i).bb_left+detList.at(i).bb_width)<=float(frameWidth_) && (detList.at(i).bb_top+detList.at(i).bb_height)<=float(frameHeight_))
//                        detListCurrFrame.push_back(detList.at(i));
            }
        }
    }

    void YoloObjectDetector::processImage() {

        std::size_t found = img_name.find("DPM");
        if (found==std::string::npos) {
            readCurrFrameDets(0.3);
        } else {
            readCurrFrameDets(0.0);
        }

//        occlutionMap();

        buffIndex_ = (buffIndex_ + 2) % 3;
        std::chrono::high_resolution_clock::time_point t3, t4, t5, t6;
        t3 = std::chrono::high_resolution_clock::now();
//        std::cout << "fetchInThread" << std::endl;
//        fetchInThread();
//        std::cout << "fetchInThread finished" << std::endl;
//        detectInThread();
//        std::cout << "detectInThread finished" << std::endl;
        t4 = std::chrono::high_resolution_clock::now();
//        displayInThread();
//        std::cout << "displayInThread finished" << std::endl;
        t5 = std::chrono::high_resolution_clock::now();
//        publishInThread();
//        dataAssociation();//mot
//        std::cout << "dataAssociation finished" << std::endl;
        t6 = std::chrono::high_resolution_clock::now();
        long durationTotDet, durationTotPub;
        durationTotDet = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3 ).count();
        durationTotPub = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5 ).count();
        totTimeDet += durationTotDet;
        totTimePub += durationTotPub;

        frame_num++;
    }

    void YoloObjectDetector::getFrameRate(){
        std::cout<<"Time per frame (for tracking) : "<<totTime/totFrames<<" ms, Tot frames: "<<totFrames<<", Tot time tracking: "<<totTime<<"ms "
        <<"Tot time det: "<<totTimeDet<<"ms " <<"Tot time pub: "<<totTimePub<<"ms"<<std::endl;
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
  detection *dets = get_network_boxes(net, buff_.w, buff_.h, demoThresh_, demoHier_, 0, 1, nboxes, 0);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  globalframe++;
  running_ = 1;
  float nms = .45;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_.data;
  network_predict(*net_, X);

//  int size_of_array = sizeof(ss)/sizeof(ss[0]);
//
//
//  for (int i=0; i < size_of_array; i++){
//      printf("%lf\n", ss[i]);
//  }
//  printf("output array size: %d\n\n", size_of_array);

  image display = buff_;//[(buffIndex_ + 2) % 3];
  int nboxes = 0;

  detection *dets = get_network_boxes(net_, display.w, display.h, demoThresh_, demoHier_, nullptr, 1, &nboxes, 1);

  if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

  draw_detections_v3(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, l.classes, 0); // 1 means output classes, here I ignore

//  if ( (enableConsoleOutput_)&&(globalframe%20==1) ) {
//    printf("\033[2J");
//    printf("\033[1;1H");
//    printf("\nFPS:%.1f\n",fps_);
//    printf("Objects:\n\n");
//      printf("FPS:%.1f\n", fps_);
//  }

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
  demoIndex_ = (demoIndex_ + 2) % demoFrame_;
  running_ = 0;
  return nullptr;
}

void *YoloObjectDetector::fetchInThread()
{
  IplImage* ROS_img = getIplImage();
  ipl_into_image(ROS_img, buff_);
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_ = actionId_;
  }
  if(!use_grey)
    rgbgr_image(buff_);//[buffIndex_]);

  letterbox_image_into(buff_, net_->w, net_->h, buffLetter_);

  buff_cv_l_ = left_rectified.clone();
//  buff_cv_r_[(buffIndex_)] = right_rectified.clone();
  buff_cv_rgb_ = camImageCopy_.clone();

//  if(counter > 2) {
//      disparityFrame[(buffIndex_ + 2) % 3] = getDepth(buff_cv_l_[(buffIndex_ + 2) % 3], buff_cv_r_[(buffIndex_ + 2) % 3]);
//  }

  counter ++;
  return nullptr;
}

void *YoloObjectDetector::displayInThread()
{
  show_image_cv(buff_, "YOLO V3", ipl_);
//    visualize_network(net_);
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
  buff_ = ipl_to_image(ROS_img);
//  buff_[1] = copy_image(buff_[0]);
//  buff_[2] = copy_image(buff_[0]);
  buffLetter_ = letterbox_image(buff_, net_->w, net_->h);
//  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
//  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
//    disparityFrame[0] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
//    disparityFrame[1] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
//    disparityFrame[2] = cv::Mat(Height, Width, CV_8UC1, cv::Scalar(0));
    buff_cv_l_ = camImageCopy_.clone();
//    buff_cv_l_[1] = camImageCopy_.clone();
//    buff_cv_l_[2] = camImageCopy_.clone();
    buff_cv_rgb_ = camImageCopy_.clone();
//    buff_cv_rgb_[1] = camImageCopy_.clone();
//    buff_cv_rgb_[2] = camImageCopy_.clone();
  ipl_ = cvCreateImage(cvSize(buff_.w, buff_.h), IPL_DEPTH_8U, buff_.c);

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

void YoloObjectDetector::dataAssociation(){

    layer cost_layer = net_->layers[2];//net_->n-2];
    int l_w = cost_layer.out_w;
    int l_h = cost_layer.out_h;
    int l_c = cost_layer.out_c;
    int fil_r = (int)l_c/2;
    int fil_c = l_c-fil_r;



//    for(int i = net_->n-1; i > 0; --i) {
//        std::cout<<net_->layers[i].type<<", ";
////        if(net.layers[i].type != COST) break;
//    }
//    std::cout<<std::endl;

//    std::cout<<"layer cost"<<std::endl;

    cv::Mat layer_cost = cv::Mat::zeros(l_h, l_w, CV_32FC(l_c));
    cv::Mat layer_cost_vis = cv::Mat::zeros(l_h, l_w, CV_8UC(l_c));

//    std::cout<<cost_layer.cost<<std::endl;//", "<<cost_layer.cost[0]<<cost_layer.cost[1]<<std::endl;
    cuda_pull_array(cost_layer.output_gpu, cost_layer.output, cost_layer.batch*cost_layer.outputs);
    for(int kk = 0; kk < l_c; ++kk){
        for(int ii = 0; ii < l_h; ++ii){
            for(int jj = 0; jj < l_w; ++jj){
//                layer_cost.at<cv::Vec<float, 21> >(ii,jj)[kk]=cost_layer.output[jj+ii*l_w+l_w*l_h*kk];//kitti - 2 class
                layer_cost.at<cv::Vec<float, 32> >(ii,jj)[kk]=cost_layer.output[jj+ii*l_w+l_w*l_h*kk];//mot - 1 class
//                memcpy();
//                std::cout<<cost_layer.output[ii+l_w*l_h*kk]<<"; ";
            }
        }
    }

//    cuda_pull_array(cost_layer.output_gpu, cost_layer.output, cost_layer.batch*cost_layer.outputs);
//    if (l_w >= 0 && l_h >= 1 && l_c >= 3) {
//        int jj;
//        for (jj = 0; jj < l_c; ++jj) {
////            image img = make_image(l_w, l_h, 3);
//            cv::Mat vis = cv::Mat::zeros(l_h, l_w, CV_8UC3);
//            memcpy(vis.data, cost_layer.output+ l_w*l_h*jj, l_w*l_h * 1 * sizeof(float));
//            char buff[256];
//            sprintf(buff, "slice-%d", jj);
//            cv::imshow(buff,vis);
////            show_image(img, buff);
//        }
////        cvWaitKey(0); // wait press-key in console
////        cvDestroyAllWindows();
//    }

//    double min, max;
//    cv::minMaxLoc(layer_cost, &min, &max);
//    layer_cost.convertTo(layer_cost_vis,CV_8U,255.0/(max-min),-255.0*min/(max-min));
//    cv::resize(layer_cost_vis, layer_cost_vis, camImageCopy_.size(), 0, 0, CV_INTER_CUBIC);


//    std::cout<<l_w<<", "<<l_h<<", "<<l_c<<std::endl;

//    cv::Mat layer_output = cv::Mat::zeros(l_h, l_w, CV_8UC3);

//    cv::Mat roi_vis[l_c];
//    split(layer_cost_vis,roi_vis);
//    for(int ch = 0; ch < l_c; ++ch){
//        std::ostringstream str;
//        str << "Layer "<<ch;
//        cv::imshow(str.str(),roi_vis[ch]);
//    }

//    std::cout<<"blob list"<<std::endl;

    for (size_t kk = 0; kk < detListCurrFrame.size(); kk++) {

        auto xmin = static_cast<int>(detListCurrFrame.at(kk).bb_left);
        auto ymin = static_cast<int>(detListCurrFrame.at(kk).bb_top);
        auto width = static_cast<int>(detListCurrFrame.at(kk).bb_width);
        auto height = static_cast<int>(detListCurrFrame.at(kk).bb_height);

        auto xmin_net = xmin*l_w/frameWidth_;
        auto ymin_net = ymin*l_h/frameHeight_;
        auto width_net = width*l_w/frameWidth_;
        auto height_net = height*l_h/frameHeight_;

        cv::Rect_<int> rect_layer = cv::Rect_<int>(xmin_net, ymin_net, width_net, height_net);
//        cv::rectangle(layer_output, rect_layer, cv::Scalar( 0, 0, 255 ), 1);
        if(rect_layer.size().area()>0){
            cv::Mat cost_roi;
            layer_cost(rect_layer).copyTo(cost_roi);

            Blob outputObs(cv::Rect(xmin, ymin, width, height));
            outputObs.category = "Pedestrian";//classLabels_[i]; //mot
            outputObs.probability = detListCurrFrame.at(kk).det_conf;
            outputObs.feature_cost = cost_roi;
            currentFrameBlobs.push_back(outputObs);
        }

    }

//    std::cout<<"tracking"<<std::endl;

    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
    Tracking();
    t2 = std::chrono::high_resolution_clock::now();
    long durationTot;
    durationTot = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1 ).count();
    totTime += durationTot;
    totFrames++;
//    std::cout<<"CreateMsg"<<std::endl;
    CreateMsg();

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

    cv::Mat rgbOutput = buff_cv_rgb_.clone();
//    cv::Mat greyOutput = buff_cv_l_[(buffIndex_ + 2) % 3].clone();
//    cv::Mat disparityMap = disparityFrame[(buffIndex_ + 2) % 3].clone();
//    cv::Mat hsv;
//    cv::cvtColor(rgbOutput, hsv, CV_BGR2HSV);

    std_msgs::Int8 msg;
    msg.data = static_cast<signed char>(num);
    objectPublisher_.publish(msg);

//    float *netowrk_output = get_network_output(*net_);
    layer cost_layer = net_->layers[net_->n-2];
    int l_w = cost_layer.out_w;
    int l_h = cost_layer.out_h;
    int l_c = cost_layer.out_c;
    int fil_r = (int)l_c/2;
    int fil_c = l_c-fil_r;

    cv::Mat layer_cost = cv::Mat::zeros(l_h, l_w, CV_32FC(l_c));
//    cv::Mat layer_cost_vis = cv::Mat::zeros(l_h, l_w, CV_8UC(l_c));

//    std::cout<<layer_cost.size()<<std::endl;
      cuda_pull_array(cost_layer.output_gpu, cost_layer.output, cost_layer.batch*cost_layer.outputs);
    for(int kk = 0; kk < l_c; ++kk){
        for(int ii = 0; ii < l_h; ++ii){
            for(int jj = 0; jj < l_w; ++jj){
//                layer_cost.at<cv::Vec<float, 21> >(ii,jj)[kk]=cost_layer.output[jj+ii*l_w+l_w*l_h*kk];//kitti - 2 class
                layer_cost.at<cv::Vec<float, 18> >(ii,jj)[kk]=cost_layer.output[jj+ii*l_w+l_w*l_h*kk];//mot - 1 class
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

//                  if ((counter>2) ) {

                      cv::Rect_<int> rect_layer = cv::Rect_<int>(xmin_net, ymin_net, xmax_net - xmin_net, ymax_net - ymin_net);
                      cv::rectangle(layer_output, rect_layer, cv::Scalar( 0, 0, 255 ), 1);
                      cv::Mat cost_roi;
                      layer_cost(rect_layer).copyTo(cost_roi);
//                      std::cout<<cost_roi<<std::endl;
//                      cv::Mat roi_vis[l_c];
//                      split(cost_vis_roi,roi_vis);
//                      for(int ch = 0; ch < l_c; ++ch){
//                          std::ostringstream str;
//                          str << "Layer "<<ch;
//                          cv::imshow(str.str(),roi_vis[ch]);
//                      }

//                      cv::Mat cost = cv::Mat::zeros((ymax_net-ymin_net)*fil_r, (xmax_net-xmin_net)*fil_c, CV_32FC1);
//                    std::cout<<std::endl<<"New Object"<<std::endl;
//                      for(int jj = 0; jj < l_c; ++jj){
//                          for(int ii = xmin_net*ymin_net; ii < xmax_net*ymax_net; ++ii){
//                              std::cout<<cost_layer.output[ii+l_w*l_h*jj]<<"; ";
////                            printf("%lf\n", l.output[ii+l.out_w*l.out_h*(l.out_c-1)]);
//                          }
//                      }


//                      cv::Rect_<int> rect = cv::Rect_<int>(xmin, ymin, xmax - xmin, ymax - ymin);
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
//                      outputObs.category = "Pedestrian";//classLabels_[i]; //mot
//                      std::cout<<outputObs.category<<", ";
                      outputObs.probability = rosBoxes_[i][j].prob;
                      outputObs.feature_cost = cost_roi;
//                      outputObs.kpDesc = desc;
//                      outputObs.nHist = hist;
//                      for (int i=0; i<kpts.size(); i++){
//                          float kptDis = (float) disparityMap.at<uchar>((int)kpts[i].pt.x,(int)kpts[i].pt.y);
//                          outputObs.keyPoints.push_back(cv::Point3f(kpts[i].pt.x,kpts[i].pt.y,kptDis));
//                      }

                      if (outputObs.category == "person") {
                          currentFrameBlobs.push_back(outputObs);
                      }



//                  } else {
//                      ROS_WARN("*********************************************************");
//                  }

              }
          }
      }
//      cv::imshow("layer_output",layer_output);

  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
//    std::cout << "************************************************num 0" << std::endl;
  }

//  std::cout<<std::endl;
    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
//  Tracking();
    t2 = std::chrono::high_resolution_clock::now();
    long durationTot;
    durationTot = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1 ).count();
    totTime += durationTot;
    totFrames++;
//    std::cout<<"Time per frame (for tracking) : "<<durationTot<<" ms"<<std::endl;
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

    void YoloObjectDetector::matchCurrentDetectionsToExisting(){

    }

    void YoloObjectDetector::matchCurrentFrameBlobsToExistingBlobs() {

        int simHeight = (int)blobs.size();
        int simWidth = (int)currentFrameBlobs.size();
        // cv::Mat simSize(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        //cv::Mat simApp(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        //cv::Mat simKeyPts(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        // cv::Mat simPos(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        cv::Mat similarity(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        cv::Mat disSimilarity(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
        cv::Mat simGeometry(simHeight, simWidth, CV_64FC1, cv::Scalar(0));
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

//                        if(blob.intNumOfConsecutiveFramesWithoutAMatch==0){
                            int xmin = blob.predictedNextPosition.x - blob.predictedWidth/2;
                            int ymin = blob.predictedNextPosition.y - blob.predictedHeight/2;
                            cv::Rect predictedBB(xmin, ymin, blob.predictedWidth, blob.predictedHeight);

                            cv::Rect intersection = predictedBB & currBlob.boundingRects.back();
                            cv::Rect unio = predictedBB | currBlob.boundingRects.back();
                            simGeometry.at<double>(r,c) = (double)intersection.area()/unio.area();
//                            std::cout<<simGeometry.at<double>(r,c)<<", "<<intersection.area()<<", "<<unio.area()<<", "<<predictedBB.area()<<", "<<currBlob.boundingRects.back().area()<<std::endl;
//                        } else {
//                            simGeometry.at<double>(r,c) = -1.0;
//                        }

                        cv::Mat blbCosts[ch], reSizedBlbCosts[ch];
                        split(blob.feature_cost,blbCosts);
//                        blob.feature_cost.copyTo(blbCosts);
//                        std::cout<<"inside the cost calculation. ch: "<<ch<<", blob ch: "<<blob.feature_cost.channels()<<std::endl;
//                        std::cout<<"blob area: "<<blob.boundingRects.back().area()<<", currBlob area: "<<currBlob.boundingRects.back().area()<<std::endl;
                        for(int ii = 0; ii < ch; ++ii){
                            if(blob.boundingRects.back().area()>currBlob.boundingRects.back().area()){
                                cv::resize(blbCosts[ii], reSizedBlbCosts[ii], currBlbCosts[ii].size(), 0, 0, CV_INTER_AREA);
                            } else {
                                cv::resize(blbCosts[ii], reSizedBlbCosts[ii], currBlbCosts[ii].size(), 0, 0, CV_INTER_CUBIC);// CV_INTER_LANCZOS4);
                            }
//                            std::cout<<ii<<std::endl<<currBlbCosts[ii]<<std::endl;
//                            std::cout<<reSizedBlbCosts[ii]<<std::endl;
                            disSimilarity.at<double>(r,c) += cv::norm(currBlbCosts[ii],reSizedBlbCosts[ii]);
                        }
//                        std::cout<<currBlbCosts[20]<<std::endl;
//                        std::cout<<r<<", "<<c<<": "<<disSimilarity.at<double>(r,c)<<"; ";
                    } else {
                        disSimilarity.at<double>(r,c) = -1.0;
//                        simGeometry.at<double>(r,c) = -1.0;
//                        std::cout<<currBlob.category<<", "<<blob.category<<std::endl;
                    }
                } else {
                    disSimilarity.at<double>(r,c) = -1.0;
//                    simGeometry.at<double>(r,c) = -1.0;
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
                        // double appSim = cv::compareHist(currBlob.nHist, blob.nHist, cv::CV_COMP_CORREL);
                        //double appSim = 1.0-cv::compareHist(currBlob.nHist, blob.nHist, cv::HISTCMP_HELLINGER ); //may not yeild accurate results
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

//        double maxFeature=0.0;//maxGeo=0.0,
//        double  minFeature=10000000.0; //minGeo=10000000.0,
        cv::Mat maxFeatureDis;
        cv::reduce(disSimilarity,maxFeatureDis,0,CV_REDUCE_MAX);

        for (int c=0; c<simWidth; c++) {
            for (int r = 0; r < simHeight; r++) {
                if (disSimilarity.at<double>(r,c)!=-1.0){
                    disSimilarity.at<double>(r,c) /= maxFeatureDis.at<double>(0,c);
                }
            }
        }

//        for (int c=0; c<simWidth; c++){
//            for (int r=0; r<simHeight; r++){
//                double disSimilarityVal = disSimilarity.at<double>(r,c);
//                if (disSimilarityVal>=0.0){
//                    if (maxFeature < disSimilarityVal) {
//                        maxFeature = disSimilarityVal;
//                    }
//                    if (minFeature > disSimilarityVal) {
//                        minFeature = disSimilarityVal;
//                    }
//                }
////                double simGeometryVal = simGeometry.at<double>(r,c);
////                if (simGeometryVal>=0.0){
////                    if (maxGeo < simGeometryVal) {
////                        maxGeo = simGeometryVal;
////                    }
////                    if (minGeo > simGeometryVal) {
////                        minGeo = simGeometryVal;
////                    }
////                }
//            }
//        }

//        std::cout<<maxGeo

        for (int c=0; c<simWidth; c++) {
            for (int r = 0; r < simHeight; r++) {
                double simGeometryVal = simGeometry.at<double>(r,c);
                double disSimilarityVal = disSimilarity.at<double>(r,c);
                if (disSimilarity.at<double>(r,c)==-1.0){
                    similarity.at<double>(r,c)=simGeometryVal/2.0;
//                    if (simGeometryVal>0.0){
//                        std::cout<<"simGeometryVal: "<<simGeometryVal<<std::endl;
//                    }
                } else{
                    similarity.at<double>(r,c)=(simGeometryVal+(1.0-disSimilarityVal))/2.0;
//                    similarity.at<double>(r,c)=simGeometryVal;
                }
//                if (maxGeo==minGeo){
//                    if (maxFeature==minFeature){
//                        similarity.at<double>(r,c) =0.0;
//                    } else {
//                        similarity.at<double>(r,c)=(1.0-((disSimilarityVal-minFeature)/(maxFeature-minFeature)))/2;
//                    }
//                } else {
//                    if (maxFeature==minFeature){
//                        similarity.at<double>(r,c)=simGeometryVal/2.0;//((simGeometryVal-minGeo)/(maxGeo-minGeo))/2;
//                    } else {
//                        if (disSimilarity.at<double>(r,c)==-1.0){
//                            similarity.at<double>(r,c)=simGeometryVal/2.0;
//                        } else{
//                            similarity.at<double>(r,c)=(simGeometryVal+(1.0-((disSimilarityVal-minFeature)/(maxFeature-minFeature))))/2;
//                        }
//                    }
//                }
//                if (simGeometryVal>0){
//                    similarity.at<double>(r,c)=(((simGeometryVal-minGeo)/(maxGeo-minGeo))
//                            +(1.0-((disSimilarityVal-minFeature)/(maxFeature-minFeature))))/2;
//                } else if (disSimilarityVal>0){
//                    similarity.at<double>(r,c)= 1.0-((disSimilarityVal-minFeature)/(maxFeature-minFeature));
//                }
            }
        }

        double min, max;
        cv::minMaxLoc(similarity, &min, &max);
        double thForSimilarityVal = std::max(0.25,max*0.5);
        double thForHungarianCost = std::max(0.25,max*0.5);

        std::vector< std::vector<double> > costMatrix;

        for (int r = 0; r < simHeight; r++)  {
            std::vector<double> costForEachTrack;
            for (int c=0; c<simWidth; c++) {
                costForEachTrack.push_back(1.0-similarity.at<double>(r,c));
//                std::cout<<costForEachTrack.at(c)<<", ";//<<std::endl;
            }
            costMatrix.push_back(costForEachTrack);
//            std::cout<<std::endl;
        }

//        std::cout<<"costMatrix: "<<costMatrix.size()<<", "<<costMatrix[0].size()<<"; simHeight: "<<simHeight<<", simWidth: "<<simWidth<<std::endl;

        HungarianAlgorithm HungAlgo;
        std::vector<int> assignment;

//        std::cout<<"starting hungarianCost: "<<std::endl;
        double hungarianCost = HungAlgo.Solve(costMatrix, assignment);
//        std::cout<<"hungarianCost: "<<hungarianCost<<std::endl;

        for (int trackID = 0; trackID < costMatrix.size(); trackID++){
//            std::cout << trackID << "," << assignment[trackID] << "\t";
            if (assignment[trackID]>-1) {
                Blob &currentFrameBlob = currentFrameBlobs.at(static_cast<unsigned long>(assignment[trackID]));
                double similarityVal = similarity.at<double>(trackID,assignment[trackID]);
                if ( (!blobs[trackID].blnAlreadyTrackedInThisFrame) && similarityVal>thForHungarianCost ) { //(minDisSimilarity < max)
                    currentFrameBlob.blnAlreadyTrackedInThisFrame = true;
                    addBlobToExistingBlobs(currentFrameBlob, blobs, trackID);
                } else {
                    addNewBlob(currentFrameBlob, blobs);
                }
            }
        }
//        std::cout<<std::endl;

        for (int c=0; c<simWidth; c++){
            Blob &currentFrameBlob = currentFrameBlobs.at(c);
            if(!currentFrameBlob.blnAlreadyTrackedInThisFrame)
                addNewBlob(currentFrameBlob, blobs);
        }


//        std::cout<<"similarity "<<frame_num<<std::endl;
//        for (int c=0; c<simWidth; c++){
//            for (int r=0; r<simHeight; r++){
//                std::cout<<similarity.at<double>(r,c)<<"; ";
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//        std::cout<<"disSimilarity "<<std::endl;
//        for (int c=0; c<simWidth; c++){
//            for (int r=0; r<simHeight; r++){
//                std::cout<<disSimilarity.at<double>(r,c)<<"; ";
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//        std::cout<<"simGeometry "<<std::endl;
//        for (int c=0; c<simWidth; c++){
//            for (int r=0; r<simHeight; r++){
//                std::cout<<simGeometry.at<double>(r,c)<<"; ";
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;



        // std::cout<<"Debug matchCurrentFrameBlobsToExistingBlobs 3"<<std::endl;

//        for (int c=0; c<simWidth; c++){
////            double minDisSimilarity = 10000000.0;
//            double maxSimilarity = 0.0;
//            int indexMinDisSimilarity = -1;
//            Blob &currentFrameBlob = currentFrameBlobs.at(c);
//            for (int r=0; r<simHeight; r++){
////                double disSimilarityVal = disSimilarity.at<double>(r,c);
//                double similarityVal = similarity.at<double>(r,c);
//                if (similarityVal>0.0){
//                    if (maxSimilarity < similarityVal) {
//                        maxSimilarity = similarityVal;
//                        indexMinDisSimilarity = r;
//                    }
//                }
//            }
//            if (indexMinDisSimilarity != -1) {
//                if ( (!blobs[indexMinDisSimilarity].blnAlreadyTrackedInThisFrame) && maxSimilarity>thForSimilarityVal ) { //(minDisSimilarity < max)
//                    addBlobToExistingBlobs(currentFrameBlob, blobs, indexMinDisSimilarity);
//                } else {
//                    addNewBlob(currentFrameBlob, blobs);
//                }
//            } else {
//                addNewBlob(currentFrameBlob, blobs);
//            }
//        }

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

//        existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
        existingBlobs[intIndex].boundingRects.push_back(currentFrameBlob.boundingRects.back());
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
            if (!currentFrameBlobs.empty()){
                blnFirstFrame = false;
                for (auto &currentFrameBlob : currentFrameBlobs)
                    blobs.push_back(currentFrameBlob);
            }
        } else {
            for (auto &existingBlob : blobs) {
                existingBlob.blnCurrentMatchFoundOrNewBlob = false;
                existingBlob.blnAlreadyTrackedInThisFrame = false;
                existingBlob.predictNextPosition();
                existingBlob.predictWidthHeight();
            }

//            std::cout<<"blob prediction finished"<<std::endl;

            if (!currentFrameBlobs.empty()){
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

//            std::cout<<"blob association finished"<<std::endl;

        }

        currentFrameBlobs.clear();
    }

    void YoloObjectDetector::CreateMsg(){

        // cv::Mat output1 = disparityFrame[(buffIndex_ + 1) % 3].clone();
        cv::Mat output = buff_cv_rgb_.clone();//camImageCopy_.clone();

        // for (long int i = 0; i < currentFrameBlobs.size(); i++) {
        //   cv::rectangle(output, currentFrameBlobs[i].currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
        // }
        // currentFrameBlobs.clear();

        /*Tracking Result*/
        for (long int i = 0; i < blobs.size(); i++) {
//            if (blobs[i].blnStillBeingTracked == true) {
            if (blobs[i].blnCurrentMatchFoundOrNewBlob) {
//                cv::rectangle(output, blobs[i].currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
                cv::rectangle(output, blobs[i].boundingRects.back(), cv::Scalar( 0, 0, 255 ), 2);
//                int xmin = blobs[i].predictedNextPosition.x - blobs[i].predictedWidth/2;
//                int ymin = blobs[i].predictedNextPosition.y - blobs[i].predictedHeight/2;
//                cv::Rect predictedBB(xmin, ymin, blobs[i].predictedWidth, blobs[i].predictedHeight);
//                cv::rectangle(output, predictedBB, cv::Scalar( 0, 255, 0 ), 2);
                // cv::rectangle(output1, blobs[i].currentBoundingRect, cv::Scalar( 255, 255, 255 ), 2);
//                for(int j=0; j<blobs[i].keyPoints.size();j++){
//                     cv::circle(output,cv::Point((int)blobs[i].keyPoints[j].x,(int)blobs[i].keyPoints[j].y),2,cv::Scalar( 0, 255, 0 ));
////                    std::cout<<blobs[i].keyPoints[j]<<"- "<<cv::Point((int)blobs[i].keyPoints[j].x,(int)blobs[i].keyPoints[j].y)<<";;; ";
//                    // output.at<cv::Vec3b>((int)blobs[i].keyPoints[j].x, (int)blobs[i].keyPoints[j].y)[2]=255;//cv::Vec3b(0,0,255);
//                }
                std::ostringstream str;
                // str << blobs[i].position_3d[2] <<"m, ID="<<i<<"; "<<blobs[i].disparity;
                str << "ID="<<i<<"; ";
                cv::putText(output, str.str(), blobs[i].centerPositions.back(), CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(0,255,0));
                // cv::putText(output1, str.str(), blobs[i].centerPositions.back(), CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 250, 255));

                //mot
                file << frame_num+1 <<", "<< i+1 << ", " << blobs[i].boundingRects.back().x << ", " << blobs[i].boundingRects.back().y << ", "
                << blobs[i].boundingRects.back().width << ", " << blobs[i].boundingRects.back().height << ", "
                << -1 << ", " << -1 << ", " << -1 << ", " << -1 << std::endl;


                //kitti
//                file << frame_num <<" "<< i << " " << blobs[i].category<< " "<< -1 << " " << -1 <<" " << -10 << " "
//                     << blobs[i].boundingRects.back().x << " " << blobs[i].boundingRects.back().y << " "
//                     << blobs[i].boundingRects.back().x + blobs[i].boundingRects.back().width << " "
//                     << blobs[i].boundingRects.back().y + blobs[i].boundingRects.back().height << " "
//                     << -1 << " " << -1 << " " << -1 << " " << "-1000 -1000 -1000" << " " << -10 << " " << blobs[i].probability << std::endl;

            }
        }


//        for (long int i = 0; i < currentFrameBlobs.size(); i++) {
//
//            cv::rectangle(output, currentFrameBlobs[i].boundingRects.back(), cv::Scalar( 0, 0, 255 ), 2);
//            //mot det
//            file << frame_num+1 <<", "<< -1 << ", " << currentFrameBlobs[i].boundingRects.back().x << ", " << currentFrameBlobs[i].boundingRects.back().y << ", "
//                 << currentFrameBlobs[i].boundingRects.back().width << ", " << currentFrameBlobs[i].boundingRects.back().height << ", "
//                 << currentFrameBlobs[i].probability << std::endl;
//        }



        cv::imshow("debug", output);
        // cv::imshow("disparity", output1);
        cv::waitKey(1);
//        currentFrameBlobs.clear();

         sprintf(im, "f%03d.png", frame_num);
         img_name = ros::package::getPath("cubicle_detect") + "/seq_1/" + im;
        // std::cout<<img_name<<std::endl;
//         cv::imwrite(img_name, output);


        frame_num ++;

    }


} /* namespace darknet_ros*/
