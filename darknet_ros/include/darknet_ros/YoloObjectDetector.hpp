/*
 * YoloObjectDetector.h
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#pragma once

// c++
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <map>

#include <fstream>
#include "../../../libSGM/include/libsgm.h"
//#include "libsgm.h"

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_geometry/stereo_camera_model.h>
#include <image_geometry/pinhole_camera_model.h>
// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

// darknet_ros_msgs
#include "darknet_ros/Blob.h"
#include "darknet_ros/Hungarian.h"
#include "darknet_ros/Obstacles.h"
//#include "../../src/track_kalman.hpp"
#include "utils/timing.h"
//#include "utils/hog.h"
// Obstacle ros msgs
#include <obstacle_msgs/MapInfo.h>
#include <obstacle_msgs/obs.h>
#include <obstacle_msgs/point3.h>
// Darknet.
#ifdef GPU
// Cuda
//#include "sgm/disparity_method.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "blas.h"
#include "box.h"
#include "image.h"
#include "darknet_ros/image_interface.h"
#include <sys/time.h>
}

//extern "C" void ipl_into_image(IplImage* src, image im);
//extern "C" image ipl_to_image(IplImage* src);
extern "C" image mat_to_image(cv::Mat mat);
extern "C" cv::Mat image_to_mat(image img);
//extern "C" void show_image_cv(image p, const char *name, IplImage *disp);

namespace darknet_ros {

    typedef struct {
        float bb_left = 0.0, bb_top = 0.0, bb_right = 0.0, bb_bottom = 0.0, det_conf = 0.0;
        std::string objCLass;
//        bool validDet = true;
    } inputDetection;

class Detection;

//! Bounding box of the detected object.
typedef struct
{
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;

class YoloObjectDetector
{
 public:
  /*!
   * Constructor.
   */
  explicit YoloObjectDetector(ros::NodeHandle nh, ros::NodeHandle nh_p);

  /*!
   * Destructor.
   */
  ~YoloObjectDetector();

    /*!
  * Callback of camera.
  * @param[in] msg image pointer.
  */
  void cameraCallback(const sensor_msgs::ImageConstPtr &image1, const sensor_msgs::ImageConstPtr &image2); //,
//                      const sensor_msgs::CameraInfoConstPtr& left_info, const sensor_msgs::CameraInfoConstPtr& right_info);

  int globalframe, Scale;
  double stereo_baseline_, u0, v0, focal;

  /*!
  * Callback of camera.
  * @param[in] msg image pointer.
  */
  cv::Mat getDepth(cv::Mat &leftFrame, cv::Mat &rightFrame);

    /**
     * @brief compute stereo baseline, ROI's and FOV's from camera calibration messages.
     */
    void loadCameraCalibration( const sensor_msgs::CameraInfoConstPtr&left_info,
                                const sensor_msgs::CameraInfoConstPtr&right_info);

     /*!
      * Generate look up table to speed up depth generation.
      */
    void DefineLUTs();

    /*!
     * Initialize the ROS connections.
     */
    void init();

private:
   /*!
   * Reads Cuda infomation.
   * @return true if successful.
   */
    bool CudaInfo();

  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Publishes the detection image.
   * @return true if successful.
   */
  bool publishDetectionImage(const cv::Mat& detectionImage);

  cv::Mat occlutionMap(cv::Rect_<int> bbox, size_t kk, bool FNcheck);

  void calculateHistogram(Blob &currentDet, cv::Mat hsv, cv::Mat mask, int widthSeperate, int heightSeperate);

  void calculateLPBH(Blob &currentDet, cv::Mat rgb, int grid_x, int grid_y);

  void Tracking();

  void matchCurrentFrameBlobsToExistingBlobs();

  void trackingFNs();

  void addNewTracks();

  void updateUnmatchedTracks();

  void CreateMsg();

  void generateStaticObsDisparityMap();

  void Process();

  void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex, bool isDet);

  void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);

  inline int distanceBetweenPoints(cv::Point point1, cv::Point point2){

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);
    return intX*intX + intY*intY;
  };

  //! ROS node handle.
  ros::NodeHandle nodeHandle_, nodeHandle_pub;

  //! Class labels.
  int numClasses_;
  int compact_numClasses_;
  std::vector<std::string> classLabels_;
  std::vector<std::string> compact_classLabels_;

  //! ROS subscriber and publisher.
//  ros::Publisher objectPublisher_;
//  ros::Publisher boundingBoxesPublisher_;
  ros::Publisher obstaclePublisher_;
  ros::Publisher disparityPublisher_;
  ros::Publisher obs_disparityPublisher_;

  std::string pub_obs_frame_id, obs_disparityFrameId;

  //! Detected objects.
  std::vector<std::vector<RosBox_> > rosBoxes_;
  std::vector<int> rosBoxCounter_;
  obstacle_msgs::MapInfo obstacleBoxesResults_;

  //! Camera related parameters.
  int disp_size, Width, Height, rem_w, rem_h, Width_crp, Height_crp;
  bool is_even_crop;

  //! Lookup Table

  double **x3DPosition;
  double **y3DPosition;
  double *depth3D;
//  double xDirectionPosition[1280][129] ={{}};
//  double yDirectionPosition[844][129] ={{}};
//  double depthTable[129] = {};
  bool blnFirstFrame;
  bool notInitiated = true;
  bool filter_dynamic_;

  //! Publisher of the bounding box image.
  ros::Publisher detectionImagePublisher_;

  std::vector<Blob> currentFrameBlobs;
  std::vector<Blob> blobs;
  std::vector<inputDetection> detListCurrFrame;
  std::vector<unsigned long> matchedTrackID;
  std::vector<Blob> matchedFNs;
  std::vector<unsigned long> matchedFrmID;
  std::vector<unsigned long> matchedFrmIDTrackID;

  obstacle_msgs::obs obstacles;

//  Util::CPPTimer timer_yolo, timer_1, timer_2;

//  Util::HOGFeatureDescriptor* hog_descriptor;
  ObstaclesDetection ObstacleDetector;

  // Yolo running on thread.
  std::thread detect_thread;
//  std::thread fetch_thread;
  std::thread stereo_thread;

  // Darknet.
  char **demoNames_;
  char **compactDemoNames_;
  image **demoAlphabet_;
  int demoClasses_;

  network *net_;
  image buff_;//[3];
  image buffLetter_;//[3];
  cv::Mat buff_cv_l_;//[3];
  cv::Mat buff_cv_r_;//[3];
  cv::Mat disparityFrame;//[3];
//  int buffId_;//[3];
//  int buffIndex_ = 0;

//  IplImage * ipl_;
//  cv::Mat ipl_cv;
  double fps_ = 0;
  double stereo_fps_ = 0;
  double classi_fps_ = 0;
  double obs_fps_ = 0;
  float demoThresh_ = 0;
  float demoHier_ = .5;
//  int running_ = 0;

  int demoDelay_ = 0;
  int demoFrame_ = 3;
  float **predictions_;
  int demoIndex_ = 0;
  int demoDone_ = 0;
  float *lastAvg2_;
  float *lastAvg_;
  float *avg_;
  int demoTotal_ = 0;
  double demoTime_;

  RosBox_ *roiBoxes_;
  bool viewImage_;
  bool enableConsoleOutput_;
  bool enableEvaluation_;
  int waitKeyDelay_;
  int fullScreen_;
  char *demoPrefix_;

  ros::Time image_time_, prvImageTime;
  std_msgs::Header imageHeader_;
  cv::Mat camImageCopy_, origLeft, origRight, camImageOrig;
  cv::Mat left_rectified, right_rectified, output, ObsDisparity;
//  bool updateOutput = true;
  boost::shared_mutex mutexImageCallback_;

  bool imageStatus_ = false;
  boost::shared_mutex mutexImageStatus_;

  bool isNodeRunning_ = true;
  boost::shared_mutex mutexNodeStatus_;

  int actionId_;
  boost::shared_mutex mutexActionStatus_;

  // double getWallTime();

  int sizeNetwork(network *net);

//  void rememberNetwork(network *net);

//  detection *avgPredictions(network *net, int *nboxes);

  void *stereoInThread();

  void *detectInThread();

  void *fetchInThread();

  void *displayInThread();

  void *trackingInThread();

  void setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                    char **names, char **less_names, int classes,
                    int delay, char *prefix, int avg_frames, float hier, int w, int h,
                    int frames, int fullscreen);

  void yolo();

  IplImage* getIplImage();

  bool getImageStatus(void);

  bool isNodeRunning(void);

  void *trackInThread();

  bool use_grey;

//  cv::Rect left_roi_, right_roi_;
  Detection* mpDetection;
//  Tracker_optflow tracker_flow;
  std::thread* mpDepth_gen_run;
  int output_verbose;
  int intIndexOfLeastDistance;
  double dblLeastDistance;
  double hogLeastDistance;
//  std::vector<float> nullHog;


// Disparity

    std::mutex mMutexDepth;

    std::vector<double> depth;
    int min_disparity;
    bool enableStereo = true;
    bool enableClassification = true;
// Disparity
    stereo_msgs::DisparityImage disparity_info;
    stereo_msgs::DisparityImage disparity_obs;
    std::ofstream file;
    std::string file_name;
    std::string img_name;
    char s[20];
    char im[20];
    int frame_num, counter;
    int trackLife = 10;
    int cellsX = 4, cellsY =3;
    cv::Size bboxResize;

};

} /* namespace darknet_ros*/
