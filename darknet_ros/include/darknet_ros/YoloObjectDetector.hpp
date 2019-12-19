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

/**
 * @namespace darknet_ros YoloObjectDetector.hpp
 */
namespace darknet_ros {

    /**
     * @brief This is defined to contain each of object detections output by Yolo classification network.
     * This is used in object tracking where some of the objects may be invalidated if found to be any duplicates.
     */
    typedef struct {
        float bb_left = 0.0, bb_top = 0.0, bb_right = 0.0, bb_bottom = 0.0, det_conf = 0.0;
        std::string objCLass;
        bool validDet = true;
    } inputDetection;

/**
 * @class Detection YoloObjectDetector.hpp
 * @brief Definition of object.
 */
class Detection;

/**
 * @brief Bounding box of the detected object by the Yolo classification network.
 */
typedef struct
{
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;


/**
 * @class YoloObjectDetector YoloObjectDetector.hpp
 * @brief The core class for multiple object classification, detection and tracking.
 */
class YoloObjectDetector {

    public:
        /**
        * @brief Constructor.
        */
        explicit YoloObjectDetector();

        /**
        * @brief Destructor. Free the reserved GPU memory.
        */
        ~YoloObjectDetector();

        /**
        * @brief Callback for camera images published in ROS. In this function the ROS image ROS messages
        * for left anf right view are stored in to OpenCV cv::Mat container and pad the images to be divisible by 4
        * to support stereo matching.
        * @param image1 Left image ROS message
        * @param image2  Right image ROS message
        */
        void cameraCallback(const sensor_msgs::ImageConstPtr &image1, const sensor_msgs::ImageConstPtr &image2); //,
//                      const sensor_msgs::CameraInfoConstPtr& left_info, const sensor_msgs::CameraInfoConstPtr& right_info);

        /**
         * @brief This function returns the disparity image calculated using SGBM (Semi Global Block Matching)
         * stereo matching algorithm.
         * @param leftFrame Rectified left camera view in grayscale. Image should be CV_8UC1 format. Width anf height of
         * the image should be divisible by 4.
         * @param rightFrame Rectified right camera view in grayscale. Image should be CV_8UC1 format. Width anf height of
         * the image should be divisible by 4.
         * @return Disparity image in CV_8UC1 format. Disparity values are in the range of 0 to 'disp_size'
         * (Defined in demo.launch under argument name 'disparity_scope', should either be 64 or 128).
         */
        cv::Mat getDepth(cv::Mat &leftFrame, cv::Mat &rightFrame);

        /**
         * @brief In this function external and internal stereo camera parameters are read from ROS messages.
         * ObstacleDetector class which is used to calculate obstacle points, road and slope is initialized.
         * @param left_info ROS message containing information on left camera and left images
         * @param right_info ROS message containing information on right camera and right images
         */
        void loadCameraCalibration(const sensor_msgs::CameraInfoConstPtr &left_info,
                                   const sensor_msgs::CameraInfoConstPtr &right_info);

        /**
         * @brief This function defines look up tables that stores 3D information (Z_depth, X_direction, Y_direction)
         * for all the possible points and disparities for the quick access.
         */
        void DefineLUTs();

        /**
        * @brief Reads and verifies the ROS parameters and parameters required for this code. (Parameters can be
         * set in 'demo.launch' file)
        * @return true if successful.
        */
        bool readParameters(ros::NodeHandle nh, ros::NodeHandle nh_p);

        /**
         * @brief Contains the current disparity image of type cv::Mat CV_8UC1. This is used in many functions
         * when 3D information is required.
         */
        cv::Mat disparityFrame;
//        cv::Mat disparityFrameOrg;
        int globalframe;
        /**
         * @brief This parameter is used to scale down the calibration parameters and image size and is a user input
         * in the demo.launch file.
         */
        int Scale;
        /**
         * @brief Baseline of the stereo rig.
         */
        double stereo_baseline_;
        /**
         * @brief Camera center in x-direction in the rectified image coordinates
         */
        double u0;
        /**
         * @brief Camera center in y-direction in the rectified image coordinates
         */
        double v0;
        /**
         * @brief Focal length of the rectified stereo rig
         */
        double focal;

    private:
        /**
        * @brief Reads Cuda infomation.
        * @return true if successful.
        */
        bool CudaInfo();

        /**
         * @brief Publishes the object detection image (classification) from Yolo.
         * @param detectionImage Input cv::Mat image with bounding boxes drawn
         * @param publisher_ ROS publisher
         * @return true if successful.
         */
        bool publishDetectionImage(const cv::Mat &detectionImage, const ros::Publisher &publisher_);

		void publishGrayImage(const cv::Mat &detectionImage, const ros::Publisher &publisher_);

        /**
         * @brief This function is deprecated and ot used in the code. Can be used to publish an image.
         * @param detectionImage Image to be published in cv::Mat
         * @param publisher_ ROS publisher
         * @return true if successful
         */
        bool publishDetectionImage_single(const cv::Mat &detectionImage, const ros::Publisher &publisher_);

        /**
         * @brief This function calculates the occlusion for each object. This information is used in
         * object tracking. If there is an intersection between another object's bounding box and that object has
         * a higher detection confidence than the object in consideration, that object part is considered occluded.
         * @param bbox Bounding box coordinates for the object in consideration following cv::Rect_ format.
         * @param kk ID of the bounding box (object) in consideration.
         * @param FNcheck Bool to check whether this function is initiated inside tracking for false negatives
         * (i.e. stable track is missed in the detector:classification network)
         * @return Mask indicating occlusion of the object. 1 indicates a non occluded pixel and 0 indicate
         * an occluded pixel.
         */
        cv::Mat occlutionMap(cv::Rect_<int> bbox, size_t kk, bool FNcheck);

        /**
         * @brief This function is used to calculate the concatenated histogram for the object in h and s values
         * in hsv color space. Multiple histograms are calculated based on a grid structure.
         * @param currentDet The detection (object) for which the concatenated histogram is generated.
         * @param hsv The detection (object) in hsv colour space.
         * @param mask Mask indicating which pixels are occluded or not.
         * @param widthSeperate Integer indicating how many cells to be considered in the x direction for the
         * grid structure used to calculate each individual histogram.
         * @param heightSeperate Integer indicating how many cells to be considered in the y direction for the
         * grid structure used to calculate each individual histogram.
         */
        void calculateHistogram(Blob &currentDet, cv::Mat hsv, cv::Mat mask, int widthSeperate, int heightSeperate);

        /**
         * @brief This function is used to calculate the concatenated linear binary pattern histogram (LBPH)
         * for the object. Multiple LBPHs are calculated based on a grid structure.
         * @param currentDet The detection (object) for which the concatenated LBPH is generated.
         * @param rgb The detection (object) in RGB colour space.
         * @param grid_x Integer indicating how many cells to be considered in the x direction for the
         * grid structure used to calculate each individual histogram.
         * @param grid_y Integer indicating how many cells to be considered in the y direction for the
         * grid structure used to calculate each individual histogram.
         */
        void calculateLPBH(Blob &currentDet, cv::Mat rgb, int grid_x, int grid_y);

        /**
         * @brief This function handles the overall tracking process. It makes use of information present
         * in currentFrameBlobs and blobs vectors.
         */
        void Tracking();

        /**
         * @brief This function perform the dissimilarity cost calculation and matching process between the tracks
         * and objects detected. This function make use of information from  currentFrameBlobs and blobs vectors
         * and called within Tracking() function.
         * Dissimilarity cost calculation is based on four feature matching methods,
         * (1) grid based multiple HS histogram matching, (2) grid based LBPH matching,
         * (3) IoU based object size matching and (4) localization matching with distance.
         * Dissimilarity cost will result in a matrix of size currentFrameBlobs.size() * blobs.size().
         * Hungarian Assignment is used on this cost matrix to find the matches.
         */
        void matchCurrentFrameBlobsToExistingBlobs();

        /**
         * @brief This function performs tracking for stable tracks that goes undetected (false negative) in the
         * current frame. Stable track is defined a s a track that has been tracked successfully over five frames.
         * This function make use of information from  currentFrameBlobs, blobs, matchedTrackID, matchedFNs,
         * matchedFrmID, matchedFrmIDTrackID vectors. Function is called within Tracking() function.
         */
        void trackingFNs();

        /**
         * @brief This function checks for any remaining detections in currentFrameBlobs vector that are
         * not matched and initiate new tracks for them. This function is called within Tracking() function.
         */
        void addNewTracks();

        /**
         * @brief This function checks and update tracks that are not present in the current frame. It also
         * removes tracks that are exiting the frame and tracks that are not been present for threshold number of
         * consecutive frame. This threshold is defined in trackLife.
         */
        void updateUnmatchedTracks();

        /**
         * @brief This function displays tracking, classification and stereo outputs. This also generates the
         * ROS message about object tracks which is to be used in 'cubicle_merge' for speed estimation combined with
         * VO.
         */
        void CreateMsg();

		void DisplayResults();

        /**
         * @brief This function generates the ObsDisparity matrix which contains disparity information that only
         * belong to static obstacles. This is to be used in SLAM.
         */
        void generateStaticObsDisparityMap();

        /**
         * @brief This is the main function of this file that manages the flow. This function is called
         * within cameraCallback() function each time new images are received.
         */
        void Process();

        /**
         * @brief This function update the tracks with the matched object detections in the current frame.
         * This is called within matchCurrentFrameBlobsToExistingBlobs() and trackingFNs() functions.
         * @param currentFrameBlob Matched object detection in the current frame
         * @param existingBlobs Complete object track list
         * @param intIndex Matched track ID or the matched index of the track list.
         * @param isDet Flag to indicate whether the matched detection is from Yolo classification or defined in
         * trackingFNs().
         */
        void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex, bool isDet);

        /**
         * @brief This function initiate new tracks when there are no match is found in the existing tracks
         * for a new object appearing. Function is called within addNewTracks().
         * @param currentFrameBlob The detection to be initiated as a new track.
         * @param existingBlobs Track list
         */
        void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);

        /**
         * @brief Calculates the distance between two 2D points.
         * @param point1 2D Point 1
         * @param point2 2D Point 2
         * @return distance between point1 and point2
         */
        inline int distanceBetweenPoints(cv::Point point1, cv::Point point2) {
            int intX = abs(point1.x - point2.x);
            int intY = abs(point1.y - point2.y);
            return intX * intX + intY * intY;
        };

        /**
         * @brief ROS node handle.
         */
        ros::NodeHandle nodeHandle_, nodeHandle_pub;
        /**
         * Required by Yolo classification and detection network. Contains total number of classes.
         */
        int numClasses_;
        /**
         * Required by Yolo classification and detection network. Contains subset of classes required.
         */
        int compact_numClasses_;
        /**
         * Required by Yolo classification and detection network. Contains total class labels.
         */
        std::vector<std::string> classLabels_;
        /**
         * Required by Yolo classification and detection network. Contains subset of class labels required.
         */
        std::vector<std::string> compact_classLabels_;
        //! ROS subscriber and publisher.
//      ros::Publisher objectPublisher_;
//      ros::Publisher boundingBoxesPublisher_;
        /**
         * @brief ROS publisher containing information on tracked objects.
         */
        ros::Publisher obstaclePublisher_;
        /**
         * @brief ROS publisher containing the stereo result.
         */
        ros::Publisher disparityPublisher_;
        /**
         * @brief OS publisher containing the stereo result of static obstacles.
         */
        ros::Publisher obs_disparityPublisher_;
        std::string pub_obs_frame_id, obs_disparityFrameId;
        /**
         * @brief Detected objects by Yolo classification network
         */
        std::vector<std::vector<RosBox_> > rosBoxes_;
        /**
         * @brief Required by Yolo classification network
         */
        std::vector<int> rosBoxCounter_;
        obstacle_msgs::MapInfo obstacleBoxesResults_;
        /**
         * @brief Disparity search range. It should either be 64 or 128. Passed as an input from demo.launch file.
         */
        int disp_size;
        /**
         * @brief Scaled width of the image
         */
        int Width;
        /**
         * @brief Scaled height of the image
         */
        int Height;
        /**
         * @brief Amount of pixels to be removed from Width to make it fully divisible by 4.
         */
        int rem_w;
        /**
         * @brief Amount of pixels to be removed from Height to make it fully divisible by 4.
         */
        int rem_h;
        /**
         * @brief Final width of the image that is divisible by 4.
         */
        int Width_crp;
        /**
         * @brief Final height of the image that is divisible by 4.
         */
        int Height_crp;
        double camHeight;
        double camAngle;
        /**
         * @brief Flag to indicate whether the input images are fully divisible by 4.
         */
        bool is_even_crop;
        /**
         * @brief Lookup table that will give the X direction distance with respect to the left camera center
         * in meters. It is a 2D lookup table based on the disparity value and the column of the rectified image
         * point.
         */
        double **x3DPosition;
        /**
         * @brief Lookup table that will give the Y direction distance with respect to the left camera center
         * in meters. It is a 2D lookup table based on the disparity value and the row of the rectified image
         * point.
         */
        double **y3DPosition;
        /**
         * @brief Lookup table that will give the Z direction (depth) distance with respect to the left camera center
         * in meters. It is a 1D lookup table based on the disparity value of a point.
         */
        double **depth3D;
//        int **recDisparity;
//        double **recDepth;
//        double xDirectionPosition[1280][129] ={{}};
//        double yDirectionPosition[844][129] ={{}};
//        double depthTable[129] = {};
        /**
         * @brief Flag to indicate the first frame.
         */
        bool blnFirstFrame;
        /**
         * @brief Flag to indicate whether the yolo() function is initiated.
         */
        bool notInitiated = true;
        /**
         * @brief Flag to indicate whether to run generateStaticObsDisparityMap() function.
         */
        bool filter_dynamic_;

//        ros::Publisher leftImagePub_, rightImagePub_;
        /**
         * @brief Publishers for visualization of the results.
         */
        ros::Publisher detectionImagePublisher_;
        /**
         * @brief Publishers for visualization of the results.
         */
        ros::Publisher disparityColorPublisher_;
        /**
         * @brief Publishers for visualization of the results.
         */
        ros::Publisher trackingPublisher_;
        /**
         * @brief Publishers for visualization of the results.
         */
        ros::Publisher obstacleMaskPublisher_;
        /**
         * @brief Publishers for visualization of the results.
         */
        ros::Publisher slopePublisher_;
        /**
         * @brief Vector containing objects (detections) in the current frame. Vector used in tracking.
         */
        std::vector<Blob> currentFrameBlobs;
        /**
         * @brief Vector containing all the tracks and their status for the current sequence or the rosbag.
         * Vector used in tracking.
         */
        std::vector<Blob> blobs;
        /**
         * @brief Vector containing objects (detections) in the current frame from Yolo classification network.
         * Vector used in tracking.
         */
        std::vector<inputDetection> detListCurrFrame;
        /**
         * @brief Vector containing matched track IDs for tracks where a detection is defined at trackingFNs()
         * function. This vector is always updated with matchedFNs vector. Vector used in tracking.
         */
        std::vector<unsigned long> matchedTrackID;
        /**
         * @brief Vector containing objects (detections) defined at trackingFNs(). This vector is always
         * updated with matchedTrackID vector. Vector used in tracking.
         */
        std::vector<Blob> matchedFNs;
        /**
         * @brief Vector containing index of the objects (detections) in the currentFrameBlobs vector matched with a
         * track in the blobs vector which are matched at trackingFNs(). This vector is always
         * updated with matchedFrmIDTrackID vector. Vector used in tracking.
         */
        std::vector<unsigned long> matchedFrmID;
        /**
         * @brief Vector containing index of the tracks in the blobs vector, matched with a object (detections) in
         * the currentFrameBlobs vector which are matched at trackingFNs(). This vector is always
         * updated with matchedFrmID vector. Vector used in tracking.
         */
        std::vector<unsigned long> matchedFrmIDTrackID;
        /**
         * @brief Obstacle detector related
         */
        obstacle_msgs::obs obstacles;
//        Util::CPPTimer timer_yolo, timer_1, timer_2;
//        Util::HOGFeatureDescriptor* hog_descriptor;

		std::vector<cv::Scalar> colors;
        /**
         * @brief Constructor for Obstacles.cpp where obstacle, road and slope detections are performed.
         */
        ObstaclesDetection ObstacleDetector;

        /**
         * @brief Thread running yolo detection network.
         */
        std::thread detect_thread;
        /**
         * @brief Thread running stereo matching.
         */
        std::thread stereo_thread;
        /**
         * @brief Parameter for Darknet.
         */
        char **demoNames_;
        /**
         * @brief Parameter for Darknet.
         */
        char **compactDemoNames_;
        /**
         * @brief Parameter for Darknet.
         */
        image **demoAlphabet_;
        /**
         * @brief Parameter for Darknet.
         */
        int demoClasses_;
        /**
         * @brief Parameter for Darknet.
         */
        network *net_;
        /**
         * @brief Parameter for Darknet.
         */
        image buff_;//[3];
        /**
         * @brief Parameter for Darknet.
         */
        image buffLetter_;//[3];
        /**
         * @brief Deprecated. Buffer for left rectified images
         */
        cv::Mat buff_cv_l_;//[3];
        /**
         * @brief Deprecated. Buffer for right rectified images
         */
        cv::Mat buff_cv_r_;//[3];
//      int buffId_;//[3];
//      int buffIndex_ = 0;
//      IplImage * ipl_;
//      cv::Mat ipl_cv;

        double fps_ = 0;
        double whole_duration_ = 0;
        double stereo_duration_ = 0;
        double classi_duration_ = 0;
        double obs_duration_ = 0;
        float demoThresh_ = 0;
        float demoHier_ = .5;
//      int running_ = 0;
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
        boost::shared_mutex mutexImageCallback_;
        bool imageStatus_ = false;
        boost::shared_mutex mutexImageStatus_;
        bool isNodeRunning_ = true;
        boost::shared_mutex mutexNodeStatus_;
        int actionId_;
        boost::shared_mutex mutexActionStatus_;
        //*****************//
        ros::Time image_time_, prvImageTime;
        std_msgs::Header imageHeader_;
        /**
         * @brief Original images from the ROS message
         */
        cv::Mat camImageCopy_, origLeft, origRight, camImageOrig;
        /**
         * @brief Rectified images used for stereo matching, classification and tracking.
         */
        cv::Mat left_rectified, right_rectified;
        /**
         * @brief Images used for result displaying purposes.
         */
        cv::Mat output, tracking_output;
        /**
         * @brief Holder for disparity information on all obstacles. Information is taken from ObstacleDetector
         * which is a constructor for Obstacles.cpp.
         */
        cv::Mat ObsDisparity;
//      bool updateOutput = true;

        // double getWallTime();
        int sizeNetwork(network *net);
//      void rememberNetwork(network *net);
//      detection *avgPredictions(network *net, int *nboxes);

        /**
         * @brief This function call for getDepth() for stereo matching and disparity calculation. Also publishes
         * disparity image in disparityPublisher_.
         * @return nullptr
         */
        void *stereoInThread();

        /**
         * @brief This function runs the Yolo network for detection and classification for the current frame.
         * @return nullptr
         */
        void *detectInThread();

        /**
         * @brief This function copy the image pixel information required to run Yolo network.
         * @return nullptr
         */
        void *fetchInThread();

        /**
         * @brief This function displays the Yolo network detection and classification output for the current frame.
         * @return nullptr
         */
        void *displayInThread();

        /**
         * @brief This function loads the Yolo network and its weights.
         * @param cfgfile
         * @param weightfile
         * @param datafile
         * @param thresh
         * @param names
         * @param less_names
         * @param classes
         * @param delay
         * @param prefix
         * @param avg_frames
         * @param hier
         * @param w
         * @param h
         * @param frames
         * @param fullscreen
         */
        void setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                          char **names, char **less_names, int classes,
                          int delay, char *prefix, int avg_frames, float hier, int w, int h,
                          int frames, int fullscreen);

        /**
         * @brief This function initiates the Yolo network and cvNamedWindows for result displaying.
         */
        void yolo();

        /**
         * @brief This function is deprecated.
         */
        IplImage *getIplImage();

        /**
         * @brief This function is deprecated.
         */
        bool getImageStatus(void);

        /**
         * @brief This function is deprecated.
         */
        bool isNodeRunning(void);

        /**
         * @brief This function generates information that us required for object tracking.
         * Tracking() is called within this function at the end. detListCurrFrame vector is created from
         * Yolo network output inside this function. currentFrameBlobs vector is created by processing
         * detListCurrFrame vector inside this function.
         * @return nullptr
         */
        void *trackInThread();

        /**
         * @brief Flag to indicate whether to use gray image or colour image.
         */
        bool use_grey;
//      cv::Rect left_roi_, right_roi_;
        Detection *mpDetection;

        /**
         * @brief Pointer for StereoSGM class which perform stereo matching and disparity image calculation task.
         */
        sgm::StereoSGM *ssgm;
//      Tracker_optflow tracker_flow;
        std::thread *mpDepth_gen_run;
        int output_verbose;
        int intIndexOfLeastDistance;
        double dblLeastDistance;
        double hogLeastDistance;
//      std::vector<float> nullHog;
        // Disparity
        std::mutex mMutexDepth;
        std::vector<double> depth;
        /**
         * @brief Integer to indicate the minimum disparity to consider when processing stereo information.
         * This value can be set at demo.launch file.
         */
        int min_disparity;
        /**
         * @brief Flag to indicate whether to run stereo matching or not. This flag can be set at demo.launch file.
         */
        bool enableStereo = true;
        /**
         * @brief Flag to indicate whether to Yolo classification and detection network or not.
         * This flag can be set at demo.launch file.
         */
        bool enableClassification = true;
        /**
         * @brief Flag to indicate whether to run negative obstacle detection inside ObstaclesDetection class.
         * This flag can be set at demo.launch file.
         */
        bool enableNeg = false; //negative obstacle detection
        stereo_msgs::DisparityImage disparity_info;
        stereo_msgs::DisparityImage disparity_obs;
        std::ofstream file;
        std::string file_name;
        std::string img_name;
        /**
         * @brief Path to the file that contain parameters for superpixel segmentation in ObstaclesDetection class.
         */
        std::string parameter_filename; //negative obstacle detection
        char s[20];
        char im[20];
        /**
         * @brief Counter for total number frames processed.
         */
        int frame_num=0;
        /**
         * @brief Counter for number of times stereoInThread() is called.
         */
        int counter=0;
        /**
         * @brief Consecutive number of frames to check before a track is invalidated so that it is no longer
         * used in the tracking process.
         */
        int trackLife = 10;
        /**
         * @brief No of cells a object bounding box should be divided in the horizontal direction
         * (along bounding box width) to calculate the grid structure for
         * appearance histograms and LBPHs generation.
         */
        int cellsX = 4;
        /**
         * @brief No of cells a object bounding box should be divided in the vertical direction
         * (along bounding box height) to calculate the grid structure for
         * appearance histograms and LBPHs generation.
         */
        int cellsY = 3;
        /**
         * @brief The size the bounding boxes are resized before generating histograms.
         */
        cv::Size bboxResize;

        unsigned long track_counter = 0;

    };

} /* namespace darknet_ros*/
