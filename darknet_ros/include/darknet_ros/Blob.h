//
// Created by hd on 1/18/18.
//

#ifndef PROJECT_BLOB_H
#define PROJECT_BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/kalman_filters.hpp>
#include <opencv2/opencv.hpp>

#define Mat_t CV_32FC

namespace darknet_ros {

    struct Blob{

        unsigned long counter;

        /**
         * @brief Holds the current frame width of the track in meters (3D absolute coordinates).
         */
        double diameter;

        /**
         * @brief Holds the current frame height of the track in meters (3D absolute coordinates).
         */
        double height;

        /**
         * @brief Holds the current frame detection confidence of the track.
         */
        double probability;
//        cv::KalmanFilter kf;//(stateSize, measSize, contrSize, CV_32F);
//        cv::Mat state;//(stateSize, 1, CV_32F);  // [x,y,v_x,v_y,w,h]
//        cv::Mat meas;//(measSize, 1, CV_32F);    // [z_x,z_y,z_w,z_h]
        cv::Ptr<cv::tracking::UnscentedKalmanFilter> uncsentedKF;
        bool t_initialized;
        std::deque<cv::Rect> t_initialRects;
        cv::Rect_<float> t_lastRectResult;
        static const size_t MIN_INIT_VALS = 4;

        /**
         * @brief Holds the predicted bounding box for the next frame of the track.
         */
        cv::Rect preditcRect;

        /**
         * @brief Vector containing whole history of the center points of the bounding boxes of the track.
         */
        std::vector<cv::Point2f> centerPositions; //2D coordinated with respect to the region of interest defined from rectified left image

        /**
         * @brief Vector containing whole history of the bounding boxes of the track.
         */
        std::vector<cv::Rect> boundingRects;

        /**
         * @brief Holds the current frame X, Y, Z of the center of the track in meters (3D absolute coordinates)
         * with respect to the local frame left camera center point.
         */
        cv::Vec3d position_3d;

        /**
         * @brief Holds the object category of the track.
         */
        std::string category;

        /**
         * @brief Holds the current frame disparity value of the center of the object bounding box of the track.
         */
        int disparity;

        /**
         * @brief Vector containing the three most relevant concatenated appearance histograms of the track.
         */
        std::vector<std::vector<cv::MatND> > hist;

        /**
         * @brief Vector containing the occlusion information for hist (appearance histograms) of the track.
         */
        std::vector<std::vector<bool> >occluded;

        /**
         * @brief Vector containing the values indicating the level of occlusion for each concatenated histograms
         * in hist (appearance histograms). Value is between 0-1.
         */
        std::vector<float> overallOcclusion;

        /**
         * @brief Holds the most relevant linear binary pattern histogram of the track.
         */
        cv::Mat lpbHist;

        /**
         * @brief Holds the predicted center position (as integers) of the next frame for the track.
         */
        cv::Point predictedNextPosition;

        /**
         * @brief Holds the predicted center position (as floats) of the next frame for the track.
         */
        cv::Point2f predictedNextPositionf;

        /**
         * @brief Holds the predicted width of the bounding box of the track in the next frame.
         */
        float predictedWidth;

        /**
         * @brief Holds the predicted height of the bounding box of the track in the next frame.
         */
        float predictedHeight;

        /**
         * @brief Holds the diagonal of the bounding box of the track in the current frame.
         */
        int dblCurrentDiagonalSize;

        /**
         * @brief Holds the aspect ratio of the bounding box of the track in the current frame.
         */
        double dblCurrentAspectRatio;

        /**
         * @brief Flag indicating either this detection is new or match found in the current frame.
         */
        bool blnCurrentMatchFoundOrNewBlob;

        /**
         * @brief Flag indicating whether the track is considered for track detection matching process.
         */
        bool blnStillBeingTracked;

        /**
         * @brief Deprecated. Flag indicating whether the detection is already matched in the current frame.
         */
        bool blnAlreadyTrackedInThisFrame;

        /**
         * @brief Flag indicating whether the detection is already matched in the current frame.
         */
        bool trackedInCurrentFrame;

        /**
         * @brief Flag indicating number of consecutive frames that the track has not being assigned any detection.
         */
        int intNumOfConsecutiveFramesWithoutAMatch;

        /**
         * @brief Flag indicating number of consecutive frames that the track has not being assigned a Yolo
         * detection.
         */
        int numOfConsecutiveFramesWithoutDetAsso;

        /**
         * @brief Constructor.
         * @param x x coordinate of the top left corner of the bounding box.
         * @param y y coordinate of the top left corner of the bounding box.
         * @param width Width of the bounding box.
         * @param height Height of the bounding box.
         */
        Blob(float x, float y, float width, float height);

        /**
         * @brief This function is deprecated.
         * @param rect0
         * @param rectv0
         */
        void CreateAugmentedUnscentedKF(cv::Rect_<float> rect0, cv::Point_<float> rectv0);

        /**
         * @brief This function is deprecated.
         * @return
         */
        cv::Rect GetRectPrediction();

        /**
         * @brief This function is deprecated.
         * @param dataCorrect
         * @return
         */
        cv::Rect UpdateAUKF(bool dataCorrect);

        /**
         * @brief This function predicts the center position of the bounding box of the track for the next frame.
         * Prediction is based on the most recent five frames information from the centerPositions vector.
         * Predicted information is stored in predictedNextPositionf and predictedNextPosition.
         */
        void predictNextPosition();

        /**
         * @brief This function predicts the width and height of the bounding box of the track for the next frame.
         * Prediction is based on the most recent five frames information from the boundingRects vector.
         * Predicted information is stored in predictedWidth and predictedHeight.
         */
        void predictWidthHeight();

    };

}





#endif //PROJECT_BLOB_H
