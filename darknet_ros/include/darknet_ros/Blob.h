//
// Created by hd on 1/18/18.
//

#ifndef PROJECT_BLOB_H
#define PROJECT_BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/tracking.hpp>

namespace darknet_ros {

    struct Blob{

        unsigned long counter;

//        int noOfPixels;

//        int max_disparity;

//        std::vector<cv::Point3f> keyPoints;
        double probability;
//        cv::Mat kpDesc;
//        cv::MatND nHist;
        cv::Mat feature_cost;

        cv::MatND histHead , histBody, histLegs;

        bool isHead, isBody, isLegs;
        bool updateHead, updateBody, updateLegs;

        cv::KalmanFilter kf;//(stateSize, measSize, contrSize, CV_32F);
        cv::Mat state;//(stateSize, 1, CV_32F);  // [x,y,v_x,v_y,w,h]
        cv::Mat meas;//(measSize, 1, CV_32F);    // [z_x,z_y,z_w,z_h]

//        std::vector<cv::Point2i> obsPoints;     //2D coordinated with respect to the region of interest defined from rectified left image
//
//        std::vector<float> obsHog;              //Record the hog features of a single blob
//
//        double depth, diameter, height, probability;
//        double ymin, ymax, xmin, xmax;          //2D coordinated with respect to left camera

        std::vector<cv::Rect> boundingRects;           //2D coordinated with respect to the region of interest defined from rectified left image

        std::vector<cv::Point> centerPositions; //2D coordinated with respect to the region of interest defined from rectified left image

//        std::vector<double> size;

//        cv::Vec3d position_3d;

        std::string category;

//        int disparity;

        cv::Point predictedNextPosition;

        int predictedWidth, predictedHeight;

        int dblCurrentDiagonalSize;

//        double dblCurrentAspectRatio;

        bool blnCurrentMatchFoundOrNewBlob;

        bool blnStillBeingTracked;

        bool blnAlreadyTrackedInThisFrame;

        int intNumOfConsecutiveFramesWithoutAMatch;

        Blob(float x, float y, float width, float height);

        Blob(cv::Rect _BBoxRect);

        void predictNextPosition(void);

        void predictWidthHeight();


    };

}





#endif //PROJECT_BLOB_H
