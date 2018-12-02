//
// Created by hd on 1/18/18.
//
#include "darknet_ros/Blob.h"

namespace darknet_ros {

    Blob::Blob(float xmin, float ymin, float width, float height) {
        counter = 0;

        currentBoundingRect = cv::Rect_<int>(static_cast<int>(xmin),
                                                      static_cast<int>(ymin),
                                                      static_cast<int>(width),
                                                      static_cast<int>(height));

        cv::Point currentCenter;

        currentCenter.x = currentBoundingRect.x + currentBoundingRect.width / 2;

        currentCenter.y = currentBoundingRect.y + currentBoundingRect.height / 2;

//        boundingRects.push_back(currentBoundingRect);

        centerPositions.push_back(currentCenter);


//        double currentSize = currentBoundingRect.width * currentBoundingRect.height;
//
//        size.push_back(currentSize);

        dblCurrentDiagonalSize = currentBoundingRect.width * currentBoundingRect.width
                                 + currentBoundingRect.height * currentBoundingRect.height;

//        dblCurrentAspectRatio = (float)currentBoundingRect.width / (float)currentBoundingRect.height;

        blnStillBeingTracked = true;

        blnCurrentMatchFoundOrNewBlob = true;

        blnAlreadyTrackedInThisFrame = false;

        intNumOfConsecutiveFramesWithoutAMatch = 0;

        int stateSize = 6, measSize = 4, contrSize = 0;
        kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);
        state = cv::Mat(stateSize, 1, CV_32F);  // [x,y,v_x,v_y,w,h];
        meas = cv::Mat(measSize, 1, CV_32F);    // [z_x,z_y,z_w,z_h]
        cv::setIdentity(kf.transitionMatrix);
        // Transition State Matrix A
        kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
        kf.measurementMatrix.at<float>(0) = 1.0f;
        kf.measurementMatrix.at<float>(7) = 1.0f;
        kf.measurementMatrix.at<float>(16) = 1.0f;
        kf.measurementMatrix.at<float>(23) = 1.0f;
        // Process Noise Covariance Matrix Q
        //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
        kf.processNoiseCov.at<float>(0) = 1e-2;
        kf.processNoiseCov.at<float>(7) = 1e-2;
        kf.processNoiseCov.at<float>(14) = 5.0f;
        kf.processNoiseCov.at<float>(21) = 5.0f;
        kf.processNoiseCov.at<float>(28) = 1e-2;
        kf.processNoiseCov.at<float>(35) = 1e-2;
        // Measures Noise Covariance Matrix R
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

        meas.at<float>(0) = xmin + width / 2;
        meas.at<float>(1) = ymin + height / 2;
        meas.at<float>(2) = width;
        meas.at<float>(3) = height;

        // >>>> Initialization
        kf.errorCovPre.at<float>(0) = 1; // px
        kf.errorCovPre.at<float>(7) = 1; // px
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1; // px
        kf.errorCovPre.at<float>(35) = 1; // px

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);
        // <<<< Initialization

        kf.statePost = state;

//        double deltaT = 0.01, omega_w =8, omega_u = 3.1623;
//        EKF = cv::KalmanFilter(3, 2, 0);
//        cv::Mat_<float> measurement(2,1);
//        measurement.setTo(cv::Scalar(0));
//        EKF.statePost.at<float>(0) = 0; // X
//        EKF.statePost.at<float>(1) = 0; // dX
//        EKF.statePost.at<float>(2) = 0; // theta
//        EKF.transitionMatrix = (cv::Mat_<float>(3, 3) << 1,1,0,   0,1,0,  0,0,1  ); //f
//        EKF.measurementMatrix = (cv::Mat_<float>(2, 3) << 1,0,0, 0,0,1  );  //H
//        EKF.processNoiseCov = (cv::Mat_<float>(3, 3) << 1,0,0, 0,0.1,0, 0,0,0.1);
//        EKF.processNoiseCov *=pow(omega_w,2);
//        setIdentity(EKF.measurementNoiseCov, cv::Scalar::all(pow(omega_u,2)));
//        setIdentity(EKF.errorCovPost, cv::Scalar::all(50));
    }

}
