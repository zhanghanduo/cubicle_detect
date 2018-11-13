//
// Created by hd on 1/18/18.
//
#include "darknet_ros/Blob.h"

namespace darknet_ros {

    Blob::Blob(float xmin, float ymin, float width, float height) {
        counter = 0;

        cv::Rect currentBoundingRect = cv::Rect_<int>(static_cast<int>(xmin),
                                                      static_cast<int>(ymin),
                                                      static_cast<int>(width),
                                                      static_cast<int>(height));

        cv::Point currentCenter;

        currentCenter.x = currentBoundingRect.x + currentBoundingRect.width / 2;

        currentCenter.y = currentBoundingRect.y + currentBoundingRect.height / 2;

        boundingRects.push_back(currentBoundingRect);

        centerPositions.push_back(currentCenter);

        isBody = false;
        isHead = false;
        isLegs = false;

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

        ///kf update: TODO: when tracking do the below
        /*// >>>> Matrix A
        kf.transitionMatrix.at<float>(2) = dT;
        kf.transitionMatrix.at<float>(9) = dT;
        // <<<< Matrix A

        state = kf.predict();

        cv::Rect predRect;
        predRect.width = state.at<float>(4);
        predRect.height = state.at<float>(5);
        predRect.x = state.at<float>(0) - predRect.width / 2;
        predRect.y = state.at<float>(1) - predRect.height / 2;

        cv::Point center;
        center.x = state.at<float>(0);
        center.y = state.at<float>(1);
        cv::circle(res, center, 2, CV_RGB(255,0,0), -1);
        cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);

        meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
        meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
        meas.at<float>(2) = (float)ballsBox[0].width;
        meas.at<float>(3) = (float)ballsBox[0].height;

        kf.correct(meas); // Kalman Correction*/



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

    Blob::Blob(cv::Rect _BBoxRect) {

        counter = 0;

        cv::Rect currentBoundingRect = _BBoxRect;

        cv::Point currentCenter;

        currentCenter.x = currentBoundingRect.x + currentBoundingRect.width / 2;

        currentCenter.y = currentBoundingRect.y + currentBoundingRect.height / 2;

        boundingRects.push_back(currentBoundingRect);

        centerPositions.push_back(currentCenter);

        isBody = false;
        isHead = false;
        isLegs = false;

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

    }



    void Blob::predictNextPosition() {

        auto numPositions = static_cast<int>(centerPositions.size());

        int deltaX = 0, deltaY=0;

//        if (numPositions == 1) {
//
//            predictedNextPosition.x = centerPositions.back().x;
//            predictedNextPosition.y = centerPositions.back().y;
//
//        }
        if (numPositions == 2) {
            deltaX = centerPositions[1].x - centerPositions[0].x;
            deltaY = centerPositions[1].y - centerPositions[0].y;
        } else if (numPositions == 3) {

            int sumOfXChanges = ((centerPositions[2].x - centerPositions[1].x) * 2) +
                                ((centerPositions[1].x - centerPositions[0].x) * 1);

            deltaX = (int)round((float)sumOfXChanges / 3.0);

            int sumOfYChanges = ((centerPositions[2].y - centerPositions[1].y) * 2) +
                                ((centerPositions[1].y - centerPositions[0].y) * 1);

            deltaY = (int)round((float)sumOfYChanges / 3.0);

        } else if (numPositions == 4) {

            int sumOfXChanges = ((centerPositions[3].x - centerPositions[2].x) * 3) +
                                ((centerPositions[2].x - centerPositions[1].x) * 2) +
                                ((centerPositions[1].x - centerPositions[0].x) * 1);

            deltaX = (int)round((float)sumOfXChanges / 6.0);

            int sumOfYChanges = ((centerPositions[3].y - centerPositions[2].y) * 3) +
                                ((centerPositions[2].y - centerPositions[1].y) * 2) +
                                ((centerPositions[1].y - centerPositions[0].y) * 1);

            deltaY = (int)round((float)sumOfYChanges / 6.0);

        } else if (numPositions >= 5) {

            int sumOfXChanges = ((centerPositions[numPositions - 1].x - centerPositions[numPositions - 2].x) * 4) +
                                ((centerPositions[numPositions - 2].x - centerPositions[numPositions - 3].x) * 3) +
                                ((centerPositions[numPositions - 3].x - centerPositions[numPositions - 4].x) * 2) +
                                ((centerPositions[numPositions - 4].x - centerPositions[numPositions - 5].x) * 1);

            deltaX = (int)round((float)sumOfXChanges / 10.0);

            int sumOfYChanges = ((centerPositions[numPositions - 1].y - centerPositions[numPositions - 2].y) * 4) +
                                ((centerPositions[numPositions - 2].y - centerPositions[numPositions - 3].y) * 3) +
                                ((centerPositions[numPositions - 3].y - centerPositions[numPositions - 4].y) * 2) +
                                ((centerPositions[numPositions - 4].y - centerPositions[numPositions - 5].y) * 1);

            deltaY = (int)round((float)sumOfYChanges / 10.0);
        }

        predictedNextPosition.x = centerPositions.back().x + deltaX*(intNumOfConsecutiveFramesWithoutAMatch+1);
        predictedNextPosition.y = centerPositions.back().y + deltaY*(intNumOfConsecutiveFramesWithoutAMatch+1);

    }

    void Blob::predictWidthHeight() {

        auto numPositions = static_cast<int>(boundingRects.size());

        int deltaX = 0, deltaY=0;

//        if (numPositions == 1) {
//
//            predictedWidth = boundingRects.back().width;
//            predictedHeight = boundingRects.back().height;
//
//        }
        if (numPositions == 2) {

            deltaX = boundingRects[1].width - boundingRects[0].width;
            deltaY = boundingRects[1].height - boundingRects[0].height;

        }
        else if (numPositions == 3) {

            int sumOfXChanges = ((boundingRects[2].width - boundingRects[1].width) * 2) +
                                ((boundingRects[1].width - boundingRects[0].width) * 1);

            deltaX = (int)round((float)sumOfXChanges / 3.0);

            int sumOfYChanges = ((boundingRects[2].height - boundingRects[1].height) * 2) +
                                ((boundingRects[1].height - boundingRects[0].height) * 1);

            deltaY = (int)round((float)sumOfYChanges / 3.0);

        }
        else if (numPositions == 4) {

            int sumOfXChanges = ((boundingRects[3].width - boundingRects[2].width) * 3) +
                                ((boundingRects[2].width - boundingRects[1].width) * 2) +
                                ((boundingRects[1].width - boundingRects[0].width) * 1);

            deltaX = (int)round((float)sumOfXChanges / 6.0);

            int sumOfYChanges = ((boundingRects[3].height - boundingRects[2].height) * 3) +
                                ((boundingRects[2].height - boundingRects[1].height) * 2) +
                                ((boundingRects[1].height - boundingRects[0].height) * 1);

            deltaY = (int)round((float)sumOfYChanges / 6.0);

        }
        else if (numPositions >= 5) {

            int sumOfXChanges = ((boundingRects[numPositions - 1].width - boundingRects[numPositions - 2].width) * 4) +
                                ((boundingRects[numPositions - 2].width - boundingRects[numPositions - 3].width) * 3) +
                                ((boundingRects[numPositions - 3].width - boundingRects[numPositions - 4].width) * 2) +
                                ((boundingRects[numPositions - 4].width - boundingRects[numPositions - 5].width) * 1);

            deltaX = (int)round((float)sumOfXChanges / 10.0);

            int sumOfYChanges = ((boundingRects[numPositions - 1].height - boundingRects[numPositions - 2].height) * 4) +
                                ((boundingRects[numPositions - 2].height - boundingRects[numPositions - 3].height) * 3) +
                                ((boundingRects[numPositions - 3].height - boundingRects[numPositions - 4].height) * 2) +
                                ((boundingRects[numPositions - 4].height - boundingRects[numPositions - 5].height) * 1);

            deltaY = (int)round((float)sumOfYChanges / 10.0);
        }

        predictedWidth = boundingRects.back().width + deltaX*(intNumOfConsecutiveFramesWithoutAMatch+1);
        predictedHeight = boundingRects.back().height + deltaY*(intNumOfConsecutiveFramesWithoutAMatch+1);

    }

}
