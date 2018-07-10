//
// Created by hd on 1/18/18.
//

#ifndef PROJECT_BLOB_H
#define PROJECT_BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

namespace cubicle_detect {

    struct Blob{

//        int noOfPixels;

//        int max_disparity;

        std::vector<cv::Point2i> obsPoints;

        std::vector<float> obsHog;

        float diameter, height, probability;

        cv::Vec3f position_3d;

        std::string category;

//        double disparity;

        cv::Rect currentBoundingRect;

        std::vector<cv::Point> centerPositions;

        cv::Point predictedNextPosition;

        int dblCurrentDiagonalSize;

//        double dblCurrentAspectRatio;

        bool blnCurrentMatchFoundOrNewBlob;

        bool blnStillBeingTracked;

        bool blnAlreadyTrackedInThisFrame;

        int intNumOfConsecutiveFramesWithoutAMatch;

        Blob(cv::Rect _BBoxRect);

        void predictNextPosition(void);





    };

}





#endif //PROJECT_BLOB_H
