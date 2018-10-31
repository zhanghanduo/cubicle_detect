//
// Created by hd on 1/18/18.
//
#include "darknet_ros/Blob.h"

namespace darknet_ros {

    Blob::Blob(cv::Rect _BBoxRect) {

        counter = 0;

        cv::Rect currentBoundingRect = _BBoxRect;

        cv::Point currentCenter;

        currentCenter.x = currentBoundingRect.x + currentBoundingRect.width / 2;

        currentCenter.y = currentBoundingRect.y + currentBoundingRect.height / 2;

        boundingRects.push_back(currentBoundingRect);

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
