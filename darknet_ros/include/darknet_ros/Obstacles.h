#ifndef OBSTACLES_H
#define OBSTACLES_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
//#include "darknet_ros/Blob.h"

//negative obstacles
#include "darknet_ros/segengine.h"
#include "darknet_ros/structures.h"
//#include "darknet_ros/RectangleDetector.h"

struct u_span {
    int u_left; int u_right; int u_d;
    int u_pixels;
    bool checked;
    int ID;
};

class obsBlobs {
public:
    int noOfPixels, max_disparity;
    std::vector<cv::Point2i> obsPoints;     //2D coordinated with respect to the region of interest defined from rectified left image

    double depth;
    double startY, endY;                    //3D coordinated with respect to left camera
    double startX, endX;                    //3D coordinated with respect to left camera

    cv::Rect currentBoundingRect;           //2D coordinated with respect to the region of interest defined from rectified left image
//    cv::Point centerPosition; //2D coordinated with respect to the region of interest defined from rectified left image

    double getDiagonalSize() {
        return pow(currentBoundingRect.width, 2) + pow(currentBoundingRect.height, 2);
    }

    double getAspectRatio() {
        return ((double)currentBoundingRect.width / (double)currentBoundingRect.height);
    }
};

class ObstaclesDetection
{
public:
    ObstaclesDetection();
    ~ObstaclesDetection();
    void Initiate(int disparity_size, double baseline, double u0, double v0, double focal, int Width, int Height, int scale, int min_disparity, bool enNeg, const std::string& Parameter_filename);
    void ExecuteDetection(cv::Mat &disparity_map, cv::Mat &img);
    cv::Mat obsDisFiltered;
    cv::Mat slope_map, left_rect_clr;
    double slope_angle =0.0, pitch_angle = 0.0;

private:
    cv::Mat GenerateUDisparityMap (cv::Mat disp_map);
    void RemoveObviousObstacles ();
    void GenerateVDisparityMap ();
    void RoadProfileCalculation ();
    void InitiateObstaclesMap();
    void RefineObstaclesMap();
    bool colinear();
    void SurfaceNormal();
    void SurfaceNormalCalculation();
    void RoadSlopeCalculation();
    void RoadSlopeInit();
    void DisplayRoad();
    void DisplayPosObs();
    void GenerateSuperpixels ();
    void SaliencyBasedDetection();
    void IntensityBasedDetection ();
    void GenerateNegativeObstacles();

    cv::Mat disparity_map, roadmap, obstaclemap, road, v_disparity_map, u_disparity_map; //road
    cv::Mat u_disparity_map_new, u_thresh_map, negObsMap, obsMask, obstacleDisparityMap; //positive obstacle detection
    cv::Mat prvSlopeMap; //slope
    cv::Mat neg_obstacle, left_rect_clr_sp, superpixelIndexImage, superpixelImage; //negative obstacle detection

    std::vector<cv::Point2i> initialRoadProfile, refinedRoadProfile; // road
    std::vector<obsBlobs> currentFrameObsBlobs; // positive obstacle
    std::vector<cv::Vec3d> randomRoadPoints; //slope
    std::vector<cv::Point2i> randomRoadPoints2D; //slope
    std::vector<int> selectedIndexes; // slope
    std::vector<std::vector<cv::Point> > contourList, contourListSaliency; //negative obstacle detection

//    std::string pubName;
    cv::Vec3d surfaceN; // slope
    bool imuDetected = false; // slope
    bool intensityBased = false, saliencyBased = false; //negative obstacle detection
    bool enNegObsDet = false; // to enable negative obstacle detection

    int roadNotVisibleDisparity = 0;
    int rdRowToDisRegard, rdStartCheckLines, intensityThVDisPoint, thHorizon;
    int rdProfileRowDistanceTh, rdProfileColDistanceTh, intensityThVDisPointForSlope;
    int disp_size, minimum_disparity;
    int road_starting_row, minNoOfPixelsForObject;
    int yResolutionForSlopeMap = 4;//how many cm per pixel -- slope
    int zResolutionForSlopeMap = 5;//how many cm per pixel -- slope
    int heightForSlope = 800, humpEndFrames = 0;//cm both direction -- slope
    int disForSlope, disForSlopeStart; // slope
    int slopeAdjHeight, slopeAdjLength;//cm -- slope
    int frameCount = 0;
    int left_offset, right_offset, bottom_offset, top_offset;//negative obstacle detection
    int minContourLengthForNegObs, minBlobDistanceForNegObs;//negative obstacle detection

    int *uDispThresh;
    int *uHysteresisLowThresh;
    int *uXDirectionNeighbourhoodThresh;
    int *uDNeighbourhoodThresh;
    int *dynamicLookUpTableRoad;
    int *dynamicLookUpTableRoadProfile;
    int *dispThreshForNeg; //negative obstacle detection

    double widthOfInterest = 15.0;//in meters in one direction
    double heightOfInterest = 2.5;//in meters in one direction
    double meanValUPrv =0.0, meanValVPrv =0.0;
    double depthForSlpoe, depthForSlopeStart; //slope
    double imuAngularVelocityY = 0.0; //slope
    double minDepthDiffToCalculateSlope;//cm -- slope

    double **xDirectionPosition;
    double **yDirectionPosition;
    double *depthTable;

    cv::Rect region_of_interest; //negative obstacle detection
    SPSegmentationParameters seg_params; //negative obstacle detection

};

#endif