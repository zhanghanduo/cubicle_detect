#ifndef OBSTACLES_H
#define OBSTACLES_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

class ObstaclesDetection
{
public:
    ObstaclesDetection();
    ~ObstaclesDetection();
    void Initiate(std::string camera_type, int disparity_size, double baseline);
    void ExecuteDetection(cv::Mat disparity_map);

private:
    cv::Mat GenerateUDisparityMap (cv::Mat disp_map);
    void RemoveObviousObstacles ();
    void GenerateVDisparityMap ();
    void RoadProfileCalculation ();
    void DisplayRoad();

    cv::Mat disparity_map, roadmap, obstaclemap, road, v_disparity_map, u_disparity_map;
    std::vector<cv::Point2i> initialRoadProfile, refinedRoadProfile;
    int roadNotVisibleDisparity = 0;
    int rdRowToDisRegard, rdStartCheckLines, intensityThVDisPoint, thHorizon;
    int rdProfileRowDistanceTh, rdProfileColDistanceTh, intensityThVDisPointForSlope;
    int disp_size;
    std::string pubName;
    int *uDispThresh;

};

#endif