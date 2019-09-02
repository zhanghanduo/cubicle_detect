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

/**
 * @brief This is to store information for each object segment in u disparity map.
 */
struct u_span {
    int u_left; int u_right; int u_d;
    int u_pixels;
    bool checked;
    int ID;
};

class obsBlobs {
public:
    /**
     * @brief Contain the number pixels the obstacle contain.
     */
    int noOfPixels;
    /**
     * @brief Maximum disparity value of the obstacle.
     */
    int max_disparity;
    /**
     * @brief Vector containing all the 2D image points of the obstacle
     */
    std::vector<cv::Point2i> obsPoints;     //2D coordinated with respect to the region of interest defined from rectified left image

    /**
     * @brief Depth in meters to the nearest point of the obstacle with respect to the left camera center.
     */
    double depth;
    /**
     * @brief Y direction 3D world distance in meters to the top side of the obstacle with respect to the
     * left camera center.
     */
    double startY;
    /**
     * @brief Y direction 3D world distance in meters to the bottom side of the obstacle with respect to the
     * left camera center.
     */
    double endY;                    //3D coordinated with respect to left camera
    /**
     * @brief X direction 3D world distance in meters to the left side of the obstacle with respect to the
     * left camera center.
     */
    double startX;
    /**
     * @brief Y direction 3D world distance in meters to the right side of the obstacle with respect to the
     * left camera center.
     */
    double endX;                    //3D coordinated with respect to left camera
    /**
     * @brief 2D bounding box of the obstacle.
     */
    cv::Rect currentBoundingRect;           //2D coordinated with respect to the region of interest defined from rectified left image
//    cv::Point centerPosition; //2D coordinated with respect to the region of interest defined from rectified left image

    /**
     * @brief This function calculates the diagonal of the bounding box.
     * @return The diagonal length of the the bounding box.
     */
    double getDiagonalSize() {
        return pow(currentBoundingRect.width, 2) + pow(currentBoundingRect.height, 2);
    }

    /**
     * @brief This function calculates the aspect ratio of the bounding box.
     * @return The aspect ratio of the bounding box.
     */
    double getAspectRatio() {
        return ((double)currentBoundingRect.width / (double)currentBoundingRect.height);
    }
};

class ObstaclesDetection
{
public:
    /**
     * @brief Constructor.
     */
    ObstaclesDetection();

    /**
     * @brief Destructor.
     */
    ~ObstaclesDetection();

    /**
     * @brief This function initiates parameters required to run ObstaclesDetection class.
     * @param disparity_size Maximum disparity value.
     * @param baseline Baseline of the stereo rig.
     * @param u0 Camera center in x-direction in the rectified image coordinates
     * @param v0 Camera center in y-direction in the rectified image coordinates
     * @param focal Focal length of the rectified stereo rig
     * @param Width Width of the rectified image
     * @param Height Height of the rectified image
     * @param scale Value to scale down the calibration parameters and image size
     * @param min_disparity Minimum disparity value to consider. All disparities less than this is dis-regarded.
     * @param enNeg Flag to indicate whether to enable negative obstacle detection
     * @param Parameter_filename Path to the file that contain parameters for superpixel segmentation.
     */
    void Initiate(int disparity_size, double baseline, double u0, double v0, double focal, int Width, int Height, int scale, int min_disparity, bool enNeg, const std::string& Parameter_filename);

    /**
     * @brief This is the main function that run obstacle detection, road detection and slope detection.
     * @param disparity_map Disparity image of the rectified left camera view
     * @param img Rectified left camera view
     */
    void ExecuteDetection(cv::Mat &disparity_map, cv::Mat &img);

    /**
     * @brief Disparity image of all obstacles of the scene.
     */
    cv::Mat obsDisFiltered;
    /**
     * @brief Image to visualize slope detection results.
     */
    cv::Mat slope_map;
    /**
     * @brief Colour rectified left image, used to visualize obstacle detection results.
     */
    cv::Mat left_rect_clr;
    /**
     * @brief Deprecated. Slope angle for the first road segment in degrees.
     */
    double slope_angle =0.0;
    /**
     * @brief Pitch for the road segment in degrees.
     */
    double pitch_angle = 0.0;

private:
    /**
     * @brief This function generates the u-Disparity image for a given disparity image.
     * @param disp_map Disparity image for which the u-Disparity image should be computed.
     * @return u-Disparity image (u_disparity_map)
     */
    cv::Mat GenerateUDisparityMap (cv::Mat disp_map);

    /**
     * @brief This function removes large obstacles from the disparity image based on the
     * u-Disparity image (u_disparity_map). Generates obstaclemap and road matrices.
     */
    void RemoveObviousObstacles ();

    /**
     * @brief This function generates the v-Disparity image (v_disparity_map)
     * from the large obstacles removed disparity image (road).
     */
    void GenerateVDisparityMap ();

    /**
     * @brief This function calculates road profile of the scene from the v-Disparity image (v_disparity_map).
     * Generates initialRoadProfile and refinedRoadProfile vectors.
     */
    void RoadProfileCalculation ();

    /**
     * @brief This function initiate the obstacle map detection process. Generates roadmap & negObsMap matrices and
     * dynamicLookUpTableRoad & dynamicLookUpTableRoadProfile vectors. Alters obstaclemap matrix.
     */
    void InitiateObstaclesMap();

    /**
     * @brief Refine the obstacle detection using u-Disparity information. Generates u_disparity_map_new,
     * u_thresh_map and obsDisFiltered matrices. Also generates currentFrameObsBlobs vector.
     */
    void RefineObstaclesMap();

    /**
     * @brief This function checks whether the points in randomRoadPoints vector are on the same line or not.
     * This function is called within SurfaceNormal() function.
     * @return True if three points are collinear (lie on the same line)
     */
    bool colinear();

    /**
     * @brief This function calculates the surface normal to the plane that contain the three points in
     * randomRoadPoints vector. This function is called within SurfaceNormalCalculation()function.
     */
    void SurfaceNormal();

    /**
     * @brief This function initiates the surface normal calculation by assigning values to
     * selectedIndexes vector. This function is called within RoadSlopeCalculation() function.
     */
    void SurfaceNormalCalculation();

    /**
     * @brief This function calculates the road surface angle, pitch angle, road plane, road surface normal
     * with respect to the current vehicle location. Road information is based on refinedRoadProfile vector.
     * Road slope is calculated dy diving the road into different segments based on the depth defined by
     * minDepthDiffToCalculateSlope. Road points in randomRoadPoints and randomRoadPoints2D vectors required
     * for surface normal calculation are assigned in this function. This function is called within
     * RoadSlopeInit() function.
     */
    void RoadSlopeCalculation();

    /**
     * @brief This function will check for imu input to remove sudden slopes in the presence of a hump.
     * SurfaceNormalCalculation() function is initiated inside this function as well.
     */
    void RoadSlopeInit();

    /**
     * @brief This function is for visualizing road detection related results.
     */
    void DisplayRoad();

    /**
     * @brief This function is for visualizing positive obstacle detection related results.
     */
    void DisplayPosObs();

    /**
     * @brief This function initiate SPSegmentationEngine class to generate superpixel segmentation.
     * Segmentation is required for saliency based negative obstacle detection method implemented in
     * SaliencyBasedDetection() function.
     */
    void GenerateSuperpixels ();

    /**
     * @brief This function implements negative obstacle detection method based on salient road region detection
     * using superpixel segmentation.
     */
    void SaliencyBasedDetection ();

    /**
     * @brief This function implements negative obstacle detection based on lower intensity values.
     */
    void IntensityBasedDetection ();

    /**
     * @brief This function will initiate either SaliencyBasedDetection() or IntensityBasedDetection() for negative
     * obstacle detection. Furthermore, it will display the negative obstacle results.
     */
    void GenerateNegativeObstacles();

    /**
     * @brief Holds the disparity image of the scene. Disparity values below minimum_disparity is removed.
     */
    cv::Mat disparity_map;
    /**
     * @brief Holds the information regarding to road region. Pixels belonging to road is marked
     * as one, otherwise as zero.
     */
    cv::Mat roadmap;
    /**
     * @brief Holds the information regarding to obstacle region. Pixels belonging to an obstacle is marked
     * as one, otherwise as zero.
     */
    cv::Mat obstaclemap;
    /**
     * @brief Holds the disparity image after removing disparity values belonging to large obstacles.
     */
    cv::Mat road;
    /**
     * @brief Holds the v-Disparity image calculated based on road disparity image.
     */
    cv::Mat v_disparity_map;
    /**
     * @brief Holds the u-Disparity image calculated based on disparity_map disparity image.
     */
    cv::Mat u_disparity_map; //road
    /**
     * @brief Holds the u-Disparity image calculated based only on the disparity values belonging to obstacles.
     */
    cv::Mat u_disparity_map_new;
    /**
     * @brief Holds the u-Disparity image of the thresholded u_disparity_map_new.
     */
    cv::Mat u_thresh_map;
    /**
     * @brief Holds the information regarding to negative obstacle region based on stereo information.
     * Pixels belonging to a negative obstacle is marked as one, otherwise as zero.
     */
    cv::Mat negObsMap;
    /**
     * @brief Holds the information regarding to refined positive obstacle region. Pixels belonging to an
     * obstacle is marked as one, otherwise as zero.
     */
    cv::Mat obsMask;
    /**
     * @brief Holds the disparity image after removing disparity values belonging to refined positive obstacles.
     */
    cv::Mat obstacleDisparityMap; //positive obstacle detection
    /**
     * @brief Holds the slope results of the most recent frame.
     */
    cv::Mat prvSlopeMap; //slope
    /**
     * @brief Holds the final negative obstacle region information. Pixels belonging to a negative obstacle is
     * marked as one, otherwise as zero.
     */
    cv::Mat neg_obstacle;
    /**
     * @brief Holds the rectified left colour image used as superpixel segmentation input.
     */
    cv::Mat left_rect_clr_sp;
    /**
     * @brief Holds the index of the suprpixel, each pixel belongs to.
     */
    cv::Mat superpixelIndexImage;
    /**
     * @brief Hold the superpixel segmented image used for visualization.
     */
    cv::Mat superpixelImage; //negative obstacle detection

    /**
     * @brief Vector indicating points belonging to road during the initial assignment.
     * Point(x,y) follows RoadPoint(road_row,road_disparity).
     */
    std::vector<cv::Point2i> initialRoadProfile;
    /**
     * @brief Vector indicating points belonging to road after refinement.
     * Point(x,y) follows RoadPoint(road_row,road_disparity).
     */
    std::vector<cv::Point2i> refinedRoadProfile; // road
    /**
     * @brief Vector holding information about positive obstacles.
     */
    std::vector<obsBlobs> currentFrameObsBlobs; // positive obstacle
    /**
     * @brief Vector holding randomly selected road points in 3D space with respect to the left camera center
     * in the local frame. Used in surface normal calculation.
     */
    std::vector<cv::Vec3d> randomRoadPoints; //slope
    /**
     * @brief Vector holding randomly selected road points in 2D image plane. Used in surface normal visualization.
     */
    std::vector<cv::Point2i> randomRoadPoints2D; //slope
    /**
     * @brief Vector holding indexes selected from randomly selected road points for surface normal calculation.
     */
    std::vector<int> selectedIndexes; // slope
    std::vector<std::vector<cv::Point> > contourList, contourListSaliency; //negative obstacle detection

//    std::string pubName;
    /**
     * @brief Surface normal vector of the road segment.
     */
    cv::Vec3d surfaceN; // slope
    /**
     * @brief Flag indicating presence of imu sensor readings.
     */
    bool imuDetected = false; // slope
    /**
     * @brief Flag indicating to use IntensityBasedDetection() function.
     */
    bool intensityBased = false;
    /**
     * @brief Flag indicating to use SaliencyBasedDetection() function.
     */
    bool saliencyBased = false; //negative obstacle detection
    /**
     * @brief Flag indicating to run negative obstacle detection.
     */
    bool enNegObsDet = false; // to enable negative obstacle detection

    /**
     * @brief Disparity value of the furthest road point
     */
    int roadNotVisibleDisparity = 0;
    /**
     * @brief Number of rows to disregard from the bottom of the image before starting to check for road.
     * This is to allow for dark region that might arise after image rectification.
     */
    int rdRowToDisRegard;
    /**
     * @brief Number of rows to check to find the best starting point to initiate road detection.
     */
    int rdStartCheckLines;
    /**
     * @brief Minimum number of columns the road is detected for each row to consider that row contain
     * visible road.
     */
    int intensityThVDisPoint;
    /**
     * @brief Maximum number of rows to discontinue road detection if road disparity doe not reduce.
     */
    int thHorizon;
    /**
     * @brief Maximum number of rows to wait if the road goes undetected before discontinuing road detection.
     */
    int rdProfileRowDistanceTh;
    /**
     * @brief Maximum number of columns to the left (lower disparities) to check for the most suitable disparity for the road.
     */
    int rdProfileColDistanceTh;
    /**
     * @brief Minimum number of columns the road is detected for each row to calculate the slope accurately.
     */
    int intensityThVDisPointForSlope;
    /**
    * @brief Disparity search range. It should either be 64 or 128. Passed as an input from demo.launch file.
    */
    int disp_size;
    /**
    * @brief Integer to indicate the minimum disparity to consider when processing stereo information.
    * This value can be set at demo.launch file.
    */
    int minimum_disparity;
    /**
     * @brief Smallest image row number of the road is visible (or the furthest road viable image row).
     */
    int road_starting_row;
    /**
     * @brief Minimum number of points required to declare a positive obstacle.
     */
    int minNoOfPixelsForObject;
    /**
     * @brief Parameter used to visualize slope results. Indicates how many centimeters each pixel represent
     * in the Y direction in the 3D world (downward direction).
     */
    int yResolutionForSlopeMap = 4;//how many cm per pixel -- slope
    /**
     * @brief Parameter used to visualize slope results. Indicates how many centimeters each pixel represent
     * in the Z direction in the 3D world (depth).
     */
    int zResolutionForSlopeMap = 5;//how many cm per pixel -- slope
    /**
     * @brief Parameter used to visualize slope results. Indicates how many centimeters height difference to
     * consider when generating slope_map.
     */
    int heightForSlope = 800;
    /**
     * @brief Parameter for counting how many frames received after detecting a hump from imu data.
     */
    int humpEndFrames = 0;//cm both direction -- slope
    /**
     * @brief Smallest disparity value for check for slope. Corresponds to the furthest to look for the slope.
     */
    int disForSlope;
    /**
     * @brief Biggest disparity value for check for slope. Corresponds to the nearest to look for the slope.
     */
    int disForSlopeStart; // slope
    /**
     * @brief Parameter to compensate any angle to the horizontal direction with the left camera optical line.
     */
    int slopeAdjHeight;
    /**
     * @brief Parameter to compensate any angle to the horizontal direction with the left camera optical line.
     */
    int slopeAdjLength;//cm -- slope
    /**
     * @brief Counter keeping track of number of frames processed.
     */
    int frameCount = 0;
    /**
     * @brief Parameter used when defining region_of_interest for negative obstacle detection.
     */
    int left_offset;
    /**
     * @brief Parameter used when defining region_of_interest for negative obstacle detection.
     */
    int right_offset;
    /**
     * @brief Parameter used when defining region_of_interest for negative obstacle detection.
     */
    int bottom_offset;
    /**
     * @brief Parameter used when defining region_of_interest for negative obstacle detection.
     */
    int top_offset;//negative obstacle detection
    /**
     * @brief Minimum number of pixels present in the contour to declare that region covered by the
     * contour belongs to a negative obstacle.
     */
    int minContourLengthForNegObs;
    int minBlobDistanceForNegObs;//negative obstacle detection

    /**
     * @brief Lookup table that gives minimum intensity of the u-Disparity map to consider that pixel belongs to
     * a large obstacle.
     */
    int *uDispThresh;
    /**
     * @brief Lookup table that is used in u-disparity based positive obstacle categorization.
     */
    int *uHysteresisLowThresh;
    /**
     * @brief Lookup table that is used in u-disparity based positive obstacle categorization.
     */
    int *uXDirectionNeighbourhoodThresh;
    /**
     * @brief Lookup table that is used in u-disparity based positive obstacle categorization.
     */
    int *uDNeighbourhoodThresh;
    /**
     * @brief Lookup table that store results of the road region. Based on the disparity value lookup table
     * return the image row for that disparity.
     */
    int *dynamicLookUpTableRoad;
    /**
     * @brief Lookup table that store results of the road region. Based on the row lookup table return the
     * road disparity for that row.
     */
    int *dynamicLookUpTableRoadProfile;
    /**
     * @brief Lookup table that gives minimum intensity of the u-Disparity map to consider that pixel belongs to
     * a negative obstacle.
     */
    int *dispThreshForNeg; //negative obstacle detection

    /**
     * @brief Parameter indicating the width to consider in each direction from the center of the left camera.
     */
    double widthOfInterest = 15.0;//in meters in one direction
    /**
     * @brief Parameter indicating the height to consider in each direction from the center of the left camera.
     */
    double heightOfInterest = 2.5;//in meters in one direction
    double meanValUPrv =0.0, meanValVPrv =0.0;
    /**
     * @brief Parameter indicating the furthest depth to check for slope.
     */
    double depthForSlpoe;
    /**
     * @brief Parameter indicating the nearest depth to check for slope.
     */
    double depthForSlopeStart; //slope
    /**
     * @brief Parameter indicating the Y directional angular velocity of the imu.
     */
    double imuAngularVelocityY = 0.0; //slope
    /**
     * @brief Parameter indicating the depth difference to consider when segmenting the road region to
     * multiple road segments for slope calculation.
     */
    double minDepthDiffToCalculateSlope;//cm -- slope

    /**
     * @brief Lookup table that will give the X direction distance with respect to the left camera center
     * in meters. It is a 2D lookup table based on the disparity value and the column of the rectified image
     * point.
     */
    double **xDirectionPosition;
    /**
    * @brief Lookup table that will give the Y direction distance with respect to the left camera center
    * in meters. It is a 2D lookup table based on the disparity value and the row of the rectified image
    * point.
    */
    double **yDirectionPosition;
    /**
    * @brief Lookup table that will give the Z direction (depth) distance with respect to the left camera center
    * in meters. It is a 1D lookup table based on the disparity value of a point.
    */
    double *depthTable;

    /**
     * @brief Bounding box for the region selected to process for negative obstacle detection.
     */
    cv::Rect region_of_interest; //negative obstacle detection
    /**
     * @brief Parameters required for SPSegmentationEngine class for superpixel segmentation.
     */
    SPSegmentationParameters seg_params; //negative obstacle detection

};

#endif