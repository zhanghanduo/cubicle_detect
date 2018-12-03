#include "darknet_ros/Obstacles.h"//"Hungarian.h"

ObstaclesDetection::ObstaclesDetection(){}

ObstaclesDetection::~ObstaclesDetection(){}

cv::Mat ObstaclesDetection::GenerateUDisparityMap (cv::Mat disp_map) {
    cv::Mat u_disp_map(disp_size+1, disp_map.cols, CV_8UC1, cv::Scalar::all(0));
    for (int v = 0; v < disp_map.cols; v++) {
        for (int u = 0; u < disp_map.rows; u++) {
            short d = disp_map.at<uchar>(u, v);
            if (d > 0 &&  d < disp_size+1) {
                if(u_disp_map.at<uchar>(d,v) < 254)
                    u_disp_map.at<uchar>(d,v) +=1;
            }
        }
    }
    return u_disp_map;
}

void ObstaclesDetection::RemoveObviousObstacles () {
    for (int c=0; c<u_disparity_map.cols; c++) {
        for (int r=0; r<u_disparity_map.rows; r++) {
            if (u_disparity_map.at<uchar>(r,c)>uDispThresh[r]){//u_disparity_thresh){
                for (int j=0; j<road.rows; j++){
                    if (road.at<uchar>(j,c)==r){
                        road.at<uchar>(j,c) = 0;
                        obstaclemap.at<uchar>(j,c)=1;
                    }
                }
            }
        }
    }
}

void ObstaclesDetection::GenerateVDisparityMap () {
    for (int v = 0; v < road.rows; ++v) {
        for (int u = 0; u < road.cols; ++u) {
            short d = road.at<uchar>(v, u);
            if (d > 0 && d < disp_size+1) {
                if(v_disparity_map.at<uchar>(v,d) < 254)
                    v_disparity_map.at<uchar>(v,d) +=1;
            }
        }
    }
}

void ObstaclesDetection::RoadProfileCalculation () {
    int road_start_intensity = v_disparity_map.at<uchar>(v_disparity_map.rows-1,0);//401
    int road_start_row = v_disparity_map.rows-1;//401
    int road_start_disparity = 0;
    for (int r=v_disparity_map.rows-rdRowToDisRegard; r>v_disparity_map.rows-rdStartCheckLines; r--) { //10,18
        for (int c=0; c<v_disparity_map.cols; c++) {
            if (road_start_intensity<v_disparity_map.at<uchar>(r,c)){
                road_start_intensity = v_disparity_map.at<uchar>(r,c);
                road_start_row = r;
                road_start_disparity = c;
            }
        }
    }

    initialRoadProfile.push_back(cv::Point2i(road_start_row, road_start_disparity));
    refinedRoadProfile.push_back(cv::Point2i(road_start_row, road_start_disparity));

    for (int r=initialRoadProfile[0].x-1; r>0; r--) {
        int max_intensity = v_disparity_map.at<uchar>(r,0);
        cv::Point2i max_intensity_point(r,0);
        for (int c=1; c<v_disparity_map.cols; c++) {
            if (max_intensity<v_disparity_map.at<uchar>(r,c)){
                max_intensity = v_disparity_map.at<uchar>(r,c);
                max_intensity_point = cv::Point2i(r,c);
            }
        }
        if (max_intensity>intensityThVDisPoint){
            if(abs(initialRoadProfile.back().x-max_intensity_point.x)<rdProfileRowDistanceTh){
//    	   for(int i=initialRoadProfile.back().y; i>max_intensity_point.y;i--) {
//               cv::Point2i max_intensity_point_new = cv::Point2i(max_intensity_point.x,i-1);
//               initialRoadProfile.push_back(max_intensity_point_new);
//           }
                initialRoadProfile.push_back(max_intensity_point);
            } else {
                r=0;
            }

            if (max_intensity>intensityThVDisPointForSlope)
                roadNotVisibleDisparity = max_intensity_point.y;
        }
    }

    int counter = 0;
    // if (initialRoadProfile.size()>1) {
    for(int i=1  ; i < initialRoadProfile.size(); i++){
        int validity = initialRoadProfile[i-1].y-initialRoadProfile[i].y;
        if (validity>=0) {
            counter = 0;
            if(validity<rdProfileColDistanceTh){
                refinedRoadProfile.push_back(initialRoadProfile[i]);
            } else {
                int search_row = initialRoadProfile[i].x;
                int search_col = initialRoadProfile[i-1].y;
                int max_intensity = v_disparity_map.at<uchar>(search_row,search_col);
                int max_intensity_disparity = search_col;
                for (int k=1 ; k < rdProfileColDistanceTh; k++){
                    int next_point = v_disparity_map.at<uchar>(search_row,search_col-k);
                    if (max_intensity<next_point){
                        max_intensity = next_point;
                        max_intensity_disparity = search_col-k ;
                    }
                }
                initialRoadProfile[i].y = max_intensity_disparity;
                refinedRoadProfile.push_back(initialRoadProfile[i]);
            }
        } else {
            initialRoadProfile[i].y=initialRoadProfile[i-1].y;
            refinedRoadProfile.push_back(initialRoadProfile[i]);
            counter ++;
        }

        if (counter >thHorizon) {
            // if (refinedRoadProfile[i].y == refinedRoadProfile[i-thHorizon].y) {
            i = initialRoadProfile.size();
            for(int j =0; j<counter;j++){
                refinedRoadProfile.pop_back();
            }
            // }
        }
    }
    // }
}

void ObstaclesDetection::DisplayRoad() {
    cv::imshow("road",road);
    cv::imshow("v_disparity_map",v_disparity_map);
    cv::imshow("u_disparity_map",u_disparity_map);
    cv::Mat intial_road_map, refined_road_map;
    cv::cvtColor(v_disparity_map, intial_road_map, CV_GRAY2RGB);
    cv::cvtColor(v_disparity_map, refined_road_map, CV_GRAY2RGB);

    for(int i=0  ; i < initialRoadProfile.size(); i++){
        intial_road_map.at<cv::Vec3b>(initialRoadProfile[i].x, initialRoadProfile[i].y) = cv::Vec3b(0,0,255);
    }
    for(int i=0  ; i < refinedRoadProfile.size(); i++){
        refined_road_map.at<cv::Vec3b>(refinedRoadProfile[i].x, refinedRoadProfile[i].y) = cv::Vec3b(0,0,255);
    }

    cv::imshow("intial_road_map", intial_road_map);
    cv::imshow("refined_road_map", refined_road_map);
    cv::imshow("disparity_map",disparity_map);
    cv::waitKey(1);
}

void ObstaclesDetection::Initiate(std::string camera_type, int disparity_size, double baseline){

//    std::cout<<camera_type<<", "<<disparity_size<<", "<<baseline<<std::endl;

    double minHeight =0.3;//0.4m
    disp_size = disparity_size;

    uDispThresh = static_cast<int *>(calloc(disp_size + 1, sizeof(int)));
    for( int i = 0; i < disp_size+1; ++i) {
        uDispThresh[i]=cvRound(minHeight*i/baseline); //Y*dx/B
    }

    if (camera_type == "long_camera") {
        rdRowToDisRegard = 10;
        rdStartCheckLines = 40;
        intensityThVDisPoint = 10;
        thHorizon = 10;
        rdProfileRowDistanceTh = 10;
        rdProfileColDistanceTh = 4;
        intensityThVDisPointForSlope = 100;
        pubName = "/long/map_msg";
    } else if (camera_type == "wide_camera") {
        rdRowToDisRegard = 30;
        rdStartCheckLines = 24;
        intensityThVDisPoint = 10;
        thHorizon = 20;
        rdProfileRowDistanceTh = 6;
        rdProfileColDistanceTh = 16;
        intensityThVDisPointForSlope = 100;
        pubName = "/wide/map_msg";
    }

}

void ObstaclesDetection::ExecuteDetection(cv::Mat disp_img){

    disp_img.copyTo(disparity_map);

    roadmap = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
    obstaclemap = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
    road = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
    v_disparity_map = cv::Mat(disparity_map.rows, disp_size, CV_8UC1, cv::Scalar::all(0));
    u_disparity_map = cv::Mat(disp_size, disparity_map.cols, CV_8UC1, cv::Scalar::all(0));

    initialRoadProfile.clear();
    refinedRoadProfile.clear();

    u_disparity_map = GenerateUDisparityMap(disparity_map);
    disparity_map.copyTo(road);
    RemoveObviousObstacles();
    GenerateVDisparityMap();
    RoadProfileCalculation();

//    DisplayRoad();

}