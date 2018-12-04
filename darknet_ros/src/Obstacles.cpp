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

void ObstaclesDetection::InitiateObstaclesMap () {
    for (int r=refinedRoadProfile[0].x; r<roadmap.rows; r++) {
        roadmap.row(r).setTo(cv::Scalar(1));
        obstaclemap.row(r).setTo(cv::Scalar(0));
//        undefinedmap.row(r).setTo(cv::Scalar(0));
    }

    int roadmap_row = 0;
    for(int i=0; i < refinedRoadProfile.size(); i++){
        dynamicLookUpTableRoad[refinedRoadProfile[i].y]=refinedRoadProfile[i].x;
        dynamicLookUpTableRoadProfile[refinedRoadProfile[i].x]=refinedRoadProfile[i].y;
        roadmap_row = refinedRoadProfile[i].x;
        int road_disparity = refinedRoadProfile[i].y ;//+ cvRound(roadProfile[i].x*0.05);
        int negativeObsLength =0;
        for (int c=0; c<roadmap.cols; c++) {
            int disparity_value = disparity_map.at<uchar>(roadmap_row,c);
//        if (disparity_value!=0){
            if (disparity_value<=road_disparity){
                roadmap.at<uchar>(roadmap_row,c) = 1;
                obstaclemap.at<uchar>(roadmap_row,c) = 0;
//                undefinedmap.at<uchar>(roadmap_row,c) = 0;
                if(disparity_value<road_disparity){// && disparity_value<30){ //-std::min(cvRound(road_disparity*0.3),2)
                    negativeObsLength++;
                } else {
                    if((c-negativeObsLength-1)>0) {
                        if (disparity_map.at<uchar>(roadmap_row, c-negativeObsLength-1) >= road_disparity) { //- std::min(cvRound(road_disparity * 0.3), 2)
                            for (int j = 1; j < negativeObsLength + 1; j++) {
                                negObsMap.at<uchar>(roadmap_row, c-j) = 1;
                            }
                        } else if (c==roadmap.cols-1){
                            for (int j = 1; j < negativeObsLength + 1; j++) {
                                negObsMap.at<uchar>(roadmap_row, c-j) = 1;
                            }
                        }
                    }
                    negativeObsLength =0;
                }
            } else if (disparity_value<(road_disparity+std::max(cvRound(road_disparity*0.3),2))) {
                negativeObsLength =0;
                if (obstaclemap.at<uchar>(roadmap_row,c)!=1){
                    //if(v_disparity_map.at<uchar>(roadmap_row,disparity_value)>20){
                    roadmap.at<uchar>(roadmap_row,c) = 1;
                    //}
                }
            } else {
                negativeObsLength =0;
                obstaclemap.at<uchar>(roadmap_row,c)=1;
            }
            //}
        }
    }

//    for (int i=disp_size-1; i>0; i--){
//        std::cout<<"i: "<<i<<", row: "<<obj.dynamicLookUpTableRoad[i]<<"; ";
//    }
//    std::cout<<std::endl;

    for (int i =refinedRoadProfile.back().y; i>=0; i--){
        dynamicLookUpTableRoad[i]=refinedRoadProfile.back().x;
    }

    for (int i =refinedRoadProfile[0].y; i<=disp_size; i++){
        dynamicLookUpTableRoad[i]=refinedRoadProfile[0].x;
    }

    for (int i=disparity_map.rows-1; i>refinedRoadProfile[0].x;  i--){
        dynamicLookUpTableRoadProfile[i]=refinedRoadProfile[0].y;
    }

    for (int i=disp_size-1; i>0; i--){
        if (dynamicLookUpTableRoad[i]==0){
            dynamicLookUpTableRoad[i] = dynamicLookUpTableRoad[i+1];
        }
    }

//    for (int i=disp_size-1; i>0; i--){
//        std::cout<<"i: "<<i<<", row: "<<obj.dynamicLookUpTableRoad[i]<<"; ";
//    }
//    std::cout<<std::endl;
//    for (int i=0; i<obj.max_height; i++){
//        std::cout<<"i: "<<i<<", dis: "<<obj.dynamicLookUpTableRoadProfile[i]<<"; ";
//    }
//    std::cout<<std::endl;

    road_starting_row = refinedRoadProfile.back().x;        //2D coordinated with respect to the region of interest defined from rectified left image
//    region_of_interest = cv::Rect(left_offset, road_starting_row, disparity_map.cols-left_offset-right_offset, disparity_map.rows-road_starting_row-bottom_offset);

    int compare_patch = 3;
    for (int c=1; c<roadmap.cols-1; c++) {
        int patch_length = 0;
        for (int r=roadmap.rows; r>roadmap_row; r--) {
            if(roadmap.at<uchar>(r,c) == 0 && r>roadmap_row+1){
                patch_length +=1;
            } else {
                if (patch_length<compare_patch){
                    if(r+patch_length+1<roadmap.rows){
                        if(roadmap.at<uchar>(r,c)==1 && roadmap.at<uchar>(r+patch_length+1,c)==1){
                            for (int i=0; i<patch_length; i++){
                                roadmap.at<uchar>(r+i+1,c) = 1;
                                obstaclemap.at<uchar>(r+i+1,c) = 0;
//                                undefinedmap.at<uchar>(r+i+1,c) = 0;
                            }
                        }
                    }
                }
                patch_length = 0;
            }
        }
    }

    for (int r=1; r<roadmap.rows-1; r++) {
        int patch_length = 0;
        for (int c=roadmap.cols-1; c>-1; c--) {
            if(roadmap.at<uchar>(r,c) == 0 && r>roadmap_row+1){
                patch_length +=1;
            } else {
                if (patch_length<compare_patch){
                    if(c+patch_length+1<roadmap.cols){
                        if(roadmap.at<uchar>(r,c)==1 && roadmap.at<uchar>(r,c+patch_length+1)==1){
                            for (int i=0; i<patch_length; i++){
                                roadmap.at<uchar>(r,c+i+1) = 1;
                                obstaclemap.at<uchar>(r,c+i+1) = 0;
//                                undefinedmap.at<uchar>(r,c+i+1) = 0;
                            }
                        }
                    }
                }
                patch_length = 0;
            }
        }
    }

    for (int r=0; r<disparity_map.rows; r++) {
        for (int c=0; c<disparity_map.cols; c++) {
            if (disparity_map.at<uchar>(r,c) > 0 && roadmap.at<uchar>(r,c)==0){
                obstaclemap.at<uchar>(r,c)=1;
            }
        }
    }

    for (int r=0; r<obstaclemap.rows; r++) {
        for (int c=0; c<obstaclemap.cols; c++) {
            if(obstaclemap.at<uchar>(r,c)==1){
                int dis = disparity_map.at<uchar>(r,c);
                if (abs(xDirectionPosition[c][dis])<widthOfInterest){ //xDirectionThresh
                    if (yDirectionPosition[r][dis]<heightOfInterest){ //yDirectionThresh
                        obstacleDisparityMap.at<uchar>(r,c) = static_cast<uchar>(dis);
                    }
                }
                //obstacleDisparityMap.at<uchar>(r,c) = dis;
            }
        }
    }

}

std::vector<int> searchElement (std::vector<std::vector<int> > iList, int element, std::vector<u_span> uList, bool visited[]) {
    std::vector<int> out;
    if(!visited[element]){//if(!uList[element].checked){
        visited[element] = true; //std::cout<<"element :"<<element<<std::endl;
        out.push_back(element);
        for(int i=0;i<iList[element].size();i++){
            std::vector<int> tmp = searchElement (iList,iList[element][i],uList,visited);
            for (int j=0; j<tmp.size();j++){
                out.push_back(tmp[j]);
            }
        }
    }
    return out;
}

std::vector<std::vector<u_span> > searchList (std::vector<u_span> uList, std::vector<std::vector<int> > iList, bool visited[]) {
    std::vector<std::vector<u_span> > output;
    for(int i=0;i<iList.size();i++){
        std::vector<u_span> tmp;
        if(!visited[i]){//if(!uList[i].checked){
            visited[i] = true; //std::cout<<" i :"<<i<<"iList[i][0] :"<<iList[i][0]<<std::endl;
            tmp.push_back(uList[i]);
            if(iList[i].size()>0){
                for (int j=0; j<iList[i].size();j++){
                    std::vector<int> list = searchElement(iList,iList[i][j],uList, visited);
                    for(int k=0;k<list.size();k++){
                        tmp.push_back(uList[list[k]]);
                    }
                }
            }
            if (tmp.size()>0){
                output.push_back(tmp);
            }

        }
    }
    return output;
}

void ObstaclesDetection::RefineObstaclesMap () {

    u_disparity_map_new = GenerateUDisparityMap(obstacleDisparityMap);
    u_thresh_map = cv::Mat::zeros(u_disparity_map_new.rows,u_disparity_map_new.cols, CV_8UC1);

    for (int r=0; r<u_disparity_map_new.rows; r++) {
        for (int c=0; c<u_disparity_map_new.cols; c++) {
            int u_disparity = u_disparity_map_new.at<uchar>(r,c);
            if(u_disparity>uDispThresh[r]){
                u_thresh_map.at<uchar>(r,c) = 1;
            } else if (u_disparity>uHysteresisLowThresh[r]){
                if (r>1 && c>1 && r<u_disparity_map_new.rows-1 && c<u_disparity_map_new.cols-1){
                    if(u_disparity_map_new.at<uchar>(r,c-1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r,c+1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r-1,c)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r+1,c)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r-1,c-1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r-1,c+1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r+1,c-1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if(u_disparity_map_new.at<uchar>(r+1,c+1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    }
                }
            }
        }
    }

    std::vector<std::vector<u_span> > USpanList;
    std::vector<u_span> USpanRowList;
    int id=0;

//    cv::Mat u_thresh_clor1;
//    cvtColor(u_thresh_map*255, u_thresh_clor1, CV_GRAY2RGB);

    for (int r=0; r<u_thresh_map.rows; r++) {
        int count_uspan=0;
        int count_uempty=0;
        for (int c=0; c<u_thresh_map.cols; c++) {
            if (u_thresh_map.at<uchar>(r,c)==1){
                count_uspan ++;
                count_uempty =0;
                if (c==u_thresh_map.cols-1){
                    if (count_uspan>2){
                        u_span uspan{};
                        uspan.u_pixels = count_uspan;
                        uspan.u_left =c-count_uspan;
                        uspan.u_right =c;
                        uspan.u_d = r;
                        uspan.checked = false;
                        uspan.ID = id;
                        USpanRowList.push_back(uspan);
//                        if(debugPosObs){
//                            cv::line( u_thresh_clor1, cv::Point(uspan.u_left, r), cv::Point(uspan.u_right, r), cv::Scalar( 0, 55, 255 ), 1, 4 );
//                        }
                        id ++;
                    }
                    count_uempty ++;
                    count_uspan =0;
                }
            } else {
                if (count_uspan>2){
                    u_span uspan{};
                    uspan.u_pixels = count_uspan;
                    uspan.u_left =c-count_uspan;
                    uspan.u_right =c;
                    uspan.u_d = r;
                    uspan.checked = false;
                    uspan.ID = id;
                    USpanRowList.push_back(uspan);
//                    if(debugPosObs){
//                        cv::line( u_thresh_clor1, cv::Point(uspan.u_left, r), cv::Point(uspan.u_right, r), cv::Scalar( 0, 55, 255 ), 1, 4 );
//                    }
                    id ++;
                }
                count_uempty ++;
                count_uspan =0;
            }
        }
    }

//    if(debugPosObs){
//        cv::imshow("u_thresh_clor1",u_thresh_clor1);
//    }

    std::vector<std::vector<int> > listOfCoreespondences;
    bool adjacencyMatrix[USpanRowList.size()][USpanRowList.size()];

    for (int i=0; i<USpanRowList.size();i++){
        for (int j=0; j<USpanRowList.size();j++){
            adjacencyMatrix[i][j]=false;
        }
    }

    for (int i=0; i<USpanRowList.size();i++){
        int ud = USpanRowList[i].u_d;
        int ul = USpanRowList[i].u_left;
        int ur = USpanRowList[i].u_right;
        int uNeigh = uXDirectionNeighbourhoodThresh[ud];
        int id2 = USpanRowList[i].ID;

        for (int j=0; j<USpanRowList.size();j++){
            int ud1 = USpanRowList[j].u_d;
            int ul1 = USpanRowList[j].u_left;
            int ur1 = USpanRowList[j].u_right;
            int id1 = USpanRowList[j].ID;
            if(ud1>=uDNeighbourhoodThresh[ud] && ud1<=ud){
                if((ul1>=ul-uNeigh && ul1 <= ur+uNeigh) || (ur1>=ul-uNeigh && ur1 <= ur+uNeigh)){
                    adjacencyMatrix[id2][id1]=true;
                    adjacencyMatrix[id1][id2]=true;
                    USpanRowList[j].checked = true;
                }
            }
        }
    }

    for (int j=0; j<USpanRowList.size();j++){
        std::vector<int> list;
        for (int i=0; i<USpanRowList.size();i++){
            if(adjacencyMatrix[j][i]){
                list.push_back(i);
            }
        }
        listOfCoreespondences.push_back(list);
    }

    auto *visited = new bool[USpanRowList.size()];
    for (int i = 0; i < USpanRowList.size(); i++)
        visited[i] = false;

    USpanList = searchList (USpanRowList, listOfCoreespondences, visited);

//    cv::Mat u_thresh_map_clr, test_output, finalOutput;
//    cv::cvtColor(left_rectified, test_output, CV_GRAY2RGB);
//    cv::cvtColor(left_rectified, finalOutput, CV_GRAY2RGB);
//    cv::cvtColor(u_thresh_map*255,u_thresh_map_clr,CV_GRAY2RGB);

    for (int i=0; i<USpanList.size();i++){
        int startCol = USpanList[i][0].u_left;
        int endCol = USpanList[i][0].u_right;
        int lowDisparity = USpanList[i][0].u_d;
        int highDisparity = USpanList[i][0].u_d;
        int endRow = dynamicLookUpTableRoad[highDisparity];
//        if(debugPosObs){
//            cv::Vec3b color( rand()&255, rand()&255, rand()&255 );
//        }
        for (int j=1; j<USpanList[i].size();j++){
            if(startCol>USpanList[i][j].u_left){
                startCol = USpanList[i][j].u_left;
            }
            if(endCol<USpanList[i][j].u_right){
                endCol = USpanList[i][j].u_right;
            }
            if (lowDisparity>USpanList[i][j].u_d){
                lowDisparity=USpanList[i][j].u_d;
            }
            if(highDisparity<USpanList[i][j].u_d){
                highDisparity=USpanList[i][j].u_d;
                endRow = dynamicLookUpTableRoad[highDisparity];
            }
        }

        int searchStartRow = endRow;
        int startRow = endRow;

//        if(debugPosObs){
//            cv::rectangle( u_thresh_map_clr, cv::Point( startCol, lowDisparity ), cv::Point( endCol, highDisparity), cv::Scalar( 0, 55, 255 ), 1, 4 );
//            std::cout<<"startCol, endCol: "<<startCol<<", "<<endCol<<"; lowDisparity, highDisparity"<<lowDisparity<<", "<<highDisparity<<"; "<<endRow<<std::endl;
//            cv::rectangle(u_thresh_map_clr,cv::Point(startCol,startRow),cv::Point(endCol,endRow),cv::Scalar(0,55,255),1,4);
//            cv::waitKey(0);
//        }

        int end =0;
        bool obstacleDetected = false;
        int no_of_pixels = 0;
        std::vector<cv::Point2i> points;
        int obj_disparity = 0;

        bool obsPointsFound = false;
        for (int r = searchStartRow; r > 0; r--) {
            for (int c=startCol; c<endCol; c++) {
                //if ((int) obstacleDisparityMap.at<uchar>(r,c) != 0){
                int obsMap_disparity = (int) obstacleDisparityMap.at<uchar>(r,c);
                if (obsMap_disparity>=lowDisparity) {
                    if (obsMap_disparity <= highDisparity) {
                        obsPointsFound = true;
                        r = 0;
                    }
                }
            }
            if(!obsPointsFound){
//                std::cout<<"new endRow : "<<r<<", old endRow :"<<endRow;
                endRow = r;
            }
        }

        for (int c=startCol; c<endCol; c++) {
            for (int r = searchStartRow; r > 0; r--) {
                int obsMap_disparity = (int) obstacleDisparityMap.at<uchar>(r,c);
                if (obsMap_disparity>=lowDisparity) {
                    if (obsMap_disparity <= highDisparity) {
                        if (obsMap_disparity > obj_disparity) {
                            obj_disparity = obsMap_disparity;
                        }
                        if(r<startRow){
                            startRow=r;
                        }
                        points.push_back(cv::Point2i(r,c));
                        no_of_pixels ++;
                        obstacleDetected = true;
                    }
                }
            }
        }

        if(no_of_pixels>minNoOfPixelsForObject){

//            std::vector<float> hog_feature;

//            cv::Rect_<int> rect_obs = cv::Rect_<int>(startCol + leftShift, startRow + topShift, endCol - startCol, endRow - startRow);
            cv::Rect_<int> rect_blob = cv::Rect_<int>(startCol, startRow, endCol - startCol, endRow - startRow);
//            cv::Mat roi = left_rectified(rect_obs).clone();
////            cv::imshow("roi",roi);
////            cv::waitKey(0);
//            //cv::GaussianBlur(roi, roi, cv::Size(3, 3), 1.0);
//            cv::resize(roi, roi, cv::Size(22, 22));     // 12-bin color histogram in mono color space (1 channel)

//            obj.hog_descriptor -> computeHOG(hog_feature, roi);

//            std::cout<<hog_feature.size()<<std::endl;

            obsBlobs obs;   //obstacle obs;
            obs.currentBoundingRect = rect_blob;
            obs.noOfPixels = no_of_pixels;
            obs.obsPoints = points;
            obs.max_disparity = obj_disparity;
            obs.depth = depthTable[obj_disparity];
            obs.startX = xDirectionPosition[startCol][obj_disparity];
            obs.endX = xDirectionPosition[endCol][obj_disparity];
            obs.startY = yDirectionPosition[startRow][obj_disparity];
            obs.endY = yDirectionPosition[endRow][obj_disparity];
//            obs.obsHog = hog_feature;

            currentFrameObsBlobs.push_back(obs);
//            if(debugPosObs){
//                for (long int q = 0; q < points.size(); q++) {
//                    test_output.at<cv::Vec3b>(points[q].x, points[q].y)=cv::Vec3b(0,0,255);
//                }
//                cv::rectangle( finalOutput, cv::Point( startCol, startRow ), cv::Point( endCol, endRow), cv::Scalar( 0, 55, 255 ), 1, 4 );
//                cv::rectangle( u_thresh_map_clr, cv::Point( startCol, lowDisparity ), cv::Point( endCol, highDisparity), cv::Scalar( 0, 255, 255 ), 1, 4 );
//                cv::imshow("finalOutput123", finalOutput);
//            }

//        Blob possibleBlob(cv::Rect(startCol, startRow, endCol-startCol, endRow-startRow));
//        if (possibleBlob.currentBoundingRect.area() > 2000 &&
//            possibleBlob.dblCurrentAspectRatio >= 0.2 &&
//            possibleBlob.dblCurrentAspectRatio <= 1.25 &&
//            possibleBlob.currentBoundingRect.width > 20 &&
//            possibleBlob.currentBoundingRect.height > 20 &&
//            possibleBlob.dblCurrentDiagonalSize > 30.0) {
//            //&&(cv::contourArea(possibleBlob.currentContour) / (double) possibleBlob.currentBoundingRect.area()) > 0.40) {
//            currentFrameBlobs.push_back(possibleBlob);
//        }
        }
    }

//    if(debugPosObs){
//        cv::imshow("u_thresh_map_clr",u_thresh_map_clr);
//        cv::imshow("test_output",test_output);
//        std::cout << "currentFrameBlobs " <<currentFrameBlobs.size()<< std::endl;
//    }
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

void ObstaclesDetection::DisplayPosObs() {
    cv::imshow("obstacleDisparityMap", obstacleDisparityMap*255/disp_size);
    cv::Mat posObsOutput;
    cv::cvtColor(obstaclemap*255, posObsOutput, CV_GRAY2RGB);

//    std::cout << "Completed GenerateObstaclesMap function in posObstacles.cpp" << std::endl;
    for (long int i = 0; i < currentFrameObsBlobs.size(); i++) {
        cv::rectangle(posObsOutput, currentFrameObsBlobs[i].currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
        for(int j=0; j<currentFrameObsBlobs[i].obsPoints.size();j++){
            posObsOutput.at<cv::Vec3b>(currentFrameObsBlobs[i].obsPoints[j].x, currentFrameObsBlobs[i].obsPoints[j].y)[2]=255;//cv::Vec3b(0,0,255);
        }
        std::ostringstream str;
        str << depthTable[currentFrameObsBlobs[i].max_disparity]<<"m";
        cv::putText(posObsOutput, str.str(), cv::Point(currentFrameObsBlobs[i].currentBoundingRect.x,currentFrameObsBlobs[i].currentBoundingRect.y+12),
                CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(0,250,0));
    }
    cv::imshow("posObsOutput", posObsOutput);
    cv::imshow("obstaclemap",obstaclemap*255);
    cv::imshow("roadmap",roadmap*255);
    cv::waitKey(1);
}

void ObstaclesDetection::Initiate(std::string camera_type, int disparity_size, double baseline,
        double u0, double v0, double focal, int Width, int Height){

//    std::cout<<camera_type<<", "<<disparity_size<<", "<<baseline<<std::endl;

    double minHeight =0.3;//0.4m
    double uHysteresisThreshRatio = 0.7;
    double minWidthToSeperate = 0.5; //0.5m
    double minDepthToSeperate = 6; //6m
    disp_size = disparity_size;

    uDispThresh = static_cast<int *>(calloc(disp_size + 1, sizeof(int)));
    for( int i = 0; i < disp_size+1; ++i) {
        uDispThresh[i]=cvRound(minHeight*i/baseline); //Y*dx/B
    }

    int ii;
    xDirectionPosition = static_cast<double **>(calloc(Width, sizeof(double *)));
    for(ii = 0; ii < Width; ii++)
        xDirectionPosition[ii] = static_cast<double *>(calloc(disp_size + 1, sizeof(double)));
    for (int r=0; r<Width; r++) {
        xDirectionPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            xDirectionPosition[r][c]=(r-u0)*baseline/c;
//        std::cout<<xDirectionPosition[r][c]<<std::endl;
        }
    }

    yDirectionPosition = static_cast<double **>(calloc(Height, sizeof(double *)));
    for(ii = 0; ii < Height; ii++)
        yDirectionPosition[ii] = static_cast<double *>(calloc(disp_size + 1, sizeof(double)));
    for (int r=0; r<Height; r++) {
//    for (int r=300; r<301; r++) {
        yDirectionPosition[r][0]=0;
        for (int c=1; c<disp_size+1; c++) {
            yDirectionPosition[r][c]=(v0-r)*baseline/c;
//      std::cout<<r<<", "<<c<<": "<<yDirectionPosition[r][c]<<"; ";//std::endl;
        }
    }

    depthTable = static_cast<double *>(calloc(disp_size + 1, sizeof(double)));
    depthTable[0] =0;
    for( int i = 1; i < disp_size+1; ++i){
        depthTable[i]=focal*baseline/i; //Y*dx/B
//      std::cout<<"i: "<<i<<", "<<depthTable[i]<<"; \n";
    }

    uHysteresisLowThresh = static_cast<int *>(calloc(disp_size + 1, sizeof(int)));
    for( int i = 0; i < disp_size+1; ++i) {
        uHysteresisLowThresh[i]=cvRound(uHysteresisThreshRatio*minHeight*i/baseline);
    }

    uXDirectionNeighbourhoodThresh = static_cast<int *>(calloc(disp_size + 1, sizeof(int)));
    for( int i = 0; i < disp_size+1; ++i) {
        uXDirectionNeighbourhoodThresh[i]=cvRound(minWidthToSeperate*i/baseline);
    }

    uDNeighbourhoodThresh = static_cast<int *>(calloc(disp_size + 1, sizeof(int)));
    for( int i = 0; i < disp_size+1; ++i) {
        uDNeighbourhoodThresh[i]=cvRound(focal*baseline*i/(focal*baseline+minDepthToSeperate*i));
    }

    dynamicLookUpTableRoad = static_cast<int *>(calloc(disp_size + 1, sizeof(int)));
    dynamicLookUpTableRoadProfile = static_cast<int *>(calloc(Height, sizeof(int)));

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

    int minNoOfPointsForRdProfile =50;
    if (refinedRoadProfile.size()>minNoOfPointsForRdProfile){

        obstacleDisparityMap = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
        negObsMap = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
        currentFrameObsBlobs.clear();

        memset(dynamicLookUpTableRoad, 0, sizeof(dynamicLookUpTableRoad));
        memset(dynamicLookUpTableRoadProfile, 0, sizeof(dynamicLookUpTableRoadProfile));

        InitiateObstaclesMap();
        RefineObstaclesMap();

//        DisplayPosObs();
    }

}