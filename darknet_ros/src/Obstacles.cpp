#include "darknet_ros/Obstacles.h"//"Hungarian.h"

ObstaclesDetection::ObstaclesDetection(){}

ObstaclesDetection::~ObstaclesDetection(){}

cv::Mat ObstaclesDetection::GenerateUDisparityMap (cv::Mat disp_map) {
    cv::Mat u_disp_map(disp_size+1, disp_map.cols, CV_8UC1, cv::Scalar::all(0));
    for (int v = 0; v < disp_map.cols; v++) {
        for (int u = 0; u < disp_map.rows; u++) {
            short d = (int)disp_map.at<uchar>(u, v);
            if (d > 0 &&  d < disp_size+1) {
                if((int)u_disp_map.at<uchar>(d,v) < 254)
                    u_disp_map.at<uchar>(d,v) +=1;
            }
        }
    }
    return u_disp_map;
}

void ObstaclesDetection::RemoveObviousObstacles () {
    for (int c=0; c<u_disparity_map.cols; c++) {
        for (int r=0; r<u_disparity_map.rows; r++) {
            if ((int)u_disparity_map.at<uchar>(r,c)>uDispThresh[r]){//u_disparity_thresh){
                for (int j=0; j<road.rows; j++){
                    if ((int)road.at<uchar>(j,c)==r){
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
            short d = (int)road.at<uchar>(v, u);
            if (d > 0 && d < disp_size+1) {
                if((int)v_disparity_map.at<uchar>(v,d) < 254)
                    v_disparity_map.at<uchar>(v,d) +=1;
            }
        }
    }
}

void ObstaclesDetection::RoadProfileCalculation () {
    int road_start_intensity = 0;//(int)v_disparity_map.at<uchar>(v_disparity_map.rows-1,0);//401
    int road_start_row = v_disparity_map.rows-1;//401
    int road_start_disparity = 0;
//    std::cout<<v_disparity_map.rows-rdRowToDisRegard<<", "<<v_disparity_map.rows-rdStartCheckLines<<std::endl;
    for (int r=(v_disparity_map.rows-rdRowToDisRegard); r>(v_disparity_map.rows-(rdStartCheckLines+rdRowToDisRegard)); r--) { //10,18
        for (int c=0; c<v_disparity_map.cols; c++) {
            if (road_start_intensity<(int)v_disparity_map.at<uchar>(r,c)){
                road_start_intensity = (int)v_disparity_map.at<uchar>(r,c);
                road_start_row = r;
                road_start_disparity = c;
            }
        }
    }

//    std::cout<<road_start_row<<", "<<road_start_disparity<<std::endl;
    initialRoadProfile.push_back(cv::Point2i(road_start_row, road_start_disparity));
    refinedRoadProfile.push_back(cv::Point2i(road_start_row, road_start_disparity));

    for (int r=initialRoadProfile[0].x-1; r>0; r--) {
        int max_intensity = (int)v_disparity_map.at<uchar>(r,0);
        cv::Point2i max_intensity_point(r,0);
        for (int c=1; c<v_disparity_map.cols; c++) {
            if (max_intensity<(int)v_disparity_map.at<uchar>(r,c)){
                max_intensity = (int)v_disparity_map.at<uchar>(r,c);
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
                int max_intensity = (int)v_disparity_map.at<uchar>(search_row,search_col);
                int max_intensity_disparity = search_col;
                for (int k=1 ; k < rdProfileColDistanceTh; k++){
                    int next_point = (int)v_disparity_map.at<uchar>(search_row,search_col-k);
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
            int disparity_value = (int)disparity_map.at<uchar>(roadmap_row,c);
//        if (disparity_value!=0){
            if (disparity_value<=road_disparity){
                roadmap.at<uchar>(roadmap_row,c) = 1;
                obstaclemap.at<uchar>(roadmap_row,c) = 0;
//                undefinedmap.at<uchar>(roadmap_row,c) = 0;
                if(disparity_value<road_disparity){// && disparity_value<30){ //-std::min(cvRound(road_disparity*0.3),2)
                    negativeObsLength++;
                } else {
                    if((c-negativeObsLength-1)>0) {
                        if ((int)disparity_map.at<uchar>(roadmap_row, c-negativeObsLength-1) >= road_disparity) { //- std::min(cvRound(road_disparity * 0.3), 2)
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
                if ((int)obstaclemap.at<uchar>(roadmap_row,c)!=1){
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
    for (int c = 1; c < roadmap.cols - 1; c++) {
        int patch_length = 0;
        for (int r = roadmap.rows; r > roadmap_row; r--) {
            if((int)roadmap.at<uchar>(r,c) == 0 && r > roadmap_row + 1) {
                patch_length +=1;
            } else {
                if (patch_length < compare_patch){
                    if(r + patch_length + 1 < roadmap.rows){
                        if((int)roadmap.at<uchar>(r, c)==1 && (int)roadmap.at<uchar>(r + patch_length+1, c)==1){
                            for (int i = 0; i < patch_length; i++){
                                roadmap.at<uchar>(r + i + 1, c) = 1;
                                obstaclemap.at<uchar>(r + i + 1,c) = 0;
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
            if((int)roadmap.at<uchar>(r,c) == 0 && r>roadmap_row+1){
                patch_length +=1;
            } else {
                if (patch_length<compare_patch){
                    if(c+patch_length+1<roadmap.cols){
                        if((int)roadmap.at<uchar>(r,c)==1 && (int)roadmap.at<uchar>(r,c+patch_length+1)==1){
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
            if ((int)disparity_map.at<uchar>(r,c) > 0 && (int)roadmap.at<uchar>(r,c)==0){
                obstaclemap.at<uchar>(r,c)=1;
            }
        }
    }

    for (int r=0; r<obstaclemap.rows; r++) {
        for (int c=0; c<obstaclemap.cols; c++) {
            if((int)obstaclemap.at<uchar>(r,c)==1){
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
            int u_disparity = (int)u_disparity_map_new.at<uchar>(r,c);
            if(u_disparity>uDispThresh[r]){
                u_thresh_map.at<uchar>(r,c) = 1;
            } else if (u_disparity>uHysteresisLowThresh[r]){
                if (r>1 && c>1 && r<u_disparity_map_new.rows-1 && c<u_disparity_map_new.cols-1){
                    if((int)u_disparity_map_new.at<uchar>(r,c-1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r,c+1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r-1,c)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r+1,c)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r-1,c-1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r-1,c+1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r+1,c-1)>uDispThresh[r]){
                        u_thresh_map.at<uchar>(r,c) = 1;
                    } else if((int)u_disparity_map_new.at<uchar>(r+1,c+1)>uDispThresh[r]){
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
            if ((int)u_thresh_map.at<uchar>(r,c)==1){
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

bool ObstaclesDetection::colinear() {

//    cv::Point2i a = randomRoadPoints2D[selectedIndexes[0]];
//    cv::Point2i b = randomRoadPoints2D[selectedIndexes[1]];
//    cv::Point2i c = randomRoadPoints2D[selectedIndexes[2]];
    cv::Vec3d a = randomRoadPoints[selectedIndexes[0]];
    cv::Vec3d b = randomRoadPoints[selectedIndexes[1]];
    cv::Vec3d c = randomRoadPoints[selectedIndexes[2]];

    double len_ab = norm(a-b);
    double len_ac = norm(a-c);
    double len_bc = norm(b-c);

    bool val = false;

    if (len_ab * 1.2 > (len_ac + len_bc))
        val = true;
    if (len_bc * 1.2 > (len_ac + len_ab))
        val = true;
    if (len_ac * 1.2 > (len_bc + len_ab))
        val = true;

//    for (int i=0;i<3;i++){
//        cv::circle(left_rect_clr,randomRoadPoints2D[selectedIndexes[i]],3,cv::Scalar(0, 0, 255),2);
//        cv::line(left_rect_clr, randomRoadPoints2D[selectedIndexes[i]], randomRoadPoints2D[selectedIndexes[(i+1)%3]], cv::Scalar(255, 255, 255), 2);
//    }

//    if (len_ab>len_ac) {
//        if (len_ab > len_bc) {
//            if (len_ab * 1.1 > (len_ac + len_bc))
//                val = true;
//        } else {
//            if (len_bc * 1.1 > (len_ac + len_ab))
//                val = true;
//        }
//    } else if (len_bc>len_ac){
//        if (len_bc * 1.1 > (len_ac + len_ab))
//            val = true;
//    } else {
//        if (len_ac * 1.1 > (len_bc + len_ab))
//            val = true;
//    }
//    ((y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2))
//    (y3 - y2)/(x3 - x2) = (y2 - y1)/(x2 - x1)
//    double val =0.0;
//    if((c.x-b.x)!=0 && (b.x-a.x)!=0)
//        val = fabs((double)(c.y-b.y)/(c.x-b.x) - (double)(b.y-a.y)/(b.x-a.x));

    return val;

}

void ObstaclesDetection::SurfaceNormal (){
    cv::Vec3d a = randomRoadPoints[selectedIndexes[0]];
    cv::Vec3d b = randomRoadPoints[selectedIndexes[1]];
    cv::Vec3d c = randomRoadPoints[selectedIndexes[2]];


//    std::cout<<"a: "<<a<<", b:"<<b<<", c:"<<c<<std::endl;
//    std::cout<<"a-b: "<<a-b<<", b-c:"<<b-c<<std::endl;

//    if(a[1]>-100.0 or b[1]>-100.0 or c[1]>-100.0)
//        std::cout<<"a: "<<a<<", b:"<<b<<", c:"<<c<<std::endl;



    surfaceN = (a-b).cross((b-c));

//    if(norm(surfaceN)>0) {
//        surfaceN = surfaceN / norm(surfaceN);
//        cv::Vec3d hori_N = cv::Vec3d(0.0,0.0,1.0);
//        cv::Vec3d hori_N = cv::Vec3d(0.0, 1.0, 0.0);
//        double angle = acos(fabs(surfaceN.dot(hori_N)));//* 180.0 / 3.14159265;
//        if(angle * 180.0 / 3.14159265 > 5){
//            std::cout << angle << ", " << angle * 180.0 / 3.14159265 << std::endl;
//            std::cout<<"a: "<<a<<", b:"<<b<<", c:"<<c<<std::endl;
//            std::cout<<"colinearity: "<<colinear()<<std::endl;

//            for (int i=0;i<3;i++){
//                cv::circle(output,randomRoadPoints2D[selectedIndexes[i]],3,cv::Scalar(0, 0, 255),2);
//                cv::line(output, randomRoadPoints2D[selectedIndexes[i]], randomRoadPoints2D[selectedIndexes[(i+1)%3]], cv::Scalar(255, 0, 0), 2);
//            }

//        }
//    }
//    std::cout<<"surfaceN: "<<surfaceN<<std::endl;
//    cv::Vec3d noN;// = N/norm(N);
//    double angle =0.0;//=acos(fabs(noN.dot(hori_N)))* 180.0 / 3.14159265;

//    if(norm(N)>0){
//        noN = N/norm(N);
//        angle =acos(fabs(noN.dot(hori_N)))* 180.0 / 3.14159265;
//    }
//    N = N*10;
//    std::cout<<"N: "<<N<<", "<<a<<", "<<b<<", "<<c<<", "<<norm(N)<<", "<<norm(hori_N)<<", "<<N.dot(hori_N)<<", "<<angle<<std::endl;
//    std::vector<cv::Vec3d> worldPoints;
//    worldPoints.push_back(N);//(cv::Point3d(N[0],N[1],N[2]));
//    std::vector<cv::Vec2d> imgPoints;
//    cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
//    rVec.at<double>(0) = -3.9277902400761393e-002;
//    rVec.at<double>(1) = 3.7803824407602084e-002;
//    rVec.at<double>(2) = 2.6445674487856268e-002;

//    cv::Mat tVec(3, 1, cv::DataType<double>::type); // Translation vector
//    tVec.at<double>(0) = t1.at<double>(0,0);
//    tVec.at<double>(1) = t1.at<double>(1,0);
//    tVec.at<double>(2) = t1.at<double>(2,0);
//    tVec.at<double>(0) = 2.1158489381208221e+000;
//    tVec.at<double>(1) = -7.6847683212704716e+000;
//    tVec.at<double>(2) = 2.6169795190294256e+001;
//    cv::projectPoints(worldPoints, r1,tVec,k1,d1,imgPoints);
//    std::cout<<"N: "<<N<<", "<<imgPoints[0]<<std::endl;
//    std::cout<<tVec<<std::endl;
//    std::cout<<r1<<std::endl;

//    cv::ellipse(output, cv::Point(output.cols/2,output.rows*5/6),cv::Size(8,16),angle,0,360,cv::Scalar( 255, 0, 0 ), CV_FILLED );

    for (int i=0;i<3;i++){
        cv::circle(left_rect_clr,randomRoadPoints2D[selectedIndexes[i]],3,cv::Scalar(0, 0, 255),2);
        cv::line(left_rect_clr, randomRoadPoints2D[selectedIndexes[i]], randomRoadPoints2D[selectedIndexes[(i+1)%3]], cv::Scalar(255, 0, 0), 2);
    }

//    sameWindow.spin();
//    sameWindow.spinOnce(1, true);
//    while(!sameWindow.wasStopped())
//    {
//        sameWindow.spinOnce(1, true);
//    }
}

void ObstaclesDetection::SurfaceNormalCalculation () {
    int size = static_cast<int>(randomRoadPoints.size());
    if(size==3){
        for (int i=0;i<3;i++)
            selectedIndexes.push_back(i);
        if (!colinear())
            SurfaceNormal();
    } else if (size>3) {
//        selectedIndexes.push_back(0);
//        selectedIndexes.push_back((int)size/2);
//        selectedIndexes.push_back(size-1);
        for (int jj=0;jj<size-1;jj++){
            if (jj==(int)size/2)
                continue;
            selectedIndexes.clear();
            selectedIndexes.push_back(jj);
            selectedIndexes.push_back((int)size/2);
            selectedIndexes.push_back(size-1);
            if (!colinear()) {
                SurfaceNormal();
                break;
            }
        }


//        if (colinear()){
//            selectedIndexes.clear();
//            selectedIndexes.push_back(1);
//            selectedIndexes.push_back((int)size/2);
//            selectedIndexes.push_back(size-1);
//            if (!colinear())
//                SurfaceNormal();
//        } else {
//            SurfaceNormal();
//        }
    }

    randomRoadPoints.clear();
    randomRoadPoints2D.clear();
    selectedIndexes.clear();


}

void ObstaclesDetection::RoadSlopeCalculation () {

    slope_map = cv::Mat::zeros(heightForSlope/2,depthForSlpoe*zResolutionForSlopeMap+250, CV_8UC3);
    int rowForDisparity35 = dynamicLookUpTableRoad[disForSlope];
//        int inRdHeight = 150;//cm
    int roadStartDis = refinedRoadProfile[0].y;
    double inRdHeight = abs(yDirectionPosition[dynamicLookUpTableRoad[disForSlopeStart]][disForSlopeStart] * 100);

    cv::putText(slope_map, "Relative Slope at;", cv::Point(slope_map.cols-180, 40), CV_FONT_HERSHEY_PLAIN, 0.6,
                CV_RGB(0, 250, 0));

//        std::cout<<"roadStartDis: "<<roadStartDis<<"; disForSlopeStart: "<< disForSlopeStart <<"; inRdHeight: "<<inRdHeight<<std::endl;
    int maxDispForSlopeStart = disForSlopeStart;
    if (disForSlopeStart > roadStartDis) {
        inRdHeight = abs(yDirectionPosition[dynamicLookUpTableRoad[roadStartDis]][roadStartDis] * 100);
        int currentY = slope_map.rows / 2;// - (int) ((1.5+yDirectionPosition[dynamicLookUpTableRoad[roadStartDis]+topShift][roadStartDis])*100/2);
        for (int d = disForSlopeStart; d > roadStartDis; d--) {
            int currentZ = (int) (depthTable[d] * 100 / zResolutionForSlopeMap);
            for (int c = slope_map.rows - 1; c > currentY; c--) {
                slope_map.at<cv::Vec3b>(c, currentZ) = cv::Vec3b(0, 255, 0);
            }
        }
        maxDispForSlopeStart = roadStartDis;
    }

    int minDispForSlope = disForSlope;
    if (roadNotVisibleDisparity > disForSlope) {
        minDispForSlope = roadNotVisibleDisparity;
    }

    double startDepth = depthTable[maxDispForSlopeStart] * 100;
    double point1Z = startDepth;
    int point1R = dynamicLookUpTableRoad[maxDispForSlopeStart];
    double point1Y = yDirectionPosition[point1R][maxDispForSlopeStart] * 100;
    cv::line(left_rect_clr, cv::Point(0, point1R), cv::Point(left_rect_clr.cols - 1, point1R), cv::Scalar(0, 200, 0), 2);
    int rowDiff = 0;
    int rowDiff2 = 10;
    int rightLeft = 230;

//        std::cout<<"point1Y: "<<point1Y<<", inRdHeight: "<<inRdHeight<<", pointY: "<<point1Y-((point1Z-startDepth)*slopeAdjHeight/slopeAdjLength)<<std::endl;

    bool toggle = true;
    for (int d = maxDispForSlopeStart; d > minDispForSlope; d--) {
        double pointZ = depthTable[d] * 100;
        int pointR = dynamicLookUpTableRoad[d];
        double pointY = yDirectionPosition[pointR][d]*100-((pointZ-startDepth)*slopeAdjHeight/slopeAdjLength);
        int currentZ = (int) (pointZ / zResolutionForSlopeMap);
        int currentY = slope_map.rows / 2 - (int) ((inRdHeight + pointY) / yResolutionForSlopeMap);

//        if(d%2 == 0){
        if(toggle){
            bool criteria = false;
            double pointX = 0.0;
            int selectedCol = roadmap.cols*0.4;
            for (int c=selectedCol;c<roadmap.cols-25;c++){
                int rdDis = (int)disparity_map.at<uchar>(pointR,c);
                if((int)roadmap.at<uchar>(pointR,c)==1 && rdDis==d){
                    criteria = true;
                    if (c>selectedCol){
                        selectedCol = c;
                        pointX = xDirectionPosition[c][d]*100;
                    }
//                    c = roadmap.cols;
                }
            }
            randomRoadPoints.push_back(cv::Vec3d(pointX,pointY,pointZ));
            randomRoadPoints2D.push_back(cv::Point2i(selectedCol,pointR));
            toggle = false;
        } else {
            bool criteria = false;
            double pointX = 0.0;
            int selectedCol = roadmap.cols*0.8;
            for (int c=roadmap.cols*0.8;c>50;c--){
                int rdDis = (int)disparity_map.at<uchar>(pointR,c);
                if((int)roadmap.at<uchar>(pointR,c)==1 && rdDis==d){
                    criteria = true;
                    if (c<selectedCol){
                        selectedCol = c;
                        pointX = xDirectionPosition[c][d]*100;
                    }
//                    c=0;
                }
            }
            randomRoadPoints.push_back(cv::Vec3d(pointX,pointY,pointZ));
            randomRoadPoints2D.push_back(cv::Point2i(selectedCol,pointR));
            toggle = true;
        }

        if (currentY < slope_map.rows && currentZ < slope_map.cols-250) {
            for (int c = slope_map.rows - 1; c > currentY; c--) {
                slope_map.at<cv::Vec3b>(c, currentZ) = cv::Vec3b(0, 255, 0);
            }

            if (rowDiff2>(slope_map.rows-80)){
                rowDiff2 = 10;
                rightLeft = 115;
            }
            if (point1Z<pointZ) {
                double slope = (cvRound(10*atan((pointY - point1Y) / (pointZ - point1Z)) * 180 / 3.14159265))/10;
                std::ostringstream strSlope;
                strSlope << (pointZ / 100) << "m: " << slope << " deg";//<<road_height;
                cv::putText(slope_map, strSlope.str(), cv::Point(slope_map.cols-rightLeft, 40 + rowDiff2), CV_FONT_HERSHEY_PLAIN, 0.6,
                            CV_RGB(0, 250, 0));
                rowDiff2 = rowDiff2 + 10;
            }
        }


        if ((pointZ - point1Z) > minDepthDiffToCalculateSlope) {
            double slope = (cvRound(10*atan((pointY - point1Y) / (pointZ - point1Z)) * 180 / 3.14159265))/10;
            std::ostringstream strSlope;
            strSlope << (int) (point1Z / 100) << "m to " << (int) (pointZ / 100) << "m: " << slope
                     << " deg";//<<road_height;
            cv::putText(slope_map, strSlope.str(), cv::Point(50, 40 + rowDiff), CV_FONT_HERSHEY_PLAIN, 0.6,
                        CV_RGB(0, 250, 0));
            rowDiff = rowDiff + 10;
//            cv::line(slopeOutput, cv::Point(0, pointR), cv::Point(slopeOutput.cols - 1, pointR), cv::Scalar(0, 200, 0), 2);
            cv::line(left_rect_clr, cv::Point(0, pointR), cv::Point(left_rect_clr.cols - 1, pointR), cv::Scalar(0, 200, 0), 2);

//            if (!imuDetected)
                SurfaceNormalCalculation();

            if(norm(surfaceN)>0){
                surfaceN = surfaceN/norm(surfaceN);
//        cv::Vec3d hori_N = cv::Vec3d(0.0,0.0,1.0);
                cv::Vec3d hori_N = cv::Vec3d(0.0,1.0,0.0);
                double angle =acos(fabs(surfaceN.dot(hori_N)));//* 180.0 / 3.14159265;
//        std::cout<<angle<<", "<<angle*180.0/3.14159265<<std::endl;

                /* Rotation using rodrigues */
                cv::Mat rot_vec = cv::Mat::zeros(1,3,CV_32F);
                rot_vec.at<float>(0,0) = (float) angle;
                rot_vec.at<float>(0,1) = 0.0;//(float) yaw;
                rot_vec.at<float>(0,2) = 0.0;

//    std::cout<<roll<<", "<<pitch<<", "<<yaw<<std::endl;

                cv::Mat rot_mat;
                cv::Rodrigues(rot_vec, rot_mat);

                /// Construct pose
                cv::Affine3f pose(rot_mat, cv::Vec3f(0.0,0.0,0.0));

                std::ostringstream strRollPitch;
                strRollPitch << "R. R. Pitch: "<<((int) (1000*angle*180.0/3.14159265))/1000.0;
                cv::putText(slope_map, strRollPitch.str(), cv::Point(50, 40 + rowDiff), CV_FONT_HERSHEY_PLAIN, 0.6,
                            CV_RGB(0, 250, 0));

            } else {
                std::ostringstream strRollPitch;
                strRollPitch << "R. R. Pitch: "<<0.0;
                cv::putText(slope_map, strRollPitch.str(), cv::Point(50, 40 + rowDiff), CV_FONT_HERSHEY_PLAIN, 0.6,
                            CV_RGB(0, 250, 0));
            }

            point1Z = pointZ;
            point1Y = pointY;
            rowDiff = rowDiff + 10;

        } else if (d == minDispForSlope + 1) {
            double slope = (cvRound(10*atan((pointY - point1Y) / (pointZ - point1Z)) * 180 / 3.14159265))/10;
            std::ostringstream strSlope;
            strSlope << (int) (point1Z / 100) << "m to " << (int) (pointZ / 100) << "m: " << slope
                     << " deg";//<<road_height;
            cv::putText(slope_map, strSlope.str(), cv::Point(50, 40 + rowDiff), CV_FONT_HERSHEY_PLAIN, 0.6,
                        CV_RGB(0, 250, 0));
//            cv::line(slopeOutput, cv::Point(0, pointR), cv::Point(slopeOutput.cols - 1, pointR), cv::Scalar(0, 200, 0), 2);
            cv::line(left_rect_clr, cv::Point(0, pointR), cv::Point(left_rect_clr.cols - 1, pointR), cv::Scalar(0, 200, 0), 2);

        }
    }

    cv::putText(slope_map, "Slope w.r.t. camera plane:", cv::Point(50,30), CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(0,250,0));
    for (int c=50;c<slope_map.cols;c=c+50){
        std::ostringstream str;
        str << c*zResolutionForSlopeMap/100<<"m";
        cv::putText(slope_map, str.str(), cv::Point(c,10), CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(250,250,250));
    }
    for (int r=20;r<slope_map.rows;r=r+20){
        std::ostringstream str;
        str << (slope_map.rows/2-r)*yResolutionForSlopeMap<<"cm";
        cv::putText(slope_map, str.str(), cv::Point(5,r), CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(250,250,250));
    }
    /*double curr_road_slope = round(atan((refinedRoadProfile[0].x-rowForDisparity35)/(refinedRoadProfile[0].y-disForSlope))*180/3.14159265 - 63.4349);
    road_slope = curr_road_slope;// - prv_road_slope;
    prv_road_slope = curr_road_slope;
  for(int i=0; i < refinedRoadProfile.size(); i++) {
    int row_r = refinedRoadProfile[i].x;
    for (int c = 1; c < roadmap.cols; c++) {
      int disparity_value = disparity_map.at<uchar>(row_r, c);
      int prv_disparity_value = disparity_map.at<uchar>(row_r, c - 1);
      if (disparity_value != 0 && prv_disparity_value != 0) {
        double currentY = yDirectionPosition[row_r][disparity_value];
        double previousY = yDirectionPosition[row_r][prv_disparity_value];
        double currentX = xDirectionPosition[c][disparity_value];
        double previousX = xDirectionPosition[c-1][prv_disparity_value];
      }
    }
  }*/
    prvSlopeMap = slope_map;

}

void ObstaclesDetection::RoadSlopeInit (){
    if (imuDetected){
        if (abs(imuAngularVelocityY)>0.1){
            humpEndFrames = 0;
            slope_map = prvSlopeMap;
//                std::cout<<"Slope Corrected Frame ID: "<< globalframe<<std::endl;
        } else {
            humpEndFrames ++;
            if (humpEndFrames<6){
                slope_map = prvSlopeMap;
//                    std::cout<<"Slope Corrected Frame ID: "<< globalframe<<std::endl;
            } else {
                RoadSlopeCalculation();
            }
        }
    } else {
        RoadSlopeCalculation();
    }
}

void ObstaclesDetection::DisplayRoad() {
//    cv::imshow("road",road);
//    cv::imshow("v_disparity_map",v_disparity_map);
//    cv::imshow("u_disparity_map",u_disparity_map);
    cv::Mat intial_road_map, refined_road_map;
    cv::cvtColor(v_disparity_map, intial_road_map, CV_GRAY2RGB);
    cv::cvtColor(v_disparity_map, refined_road_map, CV_GRAY2RGB);

//    for(int i=0  ; i < initialRoadProfile.size(); i++){
//        intial_road_map.at<cv::Vec3b>(initialRoadProfile[i].x, initialRoadProfile[i].y) = cv::Vec3b(0,0,255);
//    }
    for(int i=0  ; i < refinedRoadProfile.size(); i++){
        refined_road_map.at<cv::Vec3b>(refinedRoadProfile[i].x, refinedRoadProfile[i].y) = cv::Vec3b(0,0,255);
    }

    cv::imshow("intial_road_map", intial_road_map);
    cv::imshow("refined_road_map", refined_road_map);
//    cv::imshow("disparity_map",disparity_map);
//    cv::waitKey(1);
}

void ObstaclesDetection::DisplayPosObs() {
//    cv::imshow("obstacleDisparityMap", obstacleDisparityMap*255/disp_size);
//    cv::Mat posObsOutput;
//    cv::cvtColor(left_rect, posObsOutput, CV_GRAY2RGB);

//    std::cout << "Completed GenerateObstaclesMap function in posObstacles.cpp" << std::endl;
    for (auto & currentFrameObsBlob : currentFrameObsBlobs) {
        cv::rectangle(left_rect_clr, currentFrameObsBlob.currentBoundingRect, cv::Scalar( 0, 0, 255 ), 2);
        for(int j=0; j<currentFrameObsBlob.obsPoints.size();j++){
            left_rect_clr.at<cv::Vec3b>(currentFrameObsBlob.obsPoints[j].x, currentFrameObsBlob.obsPoints[j].y)[2]=255;//cv::Vec3b(0,0,255);
        }
        std::ostringstream str;
        str << depthTable[currentFrameObsBlob.max_disparity]<<"m";
        cv::putText(left_rect_clr, str.str(), cv::Point(currentFrameObsBlob.currentBoundingRect.x,currentFrameObsBlob.currentBoundingRect.y+12),
                CV_FONT_HERSHEY_PLAIN, 0.6, CV_RGB(0,250,0));
    }
//    cv::imshow("posObsOutput", posObsOutput);
//    cv::imshow("obstaclemap",obstaclemap*255);
//    cv::imshow("roadmap",roadmap*255);
//    cv::waitKey(1);
}

void ObstaclesDetection::Initiate(int disparity_size, double baseline,
        double u0, double v0, double focal, int Width, int Height, int scale, int min_disparity){

//    std::cout<<camera_type<<", "<<disparity_size<<", "<<baseline<<std::endl;

    double minHeight =0.3;//0.4m
    double uHysteresisThreshRatio = 0.7;
    double minWidthToSeperate = 0.5; //0.5m
    double minDepthToSeperate = 6; //6m
    disp_size = disparity_size;
    minimum_disparity = min_disparity;

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

    rdRowToDisRegard = 10/scale;
    rdStartCheckLines = 10/scale;
    intensityThVDisPoint = 10/scale;
    thHorizon = 20/scale;
    rdProfileRowDistanceTh = 6/scale;
    rdProfileColDistanceTh = 16/scale;
    intensityThVDisPointForSlope = 100/scale;
    pubName = "/wide/map_msg";
    depthForSlpoe = 18/scale; //m -- slope
    depthForSlopeStart = 5/scale; //m -- slope
    slopeAdjHeight = 30/scale; // cm -- slope
    slopeAdjLength = 1500/scale; // cm -- slope
    minDepthDiffToCalculateSlope = 400/scale; // cm -- slope
    minNoOfPixelsForObject = 80/scale;

//    if (camera_type == "long_camera") {
//        rdRowToDisRegard = 10;
//        rdStartCheckLines = 30;
//        intensityThVDisPoint = 10;
//        thHorizon = 10;
//        rdProfileRowDistanceTh = 10;
//        rdProfileColDistanceTh = 4;
//        intensityThVDisPointForSlope = 100;
//        pubName = "/long/map_msg";
//        depthForSlpoe = 30; //m -- slope
//        depthForSlopeStart = 10; //m -- slope
//        slopeAdjHeight = 90;//35; // cm -- slope
//        slopeAdjLength = 2000;//2000; // cm -- slope
//        minDepthDiffToCalculateSlope = 1000; // cm -- slope
//        minNoOfPixelsForObject = 45;
//    } else if (camera_type == "wide_camera") {
//        rdRowToDisRegard = 30;
//        rdStartCheckLines = 30;
//        intensityThVDisPoint = 10;
//        thHorizon = 20;
//        rdProfileRowDistanceTh = 6;
//        rdProfileColDistanceTh = 16;
//        intensityThVDisPointForSlope = 100;
//        pubName = "/wide/map_msg";
//        depthForSlpoe = 18; //m -- slope
//        depthForSlopeStart = 5; //m -- slope
//        slopeAdjHeight = 30; // cm -- slope
//        slopeAdjLength = 1500; // cm -- slope
//        minDepthDiffToCalculateSlope = 400; // cm -- slope
//        minNoOfPixelsForObject = 80;
//    }

    disForSlope = cvRound(focal * baseline / depthForSlpoe);
    disForSlopeStart = cvRound(focal * baseline / depthForSlopeStart);
}

void ObstaclesDetection::ExecuteDetection(cv::Mat &disp_img, cv::Mat &img){

    disp_img.copyTo(disparity_map);

    for (int r=0; r<disparity_map.rows;r++){
        for (int c=0; c<disparity_map.cols;c++){
            int dispAtPoint = (int)disparity_map.at<uchar>(r,c);
//            int dispAtPoint2 = (int)disparity_SGM.at<uchar>(r,c);
            if (dispAtPoint>disp_size)
                disparity_map.at<uchar>(r,c) = 0;
            else if (dispAtPoint<minimum_disparity)
                disparity_map.at<uchar>(r,c) = 0;
//            if (dispAtPoint2>disp_size)
//                disparity_SGM.at<uchar>(r,c) = 0;
//            else if (dispAtPoint2<min_disparity)
//                disparity_SGM.at<uchar>(r,c) = 0;
        }
    }

    cv::Mat left_rect_gray;
    img.copyTo(left_rect_gray);
//    std::cout<<img.channels()<<", "<<img.type()<<std::endl;
//    std::cout<<left_rect_clr.channels()<<", "<<left_rect_clr.type()<<std::endl;
    cv::cvtColor(left_rect_gray, left_rect_clr, cv::COLOR_GRAY2BGR);

//    std::string img_name1;
//    char im1[20];
//    sprintf(im1, "f%03d.png", frameCount);
//    img_name1 = std::string("/home/ugv/slope/") + im1;
//    cv::imwrite(img_name1, left_rect_clr);

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

    DisplayRoad();

    int minNoOfPointsForRdProfile =50;
    if (refinedRoadProfile.size()>minNoOfPointsForRdProfile){
//
        obstacleDisparityMap = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
        negObsMap = cv::Mat::zeros(disparity_map.rows,disparity_map.cols, CV_8UC1);
        currentFrameObsBlobs.clear();

        memset(dynamicLookUpTableRoad, 0, sizeof(dynamicLookUpTableRoad));
        memset(dynamicLookUpTableRoadProfile, 0, sizeof(dynamicLookUpTableRoadProfile));

        InitiateObstaclesMap();
        RefineObstaclesMap();

        DisplayPosObs();

        randomRoadPoints.clear();
        randomRoadPoints2D.clear();
        selectedIndexes.clear();
        surfaceN = cv::Vec3d(0.0,0.0,0.0);

        RoadSlopeInit();

        cv::imshow("Slope_map", slope_map);
        cv::imshow("left_rect_clr", left_rect_clr);

//        cv::imshow("disparity_map", disparity_map*255/disp_size);
//        cv::waitKey(1);

//        std::string img_name;
//        char im[20];
//        sprintf(im, "s%03d.png", frameCount);
//        img_name = std::string("/home/ugv/slope/") + im;
//        cv::imwrite(img_name, left_rect_clr);
//
    }

    frameCount++;

}