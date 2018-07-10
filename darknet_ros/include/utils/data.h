//
// Created by hd on 1/26/18.
//

#ifndef PROJECT_DATA_H
#define PROJECT_DATA_H

#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>

namespace Util {

    double median(std::vector<double> &v);

//    double median_mat( cv::Mat channel);

    /*!
     * Find the median value of the local window of a point (x,y) lying in the image I.
     * @param[in] Original image.
     * @param[in] the x direction coordinate (horizontal).
     * @param[in] the y direction coordinate (vertical).
     * @param[in] local windows radius (e.g. 2, so window size is 5 (2*2+1) ).
    */
    double median_mat (cv::Mat& I, int x, int y, int h);

}
#endif //PROJECT_DATA_H
