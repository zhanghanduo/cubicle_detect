/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef DISPARITY_METHOD_H_
#define DISPARITY_METHOD_H_

#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"
#include "debug.h"

class disparity_sgm{
public:
    disparity_sgm(uint8_t _p1, uint8_t _p2);

    ~disparity_sgm();

    void init_disparity_method(const uint8_t _p1, const uint8_t _p2);

    cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms);

    void finish_disparity_method();

    void free_memory();

private:

    uint8_t p1, p2;
    cudaStream_t stream1, stream2, stream3;//, stream4, stream5, stream6, stream7, stream8;
    uint8_t *d_im0;
    uint8_t *d_im1;
    cost_t *d_transform0;
    cost_t *d_transform1;
    uint8_t *d_cost;
    uint8_t *d_disparity;
    uint8_t *d_disparity_filtered_uchar;
    uint8_t *h_disparity;
    uint16_t *d_S;
    uint8_t *d_L0;
    uint8_t *d_L1;
    uint8_t *d_L2;
    uint8_t *d_L3;
    uint8_t *d_L4;
    uint8_t *d_L5;
    uint8_t *d_L6;
    uint8_t *d_L7;
    bool first_alloc;
    uint32_t cols, rows, size, size_cube_l;

};



#endif /* DISPARITY_METHOD_H_ */
