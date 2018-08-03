//
// Created by hd on 1/26/18.
//
#include "utils/data.h"
#include <cassert>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
namespace Util {

    double median(vector<double> &v)
    {
        if(v.empty()) {
            return 0.0;
        }

        auto n = v.size() / 2;

        nth_element(v.begin(), v.begin() + n, v.end());

        auto med = v[n];

//        if(!(v.size() & 1)) { //If the set size is even
//
//            auto max_it = max_element(v.begin(), v.begin()+n);
//
//            med = (*max_it + med) / 2.0;
//        }

        return med;
    }

    double median_mat (cv::Mat& I, int x, int y, int h){

        // For efficiency did not assert whether x and y border issue lying in the image plane!!

        std::vector<double> array;

        // accept only char type matrices
        CV_Assert(I.depth() == CV_8U);

        int i,j;
        uchar* p;

        for ( i = y - h; i < y + h; ++ i){

            p = I.ptr<uchar>(i);

            for( j = x - h; j < x + h; ++ j){

                if(p[j]==0) continue;

                array.push_back((double)p[j]);
            }
        }

        if(!array.empty())
            return (median(array));
        else
            return 0;
    }
//
//    double median( cv::Mat channel )
//    {
//        double m = (channel.rows*channel.cols) / 2;
//        int bin = 0;
//        double med = -1.0;
//
//        int histSize = 256;
//        float range[] = { 0, 256 };
//        const float* histRange = { range };
//        bool uniform = true;
//        bool accumulate = false;
//        cv::Mat hist;
//        cv::calcHist( &channel, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
//
//        for ( int i = 0; i < histSize && med < 0.0; ++i )
//        {
//            bin += cvRound( hist.at< float >( i ) );
//            if ( bin > m && med < 0.0 )
//                med = i;
//        }
//
//        return med;
//    }

    // Grid dimensions.

    class Dim {
    public:
        Dim(int b_, int size_, int h_)
                : size(size_),
                  h(h_),
                  step(calc_step(b_, h_)),
                  count(calc_count(b_, size_, h_))
        {
            assert(2 * h + 1 < b_);
            assert(count >= 1);
            assert(2 * h + count * step >= size);
            assert(2 * h + (count - 1) * step < size || count == 1);
        }

        const int size;
        const int h;
        const int step;
        const int count;

    private:
        inline static int calc_step(int b, int h) {
            return b - 2*h;
        }

        inline static int calc_count(int b, int size, int h) {
            if (size <= b) {
                return 1;
            } else {
                int interior = size - 2 * h;
                int step = calc_step(b, h);
                return (interior + step - 1) / step;
            }
        }
    };



}

