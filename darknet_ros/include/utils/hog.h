//
// Created by hd on 1/29/18.
//

#ifndef PROJECT_HOG_H
#define PROJECT_HOG_H

// OpenCV Header Directives
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

namespace Util {

    class HOGFeatureDescriptor {

    public:
        HOGFeatureDescriptor(int, int, int, float);

        cv::Mat computeHOG(
                const cv::Mat &);

        void computeHOG(
                std::vector<float> &feature_vec, const cv::Mat &img);

        //  HOG Configuration Params
    protected:
        int N_BINS;
        int ANGLE;
        int BINS_ANGLE;
        int CELL;
        int BLOCK;

    private:
        void bilinearBinVoting(
                const float &, int &, int &);

        void imageGradient(
                const cv::Mat &, cv::Mat &);

        cv::Mat blockGradient(
                const int, const int, cv::Mat &);

        void getHOG(
                const cv::Mat &, cv::Mat &, cv::Mat &);

        template<typename T>
        T computeHOGHistogramDistances(
                const cv::Mat &, std::vector<cv::Mat> &,
                const int = CV_COMP_BHATTACHARYYA);
    };


}

#endif //PROJECT_HOG_H
