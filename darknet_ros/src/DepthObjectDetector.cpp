//
// Created by hd on 14/May/17.
//
/* ----------------------------------------------------------------------------
* Obstacle Detection Copyright 2017, Nanyang Technological University, ST Kinetics
* All Rights Reserved
* Authors: Hasith, Zhang Handuo, et al. (see THANKS for the full author list)

* See LICENSE for the license information

* -------------------------------------------------------------------------- */
/**
* @file DepthObjectDetector.cpp
* @brief A stereo visual odometry example
* @date May 25, 2016
* @author Hasith, Zhang Handuo
*/

/**
 * A 3D stereo visual odometry example
 *  - robot starts at origin
 *  -moves forward, taking periodic stereo measurements
 *  -takes stereo readings of many landmarks
 */

#include "darknet_ros/DepthObjectDetector.h"
#include <ros/package.h>

using namespace std;

namespace darknet_ros{

//Default Constructor
Detection::Detection() = default;

Detection::Detection(YoloObjectDetector* pYolo, ros::NodeHandle n):
        mpYolo(pYolo),
        isReceiveImage(false),
        isDepthNew(false),
        mbFinished_(true),
        mbStopped(false),
        mbStopRequested(false),
        mbFinishRequested_(false)
{
    disparity_frame = "/disparity_id";
    Height = 844; //422;
    Width = 1280; //640;
    disp_size = 128; //64;
    min_disparity = 12; //6;
    Scale = 2; //1

    /***** Read Params from external configurations *****/

    if(n.getParam("min_disparity", min_disparity))
        ROS_INFO("Get minimal disparity value: %d", min_disparity);

    if(n.getParam("disparity_scope", disp_size))
        ROS_INFO("Get disparity scope: %d", disp_size);

    if(n.getParam("image_width", Width))
        ROS_INFO("Get image width: %d", Width);
    if(n.getParam("image_height", Height))
        ROS_INFO("Get image height: %d", Height);
    if(n.getParam("scale", Scale))
        ROS_INFO("Scale: %d", Scale);

    Width /= Scale;
    Height /= Scale;
    getParams();
    /***** Initialize the related values and thresholds *****/

//    pub = n.advertise<obstacle_msgs::MapInfo>("/wide/map_msg", 10);

}

void Detection::getParams() {

    vxiLeft = vxCreateImage(context, static_cast<vx_uint32>(Width), static_cast<vx_uint32>(Height), VX_DF_IMAGE_RGB);
    NVXIO_CHECK_REFERENCE(vxiLeft); // NOLINT
    vxiRight = vxCreateImage(context, static_cast<vx_uint32>(Width), static_cast<vx_uint32>(Height), VX_DF_IMAGE_RGB);
    NVXIO_CHECK_REFERENCE(vxiRight); // NOLINT

    vxiDisparity = vxCreateImage(context, static_cast<vx_uint32>(Width), static_cast<vx_uint32>(Height), VX_DF_IMAGE_U8);
    NVXIO_CHECK_REFERENCE(vxiDisparity); // NOLINT

    stereo = std::unique_ptr<StereoMatching>(StereoMatching::createStereoMatching
                                                     (context, params, implementationType, vxiLeft, vxiRight, vxiDisparity));
    if (!read(params)) {
        std::cout <<"Failed to open config file "<< std::endl;
    }

    vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
    vxRegisterLogCallback(context, &nvxio::stdoutLogCallback, vx_false_e);
}

bool Detection::read(StereoMatching::StereoMatchingParams &config){
    config.min_disparity = 0;
    config.max_disparity = disp_size;

    // discontinuity penalties
    config.P1 = 8;
    config.P2 = 109;
    // SAD window size
    config.sad = 5;
    // Census Transform window size
    config.ct_win_size = 3;
    // Hamming cost window size
    config.hc_win_size = 1;
    // BT-cost clip value
    config.bt_clip_value = 31;
    // validation threshold
    config.max_diff = 32000; // cross-check
    config.uniqueness_ratio = 0;
    config.scanlines_mask = 85;
    config.flags = 2;

    return true;
}

vx_status Detection::createMatFromImage(cv::Mat &mat, vx_image image) {
    vx_status status = VX_SUCCESS;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_df_image format = VX_DF_IMAGE_VIRT;
    int cv_format = CV_8U;
    vx_size planes = 0;

    vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width));
    vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height));
    vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format));
    vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &planes, sizeof(planes));

    switch (format){
        case VX_DF_IMAGE_U8:
            cv_format = CV_8U;
            break;
        case VX_DF_IMAGE_S16:
            cv_format = CV_16S;
            break;
        case VX_DF_IMAGE_RGB:
            cv_format = CV_8UC3;
            break;
        default:
            return VX_ERROR_INVALID_FORMAT;
    }

    vx_rectangle_t rect{ 0, 0, width, height };
    vx_uint8 *src[4] = {nullptr, nullptr, nullptr, nullptr };
    vx_uint32 p;
    void *ptr = nullptr;
    vx_imagepatch_addressing_t addr[4] = { 0, 0, 0, 0 };
    vx_uint32 y = 0u;

    for (p = 0u; (p < (int)planes); p++){
        vxAccessImagePatch(image, &rect, p, &addr[p], (void **)&src[p], VX_READ_ONLY);
        size_t len = addr[p].stride_x * (addr[p].dim_x * addr[p].scale_x) / VX_SCALE_UNITY;
        for (y = 0; y < height; y += addr[p].step_y){
            ptr = vxFormatImagePatchAddress2d(src[p], 0, y - rect.start_y, &addr[p]);
            memcpy(mat.data + y * mat.step, ptr, len);
        }
    }

    for (p = 0u; p < (int)planes; p++){
        vxCommitImagePatch(image, &rect, p, &addr[p], src[p]);
    }

    return status;
}

vx_image Detection::createImageFromMat(vx_context context, const cv::Mat & mat) {
    vx_imagepatch_addressing_t patch = { (vx_uint32)mat.cols, (vx_uint32)mat.rows,
                                         (vx_int32)mat.elemSize(), (vx_int32)mat.step,
                                         VX_SCALE_UNITY, VX_SCALE_UNITY,1u, 1u };
    auto *ptr = (void*)mat.ptr();
    vx_df_image format = nvx_cv::convertCVMatTypeToVXImageFormat(mat.type());
    return vxCreateImageFromHandle(context, format, &patch, (void **)&ptr, VX_IMPORT_TYPE_HOST);
}

void Detection::GenerateDisparityMap(){

    mMutexDepth.lock();

    cv::Mat disparity_SGBM(left_rectified.size(), CV_8UC1);

    vxiLeft_U8 = createImageFromMat(context, left_rectified);
    vxuColorConvert(context, vxiLeft_U8, vxiLeft);
    vxiRight_U8 = createImageFromMat(context, right_rectified);
    vxuColorConvert(context, vxiRight_U8, vxiRight);
    stereo->run();
    createMatFromImage(disparity_SGBM, vxiDisparity);

    vxSwapImageHandle(vxiLeft_U8, nullptr, nullptr, 1);
    vxSwapImageHandle(vxiRight_U8, nullptr, nullptr, 1);
    vxReleaseImage(&vxiLeft_U8);
    vxReleaseImage(&vxiRight_U8);

    disparity_SGBM.copyTo(disparity_map);

    isDepthNew = true;

    mMutexDepth.unlock();
}

void Detection::VisualizeResults() {

    cv::imshow("disparity_map",disparity_map*256/disp_size);

    cv::waitKey(1);
}

void Detection::Run(){

    mbFinished_ = false;

    while(true) {

        if (isReceiveImage) {

            GenerateDisparityMap();

            mpYolo->getDepth(disparity_map);

//            VisualizeResults();

            isReceiveImage = false;

        }

        if(Stop()){
            while(isStopped() && !CheckFinish()){
                std::this_thread::sleep_for(std::chrono::microseconds(3000));
            }
            if(CheckFinish())
                break;
        }

        if(CheckFinish())
            break;

        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    SetFinish();

}

void Detection::getImage(cv::Mat &Frame1, cv::Mat &Frame2) {

    left_rectified = Frame1.clone();

    right_rectified = Frame2.clone();

    isReceiveImage = true;
}

void Detection::SetFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished_ = true;
}

bool Detection::isFinished() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished_;
}

void Detection::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested_ = true;
}

bool Detection::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested_;
}

void Detection::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Detection::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if(mbStopRequested )
    {
        mbStopped = true;
        std::cout << "YOLO STOP" << std::endl;
        return true;
    }

    return false;
}

bool Detection::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool Detection::stopRequested()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

Detection::~Detection() = default;



}