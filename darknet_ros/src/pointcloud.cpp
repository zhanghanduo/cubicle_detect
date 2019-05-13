//
// Created by hd on 12/5/19.
//

#include "pointcloud.h"

void reprojectTo3D(ColorCloud::Ptr& basic_cloud_ptr, cv::Mat &disparity, cv::Mat &Q,
                   bool handleMissingValues) {

    int stype = disparity.type();
    int dtype = CV_32FC3;

    cv::Mat _3dImage = cv::Mat (disparity.size(), CV_MAKETYPE(dtype, 3));

    ColorCloud::Ptr cloud(new ColorCloud());

    CV_Assert(stype == CV_8UC1 || stype == CV_16SC1 || stype == CV_32SC1 ||
              stype == CV_32FC1);
    CV_Assert(Q.size() == cv::Size(4, 4));

    const float bigZ = 10000.f;
    Matx44d _Q;
    Q.convertTo(_Q, CV_64F);

    int x, cols = disparity.cols;
    CV_Assert(cols >= 0);

    std::vector<float> _sbuf(cols);
    std::vector<Vec3f> _dbuf(cols);
    float* sbuf = &_sbuf[0];
    Vec3f* dbuf = &_dbuf[0];
    double minDisparity = FLT_MAX;

    // NOTE: here we quietly assume that at least one pixel in the disparity map
    // is not defined.
    // and we set the corresponding Z's to some fixed big value.
    if (handleMissingValues)
        cv::minMaxIdx(disparity, &minDisparity, 0, 0, 0);

    for (int y = 0; y < disparity.rows; y++) {
        float *sptr = sbuf;
        Vec3f* dptr = dbuf;

        auto *rgb_ptr = left_rectified.ptr<uchar>(y);

        if (stype == CV_8UC1) {
            const uchar *sptr0 = disparity.ptr<uchar>(y);
            for (x = 0; x < cols; x++)
                sptr[x] = (float) sptr0[x];
        } else
            sptr = disparity.ptr<float>(y);

        dptr = _3dImage.ptr<Vec3f>(y);

        for (x = 0; x < cols;x++) {

            double d = sptr[x];
            Vec4d homg_pt = _Q*Vec4d(x, y, d, 1.0);

            dptr[x] = Vec3d(homg_pt.val);
            dptr[x] /= homg_pt[3];

            if( fabs(d-minDisparity) <= FLT_EPSILON )
                dptr[x][2] = bigZ;

            uint8_t r = rgb_ptr[x];

            pcl::PointXYZRGB basic_point;

            basic_point.z = (float)dptr[x][2] * 0.001f;
            if( (basic_point.z > 80) ||(basic_point.z < 2) )
                continue;
            basic_point.x = (float)dptr[x][0] * 0.001f;
            basic_point.y = (float)dptr[x][1] * 0.001f;
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                            static_cast<uint32_t>(r) << 8 | static_cast<uint32_t>(r));
            basic_point.rgb = *reinterpret_cast<float *>(&rgb);
            cloud->points.push_back(basic_point);

        }
    }
    *basic_cloud_ptr = *cloud;

//    ColorCloud::Ptr tmp(new ColorCloud());
//    voxel.setInputCloud (basic_cloud_ptr);
//    voxel.filter (*tmp);
//    basic_cloud_ptr->swap(*tmp);

}

void generate_cloud(globals& obj, ColorCloud::Ptr& basic_cloud_ptr) {

//    reprojectTo3D(cloud_ptr, disparity_map, Q, true); //obstacleDisparityMap, disparity_map

//    cv::Mat cloud(1, disparity_map.cols*disparity_map.rows, CV_32FC3);
//
//    Point3f* data = cloud.ptr<cv::Point3f>();
//    for (int r=0;r<disparity_map.rows;r++) {
//        for (int c = 0; c < disparity_map.cols; c++) {
//            int dis = (int)disparity_map.at<uchar>(r,c);
//            data[r*disparity_map.cols+c].x = (float) obj.xDirectionPosition[c][dis];
//            data[r*disparity_map.cols+c].y = (float) obj.yDirectionPosition[r][dis];
//            data[r*disparity_map.cols+c].z = (float) -obj.depth[dis];
//        }
//    }
//
//    viz::WCloud cloud_widget(cloud, viz::Color::green());
//    obj.cloudWindow.showWidget("Cloud Widget", cloud_widget);//, obj.cloud_pose_global);
//    obj.cloudWindow.spinOnce(1, true);

//    ColorCloud::Ptr cloud_= boost::make_shared <ColorCloud> ();

    for (int r=0; r<disparity_map.rows; r++) {
        auto *rgb_ptr = left_rectified.ptr<uchar>(r);

        for (int c = 0; c < disparity_map.cols; c++) {
            pcl::PointXYZRGB basic_point;
            uint8_t color = rgb_ptr[c];
            auto dis = (int)disparity_map.at<uchar>(r,c);
            basic_point.z = (float) -obj.yDirectionPosition[r][dis];
            basic_point.x = (float) obj.depth[dis];
            basic_point.y = (float) -obj.xDirectionPosition[c][dis];
            uint32_t rgb = (static_cast<uint32_t>(color) << 16 |
                            static_cast<uint32_t>(color) << 8 | static_cast<uint32_t>(color));
            basic_point.rgb = *reinterpret_cast<float *>(&rgb);
            basic_cloud_ptr->points.push_back(basic_point);
        }
    }
//    *basic_cloud_ptr = *cloud_;
}