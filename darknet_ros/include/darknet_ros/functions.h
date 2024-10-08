/*
    This file is part of spixel.

    Spixel is free software : you can redistribute it and / or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "darknet_ros/stdafx.h"
#include "darknet_ros/structures.h"

class Timer {
private:
    clock_t startTime, time;
    bool running;
public:
    Timer(bool run = true)
    { 
        if (run) Reset();
        else {
            startTime = time = 0;
            running = false;
        }
    }

    void Reset() 
    { 
        time = 0;
        startTime = clock();
        running = true;
    }

    void Stop() 
    { 
        if (running) {
            time += clock() - startTime;
            running = false;
        }
    }

    void Resume()
    {
        startTime = clock();
        running = true;
    }

    clock_t GetTime()
    {
        return time;
    }

    double GetTimeInSec()
    {
        return (double)time / CLOCKS_PER_SEC;
    }

    friend std::ostream& operator<<(std::ostream& os, const Timer& t);
};

// Cached info for connectivity of 3x3 patches
class ConnectivityCache {
    std::vector<bool> cache;
public:
    ConnectivityCache();
    bool IsConnected(int b) const { return cache[b]; }
    void Print(); // for debug
private:
    void Initialize();
};


// See definition for description
void MovePixel(Matrix<Pixel>& pixelsImg, PixelMoveData& pmd);
void MovePixelStereo(Matrix<Pixel>& pixelsImg, PixelMoveData& pmd, bool changeBoundary);

// Return true if superpixel sp is connected in region defined by upper left/lower right corners of pixelsImg
bool IsSuperpixelRegionConnected(const Matrix<Pixel>& pixelsImg, Pixel* p, int ulr, int ulc, int lrr, int lrc);

// Return true if superpixel sp is connected in region defined by upper left/lower right corners of pixelsImg
bool IsSuperpixelRegionConnectedOptimized(const Matrix<Pixel>& pixelsImg, Pixel* p, int ulr, int ulc, int lrr, int lrc);

// For debug purposes; inefficient!
int CalcSuperpixelBoundaryLength(const Matrix<Pixel>& pixelsImg, Superpixel* sp);

// Length of superpixel boundary
void CalcSuperpixelBoundaryLength(const Matrix<Pixel>& pixelsImg, Pixel* p, Superpixel* sp, Superpixel* sq,
    int& spbl, int& sqbl, int& sobl);

bool IsPatch3x3Connected(int bits);

// Plane through 3 points, returns false if p1, p2, p3 are approximately coplanar (eps normal size)
bool Plane3P(const cv::Point3d& p1, const cv::Point3d& p2, const cv::Point3d& p3, Plane_d& plane);

// Returns false if pixels.size() < 3 or no 3 points were found to 
// form a plane
bool RANSACPlane(const std::vector<cv::Point3d>& pixels, Plane_d& plane);
void InitSuperpixelPlane(SuperpixelStereo* sp, const cv::Mat1d& depthImg);

// Equation (8) for superpixel sp
void CalcDispEnergy(SuperpixelStereo* sp, const cv::Mat1d& dispImg, double noDisp);

void CalcCoSmoothnessSum(const cv::Mat1d& depthImg, double inlierThresh, SuperpixelStereo* sp, SuperpixelStereo* sq, double& eSmo, int& count);

void CalcCoSmoothnessSum2(SuperpixelStereo* sp, SuperpixelStereo* sq, double& eSmo, int& count);

bool LeastSquaresPlaneDebug(const double x1, const double y1, const double z1, const double d1,
    const double x2, const double y2, const double z2, const double d2,
    const double x3, const double y3, const double z3, const double d3,
    Plane_d& plane);

