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


#include "darknet_ros/stdafx.h"
#include "darknet_ros/segengine.h"
#include "darknet_ros/functions.h"
#include "darknet_ros/utilsSeg.h"
#include "darknet_ros/tsdeque.h"
#include <unordered_map>
#include <fstream>   
#include <thread>
#include <stdexcept>
#include <cmath>
#include <iomanip>

#define PRINT_LEVEL_PARAM_DOUBLE(field) \
    std::cout << "*" << #field << ": " << std::setprecision(4) << field << std::endl;

#define PRINT_LEVEL_PARAM_INT(field) \
    std::cout << "*" << #field << ": " << field << std::endl;

#define PRINT_PARAM(field) \
    std::cout << #field << ": " << field << std::endl;



// Constants
///////////////////////////////////////////////////////////////////////////////

static const int nDeltas[][2] = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
static const int nDeltas0[][2] = { { 0, 0 }, { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };


// Local functions
///////////////////////////////////////////////////////////////////////////////

double TotalEnergyDelta(const SPSegmentationParameters& params, PixelMoveData* pmd) 
{
    double eSum = 0.0;
    int initPSizeQuarter = pmd->p->superPixel->GetInitialSize()/4;
    
    eSum += params.regWeight * pmd->eRegDelta;
    eSum += params.appWeight * pmd->eAppDelta;
    eSum += params.lenWeight * pmd->bLenDelta;
    if (params.stereo) {
        eSum += params.dispWeight * pmd->eDispDelta;
        eSum += params.priorWeight * pmd->ePriorDelta;
        eSum += params.smoWeight * pmd->eSmoDelta;
    }

    if (pmd->pSize < initPSizeQuarter)
        eSum -= params.sizeWeight * (initPSizeQuarter - pmd->pSize);
    return eSum;
}

void EstimateBorderType(const SPSegmentationParameters& params, SuperpixelStereo* sp, int newSpSize, 
    SuperpixelStereo* sq, int newSqSize, BInfo& bInfo)
{
    double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (bInfo.hiSum / bInfo.hiCount + params.hiPriorWeight);
    double eCo = bInfo.coCount == 0 ? HUGE_VAL : (bInfo.coSum / bInfo.coCount); // (newSpSize + newSqSize); // + 0
    double eOcc = params.occPriorWeight;

    if (eCo <= eHi && eCo < eOcc) {
        bInfo.type = BTCo;
        bInfo.typePrior = 0;
    } else if (eHi <= eOcc && eHi <= eCo) {
        bInfo.type = BTHi;
        bInfo.typePrior = params.hiPriorWeight;
    } else {
        bInfo.type = BTLo;
        bInfo.typePrior = params.occPriorWeight;
    }
}

PixelMoveData* FindBestMoveData(const SPSegmentationParameters& params, PixelMoveData d[4])
{
    PixelMoveData* dArray[4];
    int dArraySize = 0;

    // Only allowed moves and energy delta should be positive (or over threshold)
    for (int m = 0; m < 4; m++) {
        PixelMoveData* md = &d[m];
        if (md->allowed && TotalEnergyDelta(params, md) > params.updateThreshold) dArray[dArraySize++] = md;
    }

    if (dArraySize == 0) return nullptr;

    return *std::max_element(dArray, dArray + dArraySize, 
        [&params](PixelMoveData* a, PixelMoveData* b) { return TotalEnergyDelta(params, a) < TotalEnergyDelta(params, b); });
}

PixelMoveData* FindBestMoveData(const SPSegmentationParameters& params, PixelMoveData* d1, PixelMoveData* d2)
{
    if (!d1->allowed || TotalEnergyDelta(params, d1) <= 0) return (!d2->allowed || TotalEnergyDelta(params, d2) <= 0) ? nullptr : d2;
    else if (!d2->allowed || TotalEnergyDelta(params, d2) <= 0) return d1;
    else return TotalEnergyDelta(params, d1) < TotalEnergyDelta(params, d2) ? d2 : d1;
}


double GetSmoEnergy(const BorderDataMap& bd, SuperpixelStereo* sp, int pSize, SuperpixelStereo* sq, int qSize)
{
    double result = 0.0;

    for (auto& bdItem : bd) {
        const BInfo& bInfo = bdItem.second;
        if (bInfo.length > 0) {
            if (bInfo.type == BTCo) result += bInfo.coSum / bInfo.coCount; // (pSize + (bdItem.first == sq ? qSize : bdItem.first->GetSize()));
            else if (bInfo.type == BTHi) result += bInfo.hiSum / bInfo.hiCount;
        }
    }
    return result;
}

// SPSegmentationParameters
///////////////////////////////////////////////////////////////////////////////

static void read(const FileNode& node, SPSegmentationParameters& x, const SPSegmentationParameters& defaultValue)
{
    if (node.empty()) x = defaultValue;
    else x.Read(node);
}

void SPSegmentationParameters::SetLevelParams(int level)
{
    for (std::pair<std::string, std::vector<double>>& pData : levelParamsDouble) {
        if (!pData.second.empty()) {
            updateDouble[pData.first](*this,
                (level < pData.second.size()) ? pData.second[level] : pData.second.back());
        }
    }
    for (std::pair<std::string, std::vector<int>>& pData : levelParamsInt) {
        if (!pData.second.empty()) {
            updateInt[pData.first](*this,
                (level < pData.second.size()) ? pData.second[level] : pData.second.back());
        }
    }
    if (debugOutput) {
        std::cout << "--- Params for level " << level << " ----" << std::endl;
        PRINT_LEVEL_PARAM_DOUBLE(appWeight);
        PRINT_LEVEL_PARAM_DOUBLE(regWeight);
        PRINT_LEVEL_PARAM_DOUBLE(lenWeight);
        PRINT_LEVEL_PARAM_DOUBLE(sizeWeight);
        PRINT_LEVEL_PARAM_DOUBLE(dispWeight);
        PRINT_LEVEL_PARAM_DOUBLE(smoWeight);
        PRINT_LEVEL_PARAM_DOUBLE(priorWeight);
        PRINT_LEVEL_PARAM_DOUBLE(occPriorWeight);
        PRINT_LEVEL_PARAM_DOUBLE(hiPriorWeight);
        PRINT_LEVEL_PARAM_INT(reSteps);
        PRINT_LEVEL_PARAM_INT(peblThreshold);
        PRINT_PARAM(superpixelNum);
        PRINT_PARAM(noDisp);
        PRINT_PARAM(stereo);
        PRINT_PARAM(inpaint);
        PRINT_PARAM(instantBoundary);
        PRINT_PARAM(iterations);
        PRINT_PARAM(inlierThreshold);
        PRINT_PARAM(maxUpdates);
        PRINT_PARAM(minPixelSize);
        PRINT_PARAM(maxPixelSize);
        PRINT_PARAM(updateThreshold);
        PRINT_PARAM(debugOutput);
        std::cout << "---------------------------" << std::endl;
    }
}


void UpdateFromNode(double& val, const FileNode& node)
{
    if (!node.empty()) val = (double)node;
}

void UpdateFromNode(int& val, const FileNode& node)
{
    if (!node.empty()) val = (int)node;
}

void UpdateFromNode(bool& val, const FileNode& node)
{
    if (!node.empty()) val = (int)node != 0;
}

// SPSegmentationEngine
///////////////////////////////////////////////////////////////////////////////


SPSegmentationEngine::SPSegmentationEngine(SPSegmentationParameters params, Mat im, Mat depthIm) :
    params(params), origImg(im)
{
    img = ConvertRGBToLab(im);//@hk
    depthImg = AdjustDisparityImage(depthIm);
    if (params.stereo) {
        if (params.inpaint) depthImgAdj = InpaintDisparityImage(depthImg);
        else depthImgAdj = FillGapsInDisparityImage(depthImg);
    }
    //depthImg = FillGapsInDisparityImage(depthImg);
    //inliers = Mat1b(depthImg.rows, depthImg.cols);
}

SPSegmentationEngine::~SPSegmentationEngine()
{
    Reset();
}

// Works for x, y > 0
inline int iCeilDiv(int x, int y)
{
    return (x + y - 1) / y;
}

void calcPixelSizes(int actualGridSize, int maxPixelSize,
    int& actualMaxPixelSize, int& actualMinPixelSize, int& maxN, int& minN)
{
    int actualDiv = iCeilDiv(actualGridSize, maxPixelSize);

    actualMaxPixelSize = iCeilDiv(actualGridSize, actualDiv);
    actualMinPixelSize = actualGridSize / actualDiv;
    maxN = actualGridSize % actualDiv;
    minN = actualDiv - maxN;
}

void SPSegmentationEngine::Initialize(Superpixel* spGenerator(int))
{
    int imageSize = img.rows * img.cols;
    int gridSize = (int)sqrt((double)imageSize / params.superpixelNum);
    int initDiv = max(2, iCeilDiv(gridSize, params.maxPixelSize));
    int maxPixelSize = iCeilDiv(gridSize, initDiv);

    initialMaxPixelSize = maxPixelSize;

    int imgSPixelsRows = iCeilDiv(img.rows, gridSize);
    int imgSPixelsCols = iCeilDiv(img.cols, gridSize);

    int imgPixelsRows = initDiv * (img.rows / gridSize) + iCeilDiv(img.rows % gridSize, maxPixelSize);
    int imgPixelsCols = initDiv * (img.cols / gridSize) + iCeilDiv(img.cols % gridSize, maxPixelSize);

    std::vector<int> rowDims(imgPixelsRows), colDims(imgPixelsCols);
    std::vector<int> rowSDims(imgSPixelsRows, initDiv), colSDims(imgSPixelsCols, initDiv);
    int maxPS, minPS, maxN, minN;
    int ri = 0, ci = 0;

    calcPixelSizes(gridSize, maxPixelSize, maxPS, minPS, maxN, minN);
    while (ri < initDiv * (img.rows / gridSize)) {
        for (int i = 0; i < maxN; i++) rowDims[ri++] = maxPS;
        for (int i = 0; i < minN; i++) rowDims[ri++] = minPS;
    }
    while (ci < initDiv * (img.cols / gridSize)) {
        for (int i = 0; i < maxN; i++) colDims[ci++] = maxPS;
        for (int i = 0; i < minN; i++) colDims[ci++] = minPS;
    }
    if (img.rows % gridSize > 0) {
        calcPixelSizes(img.rows % gridSize, maxPixelSize, maxPS, minPS, maxN, minN);
        for (int i = 0; i < maxN; i++) rowDims[ri++] = maxPS;
        for (int i = 0; i < minN; i++) rowDims[ri++] = minPS;
        rowSDims.back() = maxN + minN;
    }
    if (img.cols % gridSize > 0) {
        calcPixelSizes(img.cols % gridSize, maxPixelSize, maxPS, minPS, maxN, minN);
        for (int i = 0; i < maxN; i++) colDims[ci++] = maxPS;
        for (int i = 0; i < minN; i++) colDims[ci++] = minPS;
        colSDims.back() = maxN + minN;
    }

    // Initialize 'pixels', 'pixelsImg'
    pixelsImg = Matrix<Pixel>(imgPixelsRows, imgPixelsCols);

    int i0, j0;

    i0 = 0;
    for (int pi = 0; pi < imgPixelsRows; pi++) {
        int i1 = i0 + rowDims[pi];
        
        j0 = 0;
        for (int pj = 0; pj < imgPixelsCols; pj++) {
            int j1 = j0 + colDims[pj];
            pixelsImg(pi, pj).Initialize(pi, pj, i0, j0, i1, j1);
            j0 = j1;
        }
        i0 = i1;
    }

    // Create superpixels (from 'pixelsImg' matrix) and borders matrices
    PixelData pd;
    int superPixelIdCount = 0;

    superpixels.clear();
    superpixels.reserve(imgSPixelsCols*imgSPixelsRows);
    i0 = 0;
    for (int pi = 0; pi < imgSPixelsRows; pi++) {
        int i1 = i0 + rowSDims[pi];

        j0 = 0;
        for (int pj = 0; pj < imgSPixelsCols; pj++) {
            int j1 = j0 + colSDims[pj];
            Superpixel* sp = spGenerator(superPixelIdCount++); // Superpixel();

            // Update superpixels pointers in each pixel
            for (int i = i0; i < i1; i++) {
                for (int j = j0; j < j1; j++) {
                    pixelsImg(i, j).CalcPixelData(img, pd);
                    sp->AddPixelInit(pd);
                }
            }
            sp->FinishInitialization();

            // Init pixelsBorder matrix and border length and border info in each Pixel
            int spRSize = 0, spCSize = 0;

            for (int i = i0; i < i1; i++) {
                pixelsImg(i, j0).SetBLeft();
                pixelsImg(i, j1 - 1).SetBRight();
                spRSize += pixelsImg(i, j0).GetRSize();
            }
            for (int j = j0; j < j1; j++) {
                pixelsImg(i0, j).SetBTop();
                pixelsImg(i1 - 1, j).SetBBottom();
                spCSize += pixelsImg(i0, j).GetCSize();
            }
            sp->SetBorderLength(2 * spRSize + 2 * spCSize);
            superpixels.push_back(sp);

            j0 = j1;
        }
        i0 = i1;
    }

}

void SPSegmentationEngine::InitializePPImage()
{
    ppImg = Matrix<Pixel*>(img.rows, img.cols);
    UpdatePPImage();
}

void SPSegmentationEngine::UpdatePPImage()
{
    for (Pixel& p : pixelsImg) {
        p.UpdatePPImage(ppImg);
    }
}


void SPSegmentationEngine::Reset()
{
    for (Superpixel* sp : superpixels) {
        delete sp;
    }
}

void SPSegmentationEngine::ProcessImage()
{
    Timer t0;

    Initialize([](int id) { return new Superpixel(id); });

    t0.Stop();
    performanceInfo.init = t0.GetTimeInSec();
    t0.Resume();

    Timer t1;
    bool splitted;
    int maxPixelSize = initialMaxPixelSize;
    int level = (int)ceil(log2(maxPixelSize));

    do {
        Timer t2;

        performanceInfo.levelMaxPixelSize.push_back(maxPixelSize);
        performanceInfo.levelIterations.push_back(0);
        for (int iteration = 0; iteration < params.iterations; iteration++) {
            int iters = IterateMoves(level);
            if (iters > performanceInfo.levelIterations.back())
                performanceInfo.levelIterations.back() = iters;
        }
        if (maxPixelSize <= params.minPixelSize) splitted = false;
        else splitted = SplitPixels(maxPixelSize);
        level--;

        t2.Stop();
        performanceInfo.levelTimes.push_back(t2.GetTimeInSec());
    } while (splitted);

    t0.Stop();
    t1.Stop();
    performanceInfo.total = t0.GetTimeInSec();
    performanceInfo.imgproc = t1.GetTimeInSec();
}

void SPSegmentationEngine::PrintDebugInfo2()
{
    std::ofstream ofs("C:\\tmp\\debugse2.txt");
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        ofs << sps->plane.x << " " << sps->plane.y << " " << sps->plane.z << std::endl;
    }
    ofs.close();
}

// Called in initialization and in re-estimation between layers.
void SPSegmentationEngine::UpdateBoundaryData()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // clear neighbors
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        sps->boundaryData.clear();
    }

    // update length & hiSum (written to smoSum)
    for (Pixel& p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (q->superPixel != sp) {
                    BInfo& bdpq = sp->boundaryData[sq];
                    BInfo& bdqp = sq->boundaryData[sp];
                    double sum;
                    int size;
                    int length;

                    q->CalcHiSmoothnessSumEI(directions[dir][2], depthImg, params.inlierThreshold, sp->plane, sq->plane, sum, size, length);
                    bdpq.hiCount += size;
                    bdpq.hiSum += sum;
                    bdpq.length += length;
                    bdqp.hiCount += size;
                    bdqp.hiSum += sum;
                    bdqp.length += length;
                }
            }
        }
    }

    //#pragma omp parallel
    for (Superpixel* s : superpixels) {
        SuperpixelStereo* sp = (SuperpixelStereo*)s;
        
        for (auto& bdIter : sp->boundaryData) {
            BInfo& bInfo = bdIter.second;
            SuperpixelStereo* sq = bdIter.first;
            double eSmoCoSum;
            int eSmoCoCount;
            double eSmoHiSum = bInfo.hiSum;
            double eSmoOcc = 1; // Phi!?

            CalcCoSmoothnessSum(depthImg, params.inlierThreshold, sp, sq, bInfo.coSum, bInfo.coCount);
            eSmoCoSum = bInfo.coSum;
            eSmoCoCount = bInfo.coCount;

            //double eHi = params.smoWeight*eSmoHiSum / item.second.bSize + params.priorWeight*params.hiPriorWeight;
            //double eCo = params.smoWeight*eSmoCoSum / (sp->GetSize() + sq->GetSize());
            //double eOcc = params.smoWeight*eSmoOcc + params.priorWeight*params.occPriorWeight;
            double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (eSmoHiSum / bInfo.hiCount + params.hiPriorWeight);
            double eCo = eSmoCoCount == 0 ? HUGE_VAL : (eSmoCoSum / eSmoCoCount); //(sp->GetSize() + sq->GetSize()); // + 0
            double eOcc = params.occPriorWeight;

            if (eCo <= eHi && eCo < eOcc) {
                bInfo.type = BTCo;
                bInfo.typePrior = 0;
            } else if (eHi <= eOcc && eHi <= eCo) {
                bInfo.type = BTHi;
                bInfo.typePrior = params.hiPriorWeight;
            } else {
                bInfo.type = BTLo;
                bInfo.typePrior = params.occPriorWeight;
            }
        }
    }

}

// Called in initialization and in re-estimation between layers.
// Version which *does not* check for inliers.
void SPSegmentationEngine::UpdateBoundaryData2()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // clear neighbors
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        sps->boundaryData.clear();
    }

    // update length & hiSum (written to smoSum)
    for (Pixel& p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (q->superPixel != sp) {
                    BInfo& bdpq = sp->boundaryData[sq];
                    BInfo& bdqp = sq->boundaryData[sp];
                    double sum;
                    int size;
                    int length;

                    q->CalcHiSmoothnessSumEI2(directions[dir][2], sp->plane, sq->plane, sum, size, length);
                    bdpq.hiCount += size;
                    bdpq.hiSum += sum;
                    bdpq.length += length;
                    bdqp.hiCount += size;
                    bdqp.hiSum += sum;
                    bdqp.length += length;
                }
            }
        }
    }

    //#pragma omp parallel
    for (Superpixel* s : superpixels) {
        SuperpixelStereo* sp = (SuperpixelStereo*)s;

        for (auto& bdIter : sp->boundaryData) {
            BInfo& bInfo = bdIter.second;
            SuperpixelStereo* sq = bdIter.first;
            double eSmoCoSum;
            int eSmoCoCount;
            double eSmoHiSum = bInfo.hiSum;
            double eSmoOcc = 1; // Phi!?

            CalcCoSmoothnessSum2(sp, sq, bInfo.coSum, bInfo.coCount);
            eSmoCoSum = bInfo.coSum;
            eSmoCoCount = bInfo.coCount;

            //double eHi = params.smoWeight*eSmoHiSum / item.second.bSize + params.priorWeight*params.hiPriorWeight;
            //double eCo = params.smoWeight*eSmoCoSum / (sp->GetSize() + sq->GetSize());
            //double eOcc = params.smoWeight*eSmoOcc + params.priorWeight*params.occPriorWeight;
            double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (eSmoHiSum / bInfo.hiCount + params.hiPriorWeight);
            double eCo = eSmoCoCount == 0 ? HUGE_VAL : (eSmoCoSum / eSmoCoCount); //(sp->GetSize() + sq->GetSize()); // + 0
            double eOcc = params.occPriorWeight;

            if (eCo <= eHi && eCo < eOcc) {
                bInfo.type = BTCo;
                bInfo.typePrior = 0;
            } else if (eHi <= eOcc && eHi <= eCo) {
                bInfo.type = BTHi;
                bInfo.typePrior = params.hiPriorWeight;
            } else {
                bInfo.type = BTLo;
                bInfo.typePrior = params.occPriorWeight;
            }
        }
    }

}

// Return true if pixels were actually split.
bool SPSegmentationEngine::SplitPixels(int& newMaxPixelSize)
{
    int imgPixelsRows = 0;
    int imgPixelsCols = 0;
    int maxPixelSize = 1;

    for (int i = 0; i < pixelsImg.rows; i++) {
        int rSize = pixelsImg(i, 0).GetRSize();
     
        imgPixelsRows += (rSize == 1) ? 1 : 2;
        if (rSize > maxPixelSize) maxPixelSize = rSize;
    }
    for (int j = 0; j < pixelsImg.cols; j++) {
        int cSize = pixelsImg(0, j).GetCSize();

        imgPixelsCols += (cSize == 1) ? 1 : 2;
        if (cSize > maxPixelSize) maxPixelSize = cSize;
    }

    if (maxPixelSize == 1) 
        return false;

    Matrix<Pixel> newPixelsImg(imgPixelsRows, imgPixelsCols);

    if (params.stereo) {
        for (Superpixel*& sp : superpixels) {
            ((SuperpixelStereo*)sp)->ClearPixelSet();
        }
    }

    int newRow = 0;

    for (int i = 0; i < pixelsImg.rows; i++) {
        int newCol = 0;
        int pRowSize = pixelsImg(i, 0).GetRSize();

        for (int j = 0; j < pixelsImg.cols; j++) {
            Pixel& p = pixelsImg(i, j);
            int pColSize = p.GetCSize();

            if (pRowSize == 1 && pColSize == 1) {
                Pixel& p11 = newPixelsImg(newRow, newCol);

                p.CopyTo(img, newRow, newCol, p11);
            } else if (pColSize == 1) { // split only row
                Pixel& p11 = newPixelsImg(newRow, newCol);
                Pixel& p21 = newPixelsImg(newRow + 1, newCol);

                p.SplitRow(img, newRow, newRow + 1, newCol, p11, p21);
            } else if (pRowSize == 1) { // split only column
                Pixel& p11 = newPixelsImg(newRow, newCol);
                Pixel& p12 = newPixelsImg(newRow, newCol + 1);

                p.SplitColumn(img, newRow, newCol, newCol + 1, p11, p12);
            } else { // split row and column
                Pixel& p11 = newPixelsImg(newRow, newCol);
                Pixel& p12 = newPixelsImg(newRow, newCol + 1);
                Pixel& p21 = newPixelsImg(newRow + 1, newCol);
                Pixel& p22 = newPixelsImg(newRow + 1, newCol + 1);

                p.Split(img, newRow, newRow + 1, newCol, newCol + 1, p11, p12, p21, p22);
            }
            newCol += (pColSize > 1) ? 2 : 1;
        }
        newRow += (pRowSize > 1) ? 2 : 1;
    }
    pixelsImg = newPixelsImg;

    for (Superpixel* sp : superpixels) {
        sp->RecalculateEnergies();
    }
    if (params.stereo) {
        for (Pixel& p : pixelsImg) {
            ((SuperpixelStereo*)p.superPixel)->AddToPixelSet(&p);
        }
        UpdatePPImage();
    }
    newMaxPixelSize = iCeilDiv(maxPixelSize, 2);
    return true;
}

static int dbgImageNum = 0;

int SPSegmentationEngine::Iterate(Deque<Pixel*>& listD, Matrix<bool>& inList)
{
    PixelMoveData tryMoveData[4];
    Superpixel* nbsp[5];
    int nbspSize;
    int popCount = 0;

    while (!listD.Empty() && popCount < params.maxUpdates) {
        Pixel* p = listD.PopFront();

        popCount++;

        if (p == nullptr) 
            continue;
        
        inList(p->row, p->col) = false;
        nbsp[0] = p->superPixel;
        nbspSize = 1;
        for (int m = 0; m < 4; m++) {
            Pixel* q = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);

            if (q == nullptr) tryMoveData[m].allowed = false;
            else {
                bool newNeighbor = true;

                for (int i = 0; i < nbspSize; i++) {
                    if (q->superPixel == nbsp[i]) {
                        newNeighbor = false;
                        break;
                    }
                }
                if (!newNeighbor) tryMoveData[m].allowed = false;
                else {
                    // if (params.stereo) TryMovePixelStereo(p, q, tryMoveData[m]);
                    // else TryMovePixel(p, q, tryMoveData[m]);
                    TryMovePixel(p, q, tryMoveData[m]);
                    nbsp[nbspSize++] = q->superPixel;
                }
            }
        }

        PixelMoveData* bestMoveData = FindBestMoveData(params, tryMoveData);

        if (bestMoveData != nullptr) {
            if (params.stereo) {
                //SuperpixelStereo* sps = (SuperpixelStereo*)(bestMoveData->p->superPixel);
                //double calc = sps->CalcDispEnergy(depthImg, params.inlierThreshold, params.noDisp);
                //if (fabs(calc - sps->GetDispSum()) > 0.01) {
                //    cout << "Disp sum mismatch";
                //}
                //sps->CheckRegEnergy();
                //sps->CheckAppEnergy(img);

                //double delta = TotalEnergyDelta(params, bestMoveData);

                //if (performanceInfo.levelMaxEDelta[level] < delta)
                //    performanceInfo.levelMaxEDelta[level] = delta;

                MovePixelStereo(pixelsImg, *bestMoveData, params.instantBoundary);

                //char fname[1000];
                //sprintf(fname, "c:\\tmp\\dbgbound-%03d.png", dbgImageNum++);
                //imwrite(fname, GetSegmentedImageStereo());
                //DebugBoundary();
                //DebugDispSums();
                //DebugNeighborhoods();

                //if (++dbgImageNum < 500) {
                //    char fname[1000];
                //    sprintf(fname, "c:\\tmp\\dbgbound-%03d.png", dbgImageNum);
                //    //if (dbgImageNum >= 160) 
                //        imwrite(fname, GetSegmentedImageStereo());
                //}

            } else {
                MovePixel(pixelsImg, *bestMoveData);
            }

            listD.PushBack(p);
            for (int m = 0; m < 4; m++) {
                Pixel* qq = PixelAt(pixelsImg, p->row + nDeltas[m][0], p->col + nDeltas[m][1]);
                if (qq != nullptr && p->superPixel != qq->superPixel && !inList(qq->row, qq->col)) {
                    listD.PushBack(qq);
                    inList(qq->row, qq->col) = true;
                }
            }
        }
    }
    return popCount;
}

int SPSegmentationEngine::IterateMoves(int level)
{
    params.SetLevelParams(level);

    Deque<Pixel*> listD(pixelsImg.rows * pixelsImg.cols);
    Matrix<bool> inList(pixelsImg.rows, pixelsImg.cols);

    // Initialize pixel (block) border listD
    std::fill(inList.begin(), inList.end(), false);
    for (Pixel& p : pixelsImg) {
        Pixel* q;

        for (int m = 0; m < 4; m++) {
            q = PixelAt(pixelsImg, p.row + nDeltas[m][0], p.col + nDeltas[m][1]);
            if (q != nullptr && p.superPixel != q->superPixel) {
                listD.PushBack(&p);
                inList(q->row, q->col) = true;
                break;
            }
        }
    }

    int nIterations = Iterate(listD, inList);

    return nIterations;
}

Mat SPSegmentationEngine::GetSegmentedImage()
{
    // if (params.stereo) return GetSegmentedImageStereo();
    // else return GetSegmentedImagePlain();
    return GetSegmentedImagePlain();
}

Mat SPSegmentationEngine::GetSegmentedImagePlain()
{
    Mat result = origImg.clone();
    Vec3b blackPixel(0, 0, 255);

    for (Pixel& p : pixelsImg) {
        if (p.BLeft()) {
            for (int r = p.ulr; r < p.lrr; r++) {
                result.at<Vec3b>(r, p.ulc) = blackPixel;
                // result.at<uchar>(r, p.ulc) = 255;
            }
        }
        if (p.BRight()) {
            for (int r = p.ulr; r < p.lrr; r++) {
                result.at<Vec3b>(r, p.lrc - 1) = blackPixel;
                // result.at<uchar>(r, p.lrc - 1) = 255;
            }
        }
        if (p.BTop()) {
            for (int c = p.ulc; c < p.lrc; c++) {
                result.at<Vec3b>(p.ulr, c) = blackPixel;
                // result.at<uchar>(p.ulr, c) = 255;
            }
        }
        if (p.BBottom()) {
            for (int c = p.ulc; c < p.lrc; c++) {
                result.at<Vec3b>(p.lrr - 1, c) = blackPixel;
                // result.at<uchar>(p.lrr - 1, c) = 255;
            }
        }
    }
    return result;
}

const Vec3b& BoundaryColor(Pixel* p, Pixel* q)
{
    static const Vec3b pixelColors[] = { Vec3b(0, 0, 0), Vec3b(0, 255, 0), Vec3b(255, 0, 0), Vec3b(0, 0, 196), Vec3b(0, 0, 196) };

    if (p == nullptr || q == nullptr) 
        return pixelColors[0];

    SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
    SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

    if (sp == nullptr || sq == nullptr) 
        return pixelColors[0];

    auto bdIter = sp->boundaryData.find(sq);
    if (bdIter == sp->boundaryData.end()) 
        return pixelColors[0];
    else {
        if (bdIter->second.type > 0 || bdIter->second.type < 5) return pixelColors[bdIter->second.type];
        else return pixelColors[0];
    }
}

// Mat SPSegmentationEngine::GetSegmentedImageStereo()
// { 
//     if (!params.stereo) return GetSegmentedImagePlain();

//     Mat result = origImg.clone();

//     for (Pixel& p : pixelsImg) {
//         if (p.BLeft()) {
//             Pixel* q = PixelAt(pixelsImg, p.row, p.col - 1);
//             const Vec3b& color = BoundaryColor(&p, q);

//             for (int r = p.ulr; r < p.lrr; r++) {
//                 result.at<Vec3b>(r, p.ulc) = color;
//             }
//         }
//         if (p.BRight()) {
//             Pixel* q = PixelAt(pixelsImg, p.row, p.col + 1);
//             const Vec3b& color = BoundaryColor(&p, q);

//             for (int r = p.ulr; r < p.lrr; r++) {
//                 result.at<Vec3b>(r, p.lrc - 1) = color;
//             }
//         }
//         if (p.BTop()) {
//             Pixel* q = PixelAt(pixelsImg, p.row - 1, p.col);

//             const Vec3b& color = BoundaryColor(&p, q);
//             for (int c = p.ulc; c < p.lrc; c++) {
//                 result.at<Vec3b>(p.ulr, c) = color;
//             }
//         }
//         if (p.BBottom()) {
//             Pixel* q = PixelAt(pixelsImg, p.row + 1, p.col);
//             const Vec3b& color = BoundaryColor(&p, q);

//             for (int c = p.ulc; c < p.lrc; c++) {
//                 result.at<Vec3b>(p.lrr - 1, c) = color;
//             }
//         }
//     }
//     return result;

// }

// Mat SPSegmentationEngine::GetDisparity() const
// {
//     Mat_<unsigned short> result = Mat_<unsigned short>(ppImg.rows, ppImg.cols);

//     for (int i = 0; i < ppImg.rows; i++) {
//         for (int j = 0; j < ppImg.cols; j++) {
//             SuperpixelStereo* sps = (SuperpixelStereo*)ppImg(i, j)->superPixel;
//             double val = DotProduct(sps->plane, i, j, 1.0);
//             result(i, j) = val < 256.0 ? (val < 0 ? 0 : val * 256.0) : 65535;
//         }
//     }
//     return result;
// }


Mat SPSegmentationEngine::GetSegmentation() const
{
    // Mat result = Mat_<unsigned short>(pixelsImg.rows, pixelsImg.cols);
    Mat result = Mat::zeros(pixelsImg.rows,pixelsImg.cols, CV_16UC1);
    // std::cout<<"debug  :"<< result.at<uchar>(0,8)<<std::endl;
    std::unordered_map<Superpixel*, int> indexMap;
    int maxIndex = 0;

    for (const Pixel& p : pixelsImg) {
        if (indexMap.find(p.superPixel) == indexMap.end()) {
            indexMap[p.superPixel] = maxIndex++;
        }
    }
    for (const Pixel& p : pixelsImg) {
        result.at<ushort>(p.row, p.col) = indexMap[p.superPixel];
    }
    return result;
}

std::string SPSegmentationEngine::GetSegmentedImageInfo()
{
    std::map<Superpixel*, std::vector<Pixel*>> spMap;
    std::stringstream ss;

    for (Pixel& p : pixelsImg) {
        Superpixel* sp = p.superPixel;
        spMap[sp].push_back(&p);
    }
    ss << '{';
    bool firstSp = true;
    for (auto mPair : spMap) {
        if (firstSp) firstSp = false; else ss << ',';
        ss << '{';
        ss << mPair.first->GetAppEnergy();
        ss << ',';
        ss << mPair.first->GetRegEnergy();
        ss << ',';
        ss << mPair.first->GetSize();
        ss << ',';
        ss << mPair.first->GetBorderLength();
        ss << ',';

        double mr, mc;
        mPair.first->GetMean(mr, mc);
        ss << '{' << mr << ',' << mc << '}';

        ss << ",";
        ss << std::fixed << mPair.first->GetRegEnergy();

        ss << ',' << '{';

        bool firstP = true;
        for (Pixel* p : mPair.second) {
            if (firstP) firstP = false; else ss << ',';
            ss << p->GetPixelsAsString();
        }
        ss << '}' << '}';
    }
    ss << '}';
    return ss.str();
}

void SPSegmentationEngine::PrintDebugInfo()
{
    double appESum = 0.0;
    double regESum = 0.0;
    double dispESum = 0.0;

    for (Superpixel* sp : superpixels) {
        appESum += sp->GetAppEnergy();
        regESum += sp->GetRegEnergy();
    }
    std::cout << "Reg energy mean: " << regESum / superpixels.size() << std::endl;
    std::cout << "Disp energy mean: " << dispESum / superpixels.size() << std::endl;
}


// void SPSegmentationEngine::PrintDebugInfoStereo()
// {
//     if (!params.debugOutput)
//         return;

//     StatData stat;

//     MeanAndVariance(superpixels.begin(), superpixels.end(),
//         [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetAppEnergy(); },
//         stat);
//     cout << "App energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

//     MeanAndVariance(superpixels.begin(), superpixels.end(),
//         [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetRegEnergy(); },
//         stat);
//     cout << "Reg energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

//     MeanAndVariance(superpixels.begin(), superpixels.end(),
//         [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetBorderLength(); },
//         stat);
//     cout << "Border length mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

//     MeanAndVariance(superpixels.begin(), superpixels.end(),
//         [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetDispSum(); },
//         stat);
//     cout << "Disp energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;

//     MeanAndVariance(superpixels.begin(), superpixels.end(),
//         [](Superpixel* sp) { return ((SuperpixelStereo*)sp)->GetSmoEnergy(); },
//         stat);
//     cout << "Smo energy mean: " << stat.mean << ", variance: " << stat.var << ", min: " << stat.min << ", max: " << stat.max << endl;
// }

int SPSegmentationEngine::GetNoOfSuperpixels() const
{
    return (int)superpixels.size();
}

void SPSegmentationEngine::PrintPerformanceInfo()
{
    if (params.timingOutput && !params.debugOutput) {
        std::cout << "Processing time: " << performanceInfo.total << " sec." << std::endl;
    }
    if (params.debugOutput) {
        std::cout << "No. of superpixels: " << GetNoOfSuperpixels() << std::endl;
        std::cout << "Initialization time: " << performanceInfo.init << " sec." << std::endl;
        std::cout << "Ransac time: " << performanceInfo.ransac << " sec." << std::endl;
        std::cout << "Time of image processing: " << performanceInfo.imgproc << " sec." << std::endl;
        std::cout << "Total time: " << performanceInfo.total << " sec." << std::endl;
        std::cout << "Times for each level (in sec.): ";
        for (double& t : performanceInfo.levelTimes)
            std::cout << t << ' ';
        std::cout << std::endl;
        std::cout << "Max energy delta for each level: ";
        for (double& t : performanceInfo.levelMaxEDelta)
            std::cout << t << ' ';
        std::cout << std::endl;
        std::cout << "Iterations for each level: ";
        for (int& c : performanceInfo.levelIterations)
            std::cout << c << ' ';
        std::cout << std::endl;
        std::cout << "Max pixel sizes for each level: ";
        for (int& ps : performanceInfo.levelMaxPixelSize)
            std::cout << ps << ' ';
        std::cout << std::endl;

        int minBDSize = INT_MAX;
        int maxBDSize = 0;

        if (params.stereo) {
            for (Superpixel* sp : superpixels) {
                SuperpixelStereo* sps = (SuperpixelStereo*)sp;
                if (minBDSize > sps->boundaryData.size())
                    minBDSize = sps->boundaryData.size();
                if (maxBDSize < sps->boundaryData.size())
                    maxBDSize = sps->boundaryData.size();
            }
            std::cout << "Max boundary size: " << maxBDSize << std::endl;
            std::cout << "Min boundary size: " << minBDSize << std::endl;
        }
    }
}

void SPSegmentationEngine::UpdateInlierSums()
{
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;

        sps->sumIRow = 0; sps->sumICol = 0;         // Sum of terms computed for inlier points
        sps->sumIRow2 = 0; sps->sumICol2 = 0;
        sps->sumIRowCol = 0;
        sps->sumIRowD = 0.0, sps->sumIColD = 0.0;
        sps->sumID = 0.0;
        sps->nI = 0;
    }
    for (int i = 0; i < ppImg.rows; i++) {
        for (int j = 0; j < ppImg.cols; j++) {
            Pixel* p = ppImg(i, j);
            SuperpixelStereo* sps = (SuperpixelStereo*)p->superPixel;
            const double& disp = depthImg(i, j);

            if (disp > 0) {
                bool inlier = fabs(DotProduct(sps->plane, i, j, 1.0) - disp) < params.inlierThreshold;

                if (inlier) {
                    sps->sumIRow += i; sps->sumIRow2 += i*i;
                    sps->sumICol += j; sps->sumICol2 += j*j;
                    sps->sumIRowCol += i*j;
                    sps->sumIRowD += i*disp; sps->sumIColD += j*disp;
                    sps->sumID += disp;
                    sps->nI++;
                }
            }
        }
    }
}

// Try to move Pixel p to Superpixel containing Pixel q with coordinates (qRow, qCol)
// Note: pixel q is must be neighbor of p and p->superPixel != q->superPixel
// Fills psd, returns psd.allowed
// Note: energy deltas in psd are "energy_before - energy_after"
bool SPSegmentationEngine::TryMovePixel(Pixel* p, Pixel* q, PixelMoveData& psd)
{
    Superpixel* sp = p->superPixel;
    Superpixel* sq = q->superPixel;

    if (sp == sq || !IsSuperpixelRegionConnectedOptimized(pixelsImg, p, p->row - 1, p->col - 1, p->row + 2, p->col + 2)) {
        psd.allowed = false;
        return false;
    }

    int spSize = sp->GetSize(), sqSize = sq->GetSize();
    double spEApp = sp->GetAppEnergy(), sqEApp = sq->GetAppEnergy();
    double spEReg = sp->GetRegEnergy(), sqEReg = sq->GetRegEnergy();

    PixelChangeData pcd;
    PixelChangeData qcd;
    PixelData pd;
    int spbl, sqbl, sobl;

    p->CalcPixelData(img, pd);
    sp->GetRemovePixelData(pd, pcd);
    sq->GetAddPixelData(pd, qcd);
    CalcSuperpixelBoundaryLength(pixelsImg, p, sp, sq, spbl, sqbl, sobl);

    psd.p = p;
    psd.q = q;
    psd.pSize = pcd.newSize;
    psd.qSize = qcd.newSize;
    psd.eAppDelta = spEApp + sqEApp - pcd.newEApp - qcd.newEApp;
    psd.eRegDelta = spEReg + sqEReg - pcd.newEReg - qcd.newEReg;
    psd.bLenDelta = sqbl - spbl;
    psd.allowed = true;
    psd.pixelData = pd;
    return true;
}

// bool SPSegmentationEngine::TryMovePixelStereo(Pixel* p, Pixel* q, PixelMoveData& psd)
// {
//     SuperpixelStereo* sp = (SuperpixelStereo*)p->superPixel;
//     SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

//     if (sp == sq || !IsSuperpixelRegionConnectedOptimized(pixelsImg, p, p->row - 1, p->col - 1, p->row + 2, p->col + 2)) {
//         psd.allowed = false;
//         return false;
//     }

//     double pSize = p->GetSize(), qSize = q->GetSize();
//     int spSize = sp->GetSize(), sqSize = sq->GetSize();
//     double spEApp = sp->GetAppEnergy(), sqEApp = sq->GetAppEnergy();
//     double spEReg = sp->GetRegEnergy(), sqEReg = sq->GetRegEnergy();
//     double spEDisp = sp->GetDispSum(), sqEDisp = sq->GetDispSum();
//     double spESmo = sp->GetSmoEnergy(), sqESmo = sq->GetSmoEnergy();
//     double spEPrior = sp->GetPriorEnergy(), sqEPrior = sq->GetPriorEnergy();

//     PixelChangeDataStereo pcd;
//     PixelChangeDataStereo qcd;
//     PixelData pd;
//     int spbl, sqbl, sobl;

//     p->CalcPixelDataStereo(img, depthImg, sp->plane, sq->plane, params.inlierThreshold, params.noDisp, pd);
//     sp->GetRemovePixelDataStereo(pd, pcd);
//     sq->GetAddPixelDataStereo(pd, qcd);
//     if (params.instantBoundary) CalcBorderChangeDataStereo(pixelsImg, depthImg, params, p, q, psd.bDataP, psd.bDataQ, psd.prem);
//     CalcSuperpixelBoundaryLength(pixelsImg, p, sp, sq, spbl, sqbl, sobl);

//     psd.p = p;
//     psd.q = q;
//     psd.pSize = pcd.newSize;
//     psd.qSize = qcd.newSize;
//     psd.eAppDelta = spEApp + sqEApp - pcd.newEApp - qcd.newEApp;
//     psd.eRegDelta = spEReg + sqEReg - pcd.newEReg - qcd.newEReg;
//     psd.bLenDelta = sqbl - spbl;
//     psd.eDispDelta = spEDisp + sqEDisp - pcd.newEDisp - qcd.newEDisp;
//     psd.ePriorDelta = 0;
//     psd.eSmoDelta = (!params.instantBoundary) ? 0.0 : (spESmo + sqESmo - GetSmoEnergy(psd.bDataP, sp, psd.pSize, sq, psd.qSize) -
//         GetSmoEnergy(psd.bDataQ, sq, psd.qSize, sp, psd.pSize));
//     psd.allowed = true;
//     psd.pixelData = pd;
//     return true;
// }

void SPSegmentationEngine::DebugNeighborhoods()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // update length & hiSum (written to smoSum)
    for (Pixel& p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (sp->id != sq->id) {
                    if (sp->boundaryData.find(sq) == sp->boundaryData.end())
                        throw Exception();
                    if (sq->boundaryData.find(sp) == sq->boundaryData.end())
                        throw Exception();
                }
            }
        }
    }

}

void SPSegmentationEngine::DebugBoundary()
{
    const int directions[2][3] = { { 0, 1, BLeftFlag }, { 1, 0, BTopFlag } };

    // unordered map is not used; pair does not have hash function, but performance is not important....
    std::map<std::pair<SuperpixelStereo*, SuperpixelStereo*>, BInfo> newBoundaryData;

    // update length & hiSum (written to smoSum)
    for (Pixel p : pixelsImg) {
        SuperpixelStereo* sp = (SuperpixelStereo*)p.superPixel;

        for (int dir = 0; dir < 2; dir++) {
            Pixel* q = PixelAt(pixelsImg, p.row + directions[dir][0], p.col + directions[dir][1]);

            if (q != nullptr) {
                SuperpixelStereo* sq = (SuperpixelStereo*)q->superPixel;

                if (q->superPixel != sp) {
                    BInfo& bdpq = newBoundaryData[std::pair<SuperpixelStereo*, SuperpixelStereo*>(sp, sq)];
                    BInfo& bdqp = newBoundaryData[std::pair<SuperpixelStereo*, SuperpixelStereo*>(sq, sp)];
                    double sum;
                    int size;
                    int length;

                    q->CalcHiSmoothnessSumEI(directions[dir][2], depthImg, params.inlierThreshold, sp->plane, sq->plane, sum, size, length);
                    bdpq.hiCount += size;
                    bdpq.hiSum += sum;
                    bdpq.length += length;
                    bdqp.hiCount += size;
                    bdqp.hiSum += sum;
                    bdqp.length += length;
                }
            }
        }
    }


    for (auto& bdInfoIter : newBoundaryData) {
        SuperpixelStereo* sp = bdInfoIter.first.first;
        SuperpixelStereo* sq = bdInfoIter.first.second;
        BInfo& bInfo = bdInfoIter.second;
        double eSmoCoSum;
        int eSmoCoCount;
        double eSmoHiSum = bInfo.hiSum;
        double eSmoOcc = 1; // Phi!?

        CalcCoSmoothnessSum(depthImg, params.inlierThreshold, sp, sq, bInfo.coSum, bInfo.coCount);
        eSmoCoSum = bInfo.coSum;
        eSmoCoCount = bInfo.coCount;

        //double eHi = params.smoWeight*eSmoHiSum / item.second.bSize + params.priorWeight*params.hiPriorWeight;
        //double eCo = params.smoWeight*eSmoCoSum / (sp->GetSize() + sq->GetSize());
        //double eOcc = params.smoWeight*eSmoOcc + params.priorWeight*params.occPriorWeight;
        double eHi = bInfo.hiCount == 0 ? HUGE_VAL : (eSmoHiSum / bInfo.hiCount + params.hiPriorWeight);
        double eCo = eSmoCoCount == 0 ? HUGE_VAL : (eSmoCoSum / eSmoCoCount); //(sp->GetSize() + sq->GetSize()); // + 0
        double eOcc = params.occPriorWeight;

        if (eCo <= eHi && eCo < eOcc) {
            bInfo.type = BTCo;
            bInfo.typePrior = 0;
        } else if (eHi <= eOcc && eHi <= eCo) {
            bInfo.type = BTHi;
            bInfo.typePrior = params.hiPriorWeight;
        } else {
            bInfo.type = BTLo;
            bInfo.typePrior = params.occPriorWeight;
        }
    }

    for (Superpixel* s : superpixels) {
        SuperpixelStereo* sp = (SuperpixelStereo*)s;

        for (auto& bdIter : sp->boundaryData) {
            BInfo& bInfo = bdIter.second;
            SuperpixelStereo* sq = bdIter.first;

            auto nnbIter = newBoundaryData.find(std::pair<SuperpixelStereo*, SuperpixelStereo*>(sp, sq));
            if (nnbIter == newBoundaryData.end()) throw std::runtime_error("Can not find boundary");
            else { // nnbiter-> re-calculated // bInfo -> currently in 
                if (nnbIter->second.type != bInfo.type) {
                    throw std::runtime_error("type mismatch");
                }
                if (bInfo.type == BTCo) {
                    if (nnbIter->second.coCount != bInfo.coCount) {
                        throw std::runtime_error("energy mismatch -- coCount");
                    }
                    if (fabs(nnbIter->second.coSum - bInfo.coSum) > 0.01) {
                        throw std::runtime_error("energy mismatch -- coSum");
                    }
                }
                if (bInfo.type == BTHi) {
                    if (fabs(nnbIter->second.hiSum - bInfo.hiSum) > 0.01) {
                        throw std::runtime_error("energy mismatch -- hiSum");
                    }
                    if (nnbIter->second.hiCount != bInfo.hiCount) {
                        throw std::runtime_error("energy mismatch -- hiCount");
                    }
                }
            }
        }
    }

}

void SPSegmentationEngine::DebugDispSums()
{
    for (Superpixel* sp : superpixels) {
        SuperpixelStereo* sps = (SuperpixelStereo*)sp;
        double disp = sps->CalcDispEnergy(depthImg, params.inlierThreshold, params.noDisp);
        if (fabs(sps->sumDisp - disp) > 0.01) {
            throw std::runtime_error("disp sum mismatch");
        }
    }
}

std::vector<Superpixel*> SPSegmentationEngine::GetSuperpixels()
{
    return superpixels;
}


// Functions
///////////////////////////////////////////////////////////////////////////////


SPSegmentationParameters ReadParameters(const std::string& fileName, const SPSegmentationParameters& defaultValue)
{
    try {
        FileStorage fs(fileName, FileStorage::READ);
        SPSegmentationParameters sp;

        fs.root() >> sp;
        return sp;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return defaultValue;
    }
}


