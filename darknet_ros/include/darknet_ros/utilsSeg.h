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

#include <string>
#include <vector>
#include <functional>

//using namespace std;

// Returns list of files in dir following pattern. See implementation comment for restrictions.
void FindFiles(const std::string& dir, const std::string& pattern, std::vector<std::string>& files, bool fullPath = false);

// Returns fileName with extension replaced by newExt (extension is substring of fileName from (including) the last dot)
std::string ChangeExtension(const std::string& fileName, const std::string& newExt);

// Extracts path name from fileName (appended with '/' inf not empty)
std::string FilePath(const std::string& fileName);

std::string FileName(const std::string& fileName);

// Append '/' to the end of dirName if not present and non-empty
void EndDir(std::string& dirName);
void MkDir(const std::string& dirName);

std::string Format(const std::string& fs, ...);

// From Jian's original code
cv::Mat ConvertRGBToLab(const cv::Mat& img);

// Input is single channel ushort image, output is single channel double image
// Input is divided by 256.0 and gaps (regions with value 0) are "filled", 
cv::Mat AdjustDisparityImage(const cv::Mat& img);

cv::Mat FillGapsInDisparityImage(const cv::Mat& img);

cv::Mat InpaintDisparityImage(const cv::Mat& img);

struct StatData {
    double mean, var, min, max;
};

template<typename It, typename Func>
void MeanAndVariance(It first, It last, Func f, StatData& stat)
{
    double s = 0.0;
    int N = 0;
    stat.max = DBL_MIN;
    stat.min = DBL_MAX;

    for (It iter = first; iter != last; iter++) {
        double val = f(*iter);
        if (val > stat.max) stat.max = val;
        if (val < stat.min) stat.min = val;
        s += val;
        N++;
    }
    if (N == 0) {
        stat.mean = stat.var = 0;
        stat.max = stat.min = 0;
        return;
    }
    stat.mean = s / N;
    if (N == 1) {
        stat.var = 0;
        return;
    }
    s = 0.0;
    for (It iter = first; iter != last; iter++) {
        s += Sqr(f(*iter) - stat.mean);
    }
    stat.var = sqrt(s / (N - 1));
}

