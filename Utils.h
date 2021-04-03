#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

namespace Utils {
	Mat toRGB(Mat img) {
		Mat res;
		int cnls = img.type();
		if (cnls == CV_8UC1) {
			cvtColor(img, res, COLOR_GRAY2RGB);
		}
		else if (cnls == CV_8UC3) {
			cvtColor(img, res, COLOR_BGR2RGB);
		}
		else if (cnls == CV_8UC4) {
			cvtColor(img, res, COLOR_BGRA2RGB);
		}

		return res;
	}

	Mat resizeAspect(Mat img, int maxSideSize) {
		if (img.empty())
			return img;

		if (img.cols <= maxSideSize && img.rows <= maxSideSize)
			return img;

		Size2f s;
		if (img.cols > img.rows)
		{
			s = Size2f(maxSideSize, ((double)img.rows / img.cols) * maxSideSize);
		}
		else
		{
			s = Size2f(((double)img.cols / img.rows) * maxSideSize, maxSideSize);
		}

		Mat dst;
		resize(img, dst, s);

		return dst;
	}
}
