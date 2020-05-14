// tflite-c-win.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ObjectDetector.h"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("test.jpg");

	ObjectDetector detector = ObjectDetector("model-float.tflite", false);
	DetectResult* res = detector.detect(src);
	for (int i = 0; i < detector.DETECT_NUM; ++i) {
		int label = res[i].label;
		float score = res[i].score;
		float xmin = res[i].xmin;
		float xmax = res[i].xmax;
		float ymin = res[i].ymin;
		float ymax = res[i].ymax;

		rectangle(src, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		putText(src, to_string(label) + "-" + to_string(score) , Point(xmin, ymin), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255), 2);
	}

	imshow("test", src);
	waitKey(0);
}
