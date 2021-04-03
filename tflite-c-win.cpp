#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ObjectDetector.h"
#include "ImageSegmentation.h"
#include "StyleTransfer.h"

using namespace std;
using namespace cv;

void runObjectDetection() {
	Mat src = imread("od_test.jpg");

	ObjectDetector detector = ObjectDetector("ssd_mobilenet_v3_float.tflite", false);
	DetectResult* res = detector.detect(src);
	for (int i = 0; i < detector.DETECT_NUM; ++i) {
		int label = res[i].label;
		float score = res[i].score;
		float xmin = res[i].xmin;
		float xmax = res[i].xmax;
		float ymin = res[i].ymin;
		float ymax = res[i].ymax;

		rectangle(src, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		putText(src, to_string(label) + "-" + to_string(score), Point(xmin, ymin), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255), 2);
	}

	imshow("test", src);
	waitKey(0);
}

void runSegmentation() {
	Mat src = imread("seg_test.jpg");

	ImageSegmentation segmentation = ImageSegmentation("deeplabv3_mnv2_pascal.tflite");
	SegmentationResult res = segmentation.segmentImage(src);

	Mat mask = res.mask;

	// Expand the mask a bit
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(mask, mask, element);

	// Mask is gray so we make it 3 channels like img so we can do arithmetic ops
	cvtColor(mask, mask, COLOR_GRAY2BGR);

	Mat bgMask = ~mask;
	Mat result;
	GaussianBlur(bgMask, bgMask, Size(), 10);
	add(src, bgMask, result);

	imshow("seg", result);
	waitKey(0);
}

void runStyleTransfer() {
	Mat src = imread("seg_test.jpg");

	StyleTransfer styler = StyleTransfer("style_transfer1.tflite");

	cout << "Stylizing image, this takes few seconds..." << endl;

	Mat res = styler.stylize(src);

	if (res.empty()) {
		cout << "Something went wrong..." << endl;
	}
	else 
	{
		imshow("res", res);
		waitKey(0);
	}
}

int main()
{
	// runObjectDetection();
	// runSegmentation();
	runStyleTransfer();
}
