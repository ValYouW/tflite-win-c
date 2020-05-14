#pragma once
#include <opencv2/core.hpp>
#include "tensorflow/lite/c/c_api.h"

using namespace std;
using namespace cv;

struct SegmentationResult
{
	float segmentedArea;
	Mat mask;

	SegmentationResult(int segmentedArea, Mat mask)
	{
		SegmentationResult::segmentedArea = segmentedArea;
		SegmentationResult::mask = mask;
	}
};

class ImageSegmentation
{
public:
	// Methods
	ImageSegmentation(const char* deeplabModelPath, bool quantized = false);
	~ImageSegmentation();
	SegmentationResult segmentImage(Mat src);
private:
	// Members
	const int MODEL_SIZE = 513;
	const int MODEL_CNLS = 3;
	const int CLASS_COUNT = 21;
	const float IMAGE_MEAN = 128.0;
	const float IMAGE_STD = 128.0;
	bool m_modelQuantized = false;
	TfLiteModel* m_model;
	TfLiteInterpreter* m_interpreter;
	TfLiteTensor* m_input_tensor = nullptr;
	const TfLiteTensor* m_output_mask = nullptr;

	// Methods
	void initModel(const char* deeplabModelPath);
};
