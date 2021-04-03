#pragma once
#include <opencv2/core.hpp>
#include "tensorflow/lite/c/c_api.h"

using namespace std;
using namespace cv;

class StyleTransfer
{
public:
	// Methods
	StyleTransfer(const char* modelPath);
	~StyleTransfer();
	Mat stylize(Mat src);
private:
	// Members
	const int MODEL_CNLS = 3;

	// Limit the input image size so neither width nor height bigger than this
	// otherwise the stylization might crash
	const int MAX_IMG_SIZE_LEN = 700;

	TfLiteModel* m_model;
	TfLiteInterpreter* m_interpreter;
	TfLiteTensor* m_input_tensor = nullptr;
	const TfLiteTensor* m_output_tensor = nullptr;

	// Methods
	void initModel(const char* modelPath);
};
