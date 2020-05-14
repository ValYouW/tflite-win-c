#include <chrono>
#include <algorithm>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/c/c_api.h"
#include "ImageSegmentation.h"

using namespace cv;

ImageSegmentation::ImageSegmentation(const char* deeplabModelPath, bool quantized)
{
	m_modelQuantized = quantized;
	initModel(deeplabModelPath);
}

ImageSegmentation::~ImageSegmentation() {
	if (m_model != nullptr)
		TfLiteModelDelete(m_model);
}

void ImageSegmentation::initModel(const char* deeplabModelPath) {
	m_model = TfLiteModelCreateFromFile(deeplabModelPath);
	if (m_model == nullptr) {
		printf("Failed to load model");
		return;
	}

	// Build the interpreter
	TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 1);

	// Create the interpreter.
	m_interpreter = TfLiteInterpreterCreate(m_model, options);
	if (m_interpreter == nullptr) {
		printf("Failed to create interpreter");
		return;
	}

	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(m_interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	// Find input tensors.
	if (TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
	if (m_modelQuantized && m_input_tensor->type != kTfLiteUInt8) {
		printf("Detection model input should be kTfLiteUInt8!");
		return;
	}

	if (!m_modelQuantized && m_input_tensor->type != kTfLiteFloat32) {
		printf("Detection model input should be kTfLiteFloat32!");
		return;
	}

	if (m_input_tensor->dims->data[0] != 1 || m_input_tensor->dims->data[1] != MODEL_SIZE || m_input_tensor->dims->data[2] != MODEL_SIZE || m_input_tensor->dims->data[3] != MODEL_CNLS) {
		printf("Detection model must have input dims of 1x%ix%ix%i", MODEL_SIZE, MODEL_SIZE, MODEL_CNLS);
		return;
	}

	// Find output tensors.
	if (TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 1) {
		printf("Detection model graph needs to have 1 and only 1 output!");
		return;
	}

	m_output_mask = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);
}

SegmentationResult ImageSegmentation::segmentImage(Mat src) {
	Mat mask;
	SegmentationResult res(0, mask);
	if (m_model == nullptr) {
		return res;
	}

	int origWidth = src.cols;
	int origHeight = src.rows;

	Mat image;
	resize(src, image, Size(MODEL_SIZE, MODEL_SIZE), 0, 0, INTER_AREA);
	int cnls = image.type();
	if (cnls == CV_8UC1) {
		cvtColor(image, image, COLOR_GRAY2RGB);
	}
	else if (cnls == CV_8UC3) {
		cvtColor(image, image, COLOR_BGR2RGB);
	}
	else if (cnls == CV_8UC4) {
		cvtColor(image, image, COLOR_BGRA2RGB);
	}

	if (m_modelQuantized) {
		// Copy image into input tensor
		uchar* dst = m_input_tensor->data.uint8;
		memcpy(dst, image.data,
			sizeof(uchar) * MODEL_SIZE * MODEL_SIZE * MODEL_CNLS);
	}
	else {
		// Normalize the image based on std and mean (p' = (p-mean)/std)
		Mat fimage;
		image.convertTo(fimage, CV_32FC3, 1 / IMAGE_STD, -IMAGE_MEAN / IMAGE_STD);

		// Copy image into input tensor
		float* dst = m_input_tensor->data.f;
		memcpy(dst, fimage.data,
			sizeof(float) * MODEL_SIZE * MODEL_SIZE * MODEL_CNLS);
	}
	if (TfLiteInterpreterInvoke(m_interpreter) != kTfLiteOk) {
		printf("Error invoking detection model");
		return res;
	}

	const int64_t* maskImage = m_output_mask->data.i64;

	// Post process result
	// Class list:
	// ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tv"]
	mask = Mat(MODEL_SIZE, MODEL_SIZE, CV_8UC1, Scalar(0));
	unsigned char* maskData = mask.data;
	int segmentedPixels = 0;
	for (int y = 0; y < MODEL_SIZE; ++y) {
		for (int x = 0; x < MODEL_SIZE; ++x) {
			int idx = y * MODEL_SIZE + x;
			int64_t classId = maskImage[idx];
			if (classId == 0)
				continue;

			++segmentedPixels;
			maskData[idx] = 255;
		}
	}

	resize(mask, mask, Size(origWidth, origHeight), 0, 0, INTER_CUBIC);
	threshold(mask, mask, 128, 255, THRESH_BINARY);
	res.segmentedArea = (float)segmentedPixels / (MODEL_SIZE * MODEL_SIZE) * 100;
	res.mask = mask;
	return res;
}
