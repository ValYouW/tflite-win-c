#include "Utils.h"
#include "StyleTransfer.h"

StyleTransfer::StyleTransfer(const char* modelPath)
{
	initModel(modelPath);
}

StyleTransfer::~StyleTransfer() {
	if (m_model != nullptr)
		TfLiteModelDelete(m_model);
}

void StyleTransfer::initModel(const char* modelPath) {
	m_model = TfLiteModelCreateFromFile(modelPath);
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

	// Check input
	if (TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1) {
		printf("Style model graph needs to have 1 and only 1 input!");
		return;
	}

	// Check output
	if (TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 1) {
		printf("Style model graph needs to have 1 and only 1 output!");
		return;
	}
}

Mat StyleTransfer::stylize(Mat src) {
	src = Utils::resizeAspect(src, MAX_IMG_SIZE_LEN);
	src = Utils::toRGB(src);

	Mat res;

	// resize input tensor
	// NOTE: WE ASSUME the input tensor is at index 0 in our model
	vector<int> sizes = { 1, src.rows, src.cols, MODEL_CNLS };
	if (TfLiteInterpreterResizeInputTensor(m_interpreter, 0, sizes.data(), 4) != kTfLiteOk) {
		printf("Failed to resize input tensor!");
		return res;
	}

	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(m_interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return res;
	}

	m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
	if (m_input_tensor->type != kTfLiteFloat32) {
		printf("Detection model input should be kTfLiteFloat32!");
		return res;
	}

	if (m_input_tensor->dims->data[0] != 1 || m_input_tensor->dims->data[1] != src.rows || m_input_tensor->dims->data[2] != src.cols || m_input_tensor->dims->data[3] != MODEL_CNLS) {
		printf("Detection input tensor has wrong size");
		return res;
	}

	Mat fimage;
	src.convertTo(fimage, CV_32FC3);

	float* dst = m_input_tensor->data.f;
	memcpy(dst, (float*)fimage.data, sizeof(float) * src.rows * src.cols * MODEL_CNLS);

	// Invoke...
	if (TfLiteInterpreterInvoke(m_interpreter) != kTfLiteOk) {
		printf("Error invoking detection model");
		return res;
	}

	// Output tensor should contain the stylized image
	m_output_tensor = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);

	// Although we expect the output image size to be like the input,
	// sometimes the model adds few lines...
	int resH = m_output_tensor->dims->data[1];
	int resW = m_output_tensor->dims->data[2];

	// Create opencv image from the output tensor (data treated as float)
	Mat outImg(resH, resW, CV_32FC3, m_output_tensor->data.f);

	// Convert the result to uint8 image and 
	outImg.convertTo(res, CV_8UC3);

	// model output image is RGB, convert to BGR
	cvtColor(res, res, COLOR_RGB2BGR);
	return res;
}
