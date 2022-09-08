#include "recog_interface.h"

using namespace std;
//yolov5输出结构体


bool compile_yolo_engine(int deviceid,const char* model_path,const char* _mode, const char* _type, const char* name) {
	//定义mode,默认FP32
	TRT::Mode mode;
	if (strcmp(_mode, "FP32") == 0) {
		mode = TRT::Mode::FP32;
	}
	else if (strcmp(_mode, "FP16") == 0) {
		mode = TRT::Mode::FP16;
	}
	else if (strcmp(_mode, "INT8") == 0) {
		mode = TRT::Mode::INT8;
	}
	else
	{
		mode = TRT::Mode::FP32;
	}
	//定义type,默认V5
	Yolo::Type type;
	if (strcmp(_type, "V5") == 0) {
		type = Yolo::Type::V5;
	}
	else if (strcmp(_type, "X") == 0)
	{
		type = Yolo::Type::X;
	}
	else if (strcmp(_type, "V3") == 0)
	{
		type = Yolo::Type::V3;
	}
	else if (strcmp(_type, "V7") == 0)
	{
		type = Yolo::Type::V7;
	}
	else
	{
		type = Yolo::Type::V5;
	}
	auto mode_name = TRT::mode_string(mode);
	TRT::set_device(deviceid);

	auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {

		INFO("Int8 %d / %d", current, count);

		for (int i = 0; i < files.size(); ++i) {
			auto image = cv::imread(files[i]);
			Yolo::image_to_tensor(image, tensor, type, i);
		}
	};
	INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name, name);
	
	//判断文件是否存在
	bool file_requires(const char* model_path);

	if (not file_requires(model_path)) {
		return false;
	}

	string onnx_file = iLogger::format("%s.onnx", model_path);
	std::cout << "onnx_file = " << onnx_file << std::endl;
	string model_file = iLogger::format("%s.%s.trtmodel", model_path, mode_name);
	std::cout << "model_file = " << model_file << std::endl;
	int test_batch_size = 1;
	if (not iLogger::exists(model_file)) {
		TRT::compile(
			mode,                       // FP32、FP16、INT8
			test_batch_size,            // max batch size
			onnx_file,                  // source 
			model_file,                 // save to
			{},
			int8process,
			"inference"
		);
	}
	return true;
}

bool creat_yolo_infer_engine(int deviceid,const char* engine_file, const char* _type, std::shared_ptr<Yolo::Infer>& handle) {
	
	//定义type,默认V5
	Yolo::Type type;
	if (strcmp(_type, "V5") == 0) {
		type = Yolo::Type::V5;
	}
	else if (strcmp(_type, "X") == 0)
	{
		type = Yolo::Type::X;
	}
	else if (strcmp(_type, "V3") == 0)
	{
		type = Yolo::Type::V3;
	}
	else if (strcmp(_type, "V7") == 0)
	{
		type = Yolo::Type::V7;
	}
	else
	{
		type = Yolo::Type::V5;
	}
	handle = Yolo::create_infer(
		engine_file,                // engine file
		type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
		deviceid,                   // gpu id
		0.25f,                      // confidence threshold
		0.45f,                      // nms threshold
		Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
		1024,                       // max objects
		false                       // preprocess use multi stream
	);
	if (handle == nullptr) {
		INFOE("Engine is nullptr");
		return false;
	}
	std::cout << "engine_ip = " << &handle << std::endl;
	//handle = engine;
	//handle = &engine;
	//std::cout << "handle = " << handle << std::endl;
	//*(shared_ptr<Yolo::Infer> *)handle->commit();
	//std::cout << "handle = " << &handle << std::endl;
	
	return true;
};

bool yolo_infer_result(cv::Mat& SrcImg, std::shared_ptr<Yolo::Infer>& handle, std::vector<Output_Yolo>& output) {
	std::vector<cv::Mat> images;
	images.emplace_back(SrcImg);
	std::vector<std::shared_future<ObjectDetector::BoxArray>> boxes_array;
	boxes_array = handle->commits(images);
	boxes_array.back().get();
	for (int i = 0; i < boxes_array.size(); ++i) {

		auto& image = images[i];
		auto boxes = boxes_array[i].get();
		for (auto& obj : boxes) {
			//auto name = tyelabels[obj.class_label];
			Output_Yolo result;
			result.id = obj.class_label;
			result.confidence = obj.confidence;
			result.box.x = obj.left;
			result.box.y = obj.top;
			result.box.width = obj.right - obj.left;
			result.box.height = obj.bottom - obj.top;
			output.push_back(result);
		}
	}
	return true;
};