
//#include "G:\\TRT\\tensorRT_Pro-main\\src\\interface\\recog_interface.h"
#include "../src/interface/recog_interface.h"

int main() {
	/////////onnx2engine///////
	//std::cout << "请输入模型路径(不带.onnx后缀):" << std::endl;
	//std::string path;
	//std::cin >> path;

	//int deviceid = 0;
	//const char* model_path = path.c_str();
	////const char* engine_path = "G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\yolov5s.FP32.trtmodel";
	//const char* _mode = "FP32";
	//const char* _type = "V5";
	//const char* name = "yolov5s";
	//bool status = compile_yolo_engine(deviceid,model_path,_mode,_type, name);
	/////////onnx2engine///////

	/////////infertest///////
	int deviceid = 0;
	const char* engine_path = "D:\\XiAn_Alg_New\\model_path\\CatenaryPedestal\\clsz-1.FP32.trtmodel";
	const char* _type = "V5";
	std::shared_ptr<Yolo::Infer> handle;
	cv::Mat img = cv::imread("G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\inference\\iso.jpg");
	creat_yolo_infer_engine(deviceid, engine_path, _type, handle);
	std::vector<Output_Yolo> output;
	yolo_infer_result(img, handle,output);
	for (int i = 0; i < output.size(); i++) {
		cv::Rect rc;
		rc.x = output[i].box.x;
		rc.y = output[i].box.y;
		rc.width = output[i].box.width;
		rc.height = output[i].box.height;
		cv::rectangle(img, rc, cv::Scalar(0, 0, 255), 5, 8);
		std::string label = std::to_string(output[i].confidence);
		cv::putText(img, label, cv::Point(output[i].box.x, output[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 255), 5);
	}
	cv::imwrite("G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\result.jpg", img);
	return 0;
}