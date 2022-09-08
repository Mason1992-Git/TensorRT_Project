#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
using namespace std;
//yolov5输出结构体
struct Output {
	int id;//结果类别id
	float confidence;//结果置信度
	cv::Rect box;//矩形框
};
int main() {
	void yolo_detect_trt(cv::Mat& SrcImg,int deviceid, const string& engine_file, const string& _mode, const string& _type, const string& model, std::vector<Output>& output);
	//指定GPU
	cv::Mat img = cv::imread("G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\inference\\iso.jpg");
	int deviceid = 0;
	const string engine_file = "G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\yolov5s.FP32.trtmodel";
	const string _mode = "FP32";
	const string _type = "V5";
	const string model = "yolov5s";
	std::vector<Output> out;
	yolo_detect_trt(img,deviceid,engine_file, _mode, _type, model, out);
	std::cout << "result_size = " << out.size() << std::endl;
	for (int i = 0; i < out.size(); i++) {
		cv::Rect rc;
		rc.x = out[i].box.x;
		rc.y = out[i].box.y;
		rc.width = out[i].box.width;
		rc.height = out[i].box.height;
		cv::rectangle(img, rc, cv::Scalar(0, 0, 255), 5, 8);
		std::string label = std::to_string(out[i].confidence);
		cv::putText(img, label, cv::Point(out[i].box.x, out[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 255), 5);
	}
	cv::imwrite("G:\\TRT\\tensorRT_Pro-main\\x64\\Release\\result.jpg", img);
	return 0;
};