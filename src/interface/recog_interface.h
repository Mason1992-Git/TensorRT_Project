#pragma once
#include "../tensorRT/builder/trt_builder.hpp"
//#include <builder/trt_builder.hpp>
#include "../tensorRT/infer/trt_infer.hpp"
//#include <infer/trt_infer.hpp>
//#include <common/ilogger.hpp>
#include "../tensorRT/common/ilogger.hpp"
//#include "app_yolo/yolo.hpp"
#include "../application/app_yolo/yolo.hpp"
//#include "app_yolo/multi_gpu.hpp"
#include "../application/app_yolo/multi_gpu.hpp"
#include "../application/common/object_detector.hpp"
#ifdef EXPORTDLL_EXPORTS
#define EXPORTDLL_API __declspec(dllexport)
#else
#define EXPORTDLL_API __declspec(dllexport)
#endif // EXPORTDLL_EXPORTS

#ifdef __cplusplus
extern "C" {
#endif
	struct Output_Yolo {
		int id;//结果类别id
		float confidence;//结果置信度
		cv::Rect box;//矩形框
	};
	//onnx2engine
	bool compile_yolo_engine(int deviceid, const char* model_path, const char* _mode, const char* _type, const char* name);
	//creat_infer
	bool creat_yolo_infer_engine(int deviceid, const char* engine_file, const char* _type, std::shared_ptr<Yolo::Infer>& handle);
	//infer
	bool yolo_infer_result(cv::Mat& SrcImg, std::shared_ptr<Yolo::Infer>& handle, std::vector<Output_Yolo>& output);

#ifdef __cplusplus
}
#endif // __cplusplus
