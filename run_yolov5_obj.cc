// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

// This is a simple program that describes how to run MobileNetV1 classification
// on an image and get its top k results by predict score.
// Should be noted: Only mobileNetV1-nv12 is supported here.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <utility>
#include <vector>
#include <fstream> 

#include <opencv2/dnn.hpp>

#include "dnn/hb_dnn.h"
#include <opencv2/dnn.hpp>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#define EMPTY ""


using namespace cv;
using namespace cv::dnn;
using namespace std;
DEFINE_string(model_file, EMPTY, "model file path");
DEFINE_string(image_file, EMPTY, "Test image path");
DEFINE_int32(top_k, 5, "Top k classes, 5 by default");


cv::Mat bgr_mat;
cv::Mat rgb_mat;

//drow box
typedef struct _Box
{
	int cl_;
	float conf_;

	float xmin_;
	float ymin_;
	float xmax_;
	float ymax_;

	_Box(int cl, float conf, float xmin, float ymin, float xmax, float ymax)
		: cl_(cl)
		, conf_(conf)
		, xmin_(xmin)
		, ymin_(ymin)
		, xmax_(xmax)
		, ymax_(ymax)
	{}
}Box;

static bool sort_score(Box box1, Box box2) 
{
	return box1.conf_ > box2.conf_ ? true : false;
}

static float iou(Box box1, Box box2) 
{
	int x1 = std::max(box1.xmin_, box2.xmin_);
	int y1 = std::max(box1.ymin_, box2.ymin_);
	int x2 = std::min(box1.xmax_, box2.xmax_);
	int y2 = std::min(box1.ymax_, box2.ymax_);
	int w = std::max(0, x2 - x1);
	int h = std::max(0, y2 - y1);
	float over_area = w * h;
	return over_area / ( (box1.xmax_- box1.xmin_) * (box1.ymax_ - box1.ymin_) + (box2.xmax_ - box2.xmin_) * (box2.ymax_ - box2.ymin_) - over_area);
}

static std::vector<Box> nms(std::vector<Box>&boxes, float threshold)
{
	std::vector<Box>resluts;
	std::sort(boxes.begin(), boxes.end(), sort_score);
	while (boxes.size()> 0)
	{
		resluts.push_back(boxes[0]);
		int index = 1;
		while (index < boxes.size()) 
		{
			float iou_value = iou(boxes[0], boxes[index]);
			//cout << "iou_value=" << iou_value << endl;
			if (iou_value > threshold) 
			{
				boxes.erase(boxes.begin() + index);
			}
			else 
			{
				index++;
			}
		}
		boxes.erase(boxes.begin());
	}

	return resluts;
}



enum VLOG_LEVEL {
  EXAMPLE_SYSTEM = 0,
  EXAMPLE_REPORT = 1,
  EXAMPLE_DETAIL = 2,
  EXAMPLE_DEBUG = 3
};

#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return ret_code;                                               \
    }                                                                \
  } while (0);

typedef struct Classification {
  int id;
  float score;
  const char *class_name;

  Classification() : class_name(0), id(0), score(0.0) {}
  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  ~Classification() {}
} Classification;

std::map<int32_t, int32_t> element_size{{HB_DNN_IMG_TYPE_Y, 1},
                                        {HB_DNN_IMG_TYPE_NV12, 1},
                                        {HB_DNN_IMG_TYPE_NV12_SEPARATE, 1},
                                        {HB_DNN_IMG_TYPE_YUV444, 1},
                                        {HB_DNN_IMG_TYPE_RGB, 1},
                                        {HB_DNN_IMG_TYPE_BGR, 1},
                                        {HB_DNN_TENSOR_TYPE_S8, 1},
                                        {HB_DNN_TENSOR_TYPE_U8, 1},
                                        {HB_DNN_TENSOR_TYPE_F16, 2},
                                        {HB_DNN_TENSOR_TYPE_S16, 2},
                                        {HB_DNN_TENSOR_TYPE_U16, 2},
                                        {HB_DNN_TENSOR_TYPE_F32, 4},
                                        {HB_DNN_TENSOR_TYPE_S32, 4},
                                        {HB_DNN_TENSOR_TYPE_U32, 4},
                                        {HB_DNN_TENSOR_TYPE_F64, 8},
                                        {HB_DNN_TENSOR_TYPE_S64, 8},
                                        {HB_DNN_TENSOR_TYPE_U64, 8}};
int prepare_tensor(hbDNNTensor *input_tensor,
                   hbDNNTensor *output_tensor,
                   hbDNNHandle_t dnn_handle);


int32_t read_image_2_tensor_as_nv12(std::string &image_file,
                                    hbDNNTensor *input_tensor);

void get_topk_result(hbDNNTensor *tensor,
                     std::vector<Classification> &top_k_cls,
                     int top_k);
void nhwc_to_nchw(uint8_t *out_data0,
                  uint8_t *out_data1,
                  uint8_t *out_data2,
                  uint8_t *in_data,
                  int height,
                  int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data0++ = *(in_data++);
      *out_data1++ = *(in_data++);
      *out_data2++ = *(in_data++);
    }
  }
}
void nhwc_to_nchw_float(uint8_t *out_data0,
                  uint8_t *out_data1,
                  uint8_t *out_data2,
                  float *in_data,
                  int height,
                  int width) {
  for (int hh = 0; hh < height; ++hh) {
    for (int ww = 0; ww < width; ++ww) {
      *out_data0++ = (uint8_t)(*(in_data++));
      *out_data1++ = (uint8_t)(*(in_data++));
      *out_data2++ = (uint8_t)(*(in_data++));
    }
  }
}


/**
 * Step1: get model handle
 * Step2: prepare input and output tensor
 * Step3: set input data to input tensor
 * Step4: run inference
 * Step5: do postprocess with output data
 * Step6: release resources
 */


int main(int argc, char **argv) {
  
 
  // Parsing command line arguments
  gflags::SetUsageMessage(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << gflags::GetArgv() << std::endl;

  // Init logging
  google::InitGoogleLogging("");
  google::SetStderrLogging(0);
  google::SetVLOGLevel("*", 3);
  FLAGS_colorlogtostderr = true;
  FLAGS_minloglevel = google::INFO;
  FLAGS_logtostderr = true;

  hbPackedDNNHandle_t packed_dnn_handle;
  hbDNNHandle_t dnn_handle;
  const char **model_name_list;
  auto modelFileName = FLAGS_model_file.c_str();
  int model_count = 0;
  // Step1: get model handle
  {
    HB_CHECK_SUCCESS(
        //加载模型
        hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1),
        "hbDNNInitializeFromFiles failed");
        //获取模型名称
    HB_CHECK_SUCCESS(hbDNNGetModelNameList(
                         &model_name_list, &model_count, packed_dnn_handle),
                     "hbDNNGetModelNameList failed");
    HB_CHECK_SUCCESS(
      //获取dnn_handle
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
        "hbDNNGetModelHandle failed");
  }
  // Show how to get dnn version
  VLOG(EXAMPLE_DEBUG) << "DNN runtime version: " << hbDNNGetVersion();
  //准备输入数据
  VLOG(EXAMPLE_DEBUG)<<"准备输入数据";
  std::vector<hbDNNTensor> input_tensors;
  std::vector<hbDNNTensor> output_tensors;
  int input_count = 0;
  int output_count = 0;
  // Step2: prepare input and output tensor
  {
    HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
                     "hbDNNGetInputCount failed");
    HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                     "hbDNNGetInputCount failed");

    VLOG(EXAMPLE_DEBUG) << "input_count"<<input_count;
    VLOG(EXAMPLE_DEBUG) << "output_count"<<output_count;

    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle);
   

  }

  // Step3: set input data to input tensor
  {
    // read a single picture for input_tensor[0], for multi_input model, you
    // should set other input data according to model input properties.
    HB_CHECK_SUCCESS(
        read_image_2_tensor_as_nv12(FLAGS_image_file, input_tensors.data()),
        "read_image_2_tensor_as_nv12 failed");
        VLOG(EXAMPLE_DEBUG) << "read image to tensor as nv12 success";   
    //print input tensor
    //std::cout << input_tensors[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>

  }

  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNTensor *output = output_tensors.data();

  // Step4: run inference core
  {
    VLOG(EXAMPLE_DEBUG)<<"run inference!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
    // make sure memory data is flushed to DDR before inference
    for (int i = 0; i < input_count; i++) {
      hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }
    
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    HB_CHECK_SUCCESS(hbDNNInfer(&task_handle,
                                &output,
                                input_tensors.data(),
                                dnn_handle,
                                &infer_ctrl_param),
                     "hbDNNInfer failed");
    // wait task done
    HB_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0),
                     "hbDNNWaitTaskDone failed");
    
  }

  // Step5: do postprocess with output data
  std::vector<Classification> top_k_cls; 
  {
    // make sure CPU read data from DDR before using output tensor data
    for (int i = 0; i < output_count; i++) {
      //对缓存的BPU内存进行刷新
      hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    get_topk_result(&output[0], top_k_cls, 0);
    get_topk_result(&output[1], top_k_cls, 1);
    get_topk_result(&output[2], top_k_cls, 2);
    //for (int i = 0; i < FLAGS_top_k; i++) {
      //VLOG(EXAMPLE_REPORT) << "TOP " << i << " result id: " << top_k_cls[i].id;
    //}
  }

  // Step6: release resources
  {
    printf("release resources!!!!!!!!!!!!!!!!!!!");
    // release task handle
    HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");
    // free input mem
    for (int i = 0; i < input_count; i++) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(input_tensors[i].sysMem[0])),
                       "hbSysFreeMem failed");
    }
    // free output mem
    for (int i = 0; i < output_count; i++) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(output_tensors[i].sysMem[0])),
                       "hbSysFreeMem failed");
    }
    // release model
    HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle), "hbDNNRelease failed");
  }

  return 0;
}

//end main 


int prepare_tensor(hbDNNTensor *input_tensor,
                   hbDNNTensor *output_tensor,
                   hbDNNHandle_t dnn_handle) {
  int input_count = 0;
  int output_count = 0;
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);

  /** Tips:
   * For input memory size:
   * *   input_memSize = input[i].properties.alignedByteSize
   * For output memory size:
   * *   output_memSize = output[i].properties.alignedByteSize
   */
  hbDNNTensor *input = input_tensor;
  for (int i = 0; i < input_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i),
        "hbDNNGetInputTensorProperties failed");
    int input_memSize = input[i].properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize),
                     "hbSysAllocCachedMem failed");
    /** Tips:
     * For input tensor, aligned shape should always be equal to the real
     * shape of the user's data. If you are going to set your input data with
     * padding, this step is not necessary.
     * */
    input[i].properties.alignedShape = input[i].properties.validShape;

    // Show how to get input name
    const char *input_name;
    HB_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i),
                     "hbDNNGetInputName failed");
    VLOG(EXAMPLE_DEBUG) << "input[" << i << "] name is " << input_name;
  }

  hbDNNTensor *output = output_tensor;
  for (int i = 0; i < output_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
    int output_memSize = output[i].properties.alignedByteSize;
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize),
                     "hbSysAllocCachedMem failed");

    // Show how to get output name
    const char *output_name;
    HB_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i),
                     "hbDNNGetOutputName failed");
    VLOG(EXAMPLE_DEBUG) << "output[" << i << "] name is " << output_name;
  }
  return 0;
}

/** You can define read_image_2_tensor_as_other_type to prepare your data **/
int32_t read_image_2_tensor_as_nv12(std::string &image_file,
                                    hbDNNTensor *input_tensor) {
  hbDNNTensor *input = input_tensor;
  hbDNNTensorProperties Properties = input->properties;
  int tensor_id = 0;
  // NCHW , the struct of mobilenetv1_224x224 shape is NCHW
  int input_h = Properties.validShape.dimensionSize[2];
  int input_w = Properties.validShape.dimensionSize[3];

  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  if (bgr_mat.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "image file not exist!";
    return -1;
  }
  // resize
  cv::Mat mat;
  mat.create(input_h, input_w, bgr_mat.type());
  cv::resize(bgr_mat, mat, mat.size(), 0, 0);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
    VLOG(EXAMPLE_SYSTEM) << "input img height and width must aligned by 2!";
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

  // copy y data
  auto data = input->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
  uint8_t *u_data = nv12_data + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    if (u_data && v_data) {
      *nv12++ = *u_data++;
      *nv12++ = *v_data++;
    }
  }
  return 0;
}
 
void get_topk_result(hbDNNTensor *tensor,
                     std::vector<Classification> &top_k_cls,
                     int top_k) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  std::priority_queue<Classification,
                      std::vector<Classification>,
                      std::greater<Classification>>
      queue;
  int *shape = tensor->properties.validShape.dimensionSize;

    std::cout <<"output_tensor ################   " <<tensor->properties.validShape.dimensionSize[0]<<endl;
    std::cout <<"output_tensor ################   " <<tensor->properties.validShape.dimensionSize[1]<<endl;
    std::cout <<"output_tensor ################   " <<tensor->properties.validShape.dimensionSize[2]<<endl;
    std::cout <<"output_tensor ################   " <<tensor->properties.validShape.dimensionSize[3]<<endl; 
  // The type reinterpret_cast should be determined according to the output type
  // For example: HB_DNN_TENSOR_TYPE_F32 is float
  auto outputdata = reinterpret_cast<float *>(tensor->sysMem[0].virAddr);
  ofstream outfile;
  outfile.open("test_out.txt");
   for (int index = 0; index < 52*52*8*3 ; index++){
        
      
 
      // 向文件写入用户输入的数据
        outfile <<  outputdata[index] ;
   
      //VLOG(EXAMPLE_DEBUG)<<"data is !!!!!!!!!!!!!!!!!!" << outputdata[index];
  }
   // 关闭打开的文件
  outfile.close();
  //******postprocess add by cr************************//
  //#ifdef YOLOV5OBJ_POST
  static cv::dnn::Net net;
  static cv::Mat fineC1, fineC2, fineC3;
  cv::Mat result_map, result_Mat;
  //cv::Mat result_map1 = net.forward("Transpose_204");
	//cv::Mat result_map2 = net.forward("Transpose_207");
	//cv::Mat result_map3 = net.forward("Transpose_210");
  cv::Mat result_map1;
	cv::Mat result_map2;
	cv::Mat result_map3;

	//std::cout << "result_map1.total:" << result_map1.total() << std::endl;
	//std::cout << "result_map2.total:" << result_map2.total() << std::endl;
	//std::cout << "result_map3.total:" << result_map3.total() << std::endl;

	int anchor_num = 3;
	int output_head = 3;

  int class_num = 3;
	int img_w = 1024;
	int img_h = 1024;

	int input_imgW = 416;
	int input_imgH = 416;

	int cell_size[3][2] = { { 52, 52 },{ 26, 26 },{ 13, 13 } };
	int anchor_size[3][3][2] = {
		{ { 10, 13 },{ 16, 30 },{ 33, 23 } },
		{ { 30, 61 },{ 62, 45 },{ 59, 119 } },
		{ { 116, 90 },{ 156, 198 },{ 373, 326 } }
	};

	int stride[] = { 8, 16, 32 };
	float grid_cell[3][52][52][2] = { 0 }; //np.zeros(shape = (3, 52, 52, 2))

	float nms_thre = 0.45;
	float obj_thre[] = { 0.5, 0.5, 0.5 };

  std::vector<Box> detectResult;
  std::vector<Box> detectResultNms;


	//def grid_cell_init() :
	{
		for (int index = 0; index < output_head; index++)
		{
			for( int w = 0; w < cell_size[index][1]; w++)
				for ( int h = 0; h < cell_size[index][0]; h++)
				{
					grid_cell[index][h][w][0] = w;
					grid_cell[index][h][w][1] = h;
				}
		}
	}
    int gs = 4 + 1 + class_num;
	  float scale_h = (float)img_h / (float)input_imgH;
	  float scale_w = (float)img_w / (float)input_imgW;


//test read txt
int head = top_k;
//for (int head = 0; head < output_head; head++)
	{
		FILE * fp = NULL;
		ifstream inputFile;
		if (head == 0)
		{
			//result_map1.convertTo(result_Mat, CV_32F);
			inputFile.open("./caffe_0.txt");

		}
		else if (head == 1)
		{
			//result_map2.convertTo(result_Mat, CV_32F);
			inputFile.open("./caffe_1.txt");
		}
		else if (head == 2)
		{
			//result_map3.convertTo(result_Mat, CV_32F);
			inputFile.open("./caffe_2.txt");
		}

		
		//float * y = (float *)result_Mat.ptr<float>(0);

		std::cout << "result_Mat.total:" << result_Mat.total() << std::endl;

		static float tmp[1024 * 1024] = {0};

/*//txt
    int cnt = 0;
    if( head == 0)
    cnt = 1*3*52*52*8; 
    else if ( head == 1)
    cnt = 1*3*52*52*8/4; 
    else 
    cnt = 1*3*52*52*8/16; 

    //float * y = (float *)malloc(cnt*sizeof(float));
  */  
  int cnt = 0;
  int head = 0; 
    if (top_k == 0 )
    {
      cnt = 1*3*52*52*8; 
    }
    else if(top_k == 1)
    {
      cnt = 1*3*26*26*8;
    }
    else{
      cnt = 1*3*13*13*8;
    }

    float * y = (float *)outputdata;//use image 
		for (int i = 0; i < cnt; i++)
		{
			//fprintf(fp, "%f ", y[i]);
      //inputFile >> y[i]; //输入txt
			//tmp[i] = y[i]; //赋值 txt

			//sigmoid
			y[i] = 1.0 / (1.0 + expf(-y[i]));
		}


//int head = 2;
  //for (int head = 0; head < output_head; head++)
    //for 1*3*52*52*8
		for (int h = 0; h < cell_size[head][0]; h++)
			for (int w = 0; w < cell_size[head][1]; w++)
				for (int a = 0; a < anchor_num; a++)
				{
         //confidence
					float conf_scale = y[(a * cell_size[head][1] * cell_size[head][0] * gs) + h * cell_size[head][1] * gs + w * gs + 4];
					for (int cl = 0; cl < class_num; cl++)
					{
						float conf = y[(a * cell_size[head][1] * cell_size[head][0] * gs) + h * cell_size[head][1] * gs + w * gs + (5 + cl)] * conf_scale;

						if (conf > obj_thre[cl])
						{
							float bx = (y[(a * cell_size[head][1] * cell_size[head][0] * gs) + h * cell_size[head][1] * gs + w * gs + 0] * 2.0 - 0.5 + grid_cell[head][h][w][0]) * stride[head];
							float by = (y[(a * cell_size[head][1] * cell_size[head][0] * gs) + h * cell_size[head][1] * gs + w * gs + 1] * 2.0 - 0.5 + grid_cell[head][h][w][1]) * stride[head];
							float bw = pow((y[(a * cell_size[head][1] * cell_size[head][0] * gs) + h * cell_size[head][1] * gs + w * gs + 2] * 2), 2) * anchor_size[head][a][0];
							float bh = pow((y[(a * cell_size[head][1] * cell_size[head][0] * gs) + h * cell_size[head][1] * gs + w * gs + 3] * 2), 2) * anchor_size[head][a][1];

							float xmin = (bx - bw / 2) * scale_w;
							float ymin = (by - bh / 2) * scale_h;
							float xmax = (bx + bw / 2) * scale_w;
							float ymax = (by + bh / 2) * scale_h;

							xmin = (xmin > 0 ? xmin : 0); //xmin if xmin > 0 else 0;
							ymin = (ymin > 0 ? ymin : 0); //ymin if ymin > 0 else 0;
							xmax = (xmax < img_w ? xmax : img_w); //xmax if xmax < img_w else img_w;
							ymax = (ymax < img_h ? ymax : img_h); //ymax if ymax < img_h else img_h;

							if (xmin >= 0 && ymin >= 0 && xmax <= img_w && ymax <= img_h)
							{
								std::cout << "cl:" << cl << " conf:" << conf << " xmin:"<< xmin << " ymin:"<< ymin <<" xmax:" << xmax << " ymax:" << ymax << std::endl;

								detectResult.push_back(Box(cl, conf, xmin, ymin, xmax, ymax));
							}
						}
					}
				
        }
  }

	detectResultNms = nms(detectResult, nms_thre);
	VLOG(EXAMPLE_DEBUG) << "detectResultNms is :" << detectResultNms.size();
  
	cv::Mat imgOrigin = cv::imread("../../data/det_images/test.jpg");
  //cv::Mat imgOrigin = cv::imread(image_file);

//drow object box
	for (int i = 0; i < detectResultNms.size(); i++)
	{
		float xmin = (float)detectResultNms[i].xmin_;
		float ymin = (float)detectResultNms[i].ymin_;
		float xmax = (float)detectResultNms[i].xmax_;
		float ymax = (float)detectResultNms[i].ymax_;
    char cl[64];
		cv::Scalar color;
		if (detectResultNms[i].cl_ == 0)
		{
			color[0] = 0; color[1] = 255; color[2] = 0;
      snprintf(cl, sizeof(cl), "changer:%f", detectResultNms[i].conf_);
		}
		else if (detectResultNms[i].cl_ == 1)
		{
			color[0] = 255; color[1] = 0; color[2] = 0;
      snprintf(cl, sizeof(cl), "disable:%f", detectResultNms[i].conf_);
		}
		else if (detectResultNms[i].cl_ == 2)
		{
			color[0] = 0; color[1] = 0; color[2] = 255;
      snprintf(cl, sizeof(cl), "stopper:%f", detectResultNms[i].conf_);
		}
	
		snprintf(cl, sizeof(cl), "%d(%f)", detectResultNms[i].cl_, detectResultNms[i].conf_);
		cv::putText(imgOrigin, cl, Point2f(xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5, 8);

		//Rect vehicle(xmin, ymin, xmax, ymax);
		cv::rectangle(imgOrigin, Rect2f(xmin, ymin, xmax - xmin, ymax - ymin), color, 2);
	}

	cv::imwrite("output.jpg", imgOrigin);
}


