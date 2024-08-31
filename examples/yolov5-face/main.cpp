#include <cuda_runtime.h>
#include "src_net.h"
#include "crop_resize.h"

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/pattern_formatter.h>

struct Detection {
    float bbox[4];
    float conf;
    float landmark[10];
    float class_id;
};


struct PreParam
{
	float ratio = 1.0f;
	float dw = 0.0f;
	float dh = 0.0f;
	float height = 0;
	float width = 0;
};

float letterbox(
    const cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = true) {
  cv::Size shape = image.size();
  float r = std::min(
      (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
  if (!scale_up) {
    r = std::min(r, 1.0f);
  }

  int newUnpad[2]{
      (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

  cv::Mat tmp;
  if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
    cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
  } else {
    tmp = image.clone();
  }

  float dw = new_shape.width - newUnpad[0];
  float dh = new_shape.height - newUnpad[1];

  if (!fixed_shape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  }

  dw /= 2.0f;
  dh /= 2.0f;

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

  return 1.0f / r;
}

float* blobFromImage(cv::Mat& img) {
  float* blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
      }
    }
  }
  return blob;
}

void xywh2xyxy(std::vector<Detection>& boxes){
    for (auto& det : boxes) {
        float x1 = det.bbox[0] - det.bbox[2] / 2;  // top left x
        float y1 = det.bbox[1] - det.bbox[3] / 2;  // top left y
        float x2 = det.bbox[0] + det.bbox[2] / 2;  // bottom right x
        float y2 = det.bbox[1] + det.bbox[3] / 2;  // bottom right y
        det.bbox[0] = x1;
        det.bbox[1] = y1;
        det.bbox[2] = x2;
        det.bbox[3] = y2;
    }
}

void clipCoords(std::vector<Detection>& boxes, int img_width, int img_height) {
    for (auto& box : boxes) {
        // Clip bounding box coordinates
        if (box.bbox[0] < 0.0f)
            box.bbox[0] = 0.0f;
        else if (box.bbox[0] > static_cast<float>(img_width))
            box.bbox[0] = static_cast<float>(img_width);

        if (box.bbox[1] < 0.0f)
            box.bbox[1] = 0.0f;
        else if (box.bbox[1] > static_cast<float>(img_height))
            box.bbox[1] = static_cast<float>(img_height);

        if (box.bbox[2] < 0.0f)
            box.bbox[2] = 0.0f;
        else if (box.bbox[2] > static_cast<float>(img_width))
            box.bbox[2] = static_cast<float>(img_width);

        if (box.bbox[3] < 0.0f)
            box.bbox[3] = 0.0f;
        else if (box.bbox[3] > static_cast<float>(img_height))
            box.bbox[3] = static_cast<float>(img_height);

        // Clip landmark coordinates
        for (int i = 0; i < 10; i += 2) {
            if (box.landmark[i] < 0.0f)
                box.landmark[i] = 0.0f;
            else if (box.landmark[i] > static_cast<float>(img_width))
                box.landmark[i] = static_cast<float>(img_width);

            if (box.landmark[i + 1] < 0.0f)
                box.landmark[i + 1] = 0.0f;
            else if (box.landmark[i + 1] > static_cast<float>(img_height))
                box.landmark[i + 1] = static_cast<float>(img_height);
        }
    }
}

std::vector<Detection> scaleCoords(const std::vector<int>& img1_shape, std::vector<Detection>& coords, const std::vector<int>& img0_shape) {
    std::vector<Detection> scaled_coords;
    float gain, pad_x, pad_y;

    gain = std::min(static_cast<float>(img1_shape[0]) / img0_shape[0], static_cast<float>(img1_shape[1]) / img0_shape[1]);
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2;
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2;
    
    // std::cout << "Gain " << gain << std::endl;
    // std::cout << "Pad X " << pad_x << std::endl;
    // std::cout << "Pad Y " << pad_y << std::endl;


    for (auto& coord : coords) {
        Detection scaled_coord = coord;
        scaled_coord.bbox[0] -= pad_x;   // x padding
        scaled_coord.bbox[2] -= pad_x;   // x padding

        scaled_coord.bbox[1] -= pad_y;   // y padding
        scaled_coord.bbox[3] -= pad_y;   // y padding

        scaled_coord.bbox[0] /= gain;   // gain
        scaled_coord.bbox[1] /= gain;   // gain
        scaled_coord.bbox[2] /= gain;   // gain
        scaled_coord.bbox[3] /= gain;   // gain

        //Print landmarks
        // for(int i = 0; i < 10; i += 2){
        //     std::cout << scaled_coord.landmark[i] << " ";
        //     std::cout << scaled_coord.landmark[i+1] << " ";
        // }
        // std::cout << std::endl;
        scaled_coord.landmark[0] -= pad_x;   // x padding
        scaled_coord.landmark[2] -= pad_x;   // x padding
        scaled_coord.landmark[4] -= pad_x;   // x padding
        scaled_coord.landmark[6] -= pad_x;   // x padding
        scaled_coord.landmark[8] -= pad_x;   // x padding
        
        scaled_coord.landmark[1] -= pad_y;   // y padding
        scaled_coord.landmark[3] -= pad_y;   // y padding
        scaled_coord.landmark[5] -= pad_y;   // y padding
        scaled_coord.landmark[7] -= pad_y;   // y padding
        scaled_coord.landmark[9] -= pad_y;   // y padding
        
        scaled_coord.landmark[0] /= gain;   // gain
        scaled_coord.landmark[2] /= gain;   // gain
        scaled_coord.landmark[4] /= gain;   // gain
        scaled_coord.landmark[6] /= gain;   // gain
        scaled_coord.landmark[8] /= gain;   // gain
        
        scaled_coord.landmark[1] /= gain;   // gain
        scaled_coord.landmark[3] /= gain;   // gain
        scaled_coord.landmark[5] /= gain;   // gain
        scaled_coord.landmark[7] /= gain;   // gain
        scaled_coord.landmark[9] /= gain;   // gain
        
        // for(int i = 0; i < 10; i += 2){
        //     std::cout << scaled_coord.landmark[i] << " ";
        //     std::cout << scaled_coord.landmark[i+1] << " ";
        // }
        // std::cout << std::endl;

        scaled_coords.push_back(scaled_coord);
    }
    
    clipCoords(scaled_coords, img0_shape[1], img0_shape[0]);

    return scaled_coords;
}

float intersectionOverUnion(const Detection& det1, const Detection& det2) {
    float interBox[] = {
        std::max(det1.bbox[0] - det1.bbox[2] / 2.f , det2.bbox[0] - det2.bbox[2] / 2.f), // left
        std::min(det1.bbox[0] + det1.bbox[2] / 2.f , det2.bbox[0] + det2.bbox[2] / 2.f), // right
        std::max(det1.bbox[1] - det1.bbox[3] / 2.f , det2.bbox[1] - det2.bbox[3] / 2.f), // top
        std::min(det1.bbox[1] + det1.bbox[3] / 2.f , det2.bbox[1] + det2.bbox[3] / 2.f)  // bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1]) {
        return 0.0f;
    }

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (det1.bbox[2] * det1.bbox[3] + det2.bbox[2] * det2.bbox[3] - interBoxS);
}

bool compareDetections(const Detection& det1, const Detection& det2) {
    return det1.conf > det2.conf;
}


std::vector<Detection> nonMaxSuppression(const std::vector<Detection>& predictions, float confThresh, float iouThresh) {
    std::vector<Detection> filteredDetections;

    // Sort detections by confidence in descending order
    std::vector<Detection> sortedDetections = predictions;
    std::sort(sortedDetections.begin(), sortedDetections.end(), compareDetections);
    
    for (const Detection& det : sortedDetections) {
        // Filter out low confidence detections
        if (det.conf <= confThresh) {
            continue;
        }

        bool isOverlapping = false;
        for (const Detection& filteredDet : filteredDetections) {
            // Check if the detection overlaps with any of the filtered detections
            if (intersectionOverUnion(det, filteredDet) > iouThresh) {
                isOverlapping = true;
                break;
            }
        }

        // If the detection does not overlap with any filtered detections, add it to the list
        if (!isOverlapping) {
            filteredDetections.push_back(det);
        }
    }

    return filteredDetections;
}


#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000

std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("yolov5-face");

class my_formatter_flag : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg &msg, const std::tm &, spdlog::memory_buf_t &dest) override
    {
		char yi_log_level_text[9] = {0};
		if (msg.level == spdlog::level::level_enum::trace) {
		strcpy(yi_log_level_text, "TRACE");
		} else if (msg.level == spdlog::level::level_enum::debug) {
		strcpy(yi_log_level_text, "DEBUG");
		} else if (msg.level == spdlog::level::level_enum::warn) {
		strcpy(yi_log_level_text, "WARNING");
		} else if (msg.level == spdlog::level::level_enum::err) {
		strcpy(yi_log_level_text, "ERROR");
		} else if (msg.level == spdlog::level::level_enum::critical) {
		strcpy(yi_log_level_text, "CRITICAL");
		} else {
		strcpy(yi_log_level_text, "INFO");
		}
		dest.append(yi_log_level_text,
					yi_log_level_text + strlen(yi_log_level_text));
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<my_formatter_flag>();
    }
};


int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            // Construct full path of the file
            std::string full_path = std::string(p_dir_name) + "/" + std::string(p_file->d_name);
            
            // Check if the file is a regular file
            struct stat file_stat;
            if (stat(full_path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
                // Add only if it's a regular file
                file_names.push_back(p_file->d_name);
            }
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char *argv[]){

    std::string jsonlogpattern = { "%^{\"timestamp\": \"%Y-%m-%dT%H:%M:%S.%eZ\",  \"logLevel\": \"%*\", \"logFacility\": null,  \"function\": \"%!\", \"file\": \"%s\", \"lineNo\": %#, \"message\": \"%v\"}%$" };
	
	auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<my_formatter_flag>('*').set_pattern(jsonlogpattern);
    spdlog::set_formatter(std::move(formatter));


    uint8_t** hostbuffers;
    uint8_t** devicebuffers;

    float* crop_hostbuffer_debug = nullptr;

    int BatchSize = 2;

    // prepare host cache
    hostbuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
    for(int i = 0 ; i < BatchSize; i++)
        CUDA_CHECK(cudaMallocHost(&hostbuffers[i], MAX_IMAGE_INPUT_SIZE_THRESH * 3)); // RGB

    // prepare device cache for input image:  All input RGB data
    devicebuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
    for(int i = 0 ; i < BatchSize; i++)
        CUDA_CHECK(cudaMalloc(&devicebuffers[i], MAX_IMAGE_INPUT_SIZE_THRESH * 3)); // RGB


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int InputH = 640;
    int InputW = 640;
    //Load the backbone
    Logger g_logger;
    int min_NCHW_[4] = {BatchSize,3,InputH,InputW};
    int opt_NCHW_[4] = {BatchSize,3,InputH,InputW};
    int max_NCHW_[4] = {BatchSize,3,InputH,InputW};
    
    int* min_NCHW= &min_NCHW_[0];
    int* opt_NCHW= &opt_NCHW_[0];
    int* max_NCHW= &max_NCHW_[0];

    char* format = "NCHW";
    vector<const char*> INPUT_BLOB_NAME = {"input"};
    vector<const char*> OUTPUT_BLOB_NAME = {"output"};

    SrcNetwork detector_yolov5face(g_logger, format, min_NCHW, opt_NCHW, max_NCHW, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME);

    std::string onnx_path = argv[1];

    // if trt_path doesn't exist then build the engine
    std::string onnx_path_str(onnx_path);
    SPDLOG_LOGGER_INFO(logger,"Onnx Path: {}", onnx_path_str);
    std::string trt_path = onnx_path_str.substr(0, onnx_path_str.find_last_of('.')) + "_batchsize_" +std::to_string(BatchSize)+".trt";
    SPDLOG_LOGGER_INFO(logger,"TRT Path: {}", trt_path);

    std::ifstream trt_file(trt_path, std::ios::binary);
    IExecutionContext* detector_context;
    if (!trt_file.good()) {
        detector_yolov5face.buildEngine(strdup(onnx_path.c_str()), strdup(trt_path.c_str()));
        detector_context = detector_yolov5face.getEngineContext(strdup(trt_path.c_str()), stream);
    }else{
        detector_context = detector_yolov5face.getEngineContext(strdup(trt_path.c_str()), stream);
    }

    float* backbone_buffers[detector_yolov5face.num_inputs + detector_yolov5face.num_outputs];
    float* output_buffers[BatchSize * detector_yolov5face.num_outputs];

    CUDA_CHECK(cudaMalloc((void**)&backbone_buffers[0], max_NCHW_[0] * max_NCHW_[1] * max_NCHW_[2] * max_NCHW_[3]  * sizeof(float)));    
    int i = 0;
    for (auto& bindings : detector_yolov5face.output_bindings){
        
        size_t size = bindings.size * bindings.dsize;
        // std::cout << size << std::endl;

        CUDA_CHECK(cudaMalloc((void**)&backbone_buffers[1 + i], max_NCHW_[0] * size  *  sizeof(float)));
        CUDA_CHECK(cudaMallocHost((void**)&output_buffers[i], max_NCHW_[0] * size  *  sizeof(float)));

        i += 1;
    }

    std::string img_dir = argv[2];
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    for (size_t i = 0; i < file_names.size(); i += BatchSize) {
        
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;

        for (size_t j = i; j < i + BatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }
        
        int currentBatchSize = img_batch.size();
        SPDLOG_LOGGER_INFO(logger,"Current Batch Size: {}", currentBatchSize); 
        
        //PREPROCESS
        float *buffer_idx = (float *)backbone_buffers[0];
        std::vector<PreParam> pparam_per_batch;
        for(int index=0; index < currentBatchSize; index++){

            cv::Mat img = img_batch[index];

            if(img.empty()){
                SPDLOG_LOGGER_ERROR(logger, "Image {} is empty", img_name_batch[index]);
                continue;
            }

            if(img.channels() != 3){
                SPDLOG_LOGGER_ERROR(logger, "Image {} channels is not 3", img_name_batch[index]);
                continue;
            }

            if(img.rows * img.cols > MAX_IMAGE_INPUT_SIZE_THRESH ){
                SPDLOG_LOGGER_ERROR(logger, "Image {} size is too large", img_name_batch[index]);
                continue;
            }

            
            int rows = img.rows;
            int cols = img.cols;
                    
            const float inp_h = InputH;
            const float inp_w = InputW;
            float height = rows;
            float width = cols;

            float r = std::min(inp_h / height, inp_w / width);
            int padw = std::round(width * r);
            int padh = std::round(height * r);
            float dw = inp_w - padw;
            float dh = inp_h - padh;
            dw /= 2.0f;
            dh /= 2.0f;

            PreParam pparam;
            pparam.ratio = 1 / r;
            pparam.dw = dw;
            pparam.dh = dh;
            pparam.height = height;
            pparam.width = width;
            pparam_per_batch.push_back(pparam);

            size_t in_size = 640*640*3;
            unsigned char* aBytes = img.data;
            cv::Mat img_(rows, cols, CV_MAKETYPE(CV_8U, 3), aBytes);
            cv::Mat pr_img;
            float scale = letterbox(img_, pr_img, {640, 640}, 32, {128, 128, 128}, true);
            cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
            float* blob = blobFromImage(pr_img);
            CUDA_CHECK(cudaMemcpyAsync(backbone_buffers[0], &blob[0], in_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        }

        SPDLOG_LOGGER_INFO(logger, "Calling inference");
        //INFERENCE
        detector_context->enqueue(currentBatchSize, (void**)backbone_buffers, stream, nullptr);
        SPDLOG_LOGGER_INFO(logger, "Calling inference done");
        
        //COPY RESULTS
        for (int i = 0; i < detector_yolov5face.output_bindings.size(); i++)
        {
            size_t osize = detector_yolov5face.output_bindings[i].size * detector_yolov5face.output_bindings[i].dsize;
            // std::cout << "osize: " << osize << std::endl;
            CUDA_CHECK(cudaMemcpyAsync(output_buffers[i],
                backbone_buffers[i + 1],
                osize,
                cudaMemcpyDeviceToHost,
                stream)
            );

        }

        cudaStreamSynchronize(stream);
        
        SPDLOG_LOGGER_INFO(logger, "Calling postprocess");
        
        for(int index=0; index < currentBatchSize; index++){
            
            std::vector<Detection> predictions;
            int det_size = sizeof(Detection) / sizeof(float);

            for (int i = 0; i < 25200; i++) {
                Detection temp;
                
                memcpy(&temp, &output_buffers[index][det_size * i], det_size * sizeof(float));
                predictions.push_back(temp);

            }

            float confThreshold = std::stof("0.5");
            float iouThreshold = std::stof("0.45");
            std::vector<Detection> filteredDetections = nonMaxSuppression(predictions, confThreshold, iouThreshold);
            
            std::vector<int> original_shape = {pparam_per_batch[index].height,pparam_per_batch[index].width};
            std::vector<int> model_shape = {InputH, InputW};
            
            xywh2xyxy(filteredDetections);
            filteredDetections = scaleCoords(model_shape, filteredDetections, original_shape);


            // Display the filtered detections
            
            for (Detection& det : filteredDetections) {
                
                // std::cout << "Class ID: " << det.class_id << std::endl;
                // std::cout << "Confidence: " << det.conf << std::endl;
                // std::cout << "Bounding Box: [";
                // for (int i = 0; i < 4; ++i) {
                //     std::cout << int(det.bbox[i]) << ", ";
                // }
                // std::cout << "]" << std::endl;
                // std::cout << "Landmarks: [";
                // for (int i = 0; i < 10; ++i) {
                //     std::cout << det.landmark[i] << ", ";
                // }
                // std::cout << "]" << std::endl;
                // std::cout << std::endl;
                // Draw on image
                cv::rectangle(img_batch[index], cv::Point(int(det.bbox[0]), int(det.bbox[1])), cv::Point(int(det.bbox[2]), int(det.bbox[3])), cv::Scalar(0, 255, 0), 2);
                cv::putText(img_batch[index], std::to_string(det.class_id), cv::Point(int(det.bbox[0]), int(det.bbox[1])), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                cv::circle(img_batch[index], cv::Point(int(det.landmark[0]), int(det.landmark[1])), 2, cv::Scalar(0, 0, 255), -1);
                cv::circle(img_batch[index], cv::Point(int(det.landmark[2]), int(det.landmark[3])), 2, cv::Scalar(0, 255, 255), -1);
                cv::circle(img_batch[index], cv::Point(int(det.landmark[4]), int(det.landmark[5])), 2, cv::Scalar(255, 255, 255), -1);
                cv::circle(img_batch[index], cv::Point(int(det.landmark[6]), int(det.landmark[7])), 2, cv::Scalar(0, 134, 0), -1);
                cv::circle(img_batch[index], cv::Point(int(det.landmark[8]), int(det.landmark[9])), 2, cv::Scalar(255, 0, 0), -1);
                
            }
            cv::imwrite("/app/results/yolov5-face_" + img_name_batch[index], img_batch[index]);
            
        }
        SPDLOG_LOGGER_INFO(logger, "Calling postprocess done");
        
        
    }
    

    return 0;
}
