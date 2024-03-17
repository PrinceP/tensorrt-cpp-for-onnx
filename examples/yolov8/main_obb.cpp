#include <cuda_runtime.h>
#include "src_net.h"
#include "crop_resize.h"

#include <dirent.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/pattern_formatter.h>

#include <cmath>

struct RotatedBOX {
    cv::RotatedRect box;
	float score;
	int label;
};


struct PreParam
{
	float ratio = 1.0f;
	float dw = 0.0f;
	float dh = 0.0f;
	float height = 0;
	float width = 0;
};

int num_labels = 15;
float score_thres = 0.50f;
float iou_thres = 0.65f;

const std::vector<std::string> CLASS_NAMES = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", 
    "basketball court", "ground track field", "harbor", "bridge", 
    "large vehicle", "small vehicle", "helicopter", "roundabout", 
    "soccer ball field", "swimming pool"
};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255}
};

void draw_objects(
    const cv::Mat& srcImg,
    cv::Mat& res,
    const std::vector<RotatedBOX> results,
    const std::vector<std::string>& CLASS_NAMES,
    const std::vector<std::vector<unsigned int>>& COLORS
)
{   
    res = srcImg.clone();
    int lw = std::max(static_cast<int>(std::round((srcImg.rows + srcImg.cols) / 2 * 0.003)), 2);

    for (int i = 0; i < results.size(); ++i) {
        
        RotatedBOX Box_ = results[i];
        cv::RotatedRect box_ = Box_.box;
        cv::Point2f points[4];
        box_.points(points);
        
        int class_id = Box_.label;
        float score = Box_.score;
        cv::Scalar color = cv::Scalar(
			COLORS[class_id][0],
			COLORS[class_id][1],
			COLORS[class_id][2]
		);
        char text[256];
		sprintf( text, "%s %.1f%%", CLASS_NAMES[class_id].c_str(), score * 100);
        std::string label = text;

        cv::Point labelPosition;
        labelPosition.x = std::min(std::min(points[0].x, points[2].x), std::min(points[1].x, points[3].x));
        labelPosition.y = std::min(std::min(points[0].y, points[2].y), std::min(points[1].y, points[3].y));

        if (!label.empty()) {
            int tf = std::max(lw - 1, 1);
            for (int i = 0; i < 4; ++i) 
            {
                cv::line(res, points[i], points[(i + 1) % 4], color, tf);  
            }
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, lw / 3, tf, nullptr);
            int labelHeight = textSize.height;
            cv::rectangle(res, labelPosition, cv::Point(labelPosition.x + textSize.width + 1, labelPosition.y + int(1.5 * labelHeight)), color, -1, cv::LINE_AA);
            cv::putText(res, label, cv::Point(labelPosition.x, labelPosition.y + labelHeight), cv::FONT_HERSHEY_SIMPLEX, lw / 3, {0, 0, 0}, tf, cv::LINE_AA);

        }
    }
}



void postprocess(std::vector<RotatedBOX>& results_boxes, float* host_ptrs[1], PreParam pparam, int index, int num_channels, int num_anchors)
{
	
    std::vector<cv::RotatedRect> boxes;
	std::vector<float>  scores;
    std::vector<RotatedBOX> BOXES;
    std::vector<int> class_list;

    auto& dw = pparam.dw;
	auto& dh = pparam.dh;
	auto& width = pparam.width;
	auto& height = pparam.height;
	auto& ratio = pparam.ratio;
   
    auto* output_data = static_cast<float*>(host_ptrs[0]);
    int offset = index * num_channels * num_anchors;
    
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, output_data+offset);
    output = output.t();
    
    for (int i = 0; i < output.rows; ++i) {
        
        float x0 = output.at<float>(i, 0) - dw;
        float y0 = output.at<float>(i, 1) - dh;
        float x1 = output.at<float>(i, 2);
        float y1 = output.at<float>(i, 3);


        x0 = clamp((x0 - 0.5f) * ratio, 0.f, width);
        y0 = clamp((y0 - 0.5f) * ratio, 0.f, height);
        x1 = clamp((x1 + 0.5f) * ratio, 0.f, width);
        y1 = clamp((y1 + 0.5f) * ratio, 0.f, height);
        
        cv::Mat class_scores = output.row(i).colRange(4, 4 + num_labels);
        double minV, maxV;
        cv::Point minI, maxI;
        cv::minMaxLoc(class_scores, &minV, &maxV, &minI, &maxI);

        int class_idx = maxI.x;
        float max_class_score = maxV;
        if (max_class_score < score_thres)
            continue;
        scores.push_back(max_class_score);

        float theta_pred = output.at<float>(i, 4+num_labels);
        if(theta_pred >= M_PI && theta_pred <= 0.75*M_PI){
            theta_pred = theta_pred - M_PI;
        }
        theta_pred = theta_pred * 180 / M_PI;
        
        cv::RotatedRect current_box = cv::RotatedRect(cv::Point2f(x0, y0), cv::Size2f(x1,y1), theta_pred);
        boxes.push_back(current_box);
        RotatedBOX BOX;
        BOX.box = current_box;
        BOX.score = max_class_score;
        BOX.label = class_idx;
        BOXES.push_back(BOX); 
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, score_thres, iou_thres, indices);

    for (int idx : indices) {
        results_boxes.push_back(BOXES[idx]);
    }
   
}


#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000

std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("yolov8-obb");

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
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
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
    vector<const char*> INPUT_BLOB_NAME = {"images"};
    vector<const char*> OUTPUT_BLOB_NAME = {"output0"};

    SrcNetwork obbdetector_v8(g_logger, format, min_NCHW, opt_NCHW, max_NCHW, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME);

    std::string onnx_path = argv[1];

    // if trt_path doesn't exist then build the engine
    std::string onnx_path_str(onnx_path);
    SPDLOG_LOGGER_INFO(logger,"Onnx Path: {}", onnx_path_str);
    std::string trt_path = onnx_path_str.substr(0, onnx_path_str.find_last_of('.')) + "_batchsize_" +std::to_string(BatchSize)+".trt";
    SPDLOG_LOGGER_INFO(logger,"TRT Path: {}", trt_path);

    std::ifstream trt_file(trt_path, std::ios::binary);
    IExecutionContext* detector_context;
    if (!trt_file.good()) {
        obbdetector_v8.buildEngine(strdup(onnx_path.c_str()), strdup(trt_path.c_str()));
        detector_context = obbdetector_v8.getEngineContext(strdup(trt_path.c_str()), stream);
    }else{
        detector_context = obbdetector_v8.getEngineContext(strdup(trt_path.c_str()), stream);
    }

    float* backbone_buffers[obbdetector_v8.num_inputs + obbdetector_v8.num_outputs];
    float* output_buffers[BatchSize * obbdetector_v8.num_outputs];

    CUDA_CHECK(cudaMalloc((void**)&backbone_buffers[0], max_NCHW_[0] * max_NCHW_[1] * max_NCHW_[2] * max_NCHW_[3]  * sizeof(float)));    
    int i = 0;
    for (auto& bindings : obbdetector_v8.output_bindings){
        
        size_t size = bindings.size * bindings.dsize;

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

            size_t size_image = rows * cols * 3;
            hostbuffers[index] = img.data;
            CUDA_CHECK(cudaMemcpyAsync(devicebuffers[index], hostbuffers[index], size_image, cudaMemcpyHostToDevice, stream));
            
            float Imean_values[3]  = {0,0,0};
            float Iscale_values[3] = {1,1,1};
            cv::Rect context_crop(0, 0, cols, rows);
            
            crop_resize_kernel_img(
                devicebuffers[index], cols, rows,        //src
                buffer_idx, InputW, InputH, //dst
                context_crop,      //  crop,
                &Imean_values[0],  //  Imean_values,
                &Iscale_values[0], //  Iscale_values,
                1,  //  letterbox,
                0,  //  scale_given,
                0,  //  size,
                1,  //  is_norm: Divide by 255 
                stream
            );
            buffer_idx += InputH * InputW * 3 ;        
        }
        
        SPDLOG_LOGGER_INFO(logger, "Calling inference");
        //INFERENCE
        detector_context->enqueue(currentBatchSize, (void**)backbone_buffers, stream, nullptr);
        SPDLOG_LOGGER_INFO(logger, "Calling inference done");
        
        //COPY RESULTS
        for (int i = 0; i < obbdetector_v8.output_bindings.size(); i++)
        {
            size_t osize = obbdetector_v8.output_bindings[i].size * obbdetector_v8.output_bindings[i].dsize;
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
        int num_channels = obbdetector_v8.output_bindings[0].dims.d[1];
        int num_anchors  = obbdetector_v8.output_bindings[0].dims.d[2];
        // std::cout << "num_channels: " << num_channels << std::endl;
        // std::cout << "num_anchors: " << num_anchors << std::endl;

        
        for(int index=0; index < currentBatchSize; index++){
            std::vector<RotatedBOX> results_boxes;
            postprocess(results_boxes, output_buffers, pparam_per_batch[index], index, num_channels, num_anchors);
            
            cv::Mat res;
            draw_objects(img_batch[index], res, results_boxes, CLASS_NAMES, COLORS);

            // Save image
            cv::imwrite("/app/results/v8obb_" + img_name_batch[index], res);
            
        }
        SPDLOG_LOGGER_INFO(logger, "Calling postprocess done");
        
        
    }
    

    return 0;
}
