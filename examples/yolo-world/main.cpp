#include <cuda_runtime.h>
#include "src_net.h"
#include "crop_resize.h"

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/pattern_formatter.h>

struct Object
{
	cv::Rect_<float> rect;
	int label = 0;
	float prob = 0.0;
};

struct PreParam
{
	float ratio = 1.0f;
	float dw = 0.0f;
	float dh = 0.0f;
	float height = 0;
	float width = 0;
};

int num_labels = 3; //80;
int topk = 100;
float score_thres = 0.25f;
float iou_thres = 0.65f;

// const std::vector<std::string> CLASS_NAMES = {
//         "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
//         "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
//         "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
//         "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
//         "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
//         "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
//         "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
//         "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
//         "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
//         "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
//         "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
//         "teddy bear",     "hair drier", "toothbrush"
// };
const std::vector<std::string> CLASS_NAMES = {
    "helmet", "head","sunglasses"
};
const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}
};
    

void postprocess(std::vector<Object>& objs, float* host_ptrs[2], PreParam pparam, int index, int num_images, int num_anchors, int num_labels, float score_thres, float iou_thres, int max_dets)
{
    objs.clear();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    auto& dw = pparam.dw;
    auto& dh = pparam.dh;
    auto& width = pparam.width;
    auto& height = pparam.height;
    auto& ratio = pparam.ratio;
    // std::cout << "ratio: " << ratio << std::endl;
    // std::cout << "dw: " << dw << std::endl;
    // std::cout << "dh: " << dh << std::endl;
    // std::cout << "width: " << width << std::endl;
    // std::cout << "height: " << height << std::endl;


    auto* scores_data = static_cast<float*>(host_ptrs[0]);
    auto* bboxes_data = static_cast<float*>(host_ptrs[1]);
    
    int scores_offset = index * num_anchors * num_labels;
    int bboxes_offset = index * num_anchors * 4;

    cv::Mat scores_mat = cv::Mat(num_anchors, num_labels, CV_32F, scores_data + scores_offset);

    cv::Mat bboxes_mat = cv::Mat(num_anchors, 4, CV_32F, bboxes_data + bboxes_offset);

    std::vector<cv::Rect> class_bboxes[num_labels];
    std::vector<float> class_scores[num_labels];
    std::vector<int> class_labels[num_labels];

    // Process each anchor
    for (int i = 0; i < num_anchors; i++) {
        float* bbox_ptr = bboxes_mat.row(i).ptr<float>();
        float* score_ptr = scores_mat.row(i).ptr<float>();
        for (int cls_id = 0; cls_id < num_labels; cls_id++) {
            float score = score_ptr[cls_id];
            if (score > score_thres) {
                float x1 = bbox_ptr[0];
                float y1 = bbox_ptr[1];
                float x2 = bbox_ptr[2];
                float y2 = bbox_ptr[3];
                // std::cout << "x1: " << x1 << "y1: " << y1 << "x2: " << x2 << "y2: " << y2 << std::endl;
                x1 /= ratio;
                y1 /= ratio;
                x2 /= ratio;
                y2 /= ratio;
                // std::cout << "x1: " << x1 << "y1: " << y1 << "x2: " << x2 << "y2: " << y2 << std::endl;
                
                x1 -= dw;
                y1 -= dh;
                x2 -= dw;
                y2 -= dh;
                // std::cout << "x1: " << x1 << "y1: " << y1 << "x2: " << x2 << "y2: " << y2 << std::endl;
                
                x1 = clamp(x1, 0.f, width);
                y1 = clamp(y1, 0.f, height);
                x2 = clamp(x2, 0.f, width);
                y2 = clamp(y2, 0.f, height);

                class_bboxes[cls_id].emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
                class_scores[cls_id].emplace_back(score);
                class_labels[cls_id].emplace_back(cls_id);
            }
        }
    }

    std::vector<cv::Rect> all_bboxes;
    std::vector<float> all_scores;
    std::vector<int> all_labels;

    // Perform class-specific NMS
    for (int cls_id = 0; cls_id < num_labels; cls_id++) {
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(class_bboxes[cls_id], class_scores[cls_id], score_thres, iou_thres, nms_indices);
        for (int idx : nms_indices) {
            all_bboxes.push_back(class_bboxes[cls_id][idx]);
            all_scores.push_back(class_scores[cls_id][idx]);
            all_labels.push_back(class_labels[cls_id][idx]);
        }
    }

    // Sort by score and limit to max_dets
    std::vector<int> sorted_indices(all_scores.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&all_scores](int i1, int i2) { return all_scores[i1] > all_scores[i2]; });

    for (int i = 0; i < std::min(static_cast<int>(sorted_indices.size()), max_dets); i++) {
        int idx = sorted_indices[i];
        Object obj;
        obj.rect = all_bboxes[idx];
        obj.prob = all_scores[idx];
        obj.label = all_labels[idx];
        objs.push_back(obj);
    }
}


void draw_objects(
	const cv::Mat& image,
	cv::Mat& res,
	const std::vector<Object>& objs,
	const std::vector<std::string>& CLASS_NAMES,
	const std::vector<std::vector<unsigned int>>& COLORS
)
{
	res = image.clone();
	for (auto& obj : objs)
	{	

		cv::Scalar color = cv::Scalar(
			COLORS[obj.label][0],
			COLORS[obj.label][1],
			COLORS[obj.label][2]
		);
		cv::rectangle(
			res,
			obj.rect,
			color,
			2
		);

		char text[256];
		sprintf(
			text,
			"%s %.1f%%",
			CLASS_NAMES[obj.label].c_str(),
			obj.prob * 100
		);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(
			text,
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			1,
			&baseLine
		);

		int x = (int)obj.rect.x;
		int y = (int)obj.rect.y - 10;

		if (y > res.rows)
			y = res.rows;

		cv::rectangle(
			res,
			cv::Rect(x, y, label_size.width, label_size.height + baseLine),
			{ 0, 0, 255 },
			-1
		);

		cv::putText(
			res,
			text,
			cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			{ 255, 255, 255 },
			1
		);
	}
}


#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000

std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("yolo-world");

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

    int BatchSize = 1;

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
    vector<const char*> OUTPUT_BLOB_NAME = {"scores", "boxes"};

    SrcNetwork detector_yoloworld(g_logger, format, min_NCHW, opt_NCHW, max_NCHW, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME);

    std::string onnx_path = argv[1];

    // if trt_path doesn't exist then build the engine
    std::string onnx_path_str(onnx_path);
    SPDLOG_LOGGER_INFO(logger,"Onnx Path: {}", onnx_path_str);
    std::string trt_path = onnx_path_str.substr(0, onnx_path_str.find_last_of('.')) + "_batchsize_" +std::to_string(BatchSize)+".trt";
    SPDLOG_LOGGER_INFO(logger,"TRT Path: {}", trt_path);

    std::ifstream trt_file(trt_path, std::ios::binary);
    IExecutionContext* detector_context;
    if (!trt_file.good()) {
        detector_yoloworld.buildEngine(strdup(onnx_path.c_str()), strdup(trt_path.c_str()));
        detector_context = detector_yoloworld.getEngineContext(strdup(trt_path.c_str()), stream);
    }else{
        detector_context = detector_yoloworld.getEngineContext(strdup(trt_path.c_str()), stream);
    }

    float* backbone_buffers[detector_yoloworld.num_inputs + detector_yoloworld.num_outputs];
    float* output_buffers[BatchSize * detector_yoloworld.num_outputs];

    CUDA_CHECK(cudaMalloc((void**)&backbone_buffers[0], max_NCHW_[0] * max_NCHW_[1] * max_NCHW_[2] * max_NCHW_[3]  * sizeof(float)));    
    int i = 0;
    for (auto& bindings : detector_yoloworld.output_bindings){
        
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
            int max_size = std::max(height, width);
            float scale_factor = static_cast<float>(InputW) / max_size;
            int pad_h = (max_size - height) / 2;
            int pad_w = (max_size - width) / 2;
            
            
            PreParam pparam;
            pparam.ratio = scale_factor;
            pparam.dw = pad_w;
            pparam.dh = pad_h;
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
        for (int i = 0; i < detector_yoloworld.output_bindings.size(); i++)
        {
            size_t osize = detector_yoloworld.output_bindings[i].size * detector_yoloworld.output_bindings[i].dsize;
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
        int batchsize = detector_yoloworld.output_bindings[0].dims.d[0];
        int num_anchors  = detector_yoloworld.output_bindings[0].dims.d[1];
        int clasess  = detector_yoloworld.output_bindings[0].dims.d[2];
        
        for(int index=0; index < currentBatchSize; index++){
            std::vector<Object> objs;
            postprocess(objs, output_buffers, pparam_per_batch[index], index, batchsize, num_anchors, clasess, 0.019/*score_threshold*/, 0.7/*iou_threshold*/, 300/*max_detections*/);
            
            cv::Mat res;
            draw_objects(img_batch[index], res, objs, CLASS_NAMES, COLORS);

            // Save image
            cv::imwrite("/app/results/yoloworld_" + img_name_batch[index], res);
            
        }
        SPDLOG_LOGGER_INFO(logger, "Calling postprocess done");
        
        
    }
    

    return 0;
}
