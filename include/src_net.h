// OpenCV
#include <opencv2/opencv.hpp>
// CUDA
#include "cuda_utils.h"
#include "logging.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda.h>
// TRT
#include "NvInfer.h"
#include "NvOnnxParser.h"
// #include "NvInferRuntimeCommon.h"
#include "NvInferPlugin.h"
#include "NvInferVersion.h"
// Log
#include "spdlog/spdlog.h"
// File Read
#include <fstream>
// String read
#include <cstring>

#include "common.hpp"

#include <iomanip> // for std::put_time
#include <ctime>   // for std::localtime and std::tm

// Namespaces
using namespace std;
using namespace cv;
using namespace nvonnxparser;
using namespace nvinfer1;

extern std::shared_ptr<spdlog::logger> logger;

using Severity = nvinfer1::ILogger::Severity;
class TrtLogger : public nvinfer1::ILogger {
private:
    void log(Severity severity, const char* msg)  noexcept override
    { 
      SPDLOG_LOGGER_INFO(logger, msg);
    }
};

class SrcNetwork {

public:
  SrcNetwork(Logger g_logger, char* format, int* min_NCHW, int* opt_NCHW, int* max_NCHW, vector<const char*> INPUT_BLOB_NAME, vector<const char*> OUTPUT_BLOB_NAME);
  ~SrcNetwork();
  void buildEngine(const char* onnx_filename, char* engine_filename);
  IExecutionContext *getEngineContext(char* engine_filename, cudaStream_t stream);
  
  int num_bindings;
  int num_inputs = 0;
	int num_outputs = 0;
	std::vector<Binding> input_bindings;
	std::vector<Binding> output_bindings;

private:
  int min_batchsize_{0};
  int min_input_channel_{0};
  int min_model_width_{0};
  int min_model_height_{0};
  
  int max_batchsize_{0};
  int max_input_channel_{0};
  int max_model_width_{0};
  int max_model_height_{0};
  
  int opt_batchsize_{0};
  int opt_input_channel_{0};
  int opt_model_width_{0};
  int opt_model_height_{0};
  
  int is_FP16_{0};

  vector<const char*> INPUT_BLOB_NAME_;
  vector<const char*> OUTPUT_BLOB_NAME_;
  char *format_;

  Logger g_logger_;
  std::unique_ptr<TrtLogger> mLogger{nullptr};
  IRuntime* m_runtime_;
  ICudaEngine* m_engine_;
  IExecutionContext* m_context_;
};
