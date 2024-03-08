#include "src_net.h"

SrcNetwork::SrcNetwork(Logger g_logger, char* format, int* NCHW, vector<const char*> INPUT_BLOB_NAME, vector<const char*> OUTPUT_BLOB_NAME)
: g_logger_(g_logger), format_(format)
{ 
  batchsize_ = NCHW[0];
  input_channel_ = NCHW[1];
  model_height_ = NCHW[2];
  model_width_ = NCHW[3];
  m_runtime_ = nvinfer1::createInferRuntime(g_logger_);
  m_engine_ = nullptr;
  INPUT_BLOB_NAME_ = INPUT_BLOB_NAME;
  OUTPUT_BLOB_NAME_ = OUTPUT_BLOB_NAME;
}

SrcNetwork::~SrcNetwork() {
  if(m_context_)
    m_context_->destroy();
  if (m_engine_)
    m_engine_->destroy();
  if(m_runtime_)
    m_runtime_->destroy();
}

void SrcNetwork::buildEngine(const char *onnx_filename, char *engine_filename){

  std::ifstream file(onnx_filename, std::ios::binary);
  SPDLOG_LOGGER_INFO(logger,"Onnx path: {}", onnx_filename);
  SPDLOG_LOGGER_INFO(logger,"TRT path: {}",  engine_filename);

  
  if (!file.good()) {
    SPDLOG_LOGGER_CRITICAL(logger,"ONNX read error for {}", onnx_filename);
    return;
  }
  IBuilder* builder = createInferBuilder(g_logger_);
  builder->setMaxBatchSize(batchsize_); 
  
  uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

  INetworkDefinition* network = builder->createNetworkV2(flag);
  IParser*  parser = createParser(*network, g_logger_);
  parser->parseFromFile(onnx_filename, 3);
  for (int32_t i = 0; i < parser->getNbErrors(); ++i)
  {
    SPDLOG_LOGGER_CRITICAL(logger,"ONNX parse error : {}", parser->getError(i)->desc());
  }

  IOptimizationProfile* profile = builder->createOptimizationProfile();

  profile->setDimensions(INPUT_BLOB_NAME_[0], OptProfileSelector::kMIN, Dims4(batchsize_, input_channel_, model_height_, model_width_));
  profile->setDimensions(INPUT_BLOB_NAME_[0], OptProfileSelector::kOPT, Dims4(batchsize_, input_channel_, model_height_, model_width_));
  profile->setDimensions(INPUT_BLOB_NAME_[0], OptProfileSelector::kMAX, Dims4(batchsize_, input_channel_, model_height_, model_width_));

  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1U << 24);
  config->addOptimizationProfile(profile);
  if(is_FP16_){
    config->setFlag(BuilderFlag::kFP16);
  }
  IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
  std::ofstream p(engine_filename, std::ios::binary);
  if (!p) {
    SPDLOG_LOGGER_CRITICAL(logger,"TRT engine could not open plan output file");
    return;
  }
  p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

  delete parser;
  delete network;
  delete config;
  delete builder;
  delete serializedModel;

}


IExecutionContext* SrcNetwork::getEngineContext(char* engine_filename, cudaStream_t stream){
  
  std::ifstream file(engine_filename, std::ios::binary);
  if (!file.good()) {
    SPDLOG_LOGGER_CRITICAL(logger,"read error {}", engine_filename);
  }
  char *trt_model_stream = nullptr;
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  trt_model_stream = new char[size];
  assert(trt_model_stream);
  file.read(trt_model_stream, size);
  file.close();

  mLogger.reset(new TrtLogger());
  initLibNvInferPlugins(mLogger.get(), "");

  m_runtime_ = createInferRuntime(g_logger_);
  assert(m_runtime_ != nullptr);

  m_engine_ = m_runtime_->deserializeCudaEngine(trt_model_stream, size);
  assert(m_engine_ != nullptr);
  
  m_context_ = m_engine_->createExecutionContext();
  assert(m_context_ != nullptr);
  
  delete[] trt_model_stream;
  
  
  num_bindings = m_engine_->getNbBindings();
  
  for(int i = 0; i < num_bindings; ++i){
    Binding binding;
	nvinfer1::Dims dims;
	nvinfer1::DataType dtype = m_engine_->getBindingDataType(i);
	std::string name = m_engine_->getBindingName(i);
	binding.name = name;
	binding.dsize = type_to_size(dtype);

	bool IsInput = m_engine_->bindingIsInput(i);
	if (IsInput)
	{
		this->num_inputs += 1;
		dims = m_engine_->getProfileDimensions(
			i,
			0,
			nvinfer1::OptProfileSelector::kMAX);
		binding.size = get_size_by_dims(dims);
		binding.dims = dims;
		this->input_bindings.push_back(binding);
		// set max opt shape
		m_context_->setBindingDimensions(i, dims);

	}
	else
	{
		dims = m_context_->getBindingDimensions(i);
		binding.size = get_size_by_dims(dims);
		binding.dims = dims;
		this->output_bindings.push_back(binding);
		this->num_outputs += 1;
	}
  }
  
  m_context_->setOptimizationProfileAsync(0, stream);
  
  return m_context_;
} 