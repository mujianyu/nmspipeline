#include "nmsFullPlugin.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

// ============================================================================
// CUDA error check
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return 1; \
    } \
} while(0)

// ============================================================================
// Forward declarations of CUDA entry points (defined in nmsFull.cu)
// ============================================================================
extern "C" {
void nonBlockNmsFull(
    cudaStream_t stream,
    const int batch_size,
    const int num_boxes,
    const int per_box_stride,
    const int num_classes,
    const float score_threshold,
    const float iou_threshold,
    const int max_output_boxes,
    const float* d_input,
    int* d_output_count,
    float* d_output_boxes,
    int* d_output_classes,
    float* d_output_scores,
    int* d_sorted_indices,
    float* d_sorted_scores,
    int* d_sorted_classes,
    int* d_temp_indices,
    float* d_temp_scores,
    int* d_temp_classes,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    int* d_order
);
size_t nonBlockNmsFullGetTempSize(int num_boxes);
}

// ============================================================================
static const char* kPluginName    = "NmsFull";
static const char* kPluginVersion = "1";
static const char* kPluginNamespace = "";

// ============================================================================
// NmsFullPlugin
// ============================================================================

NmsFullPlugin::NmsFullPlugin(float score_threshold, float iou_threshold,
                             int max_output_boxes, int per_box_stride, int num_classes)
    : mScoreThreshold(score_threshold)
    , mIouThreshold(iou_threshold)
    , mMaxOutputBoxes(max_output_boxes)
    , mPerBoxStride(per_box_stride)
    , mNumClasses(num_classes)
    , mNumBoxes(0)
    , mBatchSize(0)
    , mNamespace(kPluginNamespace)
{}

NmsFullPlugin::NmsFullPlugin(const void* data, size_t length)
    : mNumBoxes(0), mBatchSize(0), mNamespace(kPluginNamespace)
{
    const char* d = static_cast<const char*>(data);
    mScoreThreshold = *reinterpret_cast<const float*>(d); d += sizeof(float);
    mIouThreshold   = *reinterpret_cast<const float*>(d); d += sizeof(float);
    mMaxOutputBoxes = *reinterpret_cast<const int*>(d);   d += sizeof(int);
    mPerBoxStride   = *reinterpret_cast<const int*>(d);   d += sizeof(int);
    mNumClasses     = *reinterpret_cast<const int*>(d);   d += sizeof(int);
    mNumBoxes       = *reinterpret_cast<const int*>(d);   d += sizeof(int);
    mBatchSize      = *reinterpret_cast<const int*>(d);   d += sizeof(int);
}

NmsFullPlugin::~NmsFullPlugin() {}

// -- IPluginV2 ---------------------------------------------------------
const char* NmsFullPlugin::getPluginType() const noexcept { return kPluginName; }
const char* NmsFullPlugin::getPluginVersion() const noexcept { return kPluginVersion; }
int NmsFullPlugin::getNbOutputs() const noexcept { return 4; }
int32_t NmsFullPlugin::initialize() noexcept { return 0; }
void NmsFullPlugin::terminate() noexcept {}
void NmsFullPlugin::destroy() noexcept { delete this; }
void NmsFullPlugin::setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
const char* NmsFullPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// -- IPluginV2Ext -------------------------------------------------------
nvinfer1::DataType NmsFullPlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
    return (index == 0 || index == 1) ? nvinfer1::DataType::kFLOAT
                                      : nvinfer1::DataType::kINT32;
}

nvinfer1::Dims NmsFullPlugin::getOutputDimensions(
    int32_t index, const nvinfer1::Dims* inputs, int32_t nbInputDims) noexcept
{
    int batch  = (nbInputDims > 0 && inputs[0].nbDims >= 1) ? inputs[0].d[0] : 1;
    int maxOut = mMaxOutputBoxes;
    switch (index) {
    case 0: return nvinfer1::Dims{3, {batch, maxOut, 4}};
    case 1: return nvinfer1::Dims{2, {batch, maxOut}};
    case 2: return nvinfer1::Dims{2, {batch, maxOut}};
    case 3: return nvinfer1::Dims{1, {batch}};
    }
    return nvinfer1::Dims{};
}

bool NmsFullPlugin::isOutputBroadcastAcrossBatch(
    int32_t outputIndex, const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept
{ return false; }

bool NmsFullPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept
{ return false; }

void NmsFullPlugin::attachToContext(cudnnContext*, cublasContext*,
                                     nvinfer1::IGpuAllocator*) noexcept {}
void NmsFullPlugin::detachFromContext() noexcept {}

// -- IPluginV2IOExt -----------------------------------------------------
void NmsFullPlugin::configurePlugin(
    const nvinfer1::PluginTensorDesc* in, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    if (nbInputs > 0 && in[0].dims.nbDims >= 3) {
        mBatchSize    = in[0].dims.d[0];
        mPerBoxStride = in[0].dims.d[1];
        mNumBoxes     = in[0].dims.d[2];
    }
}

bool NmsFullPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs, int32_t nbOutputs) const noexcept
{
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (pos == 0)
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    if (pos == 1 || pos == 2)
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    if (pos == 3 || pos == 4)
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    return false;
}

int32_t NmsFullPlugin::enqueue(
    int32_t batchSize, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int nb = mNumBoxes;
    int ps = mPerBoxStride;
    int nc = mNumClasses;

    const float* d_in   = static_cast<const float*>(inputs[0]);
    float* d_boxes      = static_cast<float*>(outputs[0]);
    float* d_scores     = static_cast<float*>(outputs[1]);
    int*   d_classes    = static_cast<int*>(outputs[2]);
    int*   d_counts     = static_cast<int*>(outputs[3]);

    CUDA_CHECK(cudaMemsetAsync(d_counts, 0, batchSize * sizeof(int), stream));

    char* ws = static_cast<char*>(workspace);
    size_t off = 0;

    size_t cubSz = nonBlockNmsFullGetTempSize(nb);
    void* d_temp_storage = ws + off; off += cubSz;

    int*   d_si  = reinterpret_cast<int*>(ws + off);   off += nb * sizeof(int);
    float* d_ss  = reinterpret_cast<float*>(ws + off); off += nb * sizeof(float);
    int*   d_sc  = reinterpret_cast<int*>(ws + off);   off += nb * sizeof(int);
    int*   d_ti  = reinterpret_cast<int*>(ws + off);   off += nb * sizeof(int);
    float* d_ts  = reinterpret_cast<float*>(ws + off); off += nb * sizeof(float);
    int*   d_tc  = reinterpret_cast<int*>(ws + off);   off += nb * sizeof(int);
    int*   d_ord = reinterpret_cast<int*>(ws + off);   off += nb * sizeof(int);

    nonBlockNmsFull(stream,
        batchSize, nb, ps, nc,
        mScoreThreshold, mIouThreshold, mMaxOutputBoxes,
        d_in, d_counts, d_boxes, d_classes, d_scores,
        d_si, d_ss, d_sc, d_ti, d_ts, d_tc,
        d_temp_storage, cubSz, d_ord);

    return 0;
}

// -- Workspace (IPluginV2Ext) ------------------------------------------
size_t NmsFullPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    int nb = mNumBoxes;
    size_t cubTemp = nonBlockNmsFullGetTempSize(nb);
    size_t scratch = nb * (sizeof(int) + sizeof(float) + sizeof(int)
                        + sizeof(int) + sizeof(float) + sizeof(int)
                        + sizeof(int));
    scratch += 256;
    return cubTemp + scratch;
}

// -- Serialization -----------------------------------------------------
nvinfer1::IPluginV2Ext* NmsFullPlugin::clone() const noexcept
{
    auto* p = new NmsFullPlugin(mScoreThreshold, mIouThreshold,
                                mMaxOutputBoxes, mPerBoxStride, mNumClasses);
    p->mNumBoxes  = mNumBoxes;
    p->mBatchSize = mBatchSize;
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void NmsFullPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<float*>(d) = mScoreThreshold; d += sizeof(float);
    *reinterpret_cast<float*>(d) = mIouThreshold;   d += sizeof(float);
    *reinterpret_cast<int*>(d)   = mMaxOutputBoxes; d += sizeof(int);
    *reinterpret_cast<int*>(d)   = mPerBoxStride;   d += sizeof(int);
    *reinterpret_cast<int*>(d)   = mNumClasses;     d += sizeof(int);
    *reinterpret_cast<int*>(d)   = mNumBoxes;       d += sizeof(int);
    *reinterpret_cast<int*>(d)   = mBatchSize;      d += sizeof(int);
}

size_t NmsFullPlugin::getSerializationSize() const noexcept
{
    return sizeof(float) * 2 + sizeof(int) * 5;
}

// ============================================================================
// NmsFullPluginCreator
// ============================================================================

nvinfer1::PluginFieldCollection NmsFullPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> NmsFullPluginCreator::mPluginAttributes;

NmsFullPluginCreator::NmsFullPluginCreator() : mNamespace(kPluginNamespace)
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "per_box_stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "num_classes", nullptr, nvinfer1::PluginFieldType::kINT32, 1});

    mFC.nbFields = static_cast<int>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

const char* NmsFullPluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* NmsFullPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
const nvinfer1::PluginFieldCollection* NmsFullPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2* NmsFullPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    float st = 0.25f, iou = 0.45f;
    int mbo = 300, pbs = 9, nc = 5;
    for (int i = 0; i < fc->nbFields; i++) {
        const char* an = fc->fields[i].name;
        if (!strcmp(an, "score_threshold"))  st  = *static_cast<const float*>(fc->fields[i].data);
        else if (!strcmp(an, "iou_threshold"))    iou = *static_cast<const float*>(fc->fields[i].data);
        else if (!strcmp(an, "max_output_boxes")) mbo = *static_cast<const int*>(fc->fields[i].data);
        else if (!strcmp(an, "per_box_stride"))   pbs = *static_cast<const int*>(fc->fields[i].data);
        else if (!strcmp(an, "num_classes"))      nc  = *static_cast<const int*>(fc->fields[i].data);
    }
    auto* p = new NmsFullPlugin(st, iou, mbo, pbs, nc);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::IPluginV2* NmsFullPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    auto* p = new NmsFullPlugin(serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void NmsFullPluginCreator::setPluginNamespace(const char* ns) noexcept { mNamespace = ns; }
const char* NmsFullPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(NmsFullPluginCreator);
