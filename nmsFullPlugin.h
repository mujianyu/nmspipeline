#ifndef NMS_FULL_PLUGIN_H
#define NMS_FULL_PLUGIN_H

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <string>
#include <vector>

// ============================================================================
// NmsFullPlugin - TensorRT IPluginV2IOExt for nonBlockNmsFull
// ============================================================================
//
// Input:  float [batch, per_box_stride, num_boxes]  (e.g. [1, 9, 8400])
// Output: float boxes  [batch, max_output_boxes, 4]
//         float scores [batch, max_output_boxes]
//         int   classes[batch, max_output_boxes]
//         int   counts  [batch]
// ============================================================================

class NmsFullPlugin : public nvinfer1::IPluginV2IOExt
{
public:
    NmsFullPlugin(float score_threshold, float iou_threshold,
                  int max_output_boxes, int per_box_stride, int num_classes);
    NmsFullPlugin(const void* data, size_t length);
    NmsFullPlugin() = delete;
    ~NmsFullPlugin() override;

    // -- IPluginV2 -----------------------------------------------
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // -- IPluginV2Ext --------------------------------------------
    nvinfer1::DataType getOutputDataType(int32_t index,
        const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;
    nvinfer1::Dims getOutputDimensions(int32_t index,
        const nvinfer1::Dims* inputs, int32_t nbInputDims) noexcept override;
    bool isOutputBroadcastAcrossBatch(int32_t outputIndex,
        const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override;
    void detachFromContext() noexcept override;

    // configurePlugin from IPluginV2Ext (old, made final in IOExt) — NOT overridden
    // configurePlugin from IPluginV2IOExt (new) — use this:

    // -- IPluginV2IOExt ------------------------------------------
    void configurePlugin(const nvinfer1::PluginTensorDesc* in, int32_t nbInputs,
                         const nvinfer1::PluginTensorDesc* out,
                         int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) const noexcept override;

    // Workspace from IPluginV2Ext (takes maxBatchSize)
    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    // enqueue from IPluginV2Ext (takes batchSize, no descriptors)
    int32_t enqueue(int32_t batchSize, const void* const* inputs,
                    void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

    // -- Serialization -------------------------------------------
    nvinfer1::IPluginV2Ext* clone() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    size_t getSerializationSize() const noexcept override;

private:
    float mScoreThreshold;
    float mIouThreshold;
    int   mMaxOutputBoxes;
    int   mPerBoxStride;
    int   mNumClasses;
    int   mNumBoxes;
    int   mBatchSize;
    std::string mNamespace;
};

// ============================================================================
// Plugin Creator
// ============================================================================

class NmsFullPluginCreator : public nvinfer1::IPluginCreator
{
public:
    NmsFullPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
        const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // NMS_FULL_PLUGIN_H
