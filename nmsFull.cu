#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cub/cub.cuh>

// ============================================================================
// 共享的工具函数
// ============================================================================

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float calculateIoU(
    float x1, float y1, float w1, float h1,
    float x2, float y2, float w2, float h2) {
    
    float left1 = x1 - w1 / 2.0f;
    float right1 = x1 + w1 / 2.0f;
    float top1 = y1 - h1 / 2.0f;
    float bottom1 = y1 + h1 / 2.0f;
    
    float left2 = x2 - w2 / 2.0f;
    float right2 = x2 + w2 / 2.0f;
    float top2 = y2 - h2 / 2.0f;
    float bottom2 = y2 + h2 / 2.0f;
    
    float inter_left = max(left1, left2);
    float inter_right = min(right1, right2);
    float inter_top = max(top1, top2);
    float inter_bottom = min(bottom1, bottom2);
    
    float inter_width = max(0.0f, inter_right - inter_left);
    float inter_height = max(0.0f, inter_bottom - inter_top);
    float intersection = inter_width * inter_height;
    
    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - intersection;
    
    if (union_area < 1e-8f) return 0.0f;
    return intersection / union_area;
}

// ============================================================================
// KERNEL 1: 不分块 NMS
// ============================================================================

__global__ void nonBlockNmsKernel(
    const int batch_size,
    const int num_boxes,
    const int per_box_stride,
    const float score_threshold,
    const float iou_threshold,
    const int max_output_boxes,
    const float* input_data,
    int* output_count,
    float* output_boxes,
    int* output_classes,
    float* output_scores,
    const int* sorted_indices,
    const float* sorted_scores,
    const int* sorted_classes
) {
    int b = blockIdx.x;
    int idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (b >= batch_size || idx >= num_boxes) return;
    
    int box_idx = sorted_indices[b * num_boxes + idx];
    float box_score = sorted_scores[b * num_boxes + idx];
    int box_class = sorted_classes[b * num_boxes + idx];
    
    if (box_score < score_threshold) return;
    
    bool suppressed = false;
    for (int j = 0; j < idx; j++) {
        int prev_box_idx = sorted_indices[b * num_boxes + j];
        float prev_score = sorted_scores[b * num_boxes + j];
        int prev_class = sorted_classes[b * num_boxes + j];
        
        if (prev_class != box_class) continue;
        if (prev_score < score_threshold) continue;
        
        int off_curr = b * per_box_stride * num_boxes + box_idx * per_box_stride;
        int off_prev = b * per_box_stride * num_boxes + prev_box_idx * per_box_stride;
        
        float x1 = input_data[off_curr];
        float y1 = input_data[off_curr + 1];
        float w1 = input_data[off_curr + 2];
        float h1 = input_data[off_curr + 3];
        
        float x2 = input_data[off_prev];
        float y2 = input_data[off_prev + 1];
        float w2 = input_data[off_prev + 2];
        float h2 = input_data[off_prev + 3];
        
        float iou = calculateIoU(x1, y1, w1, h1, x2, y2, w2, h2);
        
        if (iou > iou_threshold) {
            suppressed = true;
            break;
        }
    }
    
    if (suppressed) return;
    
    int pos = atomicAdd(&output_count[b], 1);
    if (pos < max_output_boxes) {
        int off = b * per_box_stride * num_boxes + box_idx * per_box_stride;
        output_boxes[b * max_output_boxes * 4 + pos * 4 + 0] = input_data[off];
        output_boxes[b * max_output_boxes * 4 + pos * 4 + 1] = input_data[off + 1];
        output_boxes[b * max_output_boxes * 4 + pos * 4 + 2] = input_data[off + 2];
        output_boxes[b * max_output_boxes * 4 + pos * 4 + 3] = input_data[off + 3];
        output_classes[b * max_output_boxes + pos] = box_class;
        output_scores[b * max_output_boxes + pos] = box_score;
    }
}

// ============================================================================
// KERNEL 3: 评分提取
// ============================================================================

__global__ void extractScoresKernel(
    const int batch_size,
    const int num_boxes,
    const int per_box_stride,
    const int num_classes,
    const float* input_data,
    int* sorted_indices,
    float* sorted_scores,
    int* sorted_classes
) {
    int b = blockIdx.x;
    int idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (b >= batch_size || idx >= num_boxes) return;
    
    int offset = b * per_box_stride * num_boxes + idx * per_box_stride;
    
    float max_score = 0.0f;
    int max_class = 0;
    
    for (int c = 0; c < num_classes; c++) {
        float class_conf = sigmoid(input_data[offset + 4 + c]);
        if (class_conf > max_score) {
            max_score = class_conf;
            max_class = c;
        }
    }
    
    sorted_indices[b * num_boxes + idx] = idx;
    sorted_scores[b * num_boxes + idx] = max_score;
    sorted_classes[b * num_boxes + idx] = max_class;
}

// ============================================================================
// Kernel-only wrappers (no sort — pre-sorted data from CPU)
// ============================================================================

extern "C" void extractScores(
    cudaStream_t stream,
    const int batch_size,
    const int num_boxes,
    const int per_box_stride,
    const int num_classes,
    const float* d_input,
    int* d_indices,
    float* d_scores,
    int* d_classes
) {
    int threads = 256;
    int blocks = (num_boxes + threads - 1) / threads;
    extractScoresKernel<<<dim3(batch_size, blocks), threads, 0, stream>>>(
        batch_size, num_boxes, per_box_stride, num_classes, d_input, d_indices, d_scores, d_classes);
}



// ============================================================================
// GPU Sort Kernels (Full pipeline: extract + sort/block + NMS)
// ============================================================================

// 全局排序后重排到 sorted_* 缓冲区
__global__ void globalSortReorderKernel(
    const int num_boxes,
    const int* d_order,
    const int* d_indices,
    const float* d_scores,
    const int* d_classes,
    int* sorted_idx,
    float* sorted_sc,
    int* sorted_cls
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_boxes) return;
    int src = d_order[i];
    sorted_idx[i] = d_indices[src];
    sorted_sc[i]  = d_scores[src];
    sorted_cls[i] = d_classes[src];
}





// ============================================================================
// Full Pipeline Functions (extract + sort/block + NMS, all on GPU, included in timing)
// ============================================================================

extern "C" void nonBlockNmsFull(
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
) {
    int threads = 256;

    for (int b = 0; b < batch_size; b++) {
        // Step 1: GPU extract
        int blocks = (num_boxes + threads - 1) / threads;
        extractScoresKernel<<<dim3(1, blocks), threads, 0, stream>>>(
            1, num_boxes, per_box_stride, num_classes, d_input + b * per_box_stride * num_boxes,
            d_temp_indices, d_temp_scores, d_temp_classes);

        // Step 2: cub global sort descending by score
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_temp_scores, d_sorted_scores + b * num_boxes,
            d_temp_indices, d_order, num_boxes, 0, sizeof(int) * 8, stream);

        // Step 3: reorder classes
        globalSortReorderKernel<<<(num_boxes + threads - 1) / threads, threads, 0, stream>>>(
            num_boxes, d_order, d_temp_indices, d_temp_scores, d_temp_classes,
            d_sorted_indices + b * num_boxes,
            d_sorted_scores + b * num_boxes,
            d_sorted_classes + b * num_boxes);

        // Step 4: nonBlockNmsKernel
        nonBlockNmsKernel<<<dim3(1, blocks), threads, 0, stream>>>(
            1, num_boxes, per_box_stride, score_threshold, iou_threshold, max_output_boxes,
            d_input + b * per_box_stride * num_boxes,
            d_output_count + b,
            d_output_boxes + b * max_output_boxes * 4,
            d_output_classes + b * max_output_boxes,
            d_output_scores + b * max_output_boxes,
            d_sorted_indices + b * num_boxes,
            d_sorted_scores + b * num_boxes,
            d_sorted_classes + b * num_boxes);
    }
}

extern "C" size_t nonBlockNmsFullGetTempSize(int num_boxes) {
    size_t temp_bytes = 0;
    float *d_keys = nullptr, *d_keys_out = nullptr;
    int *d_vals = nullptr, *d_vals_out = nullptr;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes,
        d_keys, d_keys_out, d_vals, d_vals_out, num_boxes);
    return temp_bytes;
}
