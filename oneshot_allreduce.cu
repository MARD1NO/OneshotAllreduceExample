/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "src/fastertransformer/utils/cuda_type_utils.cuh"
#include "cuda_fp16.h"
#include "assert.h"
#include "vector"
#include "memory"

#define CUSTOM_AR_SIZE_THRESHOLD 50331648
#define MAX_ALL_REDUCE_BLOCKS 24
#define FLAG(a) ((uint32_t)((a) % 0x146))
#define RANKS_PER_NODE 8
#define WARP_SIZE 32
#define DEFAULT_BLOCK_SIZE 1024
#define DEFALUT_ALGO_AR_SIZE_THRESHOLD 196608


static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)


#ifdef ENABLE_BF16
typedef struct bf168 {
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
} bf168;
#endif

template<typename T>
struct AllReduceParams {
    size_t    elts_total;
    size_t    elts_per_rank;
    size_t    elts_per_block;
    size_t    rank_offset;
    size_t    rank, local_rank, node_id;
    uint32_t  barrier_flag;
    uint32_t* peer_barrier_ptrs[RANKS_PER_NODE];
    T*        peer_comm_buffer_ptrs[RANKS_PER_NODE];
    T*        local_output_buffer_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hadd2(const uint32_t& a, const uint32_t& b)
{
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t fadd(const uint32_t& a, const uint32_t& b)
{
    uint32_t c;
    asm volatile("add.f32 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void st_flag_release(uint32_t& flag, uint32_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void ld_flag_acquire(uint32_t& flag, uint32_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.global.volatile.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Type Converter that packs data format to 128 bits data type
template<typename T>
struct ARTypeConverter {
    using Type = uint4;
};

#ifdef ENABLE_BF16
template<>
struct ARTypeConverter<__nv_bfloat16> {
    using Type = bf168;
};
#endif

// add two 128b data
template<typename T_IN, typename T_COMP>
inline __device__ T_IN add128b(T_IN a, T_IN b);

template<>
inline __device__ uint4 add128b<uint4, uint16_t>(uint4 a, uint4 b)
{
    uint4 c;
    c.x = hadd2(a.x, b.x);
    c.y = hadd2(a.y, b.y);
    c.z = hadd2(a.z, b.z);
    c.w = hadd2(a.w, b.w);
    return c;
}

template<>
inline __device__ uint4 add128b<uint4, uint32_t>(uint4 a, uint4 b)
{
    uint4 c;
    c.x = fadd(a.x, b.x);
    c.y = fadd(a.y, b.y);
    c.z = fadd(a.z, b.z);
    c.w = fadd(a.w, b.w);
    return c;
}

#ifdef ENABLE_BF16
template<>
inline __device__ bf168 add128b<bf168, __nv_bfloat16>(bf168 a, bf168 b)
{
    bf168 c;
    c.x = bf16hadd2(a.x, b.x);
    c.y = bf16hadd2(a.y, b.y);
    c.z = bf16hadd2(a.z, b.z);
    c.w = bf16hadd2(a.w, b.w);
    return c;
}
#endif

// init 128bits data with 0
template<typename T>
inline __device__ T init_packed_type();

template<>
inline __device__ uint4 init_packed_type()
{
    return make_uint4(0u, 0u, 0u, 0u);
}

#ifdef ENABLE_BF16
template<>
inline __device__ bf168 init_packed_type()
{
    bf168  val;
    uint4& val_u = reinterpret_cast<uint4&>(val);
    val_u        = make_uint4(0u, 0u, 0u, 0u);
    return val;
}
#endif

template<typename T>
static __global__ void oneShotAllReduceKernel(AllReduceParams<T> params)
{
    // The block index.
    const int bidx = blockIdx.x;
    // The thread index with the block.
    const int tidx = threadIdx.x;

    // The number of elements packed into one for comms
    static constexpr int NUM_ELTS = std::is_same<T, uint32_t>::value ? 4 : 8;

    // Packed data type for comms
    using PackedType = typename ARTypeConverter<T>::Type;

    // The location in the destination array (load 8 fp16 or load 4 fp32 using LDG.128).
    size_t offset = bidx * params.elts_per_block + tidx * NUM_ELTS;
    // The end of the segment computed by that block.

    // size_t max_offset = std::min((bidx + 1) * params.elts_per_block, params.elts_per_rank);
    size_t max_offset = min((bidx + 1) * params.elts_per_block, params.elts_per_rank);

    // Synchronize the ranks.
    volatile uint32_t* barrier_d = params.peer_barrier_ptrs[params.local_rank];
    if (tidx < RANKS_PER_NODE) {
        // The 1st block notifies the other ranks.
        if (bidx == 0) {
            params.peer_barrier_ptrs[tidx][params.local_rank] = params.barrier_flag;
        }

        // Busy-wait until all ranks are ready.
        while (barrier_d[tidx] < params.barrier_flag) {}
    }

    // Make sure we can move on...
    __syncthreads();

    // The source pointers. Distributed round-robin for the different warps.
    const T* src_d[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
        int rank  = (params.local_rank + ii) % RANKS_PER_NODE;
        src_d[ii] = params.peer_comm_buffer_ptrs[rank];
    }

    // Each block accumulates the values from the different GPUs on the same node.
    for (size_t iter_offset = offset; iter_offset < max_offset; iter_offset += blockDim.x * NUM_ELTS) {
        // Iterate over the different ranks/devices on the node to load the values.
        PackedType vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
            vals[ii] = reinterpret_cast<const PackedType*>(&src_d[ii][iter_offset])[0];
        }

        // Sum the values from the different ranks.
        PackedType sums = init_packed_type<PackedType>();
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
            sums = add128b<PackedType, T>(sums, vals[ii]);
        }

        // Store to the destination buffer.
        reinterpret_cast<PackedType*>(&params.local_output_buffer_ptr[iter_offset])[0] = sums;
    }
}

template<typename T>
static __global__ void twoShotAllReduceKernel(AllReduceParams<T> params)
{

    // The block index.
    const int bidx = blockIdx.x;
    // The thread index with the block.
    const int tidx = threadIdx.x;

    // The number of elements packed into one for comms
    static constexpr int NUM_ELTS = std::is_same<T, uint32_t>::value ? 4 : 8;

    // Packed data type for comms
    using PackedType = typename ARTypeConverter<T>::Type;

    // The location in the destination array (load 8 fp16 or load 4 fp32 using LDG.128).
    size_t offset = bidx * params.elts_per_block + tidx * NUM_ELTS + params.rank_offset;
    // The end of the segment computed by that block.
    size_t max_offset = min(offset + params.elts_per_block, params.elts_total);

    // Synchronize the ranks.
    volatile uint32_t* barrier_d = params.peer_barrier_ptrs[params.local_rank];
    if (tidx < RANKS_PER_NODE) {
        // The 1st block notifies the other ranks.
        if (bidx == 0) {
            params.peer_barrier_ptrs[tidx][params.local_rank] = params.barrier_flag;
        }

        // Busy-wait until all ranks are ready.
        while (barrier_d[tidx] < params.barrier_flag) {}
    }

    // Make sure we can move on...
    __syncthreads();

    // The source pointers. Distributed round-robin for the different warps.
    T* src_d[RANKS_PER_NODE];
    // The destination ranks for round-robin gathering
    size_t dst_rank[RANKS_PER_NODE];
#pragma unroll
    for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
        int rank     = (params.local_rank + ii) % RANKS_PER_NODE;
        src_d[ii]    = params.peer_comm_buffer_ptrs[rank];
        dst_rank[ii] = rank;
    }

    // Each block accumulates the values from the different GPUs on the same node.
    for (size_t local_offset = offset; local_offset < max_offset; local_offset += blockDim.x * NUM_ELTS) {

        // Iterate over the different ranks/devices on the node to load the values.
        PackedType vals[RANKS_PER_NODE];
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
            vals[ii] = reinterpret_cast<const PackedType*>(&src_d[ii][local_offset])[0];
        }

        // Sum the values from the different ranks.
        PackedType sums = init_packed_type<PackedType>();
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
            sums = add128b<PackedType, T>(sums, vals[ii]);
        }

        // Store to the local buffer.
        reinterpret_cast<PackedType*>(&src_d[0][local_offset])[0] = sums;
    }

    // sync threads to make sure all block threads have the sums
    __syncthreads();

    // barreris among the blocks with the same idx (release-acuqire semantics)
    if (tidx < RANKS_PER_NODE) {
        // The all blocks notifies the other ranks.
        uint32_t flag_block_offset = RANKS_PER_NODE + bidx * RANKS_PER_NODE;
        st_flag_release(params.barrier_flag, params.peer_barrier_ptrs[tidx] + flag_block_offset + params.local_rank);

        // Busy-wait until all ranks are ready.
        uint32_t  rank_barrier   = 0;
        uint32_t* peer_barrier_d = params.peer_barrier_ptrs[params.local_rank] + flag_block_offset + tidx;
        do {
            ld_flag_acquire(rank_barrier, peer_barrier_d);
        } while (rank_barrier != params.barrier_flag);
    }

    // sync threads to make sure all other ranks has the final partial results
    __syncthreads();

    // Gather all needed elts from other intra-node ranks
    for (size_t local_offset = offset; local_offset < max_offset; local_offset += blockDim.x * NUM_ELTS) {
#pragma unroll
        for (int ii = 0; ii < RANKS_PER_NODE; ++ii) {
            // use round-robin gathering from other ranks
            int offset_rank = local_offset + (dst_rank[ii] - params.local_rank) * params.elts_per_rank;
            reinterpret_cast<PackedType*>(&params.local_output_buffer_ptr[offset_rank])[0] =
                reinterpret_cast<PackedType*>(&src_d[dst_rank[ii]][offset_rank])[0];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void kernelLaunchConfig(
    int& blocks_per_grid, int& threads_per_block, size_t elts, int kernel_algo, size_t data_type_bytes)
{
    assert(data_type_bytes == 2 || data_type_bytes == 4);
    // NOTE: need to support FP16 and FP32
    size_t elts_per_thread = 16 / data_type_bytes;
    size_t elts_per_warp   = (16 * WARP_SIZE) / data_type_bytes;
    switch (kernel_algo) {
        case 0: {  // one stage all reduce algo
            assert(elts % elts_per_warp == 0);
            if (elts < (elts_per_thread * DEFAULT_BLOCK_SIZE)) {  // local reduce
                threads_per_block = ((elts + elts_per_warp - 1) / elts_per_warp) * WARP_SIZE;
                blocks_per_grid   = 1;
            }
            else {  // local reduce
                if (elts % (elts_per_thread * threads_per_block) == 0) {
                    blocks_per_grid =
                        (elts + elts_per_thread * threads_per_block - 1) / (elts_per_thread * threads_per_block);
                    // NOTE: need to adjust here
                    // 如果一个block不够，就需要多iter几轮
                    if (blocks_per_grid > MAX_ALL_REDUCE_BLOCKS) {
                        int iter_factor = 1;
                        while (blocks_per_grid / iter_factor > MAX_ALL_REDUCE_BLOCKS || blocks_per_grid % iter_factor) {
                            iter_factor += 1;
                        }
                        blocks_per_grid /= iter_factor;
                    }
                }
                else {
                    int total_threads = elts / elts_per_thread;
                    blocks_per_grid   = 1;
                    while (total_threads % blocks_per_grid != 0
                           || total_threads / blocks_per_grid > DEFAULT_BLOCK_SIZE) {
                        blocks_per_grid += 1;
                    }
                    threads_per_block = total_threads / blocks_per_grid;
                }
            }
            break;
        }
        case 1: {  // two stage all reduce algo
            int total_threads = elts / RANKS_PER_NODE / RANKS_PER_NODE;
            assert(elts / RANKS_PER_NODE % RANKS_PER_NODE == 0 && total_threads % WARP_SIZE == 0);

            while (total_threads % blocks_per_grid != 0 || total_threads / blocks_per_grid > DEFAULT_BLOCK_SIZE) {
                blocks_per_grid += 1;
            }

            threads_per_block = total_threads / blocks_per_grid;

            // NOTE: need to adjust here
            if (blocks_per_grid > MAX_ALL_REDUCE_BLOCKS) {
                int iter_factor = 1;
                while (blocks_per_grid / iter_factor > MAX_ALL_REDUCE_BLOCKS || blocks_per_grid % iter_factor) {
                    iter_factor += 1;
                }
                blocks_per_grid /= iter_factor;
            }
            break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams<T>& param, cudaStream_t stream)
{
    size_t elts_total      = param.elts_total;
    int    blocks_per_grid = 1, threads_per_block = DEFAULT_BLOCK_SIZE;
    int    kernel_algo = 1;
    if (elts_total <= DEFALUT_ALGO_AR_SIZE_THRESHOLD) {
        kernel_algo = 0;
    }

    kernelLaunchConfig(blocks_per_grid, threads_per_block, elts_total, kernel_algo, sizeof(T));

    if (kernel_algo == 0) {
        param.elts_per_rank  = elts_total;
        param.elts_per_block = param.elts_per_rank / blocks_per_grid;
        oneShotAllReduceKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
    }
    else {
        param.elts_per_rank  = param.elts_total / RANKS_PER_NODE;
        param.elts_per_block = param.elts_per_rank / blocks_per_grid;
        param.rank_offset    = param.rank * param.elts_per_rank;
        twoShotAllReduceKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(param);
    }
}

template<typename T>
class CustomAllReduceComm {
public:
    CustomAllReduceComm(size_t rank_size, size_t rank){
        param_.barrier_flag = 0;
        // NOTE: assume All Reduce happens within the node (DGX A100)
        param_.rank       = rank_;
        param_.local_rank = rank_;
        param_.node_id    = 0;
    }
    
    ~CustomAllReduceComm(){
        cudaPointerAttributes comm_buffer_attributes, barrier_attributes;
        check_cuda_error(cudaPointerGetAttributes(&comm_buffer_attributes, param_.peer_comm_buffer_ptrs[rank_]));
        check_cuda_error(cudaPointerGetAttributes(&barrier_attributes, param_.peer_barrier_ptrs[rank_]));
        if (comm_buffer_attributes.type == 2) {
            check_cuda_error(cudaFree(param_.peer_comm_buffer_ptrs[rank_]));
        }
        if (barrier_attributes.type == 2) {
            check_cuda_error(cudaFree(param_.peer_barrier_ptrs[rank_]));
        }
    }

    void customAllReduce(size_t elts, cudaStream_t stream){
        param_.elts_total   = elts;
        param_.barrier_flag = FLAG(param_.barrier_flag + 1);

        invokeOneOrTwoShotAllReduceKernel<T>(param_, stream);

        // swap back
        // output_tensor_->at(0).data = (const void*)tmp_tensor_data_;
    }

    void allocateAndExchangePeerAccessPointer(
        std::vector<std::shared_ptr<CustomAllReduceComm>>* custom_all_reduce_comms){
            assert(custom_all_reduce_comms->size() == rank_size_);
        assert(rank_ == 0);
        // Enable Peer to Peer Access
        enableP2P(rank_size_);
        for (size_t i = 0; i < rank_size_; i++) {
            check_cuda_error(cudaSetDevice(i));
            check_cuda_error(cudaMalloc(&(param_.peer_comm_buffer_ptrs[i]), CUSTOM_AR_SIZE_THRESHOLD));
            check_cuda_error(
                cudaMalloc(&(param_.peer_barrier_ptrs[i]), rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
            check_cuda_error(
                cudaMemset(param_.peer_barrier_ptrs[i], 0, rank_size_ * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
            T*        current_peer_comm_buffer_ptr = param_.peer_comm_buffer_ptrs[i];
            uint32_t* current_peer_barrier_ptr     = param_.peer_barrier_ptrs[i];
            // Assume current comm allocates device memory on all ranks (rank_ == 0)
            for (size_t j = 1; j < rank_size_; j++) {
                static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(j).get())
                    ->param_.peer_comm_buffer_ptrs[i] = current_peer_comm_buffer_ptr;
                static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(j).get())->param_.peer_barrier_ptrs[i] =
                    current_peer_barrier_ptr;
            }
        }

        // Set default local_output_buffer_ptr to local peer_comm_buffer_ptrs
        for (size_t i = 0; i < rank_size_; i++) {
            static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(i).get())->param_.local_output_buffer_ptr =
                static_cast<CustomAllReduceComm<T>*>(custom_all_reduce_comms->at(i).get())->param_.peer_comm_buffer_ptrs[i];
        }
    }

    // bool swapInternalBuffer(std::vector<Tensor>* tensor_buffer, size_t elts){
    //     // Check if all reduce elts meet the requirement of custom kernels
    //     // If meet, then swap the local comm buffer ptr with output tensor data pointer (avoid additional
    //     // memory movement)
    //     if (rank_size_ > 1 && elts * sizeof(T) <= CUSTOM_AR_SIZE_THRESHOLD) {
    //         tmp_tensor_data_               = (T*)(tensor_buffer->at(0).data);
    //         output_tensor_                 = tensor_buffer;
    //         tensor_buffer->at(0).data      = param_.peer_comm_buffer_ptrs[rank_];
    //         param_.local_output_buffer_ptr = tmp_tensor_data_;
    //         return true;
    //     }
    //     return false;
    // }

    bool swapInternalBuffer(T* tensor_buffer, size_t elts){
        // Check if all reduce elts meet the requirement of custom kernels
        // If meet, then swap the local comm buffer ptr with output tensor data pointer (avoid additional
        // memory movement)
        if (rank_size_ > 1 && elts * sizeof(T) <= CUSTOM_AR_SIZE_THRESHOLD) {
            tmp_tensor_data_               = tensor_buffer;
            // output_tensor_                 = tensor_buffer;

            // tensor_buffer->at(0).data      = param_.peer_comm_buffer_ptrs[rank_];
            tensor_buffer      = param_.peer_comm_buffer_ptrs[rank_];
            param_.local_output_buffer_ptr = tmp_tensor_data_;
            return true;
        }
        return false;
    }

    void enableP2P(int ngpus){
        int peer_access_available = 0;
        for (int i = 0; i < ngpus; i++) {
            cudaSetDevice(i);
            for (int j = 0; j < ngpus; j++) {
                if (i == j) {
                    continue;
                }
                cudaDeviceCanAccessPeer(&peer_access_available, i, j);
                // Custom AR Kernels need DGX A100 NVSWITCH connections
                assert(peer_access_available);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

private:
    AllReduceParams<T>   param_;
    // std::vector<Tensor>* output_tensor_;
    T*                   tmp_tensor_data_;
    size_t               rank_size_;
    size_t               rank_;
};


template<typename T>
void initCustomAllReduceComm(std::vector<std::shared_ptr<CustomAllReduceComm<T>>>* custom_all_reduce_comms,
                             int                                               enable_custom_all_reduce,
                             size_t                                            rank_size)
{
//     if (custom_all_reduce_comms == 0) {
//         // don't use custom all reduce kernels, fall back to NCCL
//         for (size_t i = 0; i < rank_size; i++) {
//             custom_all_reduce_comms->push_back(nullptr);
//         }
//         return;
//     }

//     if (rank_size != RANKS_PER_NODE) {
// #ifdef BUILD_MULTI_GPU
//         if (rank_size > 1) {
//             FT_LOG_WARNING("Custom All Reduce only supports 8 Ranks currently. Using NCCL as Comm.");
//         }
// #else
//         FT_CHECK_WITH_INFO(rank_size == 1,
//                            fmtstr("Custom All Reduce only supports 8 Ranks currently, got rank_size %ld. FT needs "
//                                   "the NCCL library to communicate among devices but has built without NCCL. "
//                                   "Please use the flag -DBUILD_MULTI_GPU=ON when compiling.",
//                                   rank_size));
// #endif
//         for (size_t i = 0; i < rank_size; i++) {
//             custom_all_reduce_comms->push_back(nullptr);
//         }
//         return;
//     }

// #if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    for (size_t i = 0; i < rank_size; i++) {
        custom_all_reduce_comms->push_back(std::make_shared<CustomAllReduceComm<T>>(rank_size, i));
    }
    custom_all_reduce_comms->at(0)->allocateAndExchangePeerAccessPointer(custom_all_reduce_comms);
// #else
//     FT_LOG_WARNING("Custom All Reduce is not supported before CUDA 11.2. Using NCCL as Comm.");
//     for (size_t i = 0; i < rank_size; i++) {
//         custom_all_reduce_comms->push_back(nullptr);
//     }
// #endif
}

template<typename T>
void test(){
    int num_gpu = 8; 
    size_t elem_num = 1024; 
    std::vector<cudaStream_t> stream_vec{}; 
    for(int i = 0; i < num_gpu; i++){
        cudaSetDevice(i); 
        cudaStream_t stream; 
        cudaStreamCreate(&stream); 
        stream_vec.push_back(stream); 
    }

    std::vector<std::shared_ptr<CustomAllReduceComm<T>>> custom_all_reduce_comms; 

    initCustomAllReduceComm<T>(&custom_all_reduce_comms, true/*enable_custom_all_reduce*/, num_gpu); 

    std::vector<T*> data_vec{}; 
    for(int i = 0; i < num_gpu; i++){
        cudaSetDevice(i); 
        T *x; 
        cudaMalloc(&x, sizeof(T) * elem_num); 
        data_vec.push_back(x); 
    }

    for(int i = 0; i < num_gpu; i++){
        cudaSetDevice(i); 
        custom_all_reduce_comms[i]->swapInternalBuffer(data_vec[i], elem_num); 
    }

    for(int i = 0; i < num_gpu; i++){
        custom_all_reduce_comms[i]->customAllReduce(elem_num, stream_vec[i]); 
    }

    for(int i = 0; i < num_gpu; i++){
        cudaSetDevice(i); 
        cudaFree(data_vec[i]); 
    }

    for(int i = 0; i < num_gpu; i++){
        cudaSetDevice(i); 
        cudaStreamDestroy(stream_vec[i]); 
    }

}

int main(){
    test<uint32_t>(); 
}