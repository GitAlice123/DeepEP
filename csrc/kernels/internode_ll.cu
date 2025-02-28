#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace internode_ll {

template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                         int* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    nvshmemx_barrier_all_block();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure low-latency mode work fine)
    nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}

template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
dispatch(void* packed_recv_x, float* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert, int* atomic_counter_per_local_expert,
         int* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int phases) {
    // SM的数量等于块的数量
    const auto sm_id = static_cast<int>(blockIdx.x);
    // thread_id是在块内（SM内）的索引
    const auto thread_id = static_cast<int>(threadIdx.x);
    // warp_id是在块内（SM内）的索引
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    // SM的数量
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    // 每个rank上的expert数量
    const auto num_local_experts = num_experts / num_ranks;
    // 该warp所属的warp组
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    // 该warp在warp组内的索引
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    // 该warp所负责的expert的索引
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    constexpr float kFP8Margin = 1e-4, kFP8Amax = 448, kFP8AmaxInv = 1.0f / 448.0f;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_int4 = kHidden / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    const size_t num_bytes_per_msg = kHidden + num_scales * sizeof(float) + sizeof(int4);
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // Expert counts
    // 一个SM里面维护一个这个数组
    __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens（转换和发送）
    // 2. The last warp for reading `topk_idx` and count for per-expert information（处理topk_idx路由决策和专家计数）
    // topk_idx:数组，存储每个 Token 的 Top-K 专家索引
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        //线程分配：每个 SM 处理一组 token，步长为 num_sms（SM 总数），实现 ​跨 SM 负载均衡。
        //​示例：若 num_sms=4，则 SM 0 处理 token 0,4,8,...，SM 1 处理 token 1,5,9,...，依此类推。
        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_int2 = reinterpret_cast<int2*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_int2) + kHidden);
            const auto rdma_x_src_idx = reinterpret_cast<int*>(rdma_x_scales + num_scales);

            // num_topk:每个 Token 的 Top-K 专家数量（比如K=3）
            // dst_expert_idx：当前warp负责的专家索引（只有小于num_topk的warp才会执行发送）
            // Overlap top-k index read and source token index write
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            /**
                ​读取数据：从全局内存加载 bfloat16 数据。
                ​计算最大值（amax）​：找到数据块内的最大绝对值，用于缩放。
                ​量化到 FP8：将 bfloat16 转换为 FP8 格式。
                ​存储结果：将量化后的数据写入 RDMA 缓冲区。
             */
            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read and calculate local amax
                auto int4_value = __ldg(x_int4 + i);
                auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                float fp32_values[kNumElemsPerRead];
                float amax = kFP8Margin, scale, scale_inv;
                #pragma unroll
                for (int j = 0; j < kNumElemsPerRead; ++ j) {
                    fp32_values[j] = static_cast<float>(bf16_values[j]);
                    amax = fmaxf(amax, fabsf(fp32_values[j]));
                }

                // Reduce amax and scale
                EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;
                if (lane_id == 0 or lane_id == 16)
                    rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                // Cast into send buffer
                int2 int2_value;
                auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerRead; j += 2) {
                    float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                    fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                }
                rdma_x_int2[i] = int2_value;
            }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            // 下面一个warp里面的所有线程都会执行这个操作
            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                /*
                    ​**atomicAdd**：原子操作，为当前专家（dst_expert_idx）分配一个唯一的槽位索引。
                    ​目的：确保多个线程发送到同一专家时，数据不会覆盖。
                    ​行为：
                        只有 lane_id == 0 的线程执行原子操作（避免所有线程争抢）。
                        其他线程的 slot_idx 初始化为 0（后续通过 __shfl_sync 广播）。
                        ​示例：若专家 3 的 atomic_counter_per_expert[3] 初始为 0，第一次原子操作后变为 1，返回 0。
                */
                // 这个slot，可以实现从源节点发送到目的节点不按序，因为来一个token就给他分配一个slot，往后依次排就行
                // 目的节点接收的时候按照槽位顺序接收即可，自动实现了按序接收
                // 所以把对顺序的保证从网络层移到了应用层的内存摆放方式上
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                // warp内所有线程广播slot_idx
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_int2);
                /**
                num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg:
                    在目标节点里面，在dst_expert_local_idx之前每个专家的缓冲区大小（每个专家都要接收所有节点的所有token）
                    slot指的是，一个专家为一个节点的一个token分配的缓冲区(留的插槽)
                前面再乘以dst_expert_local_idx，是为了找到当前专家接收所有节点数据的缓冲区起始位置
                rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg:找到当前节点的缓冲区起始位置
                 */
                // 这个slot_idx
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;
                if (dst_rank != rank) {
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                } else {
                    // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                // Increase counter after finishing
                __syncwarp();
                // 本次循环的这个token被发送出去了，所以要增加计数
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        // 每个组负责哪个专家（专家号）是确定的，expert_count记录该SM负责的每个专家的token数量
        // 因为只有每个SM的最后一个warp做计算，相当于每个SM维护一个expert_count数组
        int expert_count[kNumWarpGroups] = {0};
        // expert_begin_idx~expert_end_idx:当前SM负责的专家范围
        const auto expert_begin_idx = sm_id * kNumWarpGroups;
        const auto expert_end_idx = min(expert_begin_idx + kNumWarpGroups, num_experts);

        // Per lane count
        // 目前expert_count还是每个lane（线程）在各自维护
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)// 在本SM负责的范围内
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        // 把该warp的计数结果累加到lane_id为0的expert_count数组中
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            // 这个sum是该SM负责的这个专家要接收的token总数
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                // atomic_finish_counter_per_expert在后面收到一个就会增加1，最后等于FINISHED_SUM_TAG*2时表示所有token都接收完了
                // *2是因为在最后结束的时候，要额外增加一个FINISHED_SUM_TAG
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * kNumWarpGroups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
        if (dst_rank != rank) {
            nvshmemi_ibgda_rma_p(rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1,
                                 dst_rank, dst_expert_local_idx, 0);
            // 保证接收端有足够的空间接收
            nvshmemi_ibgda_prepare_recvs(dst_rank, dst_expert_local_idx);
        } else {
            st_na_release(rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1);
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
    }
    __syncwarp();










    // -------------------------------------------------------------------------------
    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Receiving and packing
    // 这里的responsible_expert_idx，只用于得到一个src_rank（从这个rank发来的）
    // 和一个local_expert_idx（发到我当前rank的哪个专家）的二元组，没有特殊含义
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_x_scales = packed_recv_x_scales + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_scales;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumWarpGroups], shared_recv_token_begin_idx[kNumWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        if (sub_warp_id == 1 and lane_id == 0) {
            if (src_rank != rank) {
                nvshmemi_ibgda_poll_recv(src_rank, local_expert_idx);
                num_recv_tokens = ld_acquire_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank);
                EP_DEVICE_ASSERT(num_recv_tokens != 0);
            } else {
                while ((num_recv_tokens = ld_acquire_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
            }
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(atomic_counter_per_local_expert + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(kNumWarpsPerGroup * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src = reinterpret_cast<int4*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            const auto dst = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst, src, ld_nc_global, st_na_global);

            // Copy scales
            const auto src_scales = reinterpret_cast<float*>(rdma_recv_x_uint8 + i * num_bytes_per_msg + kHidden);
            const auto dst_scales = reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
            const auto scale_stride = num_ranks * num_max_dispatch_tokens_per_rank;
            auto scale_0 = lane_id < num_scales ? ld_nc_global(src_scales + lane_id) : 0;
            auto scale_1 = (lane_id + 32) < num_scales ? ld_nc_global(src_scales + lane_id + 32) : 0;
            lane_id < num_scales ? dst_scales[lane_id * scale_stride] = scale_0 : 0.0f;
            (lane_id + 32) < num_scales ? dst_scales[(lane_id + 32) * scale_stride] = scale_1 : 0.0f;

            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(src_scales + num_scales);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();
        }
    }
}

// 这里还是每个进程都执行
void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              void* workspace, cudaStream_t stream, int phases) {
    constexpr int kNumMaxTopK = 9;
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
    EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);
    EP_HOST_ASSERT(cell_div(static_cast<int>(hidden * 2 / sizeof(int4)), 32 * (num_warps - 1)) <= 2);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // Use the last part `rdma_recv_count` as `atomic_counter_per_local_expert`
    // NOTES: this part will be cleaned in `combine`
    auto atomic_counter_per_local_expert = rdma_recv_count + num_ranks * (num_experts / num_ranks);

#define DISPATCH_LAUNCH_CASE(hidden) \
LAUNCH_KERNEL(&cfg, dispatch<kNumWarpGroups, kNumWarpsPerGroup, hidden>, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, atomic_counter_per_local_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, phases); break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    // BF16 mode: always use BF16 for hidden data (ignoring the extra flag slot)
    constexpr size_t num_bytes_per_slot = sizeof(int4) + kHidden * sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row + 4);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot + sizeof(int4);
            if (dst_rank == rank) {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else {
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, hidden * sizeof(nv_bfloat16), dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put finishing flag
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            if (dst_rank != rank) {
                nvshmemi_ibgda_rma_p(rdma_recv_flag + global_expert_idx, 1, dst_rank, local_expert_idx, 0);
            } else {
                st_na_release(rdma_recv_flag + global_expert_idx, 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive and notify PCIe usage
    if (responsible_expert_idx < num_experts) {
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0) {
            // TODO: refactor QP indices
            auto src_rank = responsible_expert_idx / num_local_experts;
            auto src_expert_idx = responsible_expert_idx % num_local_experts;
            if (src_rank != rank) {
                nvshmemi_ibgda_poll_recv(src_rank, src_expert_idx);
            } else {
                while (ld_acquire_global(rdma_recv_flag + responsible_expert_idx) == 0);
            }
        }
    }
    cg::this_grid().sync();

    // Reduce tokens with FP8 cast
    EP_DEVICE_ASSERT(num_topk <= 32 and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type + 4);

                // Reduce
                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
            }

            // Write results
            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             void* workspace, cudaStream_t stream, int phases) {
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
    constexpr int kNumMaxTopk = 9;

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine<kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>; \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace deep_ep
