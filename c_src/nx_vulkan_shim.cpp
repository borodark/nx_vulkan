/* nx_vulkan_shim.cpp — flat C ABI on top of spirit::Engine::Backend::vulkan.
 *
 * Includes spirit's Vulkan backend header at compile time. The header
 * lives at ../../../spirit/core/include/engine/Backend_par_vulkan.hpp;
 * the source at ../../../spirit/core/src/engine/Backend_par_vulkan.cpp.
 * build.rs adds those paths to the include + source list.
 */

#include "nx_vulkan_shim.h"
#include <engine/Backend_par_vulkan.hpp>
#include <cstring>
#include <map>
#include <string>
#include <utility>

using namespace Engine::Backend::vulkan;

/* The selected device name, cached after nxv_init so we can hand out
 * a stable pointer to Rust. */
static std::string g_device_name;

/* Pipeline cache keyed on (spv_path, op_spec_constant). Persistent
 * across calls — first dispatch pays the create cost, subsequent ones
 * reuse the pipeline. Cleared in nxv_destroy.
 *
 * push_size for each shader family:
 *   binary    : sizeof(uint)        (n)
 *   unary     : sizeof(uint)        (n)
 *   reduce    : sizeof(uint)        (n)        — handled by spirit's reduce()
 *   matmul    : 3 * sizeof(uint)    (M, N, K)
 *   random    : 2 * sizeof(uint)    (n, seed)
 *
 * For the cache key we add n_buffers because matmul/binary share a
 * spec-const-0 path with different binding counts. */
struct PipeKey {
    std::string path;
    unsigned int op;
    unsigned int n_buffers;
    bool operator<(const PipeKey& o) const {
        if (path != o.path) return path < o.path;
        if (op != o.op) return op < o.op;
        return n_buffers < o.n_buffers;
    }
};
static std::map<PipeKey, VkPipe*> g_pipe_cache;

static VkPipe* get_or_create_pipe(const std::string& spv_path, unsigned int op,
                                  unsigned int n_buffers) {
    PipeKey key{spv_path, op, n_buffers};
    auto it = g_pipe_cache.find(key);
    if (it != g_pipe_cache.end()) return it->second;

    VkShaderModule shader = load_shader(spv_path);
    if (!shader) return nullptr;

    /* push_size: declare 56 (max across all shader families). Vulkan
     * ignores any push range bytes the shader doesn't read. binary/
     * unary/random/transpose all use ≤12; reduce_axis uses 16;
     * fused_elementwise uses 40 (n + n_ops + 8 op slots);
     * elementwise_binary_broadcast uses 56 (n + ndim + out_shape[4] +
     * a_strides[4] + b_strides[4]). */
    uint32_t push_size = 56;

    VkPipe* pipe = new VkPipe();
    int rc = create_pipeline(pipe, shader, n_buffers, push_size, (int32_t) op);
    if (rc != 0) {
        delete pipe;
        return nullptr;
    }

    g_pipe_cache[key] = pipe;
    return pipe;
}

extern "C" {

int nxv_init(void) {
    int rc = vk_init(0);
    if (rc != 0) return rc;
    g_device_name = g_vk_ctx.device_props.deviceName;
    return 0;
}

void nxv_destroy(void) {
    /* Tear down cached pipelines first (they reference the device). */
    for (auto& kv : g_pipe_cache) {
        destroy_pipeline(kv.second);
        delete kv.second;
    }
    g_pipe_cache.clear();

    vk_destroy();
    g_device_name.clear();
}

const char* nxv_device_name(void) {
    return g_device_name.empty() ? nullptr : g_device_name.c_str();
}

int nxv_has_f64(void) {
    return g_vk_ctx.has_float64 ? 1 : 0;
}

/* Tensor primitives — heap-allocate a VkBuf so the handle survives
 * across NIF calls. Lifetime is owned by the Rust ResourceArc; when
 * the Elixir reference is GC'd, Rust calls nxv_buf_free which
 * delegates to spirit's buf_free + delete. */

void* nxv_buf_alloc(unsigned long n_bytes) {
    VkBuf* buf = new VkBuf();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    int rc = buf_alloc(buf, (VkDeviceSize) n_bytes, usage, mem);
    if (rc != 0) {
        delete buf;
        return nullptr;
    }
    return (void*) buf;
}

void nxv_buf_free(void* handle) {
    if (!handle) return;
    VkBuf* buf = (VkBuf*) handle;
    buf_free(buf);
    delete buf;
}

int nxv_buf_upload(void* handle, const void* data, unsigned long n_bytes) {
    if (!handle || !data) return -1;
    VkBuf* buf = (VkBuf*) handle;
    return upload(buf, data, (VkDeviceSize) n_bytes);
}

int nxv_buf_download(void* handle, void* data, unsigned long n_bytes) {
    if (!handle || !data) return -1;
    VkBuf* buf = (VkBuf*) handle;
    return download(buf, data, (VkDeviceSize) n_bytes);
}

int nxv_apply_binary(void* out, void* a, void* b,
                     unsigned int n, unsigned int op,
                     const char* spv_path) {
    if (!out || !a || !b || !spv_path) return -1;

    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), op, 3);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_b   = (VkBuf*) b;
    VkBuf* buf_out = (VkBuf*) out;

    /* Shader binding order: a, b, out. Push constant: n. */
    VkBuffer bufs[3] = { buf_a->buffer, buf_b->buffer, buf_out->buffer };
    unsigned int push_n = n;
    unsigned int groups = (n + 255) / 256;

    return dispatch(pipe, bufs, 3, groups, sizeof(unsigned int), &push_n);
}

int nxv_apply_unary(void* out, void* a,
                    unsigned int n, unsigned int op,
                    const char* spv_path) {
    if (!out || !a || !spv_path) return -1;
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), op, 2);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_out = (VkBuf*) out;

    /* Shader binding order: a, out. Push constant: n. */
    VkBuffer bufs[2] = { buf_a->buffer, buf_out->buffer };
    unsigned int push_n = n;
    unsigned int groups = (n + 255) / 256;

    return dispatch(pipe, bufs, 2, groups, sizeof(unsigned int), &push_n);
}

int nxv_reduce(float* out_scalar, void* in, unsigned int n, unsigned int op,
               const char* spv_path) {
    if (!out_scalar || !in || !spv_path) return -1;
    VkBuf* buf_in = (VkBuf*) in;
    *out_scalar = reduce(buf_in, (int) n, (ReduceOp) op, std::string(spv_path));
    return 0;
}

int nxv_matmul(void* out, void* a, void* b,
               unsigned int m, unsigned int n, unsigned int k,
               const char* spv_path) {
    if (!out || !a || !b || !spv_path) return -1;

    /* matmul has no spec constant; cache key is just the spv path. */
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), 0, 3);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_b   = (VkBuf*) b;
    VkBuf* buf_out = (VkBuf*) out;

    /* matmul uses 2D dispatch (gx, gy) but spirit's dispatch helper is
     * 1D-only. Inline the dispatch dance — same pattern as
     * test_matmul.cpp. Push constants: M, N, K (12 bytes). */
    auto& ctx = g_vk_ctx;

    VkBuffer bufs[3] = { buf_a->buffer, buf_b->buffer, buf_out->buffer };

    VkDescriptorBufferInfo bi[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        bi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        w[i] = {};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = pipe->descriptor_set;
        w[i].dstBinding = (uint32_t) i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = ctx.command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &ai, &cmd);

    VkCommandBufferBeginInfo bb{};
    bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bb);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->pipeline_layout, 0, 1, &pipe->descriptor_set, 0, nullptr);

    unsigned int push[3] = { m, n, k };
    vkCmdPushConstants(cmd, pipe->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    unsigned int gx = (n + 15) / 16;
    unsigned int gy = (m + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.compute_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.compute_queue);
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);

    return 0;
}

int nxv_random(void* out, unsigned int n, unsigned int seed, unsigned int dist,
               const char* spv_path) {
    if (!out || !spv_path) return -1;
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), dist, 1);
    if (!pipe) return -2;

    VkBuf* buf_out = (VkBuf*) out;

    /* Shader binding: out only. Push constants: {n, seed} = 8 bytes. */
    VkBuffer bufs[1] = { buf_out->buffer };
    struct { unsigned int n; unsigned int seed; } push = { n, seed };
    unsigned int groups = (n + 255) / 256;

    return dispatch(pipe, bufs, 1, groups, sizeof(push), &push);
}

int nxv_cast(void* out, void* a, unsigned int n, const char* spv_path) {
    if (!out || !a || !spv_path) return -1;
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), 0, 2);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_out = (VkBuf*) out;

    /* Shader binding order: a, out. Push: n. */
    VkBuffer bufs[2] = { buf_a->buffer, buf_out->buffer };
    unsigned int push_n = n;
    unsigned int groups = (n + 255) / 256;

    return dispatch(pipe, bufs, 2, groups, sizeof(unsigned int), &push_n);
}

int nxv_reduce_axis(void* out, void* a,
                    unsigned int outer, unsigned int reduce_size, unsigned int inner,
                    unsigned int op,
                    const char* spv_path) {
    if (!out || !a || !spv_path) return -1;
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), 0, 2);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_out = (VkBuf*) out;

    /* Shader binding order: a, out. Push: {outer, reduce_size, inner, op}. */
    VkBuffer bufs[2] = { buf_a->buffer, buf_out->buffer };
    unsigned int push[4] = { outer, reduce_size, inner, op };
    unsigned int n_slots = outer * inner;
    unsigned int groups = (n_slots + 255) / 256;

    return dispatch(pipe, bufs, 2, groups, sizeof(push), push);
}

int nxv_apply_binary_broadcast(void* out, void* a, void* b,
                                unsigned int op, unsigned int ndim,
                                const unsigned int* out_shape,
                                const unsigned int* a_strides,
                                const unsigned int* b_strides,
                                const char* spv_path) {
    if (!out || !a || !b || !out_shape || !a_strides || !b_strides || !spv_path)
        return -1;

    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), op, 3);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_b   = (VkBuf*) b;
    VkBuf* buf_out = (VkBuf*) out;

    /* Compute total output count from shape (multiply non-zero dims).
     * Shape entries beyond ndim are zero-padded. */
    unsigned int n = 1;
    for (unsigned int d = 0; d < ndim; d++) n *= out_shape[d];

    /* Push: {n, ndim, out_shape[4], a_strides[4], b_strides[4]} = 56 bytes. */
    struct {
        unsigned int n;
        unsigned int ndim;
        unsigned int out_shape[4];
        unsigned int a_strides[4];
        unsigned int b_strides[4];
    } push;
    push.n = n;
    push.ndim = ndim;
    for (int i = 0; i < 4; i++) {
        push.out_shape[i] = out_shape[i];
        push.a_strides[i] = a_strides[i];
        push.b_strides[i] = b_strides[i];
    }

    VkBuffer bufs[3] = { buf_a->buffer, buf_b->buffer, buf_out->buffer };
    unsigned int groups = (n + 255) / 256;

    return dispatch(pipe, bufs, 3, groups, sizeof(push), &push);
}

int nxv_fused_chain(void* out, void* a, void* b,
                    unsigned int n, unsigned int n_ops,
                    const unsigned int* ops,
                    const char* spv_path) {
    if (!out || !a || !b || !ops || !spv_path) return -1;
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), 0, 3);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_b   = (VkBuf*) b;
    VkBuf* buf_out = (VkBuf*) out;

    /* Shader bindings: a, b, out. Push: {n, n_ops, ops[8]} = 40 bytes. */
    VkBuffer bufs[3] = { buf_a->buffer, buf_b->buffer, buf_out->buffer };

    struct {
        unsigned int n;
        unsigned int n_ops;
        unsigned int ops[8];
    } push;
    push.n = n;
    push.n_ops = n_ops;
    for (int i = 0; i < 8; i++) push.ops[i] = ops[i];

    unsigned int groups = (n + 255) / 256;
    return dispatch(pipe, bufs, 3, groups, sizeof(push), &push);
}

int nxv_transpose(void* out, void* a, unsigned int m, unsigned int n,
                  const char* spv_path) {
    if (!out || !a || !spv_path) return -1;
    VkPipe* pipe = get_or_create_pipe(std::string(spv_path), 0, 2);
    if (!pipe) return -2;

    VkBuf* buf_a   = (VkBuf*) a;
    VkBuf* buf_out = (VkBuf*) out;

    /* 2D dispatch — 16×16 tiles; same dance as matmul. The existing
     * spirit dispatch() helper is 1D-only, so inline. */
    auto& ctx = g_vk_ctx;

    VkBuffer bufs[2] = { buf_a->buffer, buf_out->buffer };

    VkDescriptorBufferInfo bi[2];
    VkWriteDescriptorSet w[2];
    for (int i = 0; i < 2; i++) {
        bi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        w[i] = {};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = pipe->descriptor_set;
        w[i].dstBinding = (uint32_t) i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = ctx.command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &ai, &cmd);

    VkCommandBufferBeginInfo bb{};
    bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bb);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->pipeline_layout, 0, 1, &pipe->descriptor_set, 0, nullptr);

    unsigned int push[2] = { m, n };
    vkCmdPushConstants(cmd, pipe->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), push);

    unsigned int gx = (n + 15) / 16;
    unsigned int gy = (m + 15) / 16;
    vkCmdDispatch(cmd, gx, gy, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.compute_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.compute_queue);
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);

    return 0;
}

}
