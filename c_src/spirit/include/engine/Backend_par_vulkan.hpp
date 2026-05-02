#pragma once
#ifndef SPIRIT_CORE_ENGINE_BACKEND_PAR_VULKAN_HPP
#define SPIRIT_CORE_ENGINE_BACKEND_PAR_VULKAN_HPP

/* Backend_par_vulkan.hpp — Vulkan compute backend for Spirit.
 *
 * Third backend alongside CUDA (Backend_par.hpp) and sequential
 * (Backend_seq.hpp). Uses Vulkan compute shaders + VkFFT for
 * GPU-accelerated spin simulations on any Vulkan-capable GPU
 * (NVIDIA, AMD, Intel).
 *
 * Pattern follows Backend_par.hpp: apply(N, f), reduce(N, ...),
 * set(vf1, vf2, f). Instead of CUDA kernels or OpenMP pragmas,
 * dispatches pre-compiled SPIR-V compute shaders via vkCmdDispatch.
 *
 * Requires: Vulkan 1.1+, VkFFT (header-only, in thirdparty/).
 */

#include <vulkan/vulkan.h>
#include <cstdint>

/* Use Spirit's scalar type if available, otherwise default to float. */
#ifndef SPIRIT_SCALAR_TYPE
#define SPIRIT_SCALAR_TYPE float
#endif
using scalar = SPIRIT_SCALAR_TYPE;
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace Engine
{
namespace Backend
{
namespace vulkan
{

/* ----------------------------------------------------------------
 * Vulkan compute context — one per Spirit simulation instance.
 * Initialized once via vk_init(), destroyed via vk_destroy().
 * ---------------------------------------------------------------- */

struct VkContext
{
    VkInstance instance             = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device                = VK_NULL_HANDLE;
    VkQueue compute_queue          = VK_NULL_HANDLE;
    uint32_t queue_family_index    = 0;
    VkCommandPool command_pool     = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties device_props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
    bool has_float64 = false;

    /* Shader cache: spv_path → VkShaderModule */
    std::unordered_map<std::string, VkShaderModule> shader_cache;

    /* Reusable fence for synchronous dispatch */
    VkFence sync_fence = VK_NULL_HANDLE;

    /* Reusable command buffers — avoids per-dispatch alloc/free.
     * dispatch_cmd: compute dispatches (bind, push, dispatch).
     * xfer_cmd: upload/download copy commands. */
    VkCommandBuffer dispatch_cmd = VK_NULL_HANDLE;
    VkCommandBuffer xfer_cmd     = VK_NULL_HANDLE;
};

/* Global context — matches Spirit's pattern of global state in
 * Backend::par (CUDA uses global streams/handles). */
extern VkContext g_vk_ctx;

/* ----------------------------------------------------------------
 * GPU buffer — wraps VkBuffer + VkDeviceMemory.
 * Used as the backing store for field<T> on Vulkan.
 *
 * USAGE PATTERN: persistent device-resident buffers
 * --------------------------------------------------
 * Allocate once, dispatch many, download once. Per-op alloc + xfer
 * is ~50 ms for a 1M-element f32 buffer on an RTX 3060 Ti; the
 * dispatch itself is ~70 us. The 99.9% gap is the alloc + xfer
 * overhead — eliminated by holding the VkBuf across operations.
 *
 * Anti-pattern (don't do this in a hot loop):
 *
 *     for (int i = 0; i < iters; i++) {
 *         buf_alloc(&a); buf_alloc(&b); buf_alloc(&c);  // ~10 ms
 *         upload(&a, ha, sz); upload(&b, hb, sz);        // ~20 ms
 *         dispatch(...);                                 // ~0.07 ms
 *         download(&c, hc, sz);                          // ~10 ms
 *         buf_free(&a); buf_free(&b); buf_free(&c);     // ~10 ms
 *     }
 *
 * Persistent pattern (the right shape):
 *
 *     // once: alloc + initial upload
 *     VkBuf a, b, c;
 *     buf_alloc(&a, sz, ...); buf_alloc(&b, sz, ...); buf_alloc(&c, sz, ...);
 *     upload(&a, ha, sz);
 *     upload(&b, hb, sz);
 *
 *     // many: dispatch only
 *     for (int i = 0; i < iters; i++)
 *         dispatch(pipe, bufs, 3, groups, sizeof(uint32_t), &n);
 *
 *     // once: download + free
 *     download(&c, hc, sz);
 *     buf_free(&a); buf_free(&b); buf_free(&c);
 *
 * For Spirit's simulation loop and Nx-style tensor lifecycles, the
 * persistent pattern is the only viable shape; benchmarks show
 * ~700x speedup over the anti-pattern at 1M elements.
 * See PERSISTENT_BUFFERS_PLAN.md for the optimization roadmap.
 * ---------------------------------------------------------------- */

struct VkBuf
{
    VkBuffer buffer       = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size     = 0;
};

/* ----------------------------------------------------------------
 * Pipeline — shader + descriptor set + pipeline layout.
 * Cached per (shader_path, specialization_constant) pair.
 * ---------------------------------------------------------------- */

struct VkPipe
{
    VkDescriptorPool descriptor_pool         = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout  = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set           = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout         = VK_NULL_HANDLE;
    VkPipeline pipeline                      = VK_NULL_HANDLE;
};

/* ----------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------- */

/* Initialize the global Vulkan context. Call once at startup.
 * device_id: index into vkEnumeratePhysicalDevices result. */
int  vk_init(int device_id = 0);
void vk_destroy();

/* ----------------------------------------------------------------
 * Memory helpers
 * ---------------------------------------------------------------- */

uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props);

int  buf_alloc(VkBuf* b, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags mem_flags);
void buf_free(VkBuf* b);

/* Host ↔ Device transfer via staging buffer. */
int  upload(VkBuf* dst, const void* data, VkDeviceSize size);
int  download(VkBuf* src, void* data, VkDeviceSize size);

/* ----------------------------------------------------------------
 * Shader / pipeline management
 * ---------------------------------------------------------------- */

VkShaderModule load_shader(const std::string& spv_path);

int  create_pipeline(VkPipe* p, VkShaderModule shader,
                     uint32_t n_buffers, uint32_t push_constant_size,
                     int32_t spec_constant = 0);
void destroy_pipeline(VkPipe* p);

/* ----------------------------------------------------------------
 * Dispatch — record + submit + wait
 * ---------------------------------------------------------------- */

int dispatch(VkPipe* p, VkBuffer* buffers, uint32_t n_buffers,
             uint32_t group_count_x,
             uint32_t push_size = 0, const void* push_data = nullptr);

/* ----------------------------------------------------------------
 * Parallel primitives — matching Backend::par interface
 * ---------------------------------------------------------------- */

/* Apply a compute shader to N elements.
 * shader_id: index into the registered shader table.
 * The shader reads/writes through storage buffer bindings. */
void apply(int N, VkPipe* pipe, VkBuffer* buffers, uint32_t n_buffers);

/* Reduction ops — matches specialization constants in reduce.comp */
enum ReduceOp { REDUCE_SUM = 0, REDUCE_MIN = 1, REDUCE_MAX = 2 };

/* GPU reduction via two-pass tree reduce shader.
 * Returns the scalar result on the host.
 * reduce_spv_path: path to compiled reduce.spv shader. */
scalar reduce(VkBuf* input, int N, ReduceOp op, const std::string& reduce_spv_path);
scalar reduce_sum(VkBuf* input, int N, const std::string& reduce_spv_path);

/* Scale all elements: buf[i] *= alpha */
void scale(VkBuf* buf, int N, scalar alpha);

/* Element-wise add: out[i] = a[i] + b[i] */
void add(VkBuf* out, VkBuf* a, VkBuf* b, int N);

/* Dot product: sum(a[i] * b[i]) */
scalar dot(VkBuf* a, VkBuf* b, int N);

} // namespace vulkan
} // namespace Backend
} // namespace Engine

#endif /* SPIRIT_CORE_ENGINE_BACKEND_PAR_VULKAN_HPP */
