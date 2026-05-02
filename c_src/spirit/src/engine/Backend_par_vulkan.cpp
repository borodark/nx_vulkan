/* Backend_par_vulkan.cpp — Vulkan compute backend implementation.
 *
 * Implements the context lifecycle, buffer management, shader loading,
 * pipeline creation, and dispatch primitives defined in
 * Backend_par_vulkan.hpp.
 *
 * Build: compiled only when SPIRIT_USE_VULKAN=ON.
 * Link:  -lvulkan
 */

#ifdef SPIRIT_USE_VULKAN

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define VK_CHECK(f, msg) do { \
    VkResult _r = (f); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "spirit-vulkan: %s (VkResult=%d)\n", msg, _r); \
        return -1; \
    } \
} while(0)

namespace Engine {
namespace Backend {
namespace vulkan {

/* Global context instance */
VkContext g_vk_ctx;

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props)
{
    for (uint32_t i = 0; i < g_vk_ctx.mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (g_vk_ctx.mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    fprintf(stderr, "spirit-vulkan: failed to find suitable memory type\n");
    return 0;
}

static uint32_t find_compute_queue_family(VkPhysicalDevice device)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());

    for (uint32_t i = 0; i < count; i++) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    }
    return UINT32_MAX;
}

/* ----------------------------------------------------------------
 * Context lifecycle
 * ---------------------------------------------------------------- */

int vk_init(int device_id)
{
    auto& ctx = g_vk_ctx;

    /* Instance */
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "spirit-vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(2, 2, 0);
    app_info.pEngineName = "spirit";
    app_info.engineVersion = VK_MAKE_VERSION(2, 2, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &ctx.instance),
             "failed to create Vulkan instance");

    /* Physical device */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &dev_count, nullptr);
    if (dev_count == 0) {
        fprintf(stderr, "spirit-vulkan: no Vulkan-capable GPUs\n");
        return -1;
    }

    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(ctx.instance, &dev_count, devices.data());

    uint32_t sel = (device_id >= 0 && (uint32_t)device_id < dev_count)
                   ? (uint32_t)device_id : 0;
    ctx.physical_device = devices[sel];

    vkGetPhysicalDeviceProperties(ctx.physical_device, &ctx.device_props);
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &ctx.mem_props);

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(ctx.physical_device, &features);
    ctx.has_float64 = features.shaderFloat64;

    printf("spirit-vulkan: %s (f64=%s)\n",
           ctx.device_props.deviceName,
           ctx.has_float64 ? "yes" : "no");

    /* Compute queue */
    ctx.queue_family_index = find_compute_queue_family(ctx.physical_device);
    if (ctx.queue_family_index == UINT32_MAX) {
        fprintf(stderr, "spirit-vulkan: no compute queue family\n");
        return -1;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = ctx.queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures enabled{};
    if (ctx.has_float64) enabled.shaderFloat64 = VK_TRUE;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enabled;

    VK_CHECK(vkCreateDevice(ctx.physical_device, &device_info, nullptr, &ctx.device),
             "failed to create logical device");

    vkGetDeviceQueue(ctx.device, ctx.queue_family_index, 0, &ctx.compute_queue);

    /* Command pool */
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = ctx.queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(ctx.device, &pool_info, nullptr, &ctx.command_pool),
             "failed to create command pool");

    /* Reusable fence */
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(ctx.device, &fence_info, nullptr, &ctx.sync_fence),
             "failed to create sync fence");

    /* Reusable command buffers — one for dispatch, one for xfer */
    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = ctx.command_pool;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmd_ai, &ctx.dispatch_cmd),
             "failed to allocate dispatch command buffer");
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmd_ai, &ctx.xfer_cmd),
             "failed to allocate xfer command buffer");

    return 0;
}

void vk_destroy()
{
    auto& ctx = g_vk_ctx;

    for (auto& kv : ctx.shader_cache)
        vkDestroyShaderModule(ctx.device, kv.second, nullptr);
    ctx.shader_cache.clear();

    if (ctx.sync_fence) vkDestroyFence(ctx.device, ctx.sync_fence, nullptr);
    if (ctx.command_pool) vkDestroyCommandPool(ctx.device, ctx.command_pool, nullptr);
    if (ctx.device) vkDestroyDevice(ctx.device, nullptr);
    if (ctx.instance) vkDestroyInstance(ctx.instance, nullptr);
    ctx = {};
}

/* ----------------------------------------------------------------
 * Buffer management
 * ---------------------------------------------------------------- */

int buf_alloc(VkBuf* b, VkDeviceSize size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags mem_flags)
{
    auto& ctx = g_vk_ctx;
    b->size = size;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(ctx.device, &info, nullptr, &b->buffer), "buf_alloc: create");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device, b->buffer, &req);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(req.memoryTypeBits, mem_flags);

    VK_CHECK(vkAllocateMemory(ctx.device, &alloc, nullptr, &b->memory), "buf_alloc: alloc");
    VK_CHECK(vkBindBufferMemory(ctx.device, b->buffer, b->memory, 0), "buf_alloc: bind");

    return 0;
}

void buf_free(VkBuf* b)
{
    auto& ctx = g_vk_ctx;
    if (b->buffer) vkDestroyBuffer(ctx.device, b->buffer, nullptr);
    if (b->memory) vkFreeMemory(ctx.device, b->memory, nullptr);
    *b = {};
}

static int submit_and_wait(VkCommandBuffer cmd)
{
    auto& ctx = g_vk_ctx;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    vkResetFences(ctx.device, 1, &ctx.sync_fence);
    VK_CHECK(vkQueueSubmit(ctx.compute_queue, 1, &submit, ctx.sync_fence), "submit");
    VK_CHECK(vkWaitForFences(ctx.device, 1, &ctx.sync_fence, VK_TRUE, UINT64_MAX), "wait");
    return 0;
}

int upload(VkBuf* dst, const void* data, VkDeviceSize size)
{
    auto& ctx = g_vk_ctx;

    VkBuf staging{};
    buf_alloc(&staging, size,
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* mapped;
    vkMapMemory(ctx.device, staging.memory, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(ctx.device, staging.memory);

    VkCommandBuffer cmd = ctx.xfer_cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, staging.buffer, dst->buffer, 1, &region);

    vkEndCommandBuffer(cmd);
    submit_and_wait(cmd);

    buf_free(&staging);
    return 0;
}

int download(VkBuf* src, void* data, VkDeviceSize size)
{
    auto& ctx = g_vk_ctx;

    VkBuf staging{};
    buf_alloc(&staging, size,
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer cmd = ctx.xfer_cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src->buffer, staging.buffer, 1, &region);

    vkEndCommandBuffer(cmd);
    submit_and_wait(cmd);

    void* mapped;
    vkMapMemory(ctx.device, staging.memory, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(ctx.device, staging.memory);

    buf_free(&staging);
    return 0;
}

/* ----------------------------------------------------------------
 * Shader / pipeline
 * ---------------------------------------------------------------- */

VkShaderModule load_shader(const std::string& spv_path)
{
    auto& ctx = g_vk_ctx;

    auto it = ctx.shader_cache.find(spv_path);
    if (it != ctx.shader_cache.end())
        return it->second;

    FILE* f = fopen(spv_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "spirit-vulkan: cannot open %s\n", spv_path.c_str());
        return VK_NULL_HANDLE;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint32_t> code(sz / sizeof(uint32_t));
    fread(code.data(), 1, sz, f);
    fclose(f);

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = sz;
    info.pCode = code.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(ctx.device, &info, nullptr, &mod) != VK_SUCCESS) {
        fprintf(stderr, "spirit-vulkan: failed to create shader from %s\n", spv_path.c_str());
        return VK_NULL_HANDLE;
    }

    ctx.shader_cache[spv_path] = mod;
    return mod;
}

int create_pipeline(VkPipe* p, VkShaderModule shader,
                    uint32_t n_buffers, uint32_t push_constant_size,
                    int32_t spec_constant)
{
    auto& ctx = g_vk_ctx;
    *p = {};

    /* Descriptor layout */
    std::vector<VkDescriptorSetLayoutBinding> bindings(n_buffers);
    for (uint32_t i = 0; i < n_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = n_buffers;
    layout_ci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &layout_ci, nullptr,
             &p->descriptor_layout), "desc layout");

    /* Descriptor pool + set */
    VkDescriptorPoolSize pool_sz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_buffers};
    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_sz;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &pool_ci, nullptr,
             &p->descriptor_pool), "desc pool");

    VkDescriptorSetAllocateInfo alloc_ci{};
    alloc_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_ci.descriptorPool = p->descriptor_pool;
    alloc_ci.descriptorSetCount = 1;
    alloc_ci.pSetLayouts = &p->descriptor_layout;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &alloc_ci, &p->descriptor_set),
             "desc set alloc");

    /* Pipeline layout */
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = push_constant_size > 0 ? push_constant_size : 4;

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &p->descriptor_layout;
    pl_ci.pushConstantRangeCount = push_constant_size > 0 ? 1u : 0u;
    pl_ci.pPushConstantRanges = push_constant_size > 0 ? &push_range : nullptr;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &pl_ci, nullptr,
             &p->pipeline_layout), "pipeline layout");

    /* Specialization constant */
    VkSpecializationMapEntry spec_entry{0, 0, sizeof(int32_t)};
    VkSpecializationInfo spec_info{1, &spec_entry, sizeof(int32_t), &spec_constant};

    /* Compute pipeline */
    VkComputePipelineCreateInfo pipe_ci{};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_ci.stage.module = shader;
    pipe_ci.stage.pName = "main";
    pipe_ci.stage.pSpecializationInfo = &spec_info;
    pipe_ci.layout = p->pipeline_layout;

    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
             &pipe_ci, nullptr, &p->pipeline), "compute pipeline");

    return 0;
}

void destroy_pipeline(VkPipe* p)
{
    auto& ctx = g_vk_ctx;
    if (p->pipeline) vkDestroyPipeline(ctx.device, p->pipeline, nullptr);
    if (p->pipeline_layout) vkDestroyPipelineLayout(ctx.device, p->pipeline_layout, nullptr);
    if (p->descriptor_pool) vkDestroyDescriptorPool(ctx.device, p->descriptor_pool, nullptr);
    if (p->descriptor_layout)
        vkDestroyDescriptorSetLayout(ctx.device, p->descriptor_layout, nullptr);
    *p = {};
}

/* ----------------------------------------------------------------
 * Dispatch
 * ---------------------------------------------------------------- */

int dispatch(VkPipe* p, VkBuffer* buffers, uint32_t n_buffers,
             uint32_t group_count_x,
             uint32_t push_size, const void* push_data)
{
    auto& ctx = g_vk_ctx;

    /* Update descriptor set */
    std::vector<VkWriteDescriptorSet> writes(n_buffers);
    std::vector<VkDescriptorBufferInfo> buf_infos(n_buffers);

    for (uint32_t i = 0; i < n_buffers; i++) {
        buf_infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p->descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx.device, n_buffers, writes.data(), 0, nullptr);

    /* Record into reusable command buffer (reset, not alloc/free) */
    VkCommandBuffer cmd = ctx.dispatch_cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p->pipeline_layout, 0, 1, &p->descriptor_set, 0, nullptr);

    if (push_size > 0 && push_data)
        vkCmdPushConstants(cmd, p->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_size, push_data);

    vkCmdDispatch(cmd, group_count_x, 1, 1);
    vkEndCommandBuffer(cmd);

    submit_and_wait(cmd);

    return 0;
}

/* ----------------------------------------------------------------
 * High-level parallel primitives (stubs — wire to shaders next)
 * ---------------------------------------------------------------- */

void apply(int N, VkPipe* pipe, VkBuffer* buffers, uint32_t n_buffers)
{
    uint32_t groups = (N + 255) / 256;
    uint32_t push = (uint32_t)N;
    dispatch(pipe, buffers, n_buffers, groups, sizeof(uint32_t), &push);
}

scalar reduce(VkBuf* input, int N, ReduceOp op, const std::string& reduce_spv_path)
{
    if (N <= 0) return 0;

    VkShaderModule shader = load_shader(reduce_spv_path);
    if (!shader) return 0;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    /* Pass 1: N elements → num_groups partial results */
    uint32_t num_groups = (N + 255) / 256;

    VkBuf partial{};
    buf_alloc(&partial, num_groups * sizeof(scalar), usage, mem);

    VkPipe pipe1{};
    create_pipeline(&pipe1, shader, 2, sizeof(uint32_t), (int32_t)op);

    VkBuffer bufs1[2] = { input->buffer, partial.buffer };
    uint32_t n1 = (uint32_t)N;
    dispatch(&pipe1, bufs1, 2, num_groups, sizeof(uint32_t), &n1);
    destroy_pipeline(&pipe1);

    /* Iterate until we have a single value */
    while (num_groups > 1) {
        uint32_t next_groups = (num_groups + 255) / 256;

        VkBuf partial2{};
        buf_alloc(&partial2, next_groups * sizeof(scalar), usage, mem);

        VkPipe pipe_n{};
        create_pipeline(&pipe_n, shader, 2, sizeof(uint32_t), (int32_t)op);

        VkBuffer bufs_n[2] = { partial.buffer, partial2.buffer };
        uint32_t n_n = num_groups;
        dispatch(&pipe_n, bufs_n, 2, next_groups, sizeof(uint32_t), &n_n);
        destroy_pipeline(&pipe_n);

        buf_free(&partial);
        partial = partial2;
        num_groups = next_groups;
    }

    /* Read back single scalar */
    scalar result = 0;
    download(&partial, &result, sizeof(scalar));
    buf_free(&partial);

    return result;
}

scalar reduce_sum(VkBuf* input, int N, const std::string& reduce_spv_path)
{
    return reduce(input, N, REDUCE_SUM, reduce_spv_path);
}

void scale(VkBuf* buf, int N, scalar alpha)
{
    /* TODO: dispatch scale shader */
    (void)buf; (void)N; (void)alpha;
}

void add(VkBuf* out, VkBuf* a, VkBuf* b, int N)
{
    /* TODO: dispatch add shader (elementwise_binary, OP=0) */
    (void)out; (void)a; (void)b; (void)N;
}

scalar dot(VkBuf* a, VkBuf* b, int N)
{
    /* TODO: dispatch dot shader (map multiply + reduce sum) */
    (void)a; (void)b; (void)N;
    return 0;
}

} // namespace vulkan
} // namespace Backend
} // namespace Engine

#endif /* SPIRIT_USE_VULKAN */
