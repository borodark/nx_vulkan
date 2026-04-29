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
#include <string>

using namespace Engine::Backend::vulkan;

/* The selected device name, cached after nxv_init so we can hand out
 * a stable pointer to Rust. */
static std::string g_device_name;

extern "C" {

int nxv_init(void) {
    int rc = vk_init(0);
    if (rc != 0) return rc;
    g_device_name = g_vk_ctx.device_props.deviceName;
    return 0;
}

void nxv_destroy(void) {
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

}
