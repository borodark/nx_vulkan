/* nx_vulkan_shim.h — extern "C" interface bridging Rust to spirit's
 * C++ Vulkan backend.
 *
 * Spirit's Backend_par_vulkan.{hpp,cpp} use C++ namespaces, classes,
 * and STL types — none of which Rust's bindgen handles cleanly. This
 * header declares a flat C ABI that the Rust NIF binds against; the
 * implementation in nx_vulkan_shim.cpp is a thin C++ file that
 * delegates to Spirit's helpers.
 *
 * Naming convention: nxv_* (Nx.Vulkan).
 */

#ifndef NX_VULKAN_SHIM_H
#define NX_VULKAN_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle ---------------------------------------------------------- */

/* Initialize the global Vulkan context. Idempotent. Returns 0 on
 * success, non-zero if no Vulkan-capable device is found. */
int nxv_init(void);

/* Tear down the global Vulkan context. Idempotent. */
void nxv_destroy(void);

/* Introspection ------------------------------------------------------ */

/* Returns a pointer to the device name string, valid until the next
 * nxv_destroy. NULL if init hasn't run. */
const char* nxv_device_name(void);

/* Returns 1 if the selected device supports f64, 0 otherwise. */
int nxv_has_f64(void);

/* Tensor primitives (v0.0.2) ------------------------------------------ */
/* Stubs placed here so Rust can declare them; implementations land in
 * the next iteration once the resource type lifetime is in place. */

/* Allocate a device-local buffer of `n_bytes`. Returns an opaque
 * handle (cast from VkBuf*) or NULL on failure. */
void* nxv_buf_alloc(unsigned long n_bytes);

/* Free a buffer handle. */
void nxv_buf_free(void* handle);

/* Upload `n_bytes` of host data to the buffer. Returns 0 on success. */
int nxv_buf_upload(void* handle, const void* data, unsigned long n_bytes);

/* Download `n_bytes` from the buffer to host memory. Returns 0 on success. */
int nxv_buf_download(void* handle, void* data, unsigned long n_bytes);

/* Compute primitives (v0.0.3) ----------------------------------------- */

/* Elementwise binary op. `out`, `a`, `b` are buffers of `n` f32 elements.
 * `op` is the elementwise_binary.spv spec constant:
 *   0=add, 1=mul, 2=sub, 3=div, 4=pow, 5=max, 6=min.
 * Returns 0 on success.
 *
 * Pipeline is created on first use per (shader_path, op) and cached in
 * the shim — avoids the ~22 ms pipeline-create overhead documented in
 * spirit's RESULTS_RTX_3060_TI.md reductions section. */
int nxv_apply_binary(void* out, void* a, void* b,
                     unsigned int n, unsigned int op,
                     const char* spv_path);

#ifdef __cplusplus
}
#endif

#endif
