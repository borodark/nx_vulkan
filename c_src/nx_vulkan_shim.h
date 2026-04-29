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

/* Elementwise unary op. `out`, `a` are buffers of `n` f32 elements.
 * `op` spec constant: 0=exp, 1=log, 2=sqrt, 3=abs, 4=neg, 5=sigmoid,
 * 6=tanh, 7=relu, 8=ceil, 9=floor, 10=sign, 11=reciprocal, 12=square. */
int nxv_apply_unary(void* out, void* a,
                    unsigned int n, unsigned int op,
                    const char* spv_path);

/* Reduction. `n` f32 elements reduced to one f32 written to `out_scalar`.
 * `op`: 0=sum, 1=min, 2=max. */
int nxv_reduce(float* out_scalar, void* in,
               unsigned int n, unsigned int op,
               const char* spv_path);

/* Matmul (naive). C[M*N] = A[M*K] · B[K*N], all row-major f32. */
int nxv_matmul(void* out, void* a, void* b,
               unsigned int m, unsigned int n, unsigned int k,
               const char* spv_path);

/* Random. Fill `out` with `n` f32 values. dist=0 uniform [0,1),
 * dist=1 normal N(0,1) via Box-Muller. */
int nxv_random(void* out, unsigned int n, unsigned int seed, unsigned int dist,
               const char* spv_path);

/* 2D transpose. Input A is M×N row-major; output C is N×M row-major.
 * C[j, i] = A[i, j] for i in 0..M, j in 0..N. */
int nxv_transpose(void* out, void* a,
                  unsigned int m, unsigned int n,
                  const char* spv_path);

/* Cast f32↔f64. Two-file split: spv_path picks the direction.
 * `n` is the element count. In/out element widths differ; caller is
 * responsible for sizing buffers correctly (n*4 vs n*8). */
int nxv_cast(void* out, void* a,
             unsigned int n,
             const char* spv_path);

/* Per-axis reduction. Input is a virtual 3-D tensor (outer, reduce, inner)
 * row-major; output is (outer, inner) row-major. `op`: 0=sum, 1=max, 2=min. */
int nxv_reduce_axis(void* out, void* a,
                    unsigned int outer, unsigned int reduce_size, unsigned int inner,
                    unsigned int op,
                    const char* spv_path);

/* Fused n-way elementwise chain — up to 8 ops in one dispatch.
 * `ops` is an array of length 8 (pad with 255 = nop). Op codes:
 *   binary 0..6 (add/mul/sub/div/pow/max/min) — second operand is buf B
 *   unary  100..114 (exp/log/sqrt/abs/neg/sigmoid/tanh/relu/ceil/floor/
 *                    sign/reciprocal/square/erf/expm1)
 * The chain applies left-to-right starting from a[i], using b[i] for
 * binary steps. Output is c[i] of length n. */
int nxv_fused_chain(void* out, void* a, void* b,
                    unsigned int n, unsigned int n_ops,
                    const unsigned int* ops,
                    const char* spv_path);

#ifdef __cplusplus
}
#endif

#endif
