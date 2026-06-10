#include "kernels.cuh"

// Scatter freshly-computed K/V rows from a decode batch into their per-slot,
// per-position cells of the global KV cache.
//
// k_tmp / v_tmp : [batch × kv_dim]  — one row per active sequence
// kcache/vcache : [num_slots × num_layers × cap × kv_dim]
//
// Row b lands at  cache + ((slots[b]*num_layers + layer)*cap + positions[b]) * kv_dim.
// One block per row; threads stride over kv_dim.
__global__ void kv_scatter_fp16_kernel(__half* kcache, __half* vcache,
                                       const __half* k_tmp, const __half* v_tmp,
                                       const int* positions, const int* slots,
                                       int batch, int kv_dim, int layer,
                                       int num_layers, int cap) {
    int b = blockIdx.x;
    if (b >= batch) return;

    size_t cell = (((size_t)slots[b] * num_layers + layer) * cap + positions[b])
                  * kv_dim;
    __half* kdst = kcache + cell;
    __half* vdst = vcache + cell;
    const __half* ksrc = k_tmp + (size_t)b * kv_dim;
    const __half* vsrc = v_tmp + (size_t)b * kv_dim;

    for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
        kdst[d] = ksrc[d];
        vdst[d] = vsrc[d];
    }
}

void kv_scatter_fp16_cuda(__half* kcache, __half* vcache,
                          const __half* k_tmp, const __half* v_tmp,
                          const int* positions, const int* slots,
                          int batch, int kv_dim, int layer, int num_layers,
                          int cap) {
    const int threads = 128;
    kv_scatter_fp16_kernel<<<batch, threads>>>(
        kcache, vcache, k_tmp, v_tmp, positions, slots,
        batch, kv_dim, layer, num_layers, cap);
}
