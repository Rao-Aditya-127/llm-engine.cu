#pragma once
// Shared by every precision's kernels header (fp32, fp16, …). Only generic
// CUDA plumbing lives here — no precision-specific declarations.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Abort on any CUDA error, printing where it happened.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(err__));                \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)
