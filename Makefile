# TinyLLM build.
#   make        / make cpu  -> Phase 1 CPU binaries (g++)
#   make gpu                -> Phase 2 CUDA FP32 binary (nvcc, needs an NVIDIA GPU)
#   make gpu_fp16           -> Phase 3 CUDA FP16 binary

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Isrc -Wall
SRC      := src/main.cpp src/model.cpp src/sampler.cpp src/infer_cpu.cpp

# CUDA. -arch=sm_75 targets the T4; change it for other GPUs.
NVCC           ?= nvcc
NVCCFLAGS_BASE := -O3 -std=c++17 -Isrc -Ikernels -arch=sm_75
NVCCFLAGS      := $(NVCCFLAGS_BASE) -DUSE_CUDA
NVCCFLAGS_FP16 := $(NVCCFLAGS_BASE) -DUSE_CUDA_FP16

CU_SRC         := src/infer_gpu_fp32.cu \
                  kernels/fp32/rmsnorm.cu  kernels/fp32/rope.cu \
                  kernels/fp32/swiglu.cu   kernels/fp32/matmul.cu \
                  kernels/fp32/attention.cu
CU_SRC_FP16    := src/infer_gpu_fp16.cu \
                  kernels/fp16/rmsnorm.cu  kernels/fp16/rope.cu \
                  kernels/fp16/swiglu.cu   kernels/fp16/matmul.cu \
                  kernels/fp16/attention.cu
GPU_CPP        := src/main.cpp src/model.cpp src/sampler.cpp

BUILD := build

.PHONY: all cpu gpu gpu_fp16 clean
all: cpu
cpu: $(BUILD)/tinyllm_naive $(BUILD)/tinyllm_omp
gpu: $(BUILD)/tinyllm_gpu
gpu_fp16: $(BUILD)/tinyllm_gpu_fp16

# Naive single-threaded baseline (OpenMP pragmas are ignored without -fopenmp).
$(BUILD)/tinyllm_naive: $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

# OpenMP build — same source, the matmul/attention pragmas now run in parallel.
$(BUILD)/tinyllm_omp: $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp $(SRC) -o $@

# GPU FP32 build.
$(BUILD)/tinyllm_gpu: $(CU_SRC) $(GPU_CPP) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CU_SRC) $(GPU_CPP) -o $@

# GPU FP16 build (Phase 3).
$(BUILD)/tinyllm_gpu_fp16: $(CU_SRC_FP16) $(GPU_CPP) | $(BUILD)
	$(NVCC) $(NVCCFLAGS_FP16) $(CU_SRC_FP16) $(GPU_CPP) -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
