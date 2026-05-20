# TinyLLM build.
#   make        / make cpu  -> Phase 1 CPU binaries (g++)
#   make gpu                -> Phase 2 CUDA binary  (nvcc, needs an NVIDIA GPU)

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Isrc -Wall
SRC      := src/main.cpp src/model.cpp src/sampler.cpp src/infer_cpu.cpp

# CUDA. -arch=sm_75 targets the T4; change it for other GPUs.
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -Isrc -Ikernels -DUSE_CUDA -arch=sm_75
CU_SRC    := infer.cu kernels/rmsnorm.cu kernels/rope.cu kernels/swiglu.cu \
             kernels/matmul.cu kernels/attention.cu
GPU_CPP   := src/main.cpp src/model.cpp src/sampler.cpp

BUILD := build

.PHONY: all cpu gpu clean
all: cpu
cpu: $(BUILD)/tinyllm_naive $(BUILD)/tinyllm_omp
gpu: $(BUILD)/tinyllm_gpu

# Naive single-threaded baseline (OpenMP pragmas are ignored without -fopenmp).
$(BUILD)/tinyllm_naive: $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

# OpenMP build — same source, the matmul/attention pragmas now run in parallel.
$(BUILD)/tinyllm_omp: $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp $(SRC) -o $@

# GPU build — nvcc compiles the .cu kernels and the shared .cpp files.
$(BUILD)/tinyllm_gpu: $(CU_SRC) $(GPU_CPP) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(CU_SRC) $(GPU_CPP) -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
