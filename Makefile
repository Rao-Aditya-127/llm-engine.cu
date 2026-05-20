# TinyLLM build.
# Phase 1 is CPU-only (g++). Phases 2+ add CUDA targets here.

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Isrc -Wall
SRC      := src/main.cpp src/model.cpp src/sampler.cpp src/infer_cpu.cpp

BUILD := build

.PHONY: all cpu clean
all: cpu
cpu: $(BUILD)/tinyllm_naive $(BUILD)/tinyllm_omp

# Naive single-threaded baseline (OpenMP pragmas are ignored without -fopenmp).
$(BUILD)/tinyllm_naive: $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

# OpenMP build — same source, the matmul/attention pragmas now run in parallel.
$(BUILD)/tinyllm_omp: $(SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp $(SRC) -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
