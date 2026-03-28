# obrain-chat — Cross-platform build
# Produces self-contained binaries with llama.cpp statically linked.
#
# Usage:
#   make              # Build for current platform (macOS arm64)
#   make linux        # Cross-compile for Linux x86_64 (AVX2, AMD Zen/Intel Haswell+)
#   make all          # Build macOS + Linux
#   make dist         # Build all + copy to dist/
#   make clean        # Clean Rust build artifacts
#   make clean-all    # Clean Rust + llama.cpp build artifacts
#
# Environment:
#   LLAMA_CPP_DIR     Path to llama.cpp fork (default: ~/projects/ia/llama.cpp)
#   RELEASE=0         Build in debug mode (default: release)
#
# CPU compatibility:
#   macOS arm64     — Apple Silicon (M1+), NEON + Metal GPU
#   Linux x86_64    — AMD Zen / Intel Haswell+ (AVX2+FMA, ~2013+)
#                     Covers: Ryzen, EPYC, Core i 4th gen+, Xeon v3+

SHELL := /bin/bash

# --- Configuration ---
LLAMA_CPP_DIR ?= /Users/triviere/projects/ia/llama.cpp
RELEASE       ?= 1
JOBS          ?= $(shell sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Use rustup-managed toolchain, not Homebrew
RUSTUP_TOOLCHAIN := stable-aarch64-apple-darwin
export PATH := $(HOME)/.rustup/toolchains/$(RUSTUP_TOOLCHAIN)/bin:$(PATH)

CARGO_FLAGS :=
ifeq ($(RELEASE),1)
  CARGO_FLAGS += --release
  PROFILE := release
else
  PROFILE := debug
endif

OUT_DIR     := target
OUT_MACOS   := $(OUT_DIR)/$(PROFILE)/obrain-chat
OUT_LINUX   := $(OUT_DIR)/x86_64-unknown-linux-gnu/$(PROFILE)/obrain-chat
DIST_DIR    := dist

# --- llama.cpp build dirs ---
LLAMA_STATIC_MACOS := $(LLAMA_CPP_DIR)/build-static
LLAMA_STATIC_LINUX := $(LLAMA_CPP_DIR)/build-linux-x64-static

# --- Zig cross-compile wrappers ---
ZIG_DIR := $(abspath .zig)
ZIG_CC  := $(ZIG_DIR)/cc-linux.sh
ZIG_CXX := $(ZIG_DIR)/cxx-linux.sh
ZIG_AR  := $(ZIG_DIR)/ar-linux.sh

# Common cmake flags for Linux cross-compile
LINUX_CMAKE_FLAGS := \
	-DCMAKE_SYSTEM_NAME=Linux \
	-DCMAKE_SYSTEM_PROCESSOR=x86_64 \
	-DCMAKE_C_COMPILER=$(ZIG_CC) \
	-DCMAKE_CXX_COMPILER=$(ZIG_CXX) \
	-DCMAKE_AR=$(ZIG_AR) \
	-DCMAKE_RANLIB=true \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=OFF \
	-DGGML_BLAS=OFF \
	-DGGML_METAL=OFF \
	-DGGML_CUDA=OFF \
	-DGGML_OPENMP=OFF \
	-DGGML_AVX=ON \
	-DGGML_AVX2=ON \
	-DGGML_FMA=ON \
	-DGGML_F16C=ON \
	-DLLAMA_BUILD_TESTS=OFF \
	-DLLAMA_BUILD_EXAMPLES=OFF \
	-DLLAMA_BUILD_SERVER=OFF \
	-DLLAMA_CURL=OFF

LLAMA_TARGETS_LINUX := llama ggml ggml-base ggml-cpu
LLAMA_TARGETS_MACOS := llama ggml ggml-base ggml-cpu ggml-metal ggml-blas

# =============================================
#  Targets
# =============================================

.PHONY: all macos linux clean clean-all dist llama-static-macos llama-static-linux zig-wrappers info

all: macos linux

macos: llama-static-macos
	@echo "=== Building obrain-chat (macOS arm64, static) ==="
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) LLAMA_STATIC=1 \
		cargo build $(CARGO_FLAGS)
	@echo "✓ $(OUT_MACOS) ($$(du -h $(OUT_MACOS) | cut -f1))"

linux: llama-static-linux zig-wrappers
	@echo "=== Building obrain-chat (Linux x86_64 AVX2, static) ==="
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) LLAMA_STATIC=1 \
		cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-gnu
	@echo "✓ $(OUT_LINUX) ($$(du -h $(OUT_LINUX) | cut -f1))"

# --- Distribution ---
dist: all
	@mkdir -p $(DIST_DIR)
	cp $(OUT_MACOS) $(DIST_DIR)/obrain-chat-macos-arm64
	cp $(OUT_LINUX) $(DIST_DIR)/obrain-chat-linux-x64
	@echo ""
	@echo "=== Distribution ==="
	@ls -lh $(DIST_DIR)/
	@echo ""
	@file $(DIST_DIR)/*
	@echo ""
	@echo "CPU support:"
	@echo "  macos-arm64 : Apple Silicon M1+ (NEON + Metal GPU)"
	@echo "  linux-x64   : AMD64/Intel64 with AVX2+FMA (Zen/Haswell+, ~2013+)"

info:
	@echo "LLAMA_CPP_DIR = $(LLAMA_CPP_DIR)"
	@echo "PROFILE       = $(PROFILE)"
	@echo "JOBS          = $(JOBS)"
	@echo ""
	@echo "Targets:"
	@echo "  macOS arm64  → $(OUT_MACOS)"
	@echo "  Linux x86_64 → $(OUT_LINUX)"

# --- llama.cpp static builds ---

llama-static-macos: $(LLAMA_STATIC_MACOS)/src/libllama.a

$(LLAMA_STATIC_MACOS)/src/libllama.a:
	@echo "=== Building llama.cpp (macOS static, Metal+Accelerate) ==="
	cd $(LLAMA_CPP_DIR) && cmake -B build-static \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_BLAS=ON \
		-DGGML_METAL=ON \
		-DGGML_METAL_EMBED_LIBRARY=ON \
		-DGGML_CUDA=OFF \
		-DGGML_OPENMP=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_CURL=OFF
	cd $(LLAMA_CPP_DIR) && cmake --build build-static --config Release \
		-j$(JOBS) --target $(LLAMA_TARGETS_MACOS)

llama-static-linux: $(LLAMA_STATIC_LINUX)/src/libllama.a

$(LLAMA_STATIC_LINUX)/src/libllama.a: zig-wrappers
	@echo "=== Building llama.cpp (Linux x86_64 static, AVX2+FMA, via zig) ==="
	cd $(LLAMA_CPP_DIR) && cmake -B build-linux-x64-static $(LINUX_CMAKE_FLAGS)
	cd $(LLAMA_CPP_DIR) && cmake --build build-linux-x64-static --config Release \
		-j$(JOBS) --target $(LLAMA_TARGETS_LINUX)

# --- Zig cross-compile wrappers ---

zig-wrappers: $(ZIG_CC)

$(ZIG_CC):
	@mkdir -p $(ZIG_DIR)
	@printf '#!/bin/sh\nexec zig cc -target x86_64-linux-gnu "$$@"\n'  > $(ZIG_DIR)/cc-linux.sh  && chmod +x $(ZIG_DIR)/cc-linux.sh
	@printf '#!/bin/sh\nexec zig c++ -target x86_64-linux-gnu "$$@"\n' > $(ZIG_DIR)/cxx-linux.sh && chmod +x $(ZIG_DIR)/cxx-linux.sh
	@printf '#!/bin/sh\nexec zig ar "$$@"\n'                           > $(ZIG_DIR)/ar-linux.sh  && chmod +x $(ZIG_DIR)/ar-linux.sh

# --- Cleanup ---

clean:
	cargo clean

clean-all: clean
	rm -rf $(LLAMA_STATIC_MACOS) $(LLAMA_STATIC_LINUX) $(ZIG_DIR) $(DIST_DIR)
