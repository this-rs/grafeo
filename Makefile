# obrain-chat — Cross-platform build
#
# Two-phase pipeline:
#   Phase 1: make deps-all     → Build llama.cpp static libs (.a) for all backends
#   Phase 2: make dist-all     → Build standalone binaries for all platforms
#
# Quick targets:
#   make deps           Build llama.cpp for Apple Silicon only (native, no Docker)
#   make macos          Build Rust binary (macOS arm64)
#   make linux          Cross-compile Rust binary (Linux x86_64 AVX2, via zig)
#   make dist           Build macOS + Linux AVX2 binaries
#
# Full pipeline:
#   make deps-all       Build llama.cpp for ALL backends (native + Docker)
#   make dist-all       Build ALL standalone binaries
#
# Individual deps (Docker):
#   make deps-docker-cpu-avx2    Linux x86_64 AVX2+FMA (Zen/Haswell+)
#   make deps-docker-cpu-compat  Linux x86_64 SSE3 only (any x86_64)
#   make deps-docker-cuda        Linux x86_64 + NVIDIA CUDA 12.8
#   make deps-docker-vulkan      Linux x86_64 + Vulkan (NVIDIA/AMD/Intel)
#   make deps-docker-sycl        Linux x86_64 + Intel SYCL/oneAPI
#
# Individual binaries:
#   make linux-compat       Linux x86_64 SSE3 (cross-compile via zig)
#   make linux-vulkan       Linux x86_64 + Vulkan GPU (full Docker build)
#   make linux-cuda         Linux x86_64 + NVIDIA CUDA (full Docker build)
#   make linux-sycl         Linux x86_64 + Intel SYCL (full Docker build)
#
# GPU binaries require full Docker builds because their runtime libs
# (libvulkan, CUDA, oneAPI) are only available inside the container.
# CPU binaries can be cross-compiled from macOS via zig.
#
# Environment:
#   LLAMA_CPP_DIR     Path to llama.cpp fork (default: ~/projects/ia/llama.cpp)
#   GRAFEO_DIR        Path to grafeo workspace (default: ~/projects/lab/grafeo)
#   RELEASE=0         Build in debug mode (default: release)
#
# Architecture:
#   deps/              Prebuilt llama.cpp static libs (.a), by variant
#     macos-arm64/     Metal + Accelerate (native build)
#     linux-cpu-avx2/  AVX2+FMA via Docker
#     linux-cpu-compat/ SSE3 only via Docker
#     linux-cuda/      CUDA 12.8 via Docker
#     linux-vulkan/    Vulkan via Docker
#     linux-sycl/      Intel SYCL/oneAPI via Docker
#   dist/              Standalone binaries
#     obrain-chat-macos-arm64          Apple Silicon (Metal GPU)
#     obrain-chat-linux-x64            Linux AVX2 CPU
#     obrain-chat-linux-x64-compat     Linux SSE3 CPU (any x86_64)
#     obrain-chat-linux-x64-vulkan     Linux + Vulkan GPU (AMD/NVIDIA/Intel)
#     obrain-chat-linux-x64-cuda       Linux + NVIDIA CUDA GPU
#     obrain-chat-linux-x64-sycl       Linux + Intel SYCL GPU

SHELL := /bin/bash

# --- Configuration ---
LLAMA_CPP_DIR ?= /Users/triviere/projects/ia/llama.cpp
GRAFEO_DIR    ?= /Users/triviere/projects/lab/grafeo
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
DEPS_DIR    := deps
DIST_DIR    := dist

# --- llama.cpp build dirs ---
LLAMA_DEPS_MACOS := $(DEPS_DIR)/macos-arm64
LLAMA_DEPS_LINUX_AVX2   := $(DEPS_DIR)/linux-cpu-avx2
LLAMA_DEPS_LINUX_COMPAT := $(DEPS_DIR)/linux-cpu-compat
LLAMA_DEPS_LINUX_CUDA   := $(DEPS_DIR)/linux-cuda
LLAMA_DEPS_LINUX_VULKAN := $(DEPS_DIR)/linux-vulkan
LLAMA_DEPS_LINUX_SYCL   := $(DEPS_DIR)/linux-sycl

# Legacy build dirs (for backward compat with build.rs)
LLAMA_STATIC_MACOS := $(LLAMA_CPP_DIR)/build-static

LLAMA_TARGETS_MACOS := llama ggml ggml-base ggml-cpu ggml-metal ggml-blas

# --- Zig cross-compile wrappers (used for Rust linking, not llama.cpp) ---
ZIG_DIR := $(abspath .zig)
ZIG_CC  := $(ZIG_DIR)/cc-linux.sh
ZIG_CXX := $(ZIG_DIR)/cxx-linux.sh
ZIG_AR  := $(ZIG_DIR)/ar-linux.sh

# All Linux Docker targets force --platform linux/amd64
# (Docker Desktop on Apple Silicon defaults to ARM)
DOCKER_PLATFORM := --platform linux/amd64

# =============================================
#  Phase 1: Dependencies (llama.cpp static libs)
# =============================================

.PHONY: deps deps-all \
        deps-docker-cpu-avx2 deps-docker-cpu-compat deps-docker-cuda deps-docker-vulkan deps-docker-sycl

# --- deps: Apple Silicon only (native, no Docker) ---
deps: $(LLAMA_DEPS_MACOS)/libllama.a
	@echo ""
	@echo "=== deps complete (Apple Silicon) ==="
	@ls -lh $(LLAMA_DEPS_MACOS)/*.a

$(LLAMA_DEPS_MACOS)/libllama.a:
	@echo "=== Building llama.cpp (macOS arm64, Metal+Accelerate) ==="
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
	@mkdir -p $(LLAMA_DEPS_MACOS)
	@find $(LLAMA_STATIC_MACOS) -name '*.a' -exec cp {} $(LLAMA_DEPS_MACOS)/ \;
	@echo "✓ $(LLAMA_DEPS_MACOS)/"

# --- deps-all: ALL backends ---
deps-all: deps deps-docker-cpu-avx2 deps-docker-cpu-compat deps-docker-cuda deps-docker-vulkan deps-docker-sycl
	@echo ""
	@echo "========================================"
	@echo "  Phase 1 complete — all deps built"
	@echo "========================================"
	@echo ""
	@for d in $(DEPS_DIR)/*/; do \
		variant=$$(basename $$d); \
		count=$$(ls $$d/*.a 2>/dev/null | wc -l | tr -d ' '); \
		echo "  ✓ $$variant: $$count libs"; \
	done
	@echo ""
	@echo "Next: make dist-all"

# --- Individual Docker deps targets ---

deps-docker-cpu-avx2: $(LLAMA_DEPS_LINUX_AVX2)/libllama.a
$(LLAMA_DEPS_LINUX_AVX2)/libllama.a:
	@echo "=== [Docker] Building llama.cpp (cpu-avx2) ==="
	@mkdir -p $(LLAMA_DEPS_LINUX_AVX2)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.deps \
		--build-arg VARIANT=cpu-avx2 \
		--output "type=local,dest=$(LLAMA_DEPS_LINUX_AVX2)" \
		$(LLAMA_CPP_DIR)
	@echo "✓ $(LLAMA_DEPS_LINUX_AVX2)/"
	@ls -lh $(LLAMA_DEPS_LINUX_AVX2)/*.a

deps-docker-cpu-compat: $(LLAMA_DEPS_LINUX_COMPAT)/libllama.a
$(LLAMA_DEPS_LINUX_COMPAT)/libllama.a:
	@echo "=== [Docker] Building llama.cpp (cpu-compat) ==="
	@mkdir -p $(LLAMA_DEPS_LINUX_COMPAT)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.deps \
		--build-arg VARIANT=cpu-compat \
		--output "type=local,dest=$(LLAMA_DEPS_LINUX_COMPAT)" \
		$(LLAMA_CPP_DIR)
	@echo "✓ $(LLAMA_DEPS_LINUX_COMPAT)/"
	@ls -lh $(LLAMA_DEPS_LINUX_COMPAT)/*.a

deps-docker-cuda: $(LLAMA_DEPS_LINUX_CUDA)/libllama.a
$(LLAMA_DEPS_LINUX_CUDA)/libllama.a:
	@echo "=== [Docker] Building llama.cpp (cuda) ==="
	@mkdir -p $(LLAMA_DEPS_LINUX_CUDA)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.deps \
		--build-arg VARIANT=cuda \
		--output "type=local,dest=$(LLAMA_DEPS_LINUX_CUDA)" \
		$(LLAMA_CPP_DIR)
	@echo "✓ $(LLAMA_DEPS_LINUX_CUDA)/"
	@ls -lh $(LLAMA_DEPS_LINUX_CUDA)/*.a

deps-docker-vulkan: $(LLAMA_DEPS_LINUX_VULKAN)/libllama.a
$(LLAMA_DEPS_LINUX_VULKAN)/libllama.a:
	@echo "=== [Docker] Building llama.cpp (vulkan) ==="
	@mkdir -p $(LLAMA_DEPS_LINUX_VULKAN)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.deps \
		--build-arg VARIANT=vulkan \
		--output "type=local,dest=$(LLAMA_DEPS_LINUX_VULKAN)" \
		$(LLAMA_CPP_DIR)
	@echo "✓ $(LLAMA_DEPS_LINUX_VULKAN)/"
	@ls -lh $(LLAMA_DEPS_LINUX_VULKAN)/*.a

deps-docker-sycl: $(LLAMA_DEPS_LINUX_SYCL)/libllama.a
$(LLAMA_DEPS_LINUX_SYCL)/libllama.a:
	@echo "=== [Docker] Building llama.cpp (sycl — Intel GPU, ~18GB image) ==="
	@mkdir -p $(LLAMA_DEPS_LINUX_SYCL)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.deps \
		--build-arg VARIANT=sycl \
		--output "type=local,dest=$(LLAMA_DEPS_LINUX_SYCL)" \
		$(LLAMA_CPP_DIR)
	@echo "✓ $(LLAMA_DEPS_LINUX_SYCL)/"
	@ls -lh $(LLAMA_DEPS_LINUX_SYCL)/*.a

# =============================================
#  Phase 2: Standalone Binaries
# =============================================

.PHONY: all macos linux linux-compat linux-vulkan linux-cuda linux-sycl \
        dist dist-all zig-wrappers

all: macos linux

# --- CPU binaries (cross-compiled from macOS via zig) ---

macos: deps
	@echo "=== Building obrain-chat (macOS arm64, static) ==="
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) LLAMA_STATIC=1 \
		LLAMA_BUILD_DIR=$(abspath $(LLAMA_DEPS_MACOS)) \
		cargo build $(CARGO_FLAGS)
	@echo "✓ $(OUT_MACOS) ($$(du -h $(OUT_MACOS) | cut -f1))"

linux: deps-docker-cpu-avx2 zig-wrappers
	@echo "=== Building obrain-chat (Linux x86_64 AVX2, static) ==="
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) LLAMA_STATIC=1 \
		LLAMA_BUILD_DIR=$(abspath $(LLAMA_DEPS_LINUX_AVX2)) \
		cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-gnu
	@echo "✓ $(OUT_LINUX) ($$(du -h $(OUT_LINUX) | cut -f1))"

linux-compat: deps-docker-cpu-compat zig-wrappers
	@echo "=== Building obrain-chat (Linux x86_64 SSE3 compat, static) ==="
	LLAMA_CPP_DIR=$(LLAMA_CPP_DIR) LLAMA_STATIC=1 \
		LLAMA_BUILD_DIR=$(abspath $(LLAMA_DEPS_LINUX_COMPAT)) \
		cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-gnu
	@echo "✓ $(OUT_LINUX) ($$(du -h $(OUT_LINUX) | cut -f1))"

# --- GPU binaries (full Docker builds) ---
# These MUST build inside Docker because the GPU runtime libs
# (libvulkan.so, CUDA, oneAPI) are needed at link time and are
# only available inside the container. Cannot cross-link from macOS.

linux-vulkan:
	@echo "=== [Docker] Full build: obrain-chat (Linux x86_64, Vulkan GPU) ==="
	@mkdir -p $(DIST_DIR)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.linux \
		--build-arg VARIANT=vulkan \
		--build-context llama=$(LLAMA_CPP_DIR) \
		--build-context grafeo=$(GRAFEO_DIR) \
		--output "type=local,dest=$(DIST_DIR)" \
		.
	@mv $(DIST_DIR)/obrain-chat $(DIST_DIR)/obrain-chat-linux-x64-vulkan 2>/dev/null || true
	@echo "✓ $(DIST_DIR)/obrain-chat-linux-x64-vulkan"
	@ls -lh $(DIST_DIR)/obrain-chat-linux-x64-vulkan
	@file $(DIST_DIR)/obrain-chat-linux-x64-vulkan

linux-cuda:
	@echo "=== [Docker] Full build: obrain-chat (Linux x86_64, NVIDIA CUDA) ==="
	@mkdir -p $(DIST_DIR)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.linux \
		--build-arg VARIANT=cuda \
		--build-context llama=$(LLAMA_CPP_DIR) \
		--build-context grafeo=$(GRAFEO_DIR) \
		--output "type=local,dest=$(DIST_DIR)" \
		.
	@mv $(DIST_DIR)/obrain-chat $(DIST_DIR)/obrain-chat-linux-x64-cuda 2>/dev/null || true
	@echo "✓ $(DIST_DIR)/obrain-chat-linux-x64-cuda"
	@ls -lh $(DIST_DIR)/obrain-chat-linux-x64-cuda
	@file $(DIST_DIR)/obrain-chat-linux-x64-cuda

linux-sycl:
	@echo "=== [Docker] Full build: obrain-chat (Linux x86_64, Intel SYCL) ==="
	@mkdir -p $(DIST_DIR)
	docker build $(DOCKER_PLATFORM) \
		-f docker/Dockerfile.linux \
		--build-arg VARIANT=sycl \
		--build-context llama=$(LLAMA_CPP_DIR) \
		--build-context grafeo=$(GRAFEO_DIR) \
		--output "type=local,dest=$(DIST_DIR)" \
		.
	@mv $(DIST_DIR)/obrain-chat $(DIST_DIR)/obrain-chat-linux-x64-sycl 2>/dev/null || true
	@echo "✓ $(DIST_DIR)/obrain-chat-linux-x64-sycl"
	@ls -lh $(DIST_DIR)/obrain-chat-linux-x64-sycl
	@file $(DIST_DIR)/obrain-chat-linux-x64-sycl

# =============================================
#  Distribution
# =============================================

dist: all
	@mkdir -p $(DIST_DIR)
	cp $(OUT_MACOS) $(DIST_DIR)/obrain-chat-macos-arm64
	cp $(OUT_LINUX) $(DIST_DIR)/obrain-chat-linux-x64
	@echo ""
	@echo "=== Distribution ==="
	@ls -lh $(DIST_DIR)/obrain-chat-*
	@echo ""
	@echo "Targets:"
	@echo "  macos-arm64 : Apple Silicon M1+ (NEON + Metal GPU)"
	@echo "  linux-x64   : AMD64/Intel64 with AVX2+FMA (Zen/Haswell+, ~2013+)"

dist-all:
	@# Build in explicit order: AVX2 first, copy, then compat overwrites OUT_LINUX
	$(MAKE) all
	@mkdir -p $(DIST_DIR)
	cp $(OUT_MACOS) $(DIST_DIR)/obrain-chat-macos-arm64
	cp $(OUT_LINUX) $(DIST_DIR)/obrain-chat-linux-x64
	$(MAKE) linux-compat
	cp $(OUT_LINUX) $(DIST_DIR)/obrain-chat-linux-x64-compat
	@# GPU binaries build inside Docker and output directly to dist/
	$(MAKE) linux-vulkan
	$(MAKE) linux-cuda
	$(MAKE) linux-sycl
	@echo ""
	@echo "========================================"
	@echo "  Phase 2 complete — all binaries built"
	@echo "========================================"
	@echo ""
	@ls -lh $(DIST_DIR)/obrain-chat-*
	@echo ""
	@file $(DIST_DIR)/obrain-chat-*
	@echo ""
	@echo "Targets:"
	@echo "  macos-arm64        : Apple Silicon M1+ (NEON + Metal GPU)"
	@echo "  linux-x64          : Linux AVX2+FMA CPU (Zen/Haswell+, ~2013+)"
	@echo "  linux-x64-compat   : Linux SSE3 CPU only (any x86_64, ~2006+)"
	@echo "  linux-x64-vulkan   : Linux + Vulkan GPU (AMD/NVIDIA/Intel)"
	@echo "  linux-x64-cuda     : Linux + NVIDIA CUDA GPU"
	@echo "  linux-x64-sycl     : Linux + Intel SYCL/oneAPI GPU (Arc, Data Center)"

# =============================================
#  Info & Cleanup
# =============================================

.PHONY: info clean clean-deps clean-docker clean-all

info:
	@echo "LLAMA_CPP_DIR = $(LLAMA_CPP_DIR)"
	@echo "GRAFEO_DIR    = $(GRAFEO_DIR)"
	@echo "PROFILE       = $(PROFILE)"
	@echo "JOBS          = $(JOBS)"
	@echo ""
	@echo "Phase 1 — Dependencies (make deps-all):"
	@echo "  macos-arm64        : Apple Silicon (native, Metal+Accelerate)"
	@echo "  linux-cpu-avx2     : [Docker] AVX2+FMA (Zen/Haswell+)"
	@echo "  linux-cpu-compat   : [Docker] SSE3 only (any x86_64)"
	@echo "  linux-cuda         : [Docker] NVIDIA CUDA 12.8"
	@echo "  linux-vulkan       : [Docker] Vulkan (universal GPU)"
	@echo "  linux-sycl         : [Docker] Intel SYCL/oneAPI"
	@echo ""
	@echo "Phase 2 — Binaries (make dist-all):"
	@echo "  macos-arm64        : cargo (native)"
	@echo "  linux-x64          : cargo + zig (cross-compile)"
	@echo "  linux-x64-compat   : cargo + zig (cross-compile)"
	@echo "  linux-x64-vulkan   : [Docker] full build (deps + Rust)"
	@echo "  linux-x64-cuda     : [Docker] full build (deps + Rust)"
	@echo "  linux-x64-sycl     : [Docker] full build (deps + Rust)"
	@echo ""
	@echo "Deps status:"
	@for d in macos-arm64 linux-cpu-avx2 linux-cpu-compat linux-cuda linux-vulkan linux-sycl; do \
		if [ -d "$(DEPS_DIR)/$$d" ] && ls $(DEPS_DIR)/$$d/*.a >/dev/null 2>&1; then \
			count=$$(ls $(DEPS_DIR)/$$d/*.a | wc -l | tr -d ' '); \
			echo "  ✓ $$d ($$count libs)"; \
		else \
			echo "  ✗ $$d (not built)"; \
		fi; \
	done
	@echo ""
	@echo "Dist status:"
	@for b in macos-arm64 linux-x64 linux-x64-compat linux-x64-vulkan linux-x64-cuda linux-x64-sycl; do \
		if [ -f "$(DIST_DIR)/obrain-chat-$$b" ]; then \
			size=$$(du -h $(DIST_DIR)/obrain-chat-$$b | cut -f1); \
			echo "  ✓ $$b ($$size)"; \
		else \
			echo "  ✗ $$b (not built)"; \
		fi; \
	done

# --- Zig cross-compile wrappers (for Rust cross-linking) ---

zig-wrappers: $(ZIG_CC)

$(ZIG_CC):
	@mkdir -p $(ZIG_DIR)
	@printf '#!/bin/sh\nexec zig cc -target x86_64-linux-gnu "$$@"\n'  > $(ZIG_DIR)/cc-linux.sh  && chmod +x $(ZIG_DIR)/cc-linux.sh
	@printf '#!/bin/sh\nexec zig c++ -target x86_64-linux-gnu "$$@"\n' > $(ZIG_DIR)/cxx-linux.sh && chmod +x $(ZIG_DIR)/cxx-linux.sh
	@printf '#!/bin/sh\nexec zig ar "$$@"\n'                           > $(ZIG_DIR)/ar-linux.sh  && chmod +x $(ZIG_DIR)/ar-linux.sh

# --- Cleanup ---

clean:
	cargo clean

clean-deps:
	rm -rf $(DEPS_DIR)

clean-dist:
	rm -rf $(DIST_DIR)

clean-docker:
	-docker rmi obrain-deps-cpu-avx2 obrain-deps-cpu-compat obrain-deps-cuda obrain-deps-vulkan obrain-deps-sycl 2>/dev/null

clean-all: clean clean-deps clean-dist clean-docker
	rm -rf $(LLAMA_STATIC_MACOS) $(ZIG_DIR)
