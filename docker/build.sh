#!/usr/bin/env bash
# obrain-chat Docker build helper
#
# Builds llama.cpp static libraries (and optionally the Rust binary)
# for a given variant inside Docker, then extracts artifacts.
#
# Usage:
#   ./docker/build.sh <variant> [--libs-only]
#
# Variants: cpu-avx2, cpu-compat, cuda, vulkan
#
# Options:
#   --libs-only   Only build and extract llama.cpp static libs (.a files)
#                 Skips Rust compilation (faster, for iterative dev)
#
# Output:
#   deps/linux-<variant>/          Static libraries (.a)
#   dist/obrain-chat-linux-<variant>   Binary (unless --libs-only)

set -euo pipefail

VARIANT="${1:?Usage: $0 <variant> [--libs-only]}"
LIBS_ONLY="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/projects/ia/llama.cpp}"

DEPS_DIR="$PROJECT_DIR/deps/linux-${VARIANT}"
DIST_DIR="$PROJECT_DIR/dist"

echo "=== Building variant: ${VARIANT} ==="
echo "  LLAMA_CPP_DIR: ${LLAMA_CPP_DIR}"
echo "  Output:        ${DEPS_DIR}"

# --- Build llama.cpp static libs ---
DOCKER_TAG="obrain-llama-${VARIANT}"

# Use the specific stage for this variant
docker build \
    -f "$SCRIPT_DIR/Dockerfile.linux" \
    --target "build-${VARIANT}" \
    --build-arg "VARIANT=${VARIANT}" \
    -t "$DOCKER_TAG" \
    "$LLAMA_CPP_DIR"

# Extract static libraries from the container
mkdir -p "$DEPS_DIR"
CONTAINER_ID=$(docker create "$DOCKER_TAG")

# Copy all .a files from the build directory
for subdir in src ggml/src ggml/src/ggml-cpu ggml/src/ggml-cuda ggml/src/ggml-vulkan ggml/src/ggml-metal ggml/src/ggml-blas; do
    docker cp "$CONTAINER_ID:/build/llama.cpp/build-linux/${subdir}/" "$DEPS_DIR/" 2>/dev/null || true
done

# Flatten: move all .a files to the top-level deps dir
find "$DEPS_DIR" -name '*.a' -exec mv {} "$DEPS_DIR/" \; 2>/dev/null || true
# Clean up subdirectories
find "$DEPS_DIR" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

docker rm "$CONTAINER_ID" > /dev/null

echo "  Static libs extracted:"
ls -lh "$DEPS_DIR"/*.a 2>/dev/null || echo "  (none found)"

if [ "$LIBS_ONLY" = "--libs-only" ]; then
    echo "  --libs-only: skipping Rust build"
    exit 0
fi

# --- Build full Rust binary ---
echo ""
echo "=== Building Rust binary (${VARIANT}) ==="

DOCKER_TAG_FULL="obrain-build-${VARIANT}"

# Full build needs the project source + grafeo deps
# We use docker build with the full context
docker build \
    -f "$SCRIPT_DIR/Dockerfile.linux" \
    --target "output" \
    --build-arg "VARIANT=${VARIANT}" \
    --output "type=local,dest=${DIST_DIR}" \
    "$PROJECT_DIR"

# Rename the output binary
if [ -f "$DIST_DIR/obrain-chat" ]; then
    mv "$DIST_DIR/obrain-chat" "$DIST_DIR/obrain-chat-linux-${VARIANT}"
    echo "  Binary: $DIST_DIR/obrain-chat-linux-${VARIANT}"
    file "$DIST_DIR/obrain-chat-linux-${VARIANT}"
fi

echo ""
echo "=== Done: ${VARIANT} ==="
