#!/usr/bin/env bash
# Build WASM package with size-optimized profile.
#
# Usage:
#   ./scripts/build-wasm.sh                           # default: GQL only, web target
#   ./scripts/build-wasm.sh --features ai              # GQL + AI search
#   ./scripts/build-wasm.sh --features full            # all languages + AI
#   ./scripts/build-wasm.sh --out-dir path/to/output   # custom output directory
#   ./scripts/build-wasm.sh --target bundler --scope obrain-db
#
# Requirements: rustup target wasm32-unknown-unknown, wasm-bindgen-cli

set -euo pipefail

CRATE_DIR="crates/bindings/wasm"
OUT_DIR=""
PROFILE="minimal-size"
TARGET="web"
SCOPE=""
FEATURES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)   TARGET="$2"; shift 2 ;;
        --scope)    SCOPE="--scope $2"; shift 2 ;;
        --features) FEATURES="--features $2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --release)  PROFILE="release"; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default output directory
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="${CRATE_DIR}/pkg"
fi

echo "Building WASM (profile: ${PROFILE}, target: ${TARGET})"

# Step 1: Cargo build
CARGO_CMD="cargo build --target wasm32-unknown-unknown --profile ${PROFILE} -p obrain-wasm"
if [[ -n "$FEATURES" ]]; then
    CARGO_CMD="${CARGO_CMD} ${FEATURES}"
fi
echo "  cargo build..."
eval "$CARGO_CMD" 2>&1 | grep -E "Compiling obrain-wasm|Finished|warning:" || true

# Determine the output path (profile name maps to directory)
PROFILE_DIR="${PROFILE}"
if [[ "$PROFILE" == "release" ]]; then
    PROFILE_DIR="release"
fi
WASM_FILE="target/wasm32-unknown-unknown/${PROFILE_DIR}/obrain_wasm.wasm"

if [[ ! -f "$WASM_FILE" ]]; then
    echo "Error: ${WASM_FILE} not found"
    exit 1
fi

# Step 2: wasm-bindgen
echo "  wasm-bindgen..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen --target "$TARGET" --out-dir "$OUT_DIR" "$WASM_FILE"

# Step 3: Report sizes
RAW_SIZE=$(stat -c%s "$OUT_DIR/obrain_wasm_bg.wasm" 2>/dev/null || stat -f%z "$OUT_DIR/obrain_wasm_bg.wasm")
GZ_SIZE=$(gzip -c "$OUT_DIR/obrain_wasm_bg.wasm" | wc -c)

echo ""
echo "Output: ${OUT_DIR}/"
echo "  Raw:    $(( RAW_SIZE / 1024 )) KB"
echo "  Gzip:   $(( GZ_SIZE / 1024 )) KB"

if [[ "$GZ_SIZE" -gt 409600 ]]; then
    echo "  WARNING: Exceeds 400 KB gzip target"
elif [[ "$GZ_SIZE" -gt 307200 ]]; then
    echo "  Note: Within 300-400 KB gzip target range"
else
    echo "  OK: Under 300 KB gzip target"
fi
