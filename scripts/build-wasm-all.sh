#!/usr/bin/env bash
# Build all WASM variants for obrain-web.
#
# Produces two binaries:
#   pkg/      - AI variant (GQL + vector/text/hybrid search) for main export
#   pkg-lite/ - Browser variant (GQL only) for /lite export
#
# Usage:
#   ./scripts/build-wasm-all.sh

set -euo pipefail

WASM_DIR="crates/bindings/wasm"

echo "=== Building WASM AI variant (main export) ==="
./scripts/build-wasm.sh --features ai

# Write package.json so bundlers can resolve @obrain-db/wasm
cat > "$WASM_DIR/pkg/package.json" <<'EOF'
{
  "name": "@obrain-db/wasm",
  "version": "0.0.0",
  "type": "module",
  "main": "obrain_wasm.js",
  "module": "obrain_wasm.js",
  "types": "obrain_wasm.d.ts"
}
EOF

echo ""
echo "=== Building WASM lite variant (/lite export) ==="
./scripts/build-wasm.sh --out-dir "$WASM_DIR/pkg-lite"

# Write package.json for @obrain-db/wasm-lite
cat > "$WASM_DIR/pkg-lite/package.json" <<'EOF'
{
  "name": "@obrain-db/wasm-lite",
  "version": "0.0.0",
  "type": "module",
  "main": "obrain_wasm.js",
  "module": "obrain_wasm.js",
  "types": "obrain_wasm.d.ts"
}
EOF

echo ""
echo "Both variants built successfully."
echo "  AI variant:   $WASM_DIR/pkg/      (used by @obrain-db/web)"
echo "  Lite variant: $WASM_DIR/pkg-lite/ (used by @obrain-db/web/lite)"
