#!/usr/bin/env bash
# Build all WASM variants for grafeo-web.
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

# Write package.json so bundlers can resolve @grafeo-db/wasm
cat > "$WASM_DIR/pkg/package.json" <<'EOF'
{
  "name": "@grafeo-db/wasm",
  "version": "0.0.0",
  "type": "module",
  "main": "grafeo_wasm.js",
  "module": "grafeo_wasm.js",
  "types": "grafeo_wasm.d.ts"
}
EOF

echo ""
echo "=== Building WASM lite variant (/lite export) ==="
./scripts/build-wasm.sh --out-dir "$WASM_DIR/pkg-lite"

# Write package.json for @grafeo-db/wasm-lite
cat > "$WASM_DIR/pkg-lite/package.json" <<'EOF'
{
  "name": "@grafeo-db/wasm-lite",
  "version": "0.0.0",
  "type": "module",
  "main": "grafeo_wasm.js",
  "module": "grafeo_wasm.js",
  "types": "grafeo_wasm.d.ts"
}
EOF

echo ""
echo "Both variants built successfully."
echo "  AI variant:   $WASM_DIR/pkg/      (used by @grafeo-db/web)"
echo "  Lite variant: $WASM_DIR/pkg-lite/ (used by @grafeo-db/web/lite)"
