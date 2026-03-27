# Build all WASM variants for obrain-web.
#
# Produces two binaries:
#   pkg\      - AI variant (GQL + vector/text/hybrid search) for main export
#   pkg-lite\ - Browser variant (GQL only) for /lite export
#
# Usage:
#   .\scripts\build-wasm-all.ps1

$ErrorActionPreference = "Stop"

$WasmDir = "crates\bindings\wasm"

Write-Host "=== Building WASM AI variant (main export) ==="
& .\scripts\build-wasm.ps1 -Features ai

# Write package.json so bundlers can resolve @obrain-db/wasm
@'
{
  "name": "@obrain-db/wasm",
  "version": "0.0.0",
  "type": "module",
  "main": "obrain_wasm.js",
  "module": "obrain_wasm.js",
  "types": "obrain_wasm.d.ts"
}
'@ | Set-Content -Path "$WasmDir\pkg\package.json" -Encoding utf8

Write-Host ""
Write-Host "=== Building WASM lite variant (/lite export) ==="
& .\scripts\build-wasm.ps1 -OutDir "$WasmDir\pkg-lite"

# Write package.json for @obrain-db/wasm-lite
@'
{
  "name": "@obrain-db/wasm-lite",
  "version": "0.0.0",
  "type": "module",
  "main": "obrain_wasm.js",
  "module": "obrain_wasm.js",
  "types": "obrain_wasm.d.ts"
}
'@ | Set-Content -Path "$WasmDir\pkg-lite\package.json" -Encoding utf8

Write-Host ""
Write-Host "Both variants built successfully."
Write-Host "  AI variant:   $WasmDir\pkg\      (used by @obrain-db/web)"
Write-Host "  Lite variant: $WasmDir\pkg-lite\ (used by @obrain-db/web/lite)"
