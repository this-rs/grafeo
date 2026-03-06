# Build all WASM variants for grafeo-web.
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

# Write package.json so bundlers can resolve @grafeo-db/wasm
@'
{
  "name": "@grafeo-db/wasm",
  "version": "0.0.0",
  "type": "module",
  "main": "grafeo_wasm.js",
  "module": "grafeo_wasm.js",
  "types": "grafeo_wasm.d.ts"
}
'@ | Set-Content -Path "$WasmDir\pkg\package.json" -Encoding utf8

Write-Host ""
Write-Host "=== Building WASM lite variant (/lite export) ==="
& .\scripts\build-wasm.ps1 -OutDir "$WasmDir\pkg-lite"

# Write package.json for @grafeo-db/wasm-lite
@'
{
  "name": "@grafeo-db/wasm-lite",
  "version": "0.0.0",
  "type": "module",
  "main": "grafeo_wasm.js",
  "module": "grafeo_wasm.js",
  "types": "grafeo_wasm.d.ts"
}
'@ | Set-Content -Path "$WasmDir\pkg-lite\package.json" -Encoding utf8

Write-Host ""
Write-Host "Both variants built successfully."
Write-Host "  AI variant:   $WasmDir\pkg\      (used by @grafeo-db/web)"
Write-Host "  Lite variant: $WasmDir\pkg-lite\ (used by @grafeo-db/web/lite)"
