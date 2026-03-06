# Build WASM package with size-optimized profile.
#
# Usage:
#   .\scripts\build-wasm.ps1                           # default: GQL only, web target
#   .\scripts\build-wasm.ps1 -Features ai              # GQL + AI search
#   .\scripts\build-wasm.ps1 -Features full            # all languages + AI
#   .\scripts\build-wasm.ps1 -OutDir path\to\output    # custom output directory
#   .\scripts\build-wasm.ps1 -Target bundler -Scope grafeo-db
#
# Requirements: rustup target wasm32-unknown-unknown, wasm-bindgen-cli

param(
    [string]$Target = "web",
    [string]$Scope = "",
    [string]$Features = "",
    [string]$OutDir = "",
    [switch]$Release
)

$ErrorActionPreference = "Stop"

$CrateDir = "crates\bindings\wasm"
if (-not $OutDir) { $OutDir = "$CrateDir\pkg" }
$Profile = if ($Release) { "release" } else { "minimal-size" }

Write-Host "Building WASM (profile: $Profile, target: $Target)"

# Step 1: Cargo build
$cargoArgs = @("build", "--target", "wasm32-unknown-unknown", "--profile", $Profile, "-p", "grafeo-wasm")
if ($Features) {
    $cargoArgs += "--features"
    $cargoArgs += $Features
}
Write-Host "  cargo build..."
& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { throw "Cargo build failed" }

# Determine output path
$WasmFile = "target\wasm32-unknown-unknown\$Profile\grafeo_wasm.wasm"
if (-not (Test-Path $WasmFile)) {
    throw "Error: $WasmFile not found"
}

# Step 2: wasm-bindgen
Write-Host "  wasm-bindgen..."
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
& wasm-bindgen --target $Target --out-dir $OutDir $WasmFile
if ($LASTEXITCODE -ne 0) { throw "wasm-bindgen failed" }

# Step 3: Report sizes
$wasmPath = Join-Path $OutDir "grafeo_wasm_bg.wasm"
$rawSize = (Get-Item $wasmPath).Length
$gzBytes = [System.IO.File]::ReadAllBytes($wasmPath)
$ms = New-Object System.IO.MemoryStream
$gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionMode]::Compress)
$gz.Write($gzBytes, 0, $gzBytes.Length)
$gz.Close()
$gzSize = $ms.Length

Write-Host ""
Write-Host "Output: $OutDir\"
Write-Host "  Raw:    $([math]::Round($rawSize / 1024)) KB"
Write-Host "  Gzip:   $([math]::Round($gzSize / 1024)) KB"

if ($gzSize -gt 409600) {
    Write-Host "  WARNING: Exceeds 400 KB gzip target" -ForegroundColor Yellow
} elseif ($gzSize -gt 307200) {
    Write-Host "  Note: Within 300-400 KB gzip target range" -ForegroundColor Cyan
} else {
    Write-Host "  OK: Under 300 KB gzip target" -ForegroundColor Green
}
