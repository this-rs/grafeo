#!/usr/bin/env node

/**
 * Obrain CLI launcher for npm.
 *
 * Resolves the platform-specific binary from optional dependencies
 * and spawns it with all arguments forwarded.
 */

"use strict";

const { execFileSync } = require("child_process");
const { existsSync } = require("fs");
const { join } = require("path");
const { platform, arch } = require("os");

/** Map of Node.js platform/arch to package names and binary paths. */
const PLATFORMS = {
  "linux-x64": { pkg: "@obrain-db/cli-linux-x64", bin: "obrain" },
  "linux-arm64": { pkg: "@obrain-db/cli-linux-arm64", bin: "obrain" },
  "darwin-x64": { pkg: "@obrain-db/cli-darwin-x64", bin: "obrain" },
  "darwin-arm64": { pkg: "@obrain-db/cli-darwin-arm64", bin: "obrain" },
  "win32-x64": { pkg: "@obrain-db/cli-win32-x64", bin: "obrain.exe" },
};

function findBinary() {
  const key = `${platform()}-${arch()}`;
  const entry = PLATFORMS[key];

  if (!entry) {
    console.error(
      `error: unsupported platform ${key}\n` +
        `Supported: ${Object.keys(PLATFORMS).join(", ")}\n` +
        "Install the binary manually: cargo install obrain-cli"
    );
    process.exit(1);
  }

  // 1. Try platform-specific optional dependency
  try {
    const pkgDir = require.resolve(`${entry.pkg}/package.json`);
    const binPath = join(pkgDir, "..", entry.bin);
    if (existsSync(binPath)) {
      return binPath;
    }
  } catch {
    // Package not installed (optional dependency)
  }

  // 2. Try binary adjacent to this package (development installs)
  const adjacent = join(__dirname, "..", entry.bin);
  if (existsSync(adjacent)) {
    return adjacent;
  }

  // 3. Fall back to system PATH
  return entry.bin;
}

const binary = findBinary();

try {
  const result = execFileSync(binary, process.argv.slice(2), {
    stdio: "inherit",
    windowsHide: true,
  });
} catch (error) {
  if (error.status != null) {
    process.exit(error.status);
  }

  if (error.code === "ENOENT") {
    console.error(
      `error: obrain binary not found.\n\n` +
        `Install via one of:\n` +
        `  npm install @obrain-db/cli\n` +
        `  cargo install obrain-cli\n` +
        `  Download from https://github.com/this-rs/obrain/releases\n`
    );
    process.exit(1);
  }

  console.error(`error: ${error.message}`);
  process.exit(1);
}
