#!/usr/bin/env bash
# safe-gh-pr.sh — Wrapper around `gh pr create` that blocks PRs on upstream repos.
# This repo is a fork (this-rs/grafeo). PRs must NEVER target the original GrafeoDB/grafeo.
#
# Usage: scripts/safe-gh-pr.sh [gh pr create args...]
# Or:    source this before `gh pr create` in CI/agent contexts.

set -euo pipefail

BLOCKED_REPOS="GrafeoDB/grafeo|grafeodb/grafeo"

# Detect the target repo: explicit --repo flag, or gh's default resolution
TARGET_REPO=""
for i in "$@"; do
    if [[ "$prev" == "--repo" || "$prev" == "-R" ]]; then
        TARGET_REPO="$i"
        break
    fi
    prev="$i"
done

if [[ -z "$TARGET_REPO" ]]; then
    # No --repo flag: gh resolves from git remote. Check what gh would use.
    TARGET_REPO=$(gh repo view --json nameWithOwner -q '.nameWithOwner' 2>/dev/null || echo "")
fi

if echo "$TARGET_REPO" | grep -qEi "$BLOCKED_REPOS"; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ❌ BLOCKED: PR would target UPSTREAM repository!           ║"
    echo "║                                                             ║"
    echo "║  Target: $TARGET_REPO"
    echo "║                                                             ║"
    echo "║  This is a fork. PRs go to this-rs/grafeo ONLY.             ║"
    echo "║  Add: --repo this-rs/grafeo                                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi

# Safe — forward to real gh
exec gh pr create "$@"
