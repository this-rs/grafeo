#!/usr/bin/env bash
# Pre-push hook: runs cargo fmt check + clippy before allowing push.
# Install: cp scripts/pre-push.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
# Or:      ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push

set -euo pipefail

echo "==> pre-push: checking formatting..."
if ! cargo fmt --all -- --check 2>/dev/null; then
    echo ""
    echo "ERROR: cargo fmt check failed. Run 'cargo fmt --all' and commit."
    exit 1
fi

echo "==> pre-push: running clippy..."
if ! cargo clippy --all-features --all-targets -- -D warnings 2>/dev/null; then
    echo ""
    echo "ERROR: clippy found warnings. Fix them and commit."
    exit 1
fi

echo "==> pre-push: all checks passed."
