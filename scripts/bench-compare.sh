#!/usr/bin/env bash
# bench-compare.sh: Compare two Criterion baselines and post results as a PR comment.
#
# Usage: bench-compare.sh <baseline> <candidate> <threshold_pct> <pr_number>
#
# Requires: critcmp, gh (GitHub CLI)

set -euo pipefail

BASELINE="${1:?Usage: bench-compare.sh <baseline> <candidate> <threshold_pct> <pr_number>}"
CANDIDATE="${2:?}"
THRESHOLD="${3:?}"
PR_NUMBER="${4:?}"

# Run critcmp comparison
COMPARISON=$(critcmp "$BASELINE" "$CANDIDATE" --color never 2>&1) || true

echo "--- critcmp output ---"
echo "$COMPARISON"
echo "--- end ---"

# Build markdown body
BODY_FILE=$(mktemp)
{
  echo "## Benchmark Comparison (base vs PR)"
  echo ""
  echo "Measured on the **same runner** to eliminate hardware variance."
  echo ""
  echo "<details>"
  echo "<summary>Full results</summary>"
  echo ""
  echo '```'
  echo "$COMPARISON"
  echo '```'
  echo ""
  echo "</details>"
  echo ""
} > "$BODY_FILE"

# Parse critcmp output for regressions exceeding threshold.
# critcmp output lines look like:
#   bench_name    base  1.00  123.4ns    candidate  1.05  129.6ns
# The ratio column (e.g. 1.05) for the candidate indicates relative time.
# A ratio > 1.0 means the candidate is slower.
REGRESSIONS=$(echo "$COMPARISON" | awk -v threshold="$THRESHOLD" '
  # Skip header lines
  /^group|^-----/ { next }
  # Match data lines (start with a letter or underscore)
  /^[a-zA-Z_]/ {
    name = $1
    # Find the candidate ratio column (second ratio value in the line)
    ratio = 0
    count = 0
    for (i = 2; i <= NF; i++) {
      # Ratios look like "1.05" or "0.95"
      if ($i ~ /^[0-9]+\.[0-9]+$/ && $i != "" ) {
        count++
        if (count == 2) {
          ratio = $i + 0
          break
        }
      }
    }
    if (ratio > 0) {
      pct_change = (ratio - 1.0) * 100
      if (pct_change > threshold + 0) {
        printf "%s %.1f%%\n", name, pct_change
      }
    }
  }
')

if [ -n "$REGRESSIONS" ]; then
  {
    echo "### Regressions detected (>${THRESHOLD}%)"
    echo ""
    echo "| Benchmark | Regression |"
    echo "| --- | --- |"
    echo "$REGRESSIONS" | while IFS=' ' read -r name pct; do
      echo "| \`$name\` | +$pct |"
    done
    echo ""
  } >> "$BODY_FILE"
else
  echo "No regressions above ${THRESHOLD}% threshold." >> "$BODY_FILE"
fi

# Add marker for idempotent comment updates
echo "" >> "$BODY_FILE"
echo "<!-- obrain-bench-comparison -->" >> "$BODY_FILE"

# Post or update PR comment
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

COMMENT_ID=$(gh api "repos/${REPO}/issues/${PR_NUMBER}/comments" \
  --jq '.[] | select(.body | contains("<!-- obrain-bench-comparison -->")) | .id' \
  | head -1) || true

BODY=$(cat "$BODY_FILE")
rm -f "$BODY_FILE"

if [ -n "$COMMENT_ID" ]; then
  gh api --method PATCH "repos/${REPO}/issues/comments/${COMMENT_ID}" -f body="$BODY"
  echo "Updated existing comment $COMMENT_ID"
else
  gh pr comment "$PR_NUMBER" --body "$BODY"
  echo "Posted new comment on PR #$PR_NUMBER"
fi
