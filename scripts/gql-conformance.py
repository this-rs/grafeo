#!/usr/bin/env python3
"""GQL ISO/IEC 39075:2024 conformance tracking.

Scans test files for ``// ISO:`` annotations and cross-references them against
the conformance matrix in ``docs/user-guide/gql/conformance.md``.

Usage:
    python scripts/gql-conformance.py report      # Human-readable coverage report
    python scripts/gql-conformance.py dialect      # Generate docs/gql-dialect.json
    python scripts/gql-conformance.py validate     # CI check: valid IDs, exit code
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches Annex D table rows: | G002 | Feature name | Status |
# Status may be wrapped in ** (bold markdown).
FEATURE_ROW_RE = re.compile(
    r"^\|\s*(G[A-Z]*\d+)\s*\|([^|]+)\|([^|]+)\|",
    re.MULTILINE,
)

# Matches section headings like "### Pattern Features"
SECTION_HEADING_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)

# Matches // ISO: G049  or  // ISO: G049, G050
ISO_ANNOTATION_RE = re.compile(r"^//\s*ISO:\s*(.+)$", re.MULTILINE)

# Matches #[test] attribute
TEST_ATTR_RE = re.compile(r"^\s*#\[test\]")

# Matches fn declaration
FN_RE = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")

# Workspace version in Cargo.toml
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_conformance(path: Path) -> dict[str, dict[str, str]]:
    """Parse conformance.md and return {feature_id: {name, status, category}}.

    Only parses the "Optional Features (Annex D)" section onward, which has
    tables with an ID column.
    """
    content = path.read_text(encoding="utf-8")

    # Find where Annex D starts
    annex_start = content.find("## Optional Features")
    if annex_start == -1:
        print(
            f"Error: could not find '## Optional Features' in {path}", file=sys.stderr
        )
        sys.exit(1)

    annex_content = content[annex_start:]
    features: dict[str, dict[str, str]] = {}

    # Track current section heading
    current_category = "Uncategorized"

    # Process line by line to track section context
    for line in annex_content.splitlines():
        heading_match = SECTION_HEADING_RE.match(line)
        if heading_match:
            current_category = heading_match.group(1).strip()
            continue

        row_match = FEATURE_ROW_RE.match(line)
        if row_match:
            feature_id = row_match.group(1).strip()
            name = row_match.group(2).strip()
            raw_status = row_match.group(3).strip()
            status = _normalize_status(raw_status)
            features[feature_id] = {
                "name": name,
                "status": status,
                "category": current_category,
            }

    return features


def _normalize_status(raw: str) -> str:
    """Normalize conformance.md status to machine-friendly values."""
    # Strip markdown bold markers
    cleaned = raw.replace("**", "").strip().lower()

    if cleaned.startswith("supported"):
        return "supported"
    if cleaned.startswith("partial"):
        return "partial"
    if cleaned.startswith("not yet") or cleaned.startswith("not supported"):
        return "not_supported"
    # Fallback
    return cleaned


def scan_tests(test_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Scan .rs test files for // ISO: annotations.

    Returns {feature_id: [{file, test_name}, ...]}.
    """
    coverage: dict[str, list[dict[str, str]]] = {}

    for rs_file in sorted(test_dir.rglob("*.rs")):
        lines = rs_file.read_text(encoding="utf-8").splitlines()
        pending_ids: list[str] = []

        for i, line in enumerate(lines):
            # Check for ISO annotation
            iso_match = ISO_ANNOTATION_RE.match(line.strip())
            if iso_match:
                ids_str = iso_match.group(1)
                pending_ids = [fid.strip() for fid in ids_str.split(",")]
                continue

            # Check for #[test] - keep pending IDs alive through attributes
            if TEST_ATTR_RE.match(line):
                continue

            # Check for fn declaration following #[test]
            fn_match = FN_RE.match(line)
            if fn_match and pending_ids:
                test_name = fn_match.group(1)
                rel_path = str(rs_file.relative_to(test_dir.parent.parent))
                for fid in pending_ids:
                    coverage.setdefault(fid, []).append(
                        {
                            "file": rel_path.replace("\\", "/"),
                            "test_name": test_name,
                        }
                    )
                pending_ids = []
                continue

            # Any other non-blank, non-comment, non-attribute line clears pending
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("//")
                and not stripped.startswith("#[")
            ):
                pending_ids = []

    return coverage


def read_version(cargo_toml: Path) -> str:
    """Read workspace version from root Cargo.toml."""
    content = cargo_toml.read_text(encoding="utf-8")
    match = VERSION_RE.search(content)
    if match:
        return match.group(1)
    return "unknown"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_report(
    features: dict[str, dict[str, str]],
    coverage: dict[str, list[dict[str, str]]],
) -> None:
    """Print a human-readable coverage report."""
    supported = {fid for fid, f in features.items() if f["status"] == "supported"}
    partial = {fid for fid, f in features.items() if f["status"] == "partial"}
    not_supported = {
        fid for fid, f in features.items() if f["status"] == "not_supported"
    }
    tested = {fid for fid in coverage if fid in features}

    total = len(features)
    tested_supported = tested & supported
    tested_partial = tested & partial
    untested_supported = supported - tested

    print("=" * 60)
    print("GQL ISO/IEC 39075:2024 Conformance Report")
    print("=" * 60)
    print()
    print(f"Total Annex D features:    {total}")
    print(f"  Supported:               {len(supported)}")
    print(f"  Partial:                  {len(partial)}")
    print(f"  Not yet implemented:      {len(not_supported)}")
    print()
    print(f"Features with tests:       {len(tested)}/{len(supported) + len(partial)}")
    print(f"  Supported + tested:      {len(tested_supported)}")
    print(f"  Partial + tested:        {len(tested_partial)}")
    print()

    if untested_supported:
        print("-" * 60)
        print("Supported features WITHOUT compliance tests:")
        print("-" * 60)
        for fid in sorted(untested_supported):
            print(f"  {fid:8s} {features[fid]['name']}")
        print()

    # Show test counts per tested feature
    print("-" * 60)
    print("Features WITH compliance tests:")
    print("-" * 60)
    for fid in sorted(tested):
        count = len(coverage[fid])
        status = features[fid]["status"]
        marker = " (partial)" if status == "partial" else ""
        print(f"  {fid:8s} {count:3d} test(s)  {features[fid]['name']}{marker}")

    # Check for annotations referencing unknown IDs
    unknown = set(coverage.keys()) - set(features.keys())
    if unknown:
        print()
        print("WARNING: Annotations reference unknown feature IDs:")
        for fid in sorted(unknown):
            for entry in coverage[fid]:
                print(f"  {fid} in {entry['file']}::{entry['test_name']}")


def cmd_dialect(
    features: dict[str, dict[str, str]],
    coverage: dict[str, list[dict[str, str]]],
    version: str,
    output_path: Path,
) -> None:
    """Generate the dialect JSON file."""
    supported_count = sum(1 for f in features.values() if f["status"] == "supported")
    partial_count = sum(1 for f in features.values() if f["status"] == "partial")
    not_supported_count = sum(
        1 for f in features.values() if f["status"] == "not_supported"
    )
    tested_count = sum(1 for fid in features if fid in coverage)

    feature_list = []
    for fid in sorted(features.keys()):
        f = features[fid]
        tests = coverage.get(fid, [])
        feature_list.append(
            {
                "id": fid,
                "name": f["name"],
                "category": f["category"],
                "status": f["status"],
                "tested": len(tests) > 0,
                "test_count": len(tests),
            }
        )

    dialect = {
        "dialect": {
            "name": "ObrainDB",
            "version": version,
            "language": "GQL",
            "standard": "ISO/IEC 39075:2024",
            "homepage": "https://obrain.dev",
            "repository": "https://github.com/ObrainDB/obrain",
        },
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "total_features": len(features),
            "supported": supported_count,
            "partial": partial_count,
            "not_supported": not_supported_count,
            "tested": tested_count,
        },
        "features": feature_list,
    }

    output_path.write_text(
        json.dumps(dialect, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Generated {output_path} ({len(feature_list)} features, {tested_count} tested)"
    )


def cmd_validate(
    features: dict[str, dict[str, str]],
    coverage: dict[str, list[dict[str, str]]],
) -> int:
    """Validate annotations. Returns exit code (0 = pass, 1 = fail)."""
    errors = 0

    unknown = set(coverage.keys()) - set(features.keys())
    for fid in sorted(unknown):
        for entry in coverage[fid]:
            print(
                f"ERROR: Unknown feature ID '{fid}' in {entry['file']}::{entry['test_name']}"
            )
            errors += 1

    if errors:
        print(f"\n{errors} validation error(s) found.")
        return 1

    tested = sum(1 for fid in features if fid in coverage)
    total_testable = sum(1 for f in features.values() if f["status"] != "not_supported")
    print(
        f"OK: All annotations reference valid feature IDs. "
        f"({tested}/{total_testable} testable features covered)"
    )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GQL ISO/IEC 39075:2024 conformance tracking",
    )
    parser.add_argument(
        "command",
        choices=["report", "dialect", "validate"],
        help="Subcommand to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for dialect JSON (default: docs/gql-dialect.json)",
    )
    args = parser.parse_args()

    # Resolve project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    conformance_path = project_root / "docs" / "user-guide" / "gql" / "conformance.md"
    test_dir = project_root / "crates" / "obrain-engine" / "tests"
    cargo_toml = project_root / "Cargo.toml"

    if not conformance_path.exists():
        print(f"Error: conformance file not found: {conformance_path}", file=sys.stderr)
        sys.exit(1)
    if not test_dir.exists():
        print(f"Error: test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)

    features = parse_conformance(conformance_path)
    coverage = scan_tests(test_dir)

    if args.command == "report":
        cmd_report(features, coverage)
    elif args.command == "dialect":
        output = args.output or (project_root / "docs" / "gql-dialect.json")
        version = read_version(cargo_toml)
        cmd_dialect(features, coverage, version, output)
    elif args.command == "validate":
        sys.exit(cmd_validate(features, coverage))


if __name__ == "__main__":
    main()
