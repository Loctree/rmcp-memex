#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY_NAME="rmcp-memex"
OUTPUT_DIR="$REPO_ROOT/dist"
TARGET=""
SKIP_BUILD=0
SIGN_MACOS=0
SIGN_IDENTITY=""

usage() {
  cat <<'EOF'
Usage: scripts/build-release-bundle.sh [--target <triple>] [--output-dir <dir>] [--skip-build] [--sign-macos] [--sign-identity <name>]

Builds a prebuilt release tarball containing:
  - rmcp-memex
  - README.md
  - LICENSE
  - LICENSE-APACHE
  - install.sh

Examples:
  scripts/build-release-bundle.sh
  scripts/build-release-bundle.sh --target x86_64-unknown-linux-gnu
  scripts/build-release-bundle.sh --skip-build --target aarch64-apple-darwin --output-dir dist --sign-macos
EOF
}

detect_target() {
  rustc -vV | awk '/^host:/ { print $2; exit }'
}

default_sign_identity() {
  if [[ -n "${RMCP_MEMEX_SIGN_IDENTITY:-}" ]]; then
    printf "%s" "$RMCP_MEMEX_SIGN_IDENTITY"
    return
  fi

  if [[ -f "$HOME/.keys/signing-identity.txt" ]]; then
    head -n 1 "$HOME/.keys/signing-identity.txt"
  fi
}

sha256_manifest() {
  local out_dir="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    (
      cd "$out_dir"
      sha256sum *.tar.gz > "${BINARY_NAME}-sha256sums.txt"
    )
  elif command -v shasum >/dev/null 2>&1; then
    (
      cd "$out_dir"
      shasum -a 256 *.tar.gz | awk '{ print $1 "  " $2 }' > "${BINARY_NAME}-sha256sums.txt"
    )
  else
    echo "warning: no SHA256 tool found; skipping checksum manifest generation" >&2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --sign-macos)
      SIGN_MACOS=1
      shift
      ;;
    --sign-identity)
      SIGN_IDENTITY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  TARGET="$(detect_target)"
fi

if [[ -z "$SIGN_IDENTITY" ]]; then
  SIGN_IDENTITY="$(default_sign_identity)"
fi

cd "$REPO_ROOT"

if [[ "$SKIP_BUILD" -ne 1 ]]; then
  cargo build --locked --release --bin "$BINARY_NAME" --target "$TARGET"
fi

BINARY_PATH="$REPO_ROOT/target/$TARGET/release/$BINARY_NAME"
if [[ ! -f "$BINARY_PATH" ]]; then
  echo "error: binary not found at $BINARY_PATH" >&2
  echo "hint: run without --skip-build or verify the target triple" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

cp "$BINARY_PATH" "$STAGE_DIR/$BINARY_NAME"
cp README.md LICENSE LICENSE-APACHE install.sh "$STAGE_DIR/"
chmod +x "$STAGE_DIR/$BINARY_NAME" "$STAGE_DIR/install.sh"

if [[ "$SIGN_MACOS" -eq 1 && "$TARGET" == *apple-darwin* ]]; then
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: macOS signing requires running on Darwin" >&2
    exit 1
  fi
  if [[ -z "$SIGN_IDENTITY" ]]; then
    echo "error: no signing identity configured; set RMCP_MEMEX_SIGN_IDENTITY or ~/.keys/signing-identity.txt" >&2
    exit 1
  fi

  codesign --force --timestamp --options runtime --sign "$SIGN_IDENTITY" "$STAGE_DIR/$BINARY_NAME"
fi

ARCHIVE_NAME="${BINARY_NAME}-${TARGET}.tar.gz"
tar -C "$STAGE_DIR" -czf "$OUTPUT_DIR/$ARCHIVE_NAME" \
  "$BINARY_NAME" README.md LICENSE LICENSE-APACHE install.sh

sha256_manifest "$OUTPUT_DIR"

echo "Built release bundle: $OUTPUT_DIR/$ARCHIVE_NAME"
