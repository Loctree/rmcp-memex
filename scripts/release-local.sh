#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY_NAME="rmcp-memex"
GITHUB_REPO="${RMCP_MEMEX_GITHUB_REPO:-VetCoders/rmcp-memex}"
SIGN_IDENTITY="${RMCP_MEMEX_SIGN_IDENTITY:-}"
OUTPUT_DIR=""
NOTES_FILE=""
DRAFT=0
UPLOAD=0
TARGETS=()

usage() {
  cat <<'EOF'
Usage: scripts/release-local.sh [--target <triple>]... [--output-dir <dir>] [--upload] [--draft] [--notes-file <path>] [--sign-identity <name>]

Builds release bundles locally, signs macOS binaries with the operator identity,
and optionally uploads the finished artifacts to GitHub Releases via `gh`.

Examples:
  scripts/release-local.sh
  scripts/release-local.sh --target aarch64-apple-darwin --target x86_64-apple-darwin
  scripts/release-local.sh --target aarch64-apple-darwin --upload --draft
EOF
}

version_from_cargo() {
  awk -F'"' '/^version = / { print $2; exit }' "$REPO_ROOT/Cargo.toml"
}

host_target() {
  rustc -vV | awk '/^host:/ { print $2; exit }'
}

default_sign_identity() {
  if [[ -n "$SIGN_IDENTITY" ]]; then
    printf "%s" "$SIGN_IDENTITY"
    return
  fi

  if [[ -f "$HOME/.keys/signing-identity.txt" ]]; then
    head -n 1 "$HOME/.keys/signing-identity.txt"
  fi
}

extract_changelog_section() {
  local version="$1"

  awk -v version="$version" '
    $0 ~ "^## \\[" version "\\]" { found=1; next }
    found && /^## \[/ { exit }
    found { print }
  ' "$REPO_ROOT/CHANGELOG.md"
}

render_release_notes() {
  local version="$1"
  local tag="$2"
  local notes_path="$3"
  local changelog_section

  changelog_section="$(extract_changelog_section "$version")"

  {
    printf "## rmcp-memex %s\n\n" "$version"
    printf "Built locally and signed on the operator machine."
    if [[ -n "$SIGN_IDENTITY" ]]; then
      printf " macOS artifacts signed with `%s`." "$SIGN_IDENTITY"
    fi
    printf "\n\n### Installation\n\n"
    printf '```bash\n'
    printf 'curl -LsSf https://raw.githubusercontent.com/%s/main/install.sh | sh\n' "$GITHUB_REPO"
    printf '\n'
    printf 'curl -LO https://github.com/%s/releases/download/%s/%s-aarch64-apple-darwin.tar.gz\n' "$GITHUB_REPO" "$tag" "$BINARY_NAME"
    printf 'curl -LO https://github.com/%s/releases/download/%s/%s-sha256sums.txt\n' "$GITHUB_REPO" "$tag" "$BINARY_NAME"
    printf 'grep %s-aarch64-apple-darwin.tar.gz %s-sha256sums.txt\n' "$BINARY_NAME" "$BINARY_NAME"
    printf '```\n'

    if [[ -n "$changelog_section" ]]; then
      printf "\n### Changelog\n%s\n" "$changelog_section"
    fi
  } > "$notes_path"
}

ensure_tag_exists() {
  local tag="$1"

  git rev-parse --verify "$tag^{tag}" >/dev/null 2>&1 || \
    git rev-parse --verify "$tag^{commit}" >/dev/null 2>&1 || {
      echo "error: git tag $tag does not exist locally" >&2
      exit 1
    }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGETS+=("${2:-}")
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --upload)
      UPLOAD=1
      shift
      ;;
    --draft)
      DRAFT=1
      shift
      ;;
    --notes-file)
      NOTES_FILE="${2:-}"
      shift 2
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

cd "$REPO_ROOT"

VERSION="$(version_from_cargo)"
TAG="v${VERSION}"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/dist/$TAG"
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=("$(host_target)")
fi

SIGN_IDENTITY="$(default_sign_identity)"
mkdir -p "$OUTPUT_DIR"

for target in "${TARGETS[@]}"; do
  sign_args=()

  if [[ -z "$target" ]]; then
    echo "error: empty --target value" >&2
    exit 1
  fi

  cargo build --locked --release --bin "$BINARY_NAME" --target "$target"

  if [[ "$target" == *apple-darwin* ]]; then
    if [[ -n "$SIGN_IDENTITY" ]]; then
      sign_args=(--sign-identity "$SIGN_IDENTITY")
    fi

    "$SCRIPT_DIR/build-release-bundle.sh" \
      --skip-build \
      --target "$target" \
      --output-dir "$OUTPUT_DIR" \
      --sign-macos \
      "${sign_args[@]}"
  else
    "$SCRIPT_DIR/build-release-bundle.sh" \
      --skip-build \
      --target "$target" \
      --output-dir "$OUTPUT_DIR"
  fi
done

if [[ "$UPLOAD" -eq 1 ]]; then
  command -v gh >/dev/null 2>&1 || {
    echo "error: gh CLI is required for --upload" >&2
    exit 1
  }

  ensure_tag_exists "$TAG"

  notes_tmp=""
  if [[ -z "$NOTES_FILE" ]]; then
    notes_tmp="$(mktemp)"
    NOTES_FILE="$notes_tmp"
    render_release_notes "$VERSION" "$TAG" "$NOTES_FILE"
  fi

  release_args=()
  if [[ "$DRAFT" -eq 1 ]]; then
    release_args+=(--draft)
  fi

  mapfile -t asset_files < <(find "$OUTPUT_DIR" -maxdepth 1 \( -name '*.tar.gz' -o -name "${BINARY_NAME}-sha256sums.txt" \) | sort)
  if [[ ${#asset_files[@]} -eq 0 ]]; then
    echo "error: no release assets found in $OUTPUT_DIR" >&2
    exit 1
  fi

  if gh release view "$TAG" --repo "$GITHUB_REPO" >/dev/null 2>&1; then
    gh release upload "$TAG" "${asset_files[@]}" --repo "$GITHUB_REPO" --clobber
  else
    gh release create "$TAG" "${asset_files[@]}" \
      --repo "$GITHUB_REPO" \
      --title "rmcp-memex $VERSION" \
      --notes-file "$NOTES_FILE" \
      "${release_args[@]}"
  fi

  if [[ -n "$notes_tmp" ]]; then
    rm -f "$notes_tmp"
  fi
fi

echo "Local release artifacts ready in: $OUTPUT_DIR"
