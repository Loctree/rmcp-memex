#!/usr/bin/env bash
# rmcp-memex installer for prebuilt GitHub Release bundles
# curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
# or with a specific release tag:
# RMCP_MEMEX_VERSION=v0.5.1 curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh

set -euo pipefail

VERSION="${RMCP_MEMEX_VERSION:-latest}"
INSTALL_DIR="${RMCP_MEMEX_INSTALL_DIR:-$HOME/.cargo/bin}"
GITHUB_REPO="VetCoders/rmcp-memex"
BINARY_NAME="rmcp-memex"
CHECKSUM_FILE="rmcp-memex-sha256sums.txt"
COMPAT_ALIASES=("rmcp_memex")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    printf "${BLUE}==>${NC} %s\n" "$1"
}

success() {
    printf "${GREEN}==>${NC} %s\n" "$1"
}

warn() {
    printf "${YELLOW}warning:${NC} %s\n" "$1"
}

error() {
    printf "${RED}error:${NC} %s\n" "$1" >&2
    exit 1
}

detect_platform() {
    local os arch

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) error "Unsupported architecture: $arch" ;;
    esac

    case "$os-$arch" in
        darwin-aarch64) echo "aarch64-apple-darwin" ;;
        darwin-x86_64) echo "x86_64-apple-darwin" ;;
        linux-x86_64) echo "x86_64-unknown-linux-gnu" ;;
        linux-aarch64) echo "aarch64-unknown-linux-gnu" ;;
        *) error "Unsupported platform: $os-$arch" ;;
    esac
}

download_file() {
    local url="$1"
    local destination="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$url" -o "$destination"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$destination"
    else
        error "Neither curl nor wget is available."
    fi
}

get_latest_version() {
    local api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"
    local payload

    if ! payload=$(download_to_stdout "$api_url" 2>/dev/null); then
        echo ""
        return
    fi

    printf "%s" "$payload" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/' || true
}

download_to_stdout() {
    local url="$1"

    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$url"
    else
        return 1
    fi
}

release_asset_url() {
    local version="$1"
    local asset="$2"

    if [ "$version" = "latest" ]; then
        printf "https://github.com/%s/releases/latest/download/%s" "$GITHUB_REPO" "$asset"
    else
        printf "https://github.com/%s/releases/download/%s/%s" "$GITHUB_REPO" "$version" "$asset"
    fi
}

sha256_file() {
    local file="$1"

    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    else
        return 1
    fi
}

verify_checksum() {
    local archive="$1"
    local manifest="$2"
    local artifact_name="$3"
    local expected actual

    expected=$(awk -v name="$artifact_name" '$2 == name { print $1 }' "$manifest")
    if [ -z "$expected" ]; then
        error "Checksum manifest does not contain an entry for $artifact_name"
    fi

    if ! actual=$(sha256_file "$archive"); then
        warn "No SHA256 tool found; skipping checksum verification"
        return 0
    fi

    if [ "$expected" != "$actual" ]; then
        error "Checksum mismatch for $artifact_name"
    fi

    success "Checksum verified"
}

is_in_path() {
    case ":$PATH:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

install_compat_aliases() {
    local binary_path="$1"
    local alias_name

    for alias_name in "${COMPAT_ALIASES[@]}"; do
        ln -sfn "$binary_path" "$INSTALL_DIR/$alias_name"
    done
}

main() {
    local target version archive_name checksum_url archive_url temp_dir extracted_binary checksum_path installed_version

    printf "\nrmcp-memex installer\n"
    printf "Shared RAG memory for MCP agents.\n\n"

    info "Detecting platform"
    target=$(detect_platform)
    success "Platform: $target"

    version="$VERSION"
    if [ "$version" = "latest" ]; then
        info "Resolving latest GitHub Release"
        version=$(get_latest_version)
        if [ -z "$version" ]; then
            warn "Could not resolve the latest release tag; falling back to latest/download"
            version="latest"
        else
            success "Latest release: $version"
        fi
    fi

    archive_name="${BINARY_NAME}-${target}.tar.gz"
    archive_url=$(release_asset_url "$version" "$archive_name")
    checksum_url=$(release_asset_url "$version" "$CHECKSUM_FILE")

    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    info "Downloading ${archive_name}"
    download_file "$archive_url" "$temp_dir/$archive_name" || error "Failed to download $archive_url"

    checksum_path="$temp_dir/$CHECKSUM_FILE"
    if download_file "$checksum_url" "$checksum_path" 2>/dev/null; then
        info "Verifying release checksum"
        verify_checksum "$temp_dir/$archive_name" "$checksum_path" "$archive_name"
    else
        warn "Checksum manifest unavailable; continuing without verification"
    fi

    info "Extracting release archive"
    mkdir -p "$temp_dir/extract"
    tar xzf "$temp_dir/$archive_name" -C "$temp_dir/extract"

    extracted_binary=$(find "$temp_dir/extract" -name "$BINARY_NAME" -type f | head -1)
    if [ -z "$extracted_binary" ]; then
        error "Binary $BINARY_NAME not found inside $archive_name"
    fi

    mkdir -p "$INSTALL_DIR"
    cp "$extracted_binary" "$INSTALL_DIR/$BINARY_NAME"
    chmod +x "$INSTALL_DIR/$BINARY_NAME"
    install_compat_aliases "$INSTALL_DIR/$BINARY_NAME"
    success "Installed ${BINARY_NAME} to $INSTALL_DIR/$BINARY_NAME"
    info "Legacy compatibility alias: $INSTALL_DIR/rmcp_memex"

    installed_version=$("$INSTALL_DIR/$BINARY_NAME" --version 2>/dev/null || echo "unknown")
    info "Installed version: $installed_version"

    if ! is_in_path "$INSTALL_DIR"; then
        warn "Install directory is not in PATH"
        printf 'Add this to your shell profile:\n\n'
        printf '  export PATH="%s:$PATH"\n\n' "$INSTALL_DIR"
    fi

    printf "Next steps\n"
    printf "1. Start the MCP server:\n"
    printf "   %s serve\n\n" "$BINARY_NAME"
    printf "2. Or start the shared HTTP/SSE daemon:\n"
    printf "   %s serve --http-port 6660 --http-only\n\n" "$BINARY_NAME"
    printf "3. Example MCP host config:\n\n"
    cat <<JSON
{"mcpServers":{"rmcp-memex":{"command":"$INSTALL_DIR/$BINARY_NAME","args":["serve"]}}}
JSON
    printf "\n"

    if [ -t 0 ] && [ -t 1 ]; then
        printf "Run the configuration wizard now? [y/N] "
        read -r answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            exec "$INSTALL_DIR/$BINARY_NAME" wizard
        fi
    fi
}

main "$@"
