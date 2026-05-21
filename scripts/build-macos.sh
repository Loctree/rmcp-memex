#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY_NAME="rmcp-memex"
APP_DIR="${RMCP_MEMEX_APP_DIR:-$HOME/.mcp-servers/rmcp-memex.app}"
VERSION="$(awk -F'"' '/^version = / { print $2; exit }' "$REPO_ROOT/Cargo.toml")"
SIGN_IDENTITY="${RMCP_MEMEX_SIGN_IDENTITY:-}"

if [[ -z "$SIGN_IDENTITY" && -f "$HOME/.keys/signing-identity.txt" ]]; then
  SIGN_IDENTITY="$(head -n 1 "$HOME/.keys/signing-identity.txt")"
fi

cd "$REPO_ROOT"

echo "Building ${BINARY_NAME} for macOS..."
cargo build --locked --release --bin "$BINARY_NAME"

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "target/release/$BINARY_NAME" "$APP_DIR/Contents/MacOS/$BINARY_NAME"
chmod +x "$APP_DIR/Contents/MacOS/$BINARY_NAME"

cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>space.div0.rmcp-memex</string>
    <key>CFBundleExecutable</key>
    <string>${BINARY_NAME}</string>
    <key>CFBundleName</key>
    <string>rmcp-memex</string>
    <key>CFBundleDisplayName</key>
    <string>rmcp-memex</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSBackgroundOnly</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticTermination</key>
    <false/>
</dict>
</plist>
EOF

if command -v codesign >/dev/null 2>&1; then
  if [[ -n "$SIGN_IDENTITY" ]]; then
    codesign --force --deep --timestamp --options runtime --sign "$SIGN_IDENTITY" "$APP_DIR"
  else
    echo "warning: no signing identity configured; falling back to ad-hoc signing" >&2
    codesign --force --deep --sign - "$APP_DIR"
  fi
fi

echo "Done! App bundle created at: $APP_DIR"
echo ""
echo "To use with an MCP host, add to config:"
echo '  "command": "'"$APP_DIR/Contents/MacOS/$BINARY_NAME"'"'
