#!/bin/bash
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
# ============================================================================
# memex-startup.sh - Project context loader at session start
# ============================================================================
#
# PHILOSOPHY:
#   At session start, load institutional knowledge about the current project.
#   No need to carry CLAUDE.md everywhere - memex remembers everything.
#
# HOW IT WORKS:
#   1. Detects project name from git repo or directory name
#   2. Searches memex via CLI for memories about this project
#   3. Outputs context to stderr (exit 2) for Claude to see
#
# TRIGGERS:
#   - SessionStart hook
#
# ============================================================================
# Created by M&K (c)2026 The LibraxisAI Team
# ============================================================================

set -uo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
MEMEX_DB_PATH="${MEMEX_DB_PATH:-$HOME/.ai-memories/lancedb}"
MEMEX_NAMESPACE="${MEMEX_NAMESPACE:-cloud}"
MEMEX_LIMIT="${MEMEX_LIMIT:-3}"
CACHE_FILE="/tmp/memex-startup-$(echo "$PWD" | md5 2>/dev/null || echo "$PWD" | md5sum | cut -c1-8).cache"
CACHE_TTL=3600  # 1 hour

# Quick exits
command -v rmcp-memex &>/dev/null || exit 0
[[ ! -d "$MEMEX_DB_PATH" ]] && exit 0

# ============================================================================
# CACHE CHECK - only load once per project per hour
# ============================================================================
if [[ -f "$CACHE_FILE" ]]; then
    # macOS stat vs Linux stat
    if stat -f%m "$CACHE_FILE" &>/dev/null; then
        CACHE_AGE=$(($(date +%s) - $(stat -f%m "$CACHE_FILE")))
    else
        CACHE_AGE=$(($(date +%s) - $(stat -c%Y "$CACHE_FILE")))
    fi
    if [[ $CACHE_AGE -lt $CACHE_TTL ]]; then
        exit 0  # Already loaded this session
    fi
fi

# ============================================================================
# PROJECT DETECTION
# ============================================================================
PROJECT_NAME=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename 2>/dev/null || basename "$PWD")
PROJECT_NAME=$(echo "$PROJECT_NAME" | tr '-_' ' ')

# ============================================================================
# MEMEX SEARCH (via CLI)
# ============================================================================
SEARCH_RESULT=$(rmcp-memex search \
    -n "$MEMEX_NAMESPACE" \
    -q "$PROJECT_NAME" \
    -k "$MEMEX_LIMIT" \
    --db-path "$MEMEX_DB_PATH" 2>/dev/null) || exit 0

# Check if we got results
if ! echo "$SEARCH_RESULT" | grep -q '\[vec:\|score:'; then
    touch "$CACHE_FILE"
    exit 0
fi

# ============================================================================
# OUTPUT (plain text to stderr, exit 2)
# ============================================================================
CONTEXT="
🧠 MEMEX PROJECT CONTEXT: $PROJECT_NAME
Loaded from institutional memory ($(date '+%H:%M')):

$SEARCH_RESULT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use 'rmcp-memex search' for more. Loads once per session.
"

touch "$CACHE_FILE"
echo "$CONTEXT" >&2
exit 2
