#!/bin/bash
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
# ============================================================================
# memex-context.sh - Memory augmentation for Claude Code
# ============================================================================
#
# PHILOSOPHY:
#   Grep/Read find CODE → "what's in this file?"
#   Memex adds MEMORY  → "what do we know about this from past conversations?"
#
#   Together: CODE + INSTITUTIONAL KNOWLEDGE for deeper understanding.
#
# HOW IT WORKS:
#   This is a PostToolUse hook for Claude Code's Grep tool.
#   1. Runs AFTER grep executes
#   2. Searches memex via CLI for related memories
#   3. Outputs context to stderr with exit code 2
#   4. Claude sees BOTH grep results AND historical context
#
# INSTALLATION:
#   1. Copy this file to ~/.claude/hooks/memex-context.sh
#   2. chmod +x ~/.claude/hooks/memex-context.sh
#   3. Add to ~/.claude/settings.json hooks section
#
# CONFIGURATION (via environment):
#   MEMEX_AUGMENT=0        - Disable all augmentation
#   MEMEX_DB_PATH          - Path to lancedb (default: ~/.ai-memories/lancedb)
#   MEMEX_NAMESPACE=cloud  - Namespace to search
#   MEMEX_LIMIT=3          - Max memories to return
#
# REQUIRES:
#   - rmcp-memex CLI installed (auto-detected in ~/.cargo/bin, /opt/homebrew/bin)
#   - jq for JSON parsing (optional)
#
# ============================================================================
# Created by M&K (c)2026 The LibraxisAI Team
# ============================================================================

set -uo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
MEMEX_AUGMENT="${MEMEX_AUGMENT:-1}"
MEMEX_DB_PATH="${MEMEX_DB_PATH:-$HOME/.ai-memories/lancedb}"
MEMEX_NAMESPACE="${MEMEX_NAMESPACE:-cloud}"
MEMEX_LIMIT="${MEMEX_LIMIT:-3}"

# Quick exits
[[ "$MEMEX_AUGMENT" == "0" ]] && exit 0
command -v rmcp-memex &>/dev/null || exit 0
[[ ! -d "$MEMEX_DB_PATH" ]] && exit 0

# ============================================================================
# INPUT PARSING
# ============================================================================
INPUT=$(cat)

# Extract tool_input from PostToolUse JSON structure
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input // empty' 2>/dev/null)
[[ -z "$TOOL_INPUT" ]] && TOOL_INPUT="$INPUT"

# Extract pattern from Grep tool
PATTERN=$(echo "$TOOL_INPUT" | jq -r '.pattern // empty' 2>/dev/null)

# Also try to extract file path for Read tool context
FILE_PATH=$(echo "$TOOL_INPUT" | jq -r '.file_path // empty' 2>/dev/null)

# Build search query from available context
QUERY=""
if [[ -n "$PATTERN" ]]; then
    QUERY="$PATTERN"
elif [[ -n "$FILE_PATH" ]]; then
    QUERY=$(basename "$FILE_PATH" | sed 's/\.[^.]*$//')
fi

# Nothing to search without a query
[[ -z "$QUERY" ]] && exit 0
[[ ${#QUERY} -lt 3 ]] && exit 0

# ============================================================================
# MEMEX SEARCH (via CLI)
# ============================================================================
SEARCH_RESULT=$(rmcp-memex search \
    -n "$MEMEX_NAMESPACE" \
    -q "$QUERY" \
    -k "$MEMEX_LIMIT" \
    --db-path "$MEMEX_DB_PATH" 2>/dev/null) || exit 0

# Check if we got results (look for score indicators)
if ! echo "$SEARCH_RESULT" | grep -q '\[vec:\|score:'; then
    exit 0
fi

# ============================================================================
# FORMAT OUTPUT
# ============================================================================
CONTEXT="
🧠 MEMEX MEMORY CONTEXT
Related memories about '$QUERY' from namespace '$MEMEX_NAMESPACE':

$SEARCH_RESULT
"

# Output to stderr with exit code 2 (shows to model immediately)
echo "$CONTEXT"
echo "$CONTEXT" >&2
exit 2
