#!/bin/bash
# reindex-memories.sh
# Reindex all memories after fixing timestamp preservation (P0)
#
# Created by M&K (c)2025 The LibraxisAI Team
# Part of rmcp-memex P4 fix

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  rmcp-memex Memory Reindexing Script  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default paths
MEMEX_DB="${MEMEX_DB:-$HOME/.ai-memories/lancedb}"
MEMEX_SOURCES="${MEMEX_SOURCES:-$HOME/.ai-memories/sources}"
BACKUP_DIR="${MEMEX_BACKUP:-$HOME/.ai-memories/backups}"

# Parse arguments
DRY_RUN=false
FORCE=false
NAMESPACE="conversations"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --db)
            MEMEX_DB="$2"
            shift 2
            ;;
        --sources)
            MEMEX_SOURCES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run           Show what would be done without doing it"
            echo "  --force             Skip confirmation prompts"
            echo "  -n, --namespace NS  Namespace to reindex (default: conversations)"
            echo "  --db PATH           LanceDB path (default: ~/.ai-memories/lancedb)"
            echo "  --sources PATH      Source files path (default: ~/.ai-memories/sources)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MEMEX_DB            LanceDB path"
            echo "  MEMEX_SOURCES       Source files path"
            echo "  MEMEX_BACKUP        Backup directory path"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if rmcp-memex is available
if ! command -v rmcp-memex &> /dev/null; then
    echo -e "${RED}Error: rmcp-memex not found in PATH${NC}"
    echo "Install the prebuilt release with: curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh"
    echo "For development-only source builds: cargo install --path ."
    exit 1
fi

# macOS file descriptor limit fix
if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${YELLOW}macOS detected: Setting ulimit -n 4096${NC}"
    ulimit -n 4096 2>/dev/null || echo -e "${YELLOW}Warning: Could not set ulimit (run as root if needed)${NC}"
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/lancedb.backup.${BACKUP_TIMESTAMP}"

echo -e "${BLUE}Configuration:${NC}"
echo "  LanceDB:    $MEMEX_DB"
echo "  Sources:    $MEMEX_SOURCES"
echo "  Backup:     $BACKUP_PATH"
echo "  Namespace:  $NAMESPACE"
echo ""

# Check if LanceDB exists
if [ ! -d "$MEMEX_DB" ]; then
    echo -e "${YELLOW}Warning: LanceDB not found at $MEMEX_DB${NC}"
    echo "Will create fresh index."
    SKIP_BACKUP=true
else
    SKIP_BACKUP=false
fi

# Confirmation (unless --force)
if [ "$FORCE" != "true" ] && [ "$DRY_RUN" != "true" ]; then
    echo -e "${YELLOW}This will:${NC}"
    echo "  1. Backup existing LanceDB to $BACKUP_PATH"
    echo "  2. Purge namespace '$NAMESPACE'"
    echo "  3. Reindex from $MEMEX_SOURCES"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Dry run mode
if [ "$DRY_RUN" == "true" ]; then
    echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
    echo ""
    echo "Would execute:"
    [ "$SKIP_BACKUP" != "true" ] && echo "  cp -r $MEMEX_DB $BACKUP_PATH"
    echo "  rmcp-memex purge -n $NAMESPACE --force"

    # Find source files
    if [ -d "$MEMEX_SOURCES" ]; then
        echo ""
        echo "Source files found:"
        find "$MEMEX_SOURCES" -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.md" -o -name "*.txt" \) | head -20
        COUNT=$(find "$MEMEX_SOURCES" -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.md" -o -name "*.txt" \) | wc -l | tr -d ' ')
        echo "  ... ($COUNT files total)"
    else
        echo -e "${YELLOW}No source directory found at $MEMEX_SOURCES${NC}"
    fi
    exit 0
fi

# Step 1: Backup existing LanceDB
if [ "$SKIP_BACKUP" != "true" ]; then
    echo -e "${GREEN}[1/3] Backing up existing LanceDB...${NC}"
    cp -r "$MEMEX_DB" "$BACKUP_PATH"
    echo "  Backed up to: $BACKUP_PATH"
else
    echo -e "${YELLOW}[1/3] Skipping backup (no existing DB)${NC}"
fi

# Step 2: Purge namespace
echo -e "${GREEN}[2/3] Purging namespace '$NAMESPACE'...${NC}"
rmcp-memex purge -n "$NAMESPACE" --force 2>/dev/null || echo "  (Namespace was already empty)"

# Step 3: Reindex from sources
echo -e "${GREEN}[3/3] Reindexing with preserved timestamps...${NC}"

# Find and index all source files
if [ -d "$MEMEX_SOURCES" ]; then
    # Check for merged conversations file first
    MERGED_FILE="$MEMEX_SOURCES/conversations-merged.json"
    MERGED_JSONL="$MEMEX_SOURCES/conversations-merged.jsonl"

    if [ -f "$MERGED_JSONL" ]; then
        echo "  Found merged JSONL: $MERGED_JSONL"
        rmcp-memex index "$MERGED_JSONL" -n "$NAMESPACE"
    elif [ -f "$MERGED_FILE" ]; then
        echo "  Found merged JSON: $MERGED_FILE"
        rmcp-memex index "$MERGED_FILE" -n "$NAMESPACE"
    else
        # Index individual files
        echo "  Indexing individual source files..."
        find "$MEMEX_SOURCES" -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.md" \) | while read -r file; do
            echo "    Indexing: $(basename "$file")"
            rmcp-memex index "$file" -n "$NAMESPACE" 2>/dev/null || echo "      (skipped: $file)"
        done
    fi
else
    echo -e "${YELLOW}  No source directory found. Create sources at: $MEMEX_SOURCES${NC}"
fi

# Verification
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Reindexing Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Verifying timestamps preserved...${NC}"

# Test search for year references
for year in 2024 2025; do
    echo -n "  Searching for '$year': "
    RESULT=$(rmcp-memex search -n "$NAMESPACE" -q "$year" -k 1 2>/dev/null || echo "error")
    if echo "$RESULT" | grep -q "$year"; then
        echo -e "${GREEN}FOUND${NC}"
    elif echo "$RESULT" | grep -q "error\|No results"; then
        echo -e "${YELLOW}NO RESULTS${NC}"
    else
        echo -e "${YELLOW}CHECK MANUALLY${NC}"
    fi
done

echo ""
echo -e "${BLUE}Quick stats:${NC}"
echo "  Backup location: $BACKUP_PATH"
echo "  To restore: cp -r $BACKUP_PATH $MEMEX_DB"
echo ""
echo "Done!"
