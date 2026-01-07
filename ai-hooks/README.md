# rmcp-memex AI Hooks

Claude Code hooks for rmcp-memex integration.

## Hooks

| Hook | Event | Purpose |
|------|-------|---------|
| `memex-startup.sh` | SessionStart | Loads project context from memex at session start |
| `memex-context.sh` | PostToolUse:Grep | Augments search results with memex context |

## Installation

```bash
cp ai-hooks/*.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/memex-*.sh
```

## Configuration

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [{ "type": "command", "command": "~/.claude/hooks/memex-startup.sh" }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Grep",
        "hooks": [{ "type": "command", "command": "~/.claude/hooks/memex-context.sh" }]
      }
    ]
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_DB_PATH` | `~/.ai-memories/lancedb` | Path to LanceDB |
| `MEMEX_NAMESPACE` | `cloud` | Default namespace to search |
| `MEMEX_LIMIT` | `3` | Max results to return |

---

Created by M&K (c)2026 The LibraxisAI Team
