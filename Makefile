# rmcp-memex Makefile
# ============================================================================
# Service management, build, and maintenance targets
# Created by M&K (c)2026 VetCoders
# ============================================================================
#
# RAM DISK MODE (Dragon 512GB):
#   make ramdisk-up    - Create 50GB RAM disk, copy DB, start service
#   make ramdisk-down  - Sync to disk, unmount RAM disk, stop service
#   make snapshot      - Sync RAM disk to disk (backup)
#
# ============================================================================

SHELL := /bin/bash
BINARY := rmcp-memex
INSTALL_PATH := $(HOME)/.cargo/bin/$(BINARY)
LAUNCHD_PLIST := $(HOME)/Library/LaunchAgents/ai.libraxis.rmcp-memex.plist

# Disk paths
DB_PATH_DISK := $(HOME)/.ai-memories/lancedb
LOG_DIR := $(HOME)/.ai-memories/logs
HTTP_PORT := 8987

# RAM disk config (50GB = 104857600 blocks of 512 bytes)
RAMDISK_NAME := MemexRAM
RAMDISK_SIZE_GB := 50
RAMDISK_BLOCKS := $(shell echo $$(($(RAMDISK_SIZE_GB) * 1024 * 1024 * 2)))
RAMDISK_MOUNT := /Volumes/$(RAMDISK_NAME)
DB_PATH_RAM := $(RAMDISK_MOUNT)/lancedb

# Auto-detect: use RAM disk if mounted, otherwise disk
DB_PATH := $(shell if [ -d "$(RAMDISK_MOUNT)" ]; then echo "$(DB_PATH_RAM)"; else echo "$(DB_PATH_DISK)"; fi)

.PHONY: help build install start stop restart status logs health dashboard clean optimize \
        ramdisk-create ramdisk-mount ramdisk-up ramdisk-down snapshot ramdisk-status

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help
	@echo "rmcp-memex Management"
	@echo "===================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# BUILD & INSTALL
# ============================================================================

build: ## Build release binary
	cargo build --release

install: build ## Build and install to ~/.cargo/bin
	@cp ./target/release/$(BINARY) $(INSTALL_PATH)
	@echo "Installed to $(INSTALL_PATH)"

# ============================================================================
# SERVICE MANAGEMENT (launchd)
# ============================================================================

start: ## Start memex service via launchd
	@if lsof -i :$(HTTP_PORT) >/dev/null 2>&1; then \
		echo "Service already running on port $(HTTP_PORT)"; \
	else \
		launchctl bootstrap gui/$$(id -u) $(LAUNCHD_PLIST) 2>/dev/null || \
		launchctl kickstart gui/$$(id -u)/ai.libraxis.rmcp-memex 2>/dev/null || \
		$(INSTALL_PATH) serve --db-path $(DB_PATH) --http-port $(HTTP_PORT) --http-only & \
		sleep 3; \
		echo "Started memex on port $(HTTP_PORT)"; \
	fi

stop: ## Stop memex service
	@-launchctl bootout gui/$$(id -u)/ai.libraxis.rmcp-memex 2>/dev/null
	@-pkill -f "$(BINARY) serve" 2>/dev/null
	@echo "Stopped memex service"

restart: stop ## Restart memex service
	@sleep 2
	@$(MAKE) start

status: ## Show service status
	@echo "=== Launchd Status ==="
	@launchctl list 2>/dev/null | grep -E "rmcp|memex" || echo "No launchd service"
	@echo ""
	@echo "=== Port $(HTTP_PORT) ==="
	@lsof -i :$(HTTP_PORT) 2>/dev/null || echo "Port $(HTTP_PORT) not in use"
	@echo ""
	@echo "=== Health Check ==="
	@curl -s --max-time 2 http://localhost:$(HTTP_PORT)/health 2>/dev/null || echo "Server not responding"

health: ## Quick health check
	@curl -s --max-time 3 http://localhost:$(HTTP_PORT)/health | jq . 2>/dev/null || echo "Server not responding"

logs: ## Tail server logs
	@tail -f $(LOG_DIR)/rmcp-memex.stderr.log

logs-error: ## Show recent errors in logs
	@grep -iE "error|panic|fail" $(LOG_DIR)/rmcp-memex.stderr.log | tail -20

# ============================================================================
# DASHBOARD & UI
# ============================================================================

dashboard: ## Open dashboard in browser
	@open http://localhost:$(HTTP_PORT)/

# ============================================================================
# RAM DISK (Dragon 512GB - full DB in RAM)
# ============================================================================

ramdisk-create: ## Create 50GB RAM disk (requires sudo for mount)
	@if [ -d "$(RAMDISK_MOUNT)" ]; then \
		echo "RAM disk already exists at $(RAMDISK_MOUNT)"; \
	else \
		echo "Creating $(RAMDISK_SIZE_GB)GB RAM disk..."; \
		DEVICE=$$(hdiutil attach -nomount ram://$(RAMDISK_BLOCKS)); \
		diskutil erasevolume HFS+ "$(RAMDISK_NAME)" $$DEVICE; \
		echo "RAM disk created at $(RAMDISK_MOUNT)"; \
	fi

ramdisk-load: ## Copy LanceDB from disk to RAM disk
	@if [ ! -d "$(RAMDISK_MOUNT)" ]; then \
		echo "ERROR: RAM disk not mounted. Run 'make ramdisk-create' first."; \
		exit 1; \
	fi
	@echo "Copying LanceDB to RAM disk..."
	@mkdir -p "$(DB_PATH_RAM)"
	@if [ -d "$(DB_PATH_DISK)" ]; then \
		rsync -av --progress "$(DB_PATH_DISK)/" "$(DB_PATH_RAM)/"; \
		echo "Copied $$(du -sh $(DB_PATH_RAM) | cut -f1) to RAM disk"; \
	else \
		echo "No existing DB at $(DB_PATH_DISK), starting fresh"; \
	fi

ramdisk-up: ramdisk-create ramdisk-load ## Create RAM disk, load DB, start service
	@$(MAKE) stop 2>/dev/null || true
	@sleep 1
	@echo "Starting memex with RAM disk..."
	@$(INSTALL_PATH) serve --db-path "$(DB_PATH_RAM)" --http-port $(HTTP_PORT) --http-only &
	@sleep 3
	@echo ""
	@echo "=== MEMEX RAM MODE ==="
	@echo "RAM disk:  $(RAMDISK_MOUNT) ($(RAMDISK_SIZE_GB)GB)"
	@echo "DB path:   $(DB_PATH_RAM)"
	@echo "Dashboard: http://localhost:$(HTTP_PORT)/"
	@echo ""
	@echo "IMPORTANT: Run 'make snapshot' periodically to save to disk!"

snapshot: ## Sync RAM disk to disk (backup)
	@if [ ! -d "$(RAMDISK_MOUNT)" ]; then \
		echo "No RAM disk mounted, nothing to snapshot"; \
		exit 0; \
	fi
	@echo "Syncing RAM disk to disk..."
	@rsync -av --delete "$(DB_PATH_RAM)/" "$(DB_PATH_DISK)/"
	@echo "Snapshot saved to $(DB_PATH_DISK)"
	@echo "Size: $$(du -sh $(DB_PATH_DISK) | cut -f1)"

ramdisk-down: snapshot ## Sync to disk and unmount RAM disk
	@$(MAKE) stop 2>/dev/null || true
	@if [ -d "$(RAMDISK_MOUNT)" ]; then \
		echo "Unmounting RAM disk..."; \
		hdiutil detach "$(RAMDISK_MOUNT)"; \
		echo "RAM disk unmounted. Data saved to $(DB_PATH_DISK)"; \
	else \
		echo "No RAM disk to unmount"; \
	fi

ramdisk-status: ## Show RAM disk status
	@echo "=== RAM Disk Status ==="
	@if [ -d "$(RAMDISK_MOUNT)" ]; then \
		echo "Status: MOUNTED"; \
		echo "Mount:  $(RAMDISK_MOUNT)"; \
		echo "Size:   $$(df -h $(RAMDISK_MOUNT) | tail -1 | awk '{print $$2}')"; \
		echo "Used:   $$(df -h $(RAMDISK_MOUNT) | tail -1 | awk '{print $$3}')"; \
		echo "Free:   $$(df -h $(RAMDISK_MOUNT) | tail -1 | awk '{print $$4}')"; \
		echo "DB:     $$(du -sh $(DB_PATH_RAM) 2>/dev/null | cut -f1 || echo 'empty')"; \
	else \
		echo "Status: NOT MOUNTED"; \
		echo "Run 'make ramdisk-up' to enable RAM mode"; \
	fi
	@echo ""
	@echo "=== Disk Backup ==="
	@echo "Path: $(DB_PATH_DISK)"
	@echo "Size: $$(du -sh $(DB_PATH_DISK) 2>/dev/null | cut -f1 || echo 'empty')"

auto-snapshot: ## Start background snapshot daemon (every 15 min)
	@if [ ! -d "$(RAMDISK_MOUNT)" ]; then \
		echo "No RAM disk mounted, auto-snapshot not needed"; \
		exit 0; \
	fi
	@echo "Starting auto-snapshot daemon (every 15 minutes)..."
	@(while true; do \
		sleep 900; \
		if [ -d "$(RAMDISK_MOUNT)" ]; then \
			echo "$$(date): Auto-snapshot starting..." >> $(LOG_DIR)/snapshot.log; \
			rsync -a --delete "$(DB_PATH_RAM)/" "$(DB_PATH_DISK)/" 2>> $(LOG_DIR)/snapshot.log; \
			echo "$$(date): Auto-snapshot complete" >> $(LOG_DIR)/snapshot.log; \
		fi; \
	done) &
	@echo "Auto-snapshot daemon started (PID: $$!)"
	@echo "Logs: $(LOG_DIR)/snapshot.log"

# ============================================================================
# DATABASE MAINTENANCE
# ============================================================================

optimize: ## Optimize LanceDB (compact + cleanup)
	$(INSTALL_PATH) optimize --db-path $(DB_PATH)

stats: ## Show database stats
	$(INSTALL_PATH) stats --db-path $(DB_PATH)

namespaces: ## List all namespaces
	$(INSTALL_PATH) namespaces --stats --db-path $(DB_PATH)

# ============================================================================
# DEVELOPMENT
# ============================================================================

dev: ## Run in development mode (foreground, debug logs)
	RUST_LOG=debug cargo run -- serve --db-path $(DB_PATH) --http-port $(HTTP_PORT) --http-only

test: ## Run tests
	cargo test

check: ## Run cargo check
	cargo check

clean: ## Clean build artifacts
	cargo clean

# ============================================================================
# QUICK COMMANDS
# ============================================================================

up: install restart ## Build, install, and restart service
	@echo "Service updated and restarted"

down: stop ## Alias for stop

ps: status ## Alias for status

ram: ramdisk-up auto-snapshot ## Full RAM mode: create disk, load DB, start service + auto-snapshot
	@echo ""
	@echo "=== MEMEX FULL RAM MODE ACTIVE ==="
	@echo "All 25GB+ in RAM. Auto-snapshot every 15 min."
	@echo "Run 'make ramdisk-down' before shutdown!"
