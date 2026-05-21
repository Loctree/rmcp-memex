<!-- Copyright (c) 2025-2026 VetCoders (https://vetcoders.io) -->
# Contributing

Thanks for helping improve `rmcp-memex`.

## Prerequisites

- Rust toolchain with edition `2024` support
- `cargo`

## Setup

```bash
cargo build
```

## Quality Gate

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo check --no-default-features
cargo package --allow-dirty
```

Use the default `rustfmt` style before opening a pull request.

If you have a local Ollama instance with the release embedding model pulled, also
run the ignored transport parity integration suite before changing MCP request
routing or transport behavior:

```bash
cargo test --test transport_parity -- --ignored
```

## Pull Requests

1. Fork the repository.
2. Create a topic branch from `main`.
3. Keep changes focused, add or update tests when behavior changes, and update docs if needed.
4. Open a pull request against `main`.

## License

By submitting a contribution, you agree that it will be licensed under
`MIT OR Apache-2.0`, consistent with the project license.
