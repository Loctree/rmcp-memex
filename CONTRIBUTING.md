<!-- Copyright (c) 2025-2026 VetCoders (https://vetcoders.io) -->
# Contributing

Thanks for helping improve `rmcp-memex`.

## Prerequisites

- Rust toolchain with edition `2024` support
- `cargo`

## Setup

```bash
cargo build
cargo test
```

## Quality Gate

```bash
cargo clippy -- -D warnings
```

Use the default `rustfmt` style before opening a pull request.

## Pull Requests

1. Fork the repository.
2. Create a topic branch from `main`.
3. Keep changes focused, add or update tests when behavior changes, and update docs if needed.
4. Open a pull request against `main`.

## License

By submitting a contribution, you agree that it will be licensed under
`MIT OR Apache-2.0`, consistent with the project license.
