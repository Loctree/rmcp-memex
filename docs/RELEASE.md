# Release Runbook

`rmcp-memex` is ready to release when the code path, distribution path, and
public-facing path all agree on the same product shape: one canonical binary,
one installer, one transport contract, and one story for new users.

The canonical public install path is the prebuilt GitHub Release bundle.
Source builds are for maintainers and local development, not the first-time
user experience.

## Preflight

Run these commands from the repo root before cutting a tag:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo check --no-default-features
cargo package --allow-dirty
```

If Ollama is available locally with the release embedding model already pulled,
also run:

```bash
cargo test --test transport_parity -- --ignored
```

## Versioning

1. Update `Cargo.toml` with the release version.
2. Move the relevant notes from `Unreleased` in `CHANGELOG.md` into a dated release section.
3. Commit the release prep.
4. Tag with `vX.Y.Z` or `vX.Y.Z-rc.N`.

## Publish Path

Releases are built and signed locally on the operator machine. GitHub Actions is
not the build authority for release artifacts.

Recommended operator flow:

```bash
git tag vX.Y.Z
scripts/release-local.sh --target "$(rustc -vV | awk '/^host:/ { print $2; exit }')"
scripts/release-local.sh --target aarch64-apple-darwin --target x86_64-apple-darwin --upload --draft
```

Current local release tooling:

- `scripts/release-local.sh` builds one or more targets locally, signs macOS binaries with the operator identity from `~/.keys/signing-identity.txt`, and can upload artifacts with `gh`
- `scripts/build-release-bundle.sh` packages the binary with `README`, licenses, and installer script
- the bundle generator emits `rmcp-memex-sha256sums.txt` next to the tarballs

## Smoke Tests After Publish

Verify the public release surface, not just the repo:

```bash
# Installer path
curl -LsSf https://raw.githubusercontent.com/VetCoders/rmcp-memex/main/install.sh | sh
rmcp-memex --version

# Manual artifact path
curl -LO https://github.com/VetCoders/rmcp-memex/releases/latest/download/rmcp-memex-x86_64-unknown-linux-gnu.tar.gz
curl -LO https://github.com/VetCoders/rmcp-memex/releases/latest/download/rmcp-memex-sha256sums.txt
grep rmcp-memex-x86_64-unknown-linux-gnu.tar.gz rmcp-memex-sha256sums.txt
```

Then confirm:

- `rmcp-memex serve` starts cleanly
- installer-based installs expose `rmcp_memex` as a legacy compatibility symlink
- the direct release bundle installs cleanly without requiring a local Rust toolchain build
- no release artifact, README snippet, or landing-page copy refers to `rust-memex`, `rmmx`, or `rmemex`
- `/health` responds when running with `--http-port 8997`
- the landing page publishes via `.github/workflows/pages.yml`

## Public Surface

The project now ships with a static landing page source in `docs/index.html`.
Keep it synchronized with the actual install path, transport story, and release
badges. If the product story changes, update the page in the same PR as the code.
