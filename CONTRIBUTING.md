# Contributing

We welcome contributions! Please follow these steps to get started:

1. **Fork the repository**
2. **Create a feature branch** (e.g., `git checkout -b feature/my-feature`)
3. **Make your changes**
4. **Run code quality checks**:
   - Check formatting with `cargo fmt --all -- --check`
   - Run tests with `cargo test --workspace`
   - Lint with `cargo clippy --workspace --all-targets --all-features -- -D warnings`
5. **Commit your changes** with clear, descriptive commit messages.
6. **Open a Pull Request** against the `main` branch.

### Before Submitting a Pull Request
- Verify that the project builds and runs on macOS (primary platform).
- Treat Linux and Windows behavior as experimental and report what was tested.
- Ensure any new code includes appropriate documentation and tests.
- Update the changelog if your changes are user‑visible.
- Follow the existing code style and conventions.

### Code of Conduct
We expect all contributors to adhere to the repository's [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md), which is based on version 2.0.

Thank you for helping make SilentKeys better!
