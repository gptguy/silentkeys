<div align="center">

# 🗝️ SilentKeys

Real-time, fully offline dictation for macOS.

Built with Rust, Tauri and Parakeet.  
Fast, private and open source. No cloud, no telemetry.

[![Platform](https://img.shields.io/badge/platform-macOS-blue.svg)](https://www.apple.com/macos/)
[![Built with Rust](https://img.shields.io/badge/built_with-Rust-orange.svg)](https://www.rust-lang.org/)
[![UI - Tauri v2](https://img.shields.io/badge/ui-Tauri_v2-FFC131.svg)](https://v2.tauri.app/)
[![Frontend - Leptos](https://img.shields.io/badge/frontend-Leptos-red.svg)](https://leptos.dev/)
[![Speech Engine - Parakeet](https://img.shields.io/badge/speech_engine-Parakeet-6f42c1.svg)](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
[![License - MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/gptguy/silentkeys?style=social)](https://github.com/gptguy/silentkeys/stargazers)

<p align="center">
  <img src="docs/app-screenshot.png" width="700" alt="SilentKeys Screenshot">
</p>

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Why SilentKeys?](#-why-silentkeys)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Roadmap](#️-roadmap)
- [Who Is This For?](#-who-is-this-for)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 💡 About

SilentKeys is an open-source macOS dictation application that performs all audio capture and transcription locally on your machine. It is designed for everyday use, but remains under active development.

> ✅ **Status: Beta**  
> Stable for everyday dictation on Apple Silicon Macs. Some features and configuration options are still evolving.

---

## ✨ Features

- 🏎 **Real-time dictation**: Types into any application as you speak.
- 🔒 **Offline only**: Audio and text remain on your device.
- 🧠 **High-quality models**: Parakeet running on ONNX Runtime, optimized for Apple Silicon.
- 🖥 **macOS-first**: Global push-to-talk, streaming output, minimal latency.
- ⚙️ **Rust core**: Optimized for low latency and low overhead.
- 🪶 **Lightweight UI**: Tauri v2 and Leptos; no Electron.
-  **No telemetry**: No analytics and no background network calls.
- 🆓 **Open source**: MIT-licensed.

---

## 🚀 Why SilentKeys?

Most dictation tools require trade-offs around privacy, transparency, flexibility, or stack complexity. SilentKeys is designed to minimize those compromises:

- **Local-only processing**: all audio capture and transcription happen on your Mac; audio and transcripts never leave your machine.
- **Transparency and auditability**: the full codebase is open, making it straightforward to review how audio is handled and how the app interacts with the network and filesystem.
- **Workflow-friendly streaming**: text streams into any application in real time, so you can use your existing editors, email clients, and tools without plugins or extensions.
- **Lean, modern stack**: built on Rust, Tauri, Leptos, and ONNX Runtime rather than heavier Python or Electron stacks, keeping the app relatively small, predictable, and maintainable.

---

## 🧩 How It Works

The speech engine uses Parakeet models running via ONNX Runtime within a Rust core. At a high level:

1. A small Rust daemon listens for a global push-to-talk shortcut.
2. When active, audio is captured from the system microphone with low-latency buffers.
3. Audio chunks are fed through a Parakeet model via ONNX Runtime.
4. Partial transcripts are streamed back to the Tauri frontend.
5. The frontend drives "virtual typing" into the currently focused application using macOS Accessibility APIs.

---

## ⚡ Quick Start

SilentKeys currently supports:

- **macOS 14.0+ (Sonoma)** (Recommended) or macOS 13 (Ventura)
- **Apple Silicon** (M1/M2/M3/M4 recommended)
- Intel Macs: **Experimental** / Not actively tested

You will also need a working Rust toolchain installed via `rustup`.

```bash
# Clone the repository
git clone https://github.com/gptguy/silentkeys
cd silentkeys

# Run in development mode
cargo tauri dev
```

Once that is working, you can build a release bundle:

```bash
cargo tauri build
```

The bundled app will end up under `src-tauri/target/release/bundle/`.

---

## 🎙 Usage

Once SilentKeys is running:

- Press **⌥Z** (Option+Z, default) to start dictation.
- Speak normally. Text will stream into the currently focused app.
- Press the shortcut again (or stop speaking) to end the dictation session.

You can configure:

- **Global hotkey**: under *Preferences → Shortcuts*
- **Model**: under *Preferences → Speech Engine*
- **Streaming behavior**: stream character-by-character or sentence-by-sentence

---

## 📥 Installation

### Download

An official release channel is not yet available. When it is ready, signed DMG builds and a Homebrew tap will be linked here.

### Build from Source

#### Prerequisites

To build SilentKeys from source you need:

- macOS with Xcode command line tools.
- [Rust](https://www.rust-lang.org/tools/install) (latest stable via `rustup`).
- [Trunk](https://trunkrs.dev/) and WASM target:
  ```bash
  rustup target add wasm32-unknown-unknown
  cargo install trunk
  ```
- [Tauri CLI](https://v2.tauri.app/start/prerequisites/) version 2:
  ```bash
  cargo install tauri-cli --locked
  ```

#### Build Steps

```bash
# 1. Clone the repository
git clone https://github.com/gptguy/silentkeys
cd silentkeys

# 2. Make sure prerequisites are installed (see above)

# 3. Build the application
cargo tauri build

# 4. The built app will be located at
# src-tauri/target/release/bundle/macos/SilentKeys.app
```

#### Development Mode

During development it is usually enough to run:

```bash
cargo tauri dev
```

---

## 🛠️ Roadmap

**Core**

- [ ] Configurable global hotkeys and per-app profiles
- [ ] Basic settings UI for model selection, latency vs accuracy, and streaming style
- [ ] Error handling and crash reporting (opt-in, local-only)

**Performance**

- [ ] Benchmarking Parakeet vs alternative ASR models
- [ ] Smarter batching and chunking for long sessions
- [ ] Better use of ONNX Runtime EPs (CoreML, Metal, etc.)

**Distribution**

- [ ] Signed, notarized DMG releases
- [ ] Homebrew tap: `brew install --cask silentkeys`
- [ ] Automatic updates for release builds

**Platforms**

- [ ] Linux and Windows builds (best-effort, not yet prioritized)
---

## 🔐 Privacy

SilentKeys is designed to be private by default:

- **No cloud services** – all audio and transcripts are processed locally.
- **No analytics** – no metrics, crash reports, or usage data are sent anywhere.
- **No auto-updaters** – release builds do not phone home.
- **Offline first** – network access is **not** required to run the app.

## 🧑‍🤝‍🧑 Who Is This For?

SilentKeys is intended for:

- Privacy-conscious professionals in fields such as law, healthcare, or journalism.
- Writers and creators who prefer dictating directly into an editor.
- Accessibility-focused users who need a desktop dictation option that does not depend on a cloud account.

---

## 🤝 Contributing

Contributions of all sizes are welcome. Useful areas to help with include:

- Rust performance tuning and profiling.
- Tauri and macOS UX polish.
- Experiments around ONNX Runtime performance and different hardware backends.
- Testing on Intel Macs, Linux and Windows.
- Documentation and localization.

### Getting Started

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Commit your changes (`git commit -m 'Describe my change'`).
4. Push to your fork (`git push origin feature/my-change`).
5. Open a pull request.

Before opening a PR, please:

- Run `cargo fmt` and `cargo clippy` in `src-tauri/`.
- Ensure `cargo test` passes for any affected crates.
- For UI changes, include a brief description or screenshot in the PR.

---

## 📣 Why Open Source?

Desktop dictation, by definition, processes highly sensitive content. Keeping the code open makes it easier to audit how the application behaves, especially around network access and data handling.

SilentKeys is released under the MIT license to make it straightforward to fork, package, and integrate into other workflows or products.

---

## 🛡️ License

Licensed under the [MIT License](LICENSE).

You are free to use, modify, integrate and distribute according to that license.

---

## ⭐ Acknowledgments

SilentKeys builds on the work of:

- [Parakeet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) and the wider speech community.
- [ONNX Runtime](https://onnxruntime.ai/) for high performance inference.
- [Tauri](https://tauri.app/), [Rust](https://www.rust-lang.org/), [Leptos](https://leptos.dev/) and the WebView ecosystem.

---

<div align="center">

## 🙌 SilentKeys

Dictation that stays on your machine.  
Fast, offline and open.

[⬆ Back to Top](#-silentkeys)

</div>
