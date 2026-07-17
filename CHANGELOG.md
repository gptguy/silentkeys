# Changelog

## 0.3.0 - 2026-07-16

### Added

- Nemotron 3.5 multilingual streaming speech recognition through `parakeet-rs`.
- Signed in-app updates sourced from GitHub Releases.
- Update controls and release-only automatic signed update installation with restart.
- A pinned model manifest with expected file sizes and SHA-256 digests.
- A persisted speech-language selector with deterministic English (US) as the
  default, plus system-language, automatic-detection, and model-declared codes.

### Changed

- Aligned experimental Linux release builds with ONNX Runtime's glibc 2.39
  minimum (Ubuntu 24.04 or Debian 13).
- Updated the Rust dependency graph to the latest compatible releases.
- Hardened resumable model downloads with temporary files, digest validation,
  atomic installation, and repair of corrupt cached assets.
- Reused successful model verification while the pinned revision, manifest,
  file sizes, and modification times remain unchanged, avoiding a full model
  hash on every launch.
- Reported model preparation and failure states in the recorder badge instead
  of showing a green idle state before the model is ready.
- Hardened recorder startup, shutdown, and audio-worker error propagation.
- Reported microphone ring-buffer overruns instead of accepting incomplete audio.
- Reduced streaming orchestration to one owned decoding worker that drains and
  joins before the canonical offline final transcription.
- Prevented recording and update installation from overlapping, while keeping
  dictation available during update downloads.
- Removed the previous ASR backend, decoder, token-timestamp, hotword, and word-hypothesis code.
- Reduced the ASR API to a Nemotron-only text transcription interface.
- Removed redundant direct ONNX Runtime initialization and dependency ownership.
- Removed redundant unstable/full-transcript streaming patches and their repeated cloning.
- Consolidated UI and shortcut recording flows behind one typed dictation transaction.
- Kept update exclusion active through transcription and output delivery.
- Replaced character-range transcript patches with direct append/replace updates.
- Removed idle model unloading and its startup race; the model remains resident
  to keep push-to-talk latency predictable.
- Extended the model-load wait boundary so integrity checks cannot falsely time
  out during normal disk contention.
- Made virtual typing acknowledged before transcript state advances, and
  corrected divergent partial text from the final transcript, including
  clearing stale streamed text when the final transcript is empty.
- Surfaced dictation and speech-model failures in the UI instead of logging
  them silently, including failures on the global-shortcut path.
- Added a production Content Security Policy.

### Security

- Removed transcript content from application logs.
