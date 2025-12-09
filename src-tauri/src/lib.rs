pub mod asr;
pub mod streaming;
pub mod vad;

pub mod app;
#[cfg(desktop)]
pub mod commands;
pub mod desktop;
pub mod diagnostics;
mod engine;
pub mod recording;
pub mod settings;

pub use app::run;
