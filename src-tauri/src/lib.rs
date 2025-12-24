pub mod asr;
pub mod audio_processing;
pub mod error;
pub mod streaming;

pub mod vad;

pub mod app;
pub mod commands;
pub mod desktop;
mod engine;
pub mod recording;
pub mod settings;

pub use app::run;
