pub mod asr;
pub mod streaming;
pub mod vad;

mod app;
mod commands;
#[cfg(desktop)]
pub mod desktop;
mod engine;
pub mod recording;
pub mod settings;

pub use app::run;
