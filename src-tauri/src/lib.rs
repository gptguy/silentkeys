pub mod asr;
pub mod audio_processing;
pub mod streaming;

#[doc(hidden)]
pub mod activity;
pub mod app;
pub mod commands;
pub mod desktop;
mod dictation;
mod engine;
pub mod errors;
pub mod recording;
pub mod settings;
#[doc(hidden)]
pub mod updater;

pub use app::run;
