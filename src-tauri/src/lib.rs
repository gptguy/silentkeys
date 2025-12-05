pub mod asr;

mod app;
mod commands;
#[cfg(desktop)]
mod desktop;
mod engine;
mod recording;
pub mod settings;

pub use app::run;
