pub mod decoding;
pub mod pipeline;
pub mod word_hypothesis;
pub mod words;

pub use pipeline::StreamingPipeline;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionPatch {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub stable: bool,
}
