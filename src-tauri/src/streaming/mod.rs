pub mod decoding;
pub mod hypothesis;
pub mod pipeline;
pub mod word_aggregation;
pub mod word_hypothesis;

pub use pipeline::StreamingPipeline;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionPatch {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub stable: bool,
}
