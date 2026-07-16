mod model_store;
mod recognizer;

pub use model_store::{
    default_model_root, fallback_model_root, invalid_model_files_for_tests,
    model_file_matches_for_tests, resolve_model_dir, verification_receipt_matches_for_tests,
    write_verification_receipt_for_tests,
};
pub(crate) use model_store::{invalidate_model_verification, resolve_model_dir_with_progress};
pub(crate) use recognizer::STREAM_CHUNK_SAMPLES;
pub use recognizer::{
    language_candidates_for_tests, language_options_for_tests, AsrError, AsrModel,
};

pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;
