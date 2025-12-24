pub mod download;
mod paths;

pub use paths::{
    default_model_root, fallback_model_root, missing_model_files_for_tests, resolve_model_dir,
};

const MODEL_BASE_URL: &str =
    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";

const MODEL_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "encoder-model.onnx",
    "decoder_joint-model.onnx",
    "nemo128.onnx",
    "vocab.txt",
];

const MAX_RETRIES: usize = 3;
const RETRY_BACKOFF_SECS: u64 = 2;
