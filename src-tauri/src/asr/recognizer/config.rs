use ndarray::ShapeError;
use std::collections::HashSet;

use super::model::AsrModel;

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub timestamps: Vec<f32>,
    pub tokens: Vec<String>,
}

impl Transcript {
    pub fn offset_timestamps(&mut self, offset_sec: f32) {
        if offset_sec.abs() < f32::EPSILON {
            return;
        }
        for timestamp in &mut self.timestamps {
            *timestamp += offset_sec;
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ShapeError),
    #[error("Model input not found: {0}")]
    InputNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Failed to get tensor shape for input: {0}")]
    TensorShape(String),
    #[error("Unsupported sample rate {0} Hz, expected 16000 Hz")]
    SampleRate(u32),
    #[error("Audio decode error: {0}")]
    Audio(String),
    #[error("Model snapshot not found under {0}")]
    SnapshotNotFound(String),
    #[error("Model download failed: {0}")]
    Download(String),
}

impl AsrError {
    pub fn user_message(&self) -> &'static str {
        match self {
            Self::Download(_) => {
                "Could not download the speech model. Check your internet connection and try again."
            }
            Self::SnapshotNotFound(_) => {
                "Speech model files are missing or corrupted. Click Retry to download them again."
            }
            Self::Audio(_) | Self::SampleRate(_) => {
                "Could not decode the recording. Please try recording again."
            }
            Self::Ort(_)
            | Self::InputNotFound(_)
            | Self::OutputNotFound(_)
            | Self::TensorShape(_)
            | Self::Shape(_) => {
                "The speech engine failed to run. Try restarting the app or downloading the model again."
            }
            Self::Io(_) => {
                "The app could not read or write its local files. Check disk space and permissions."
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub temperature: f32,
    pub max_tokens_per_step: usize,
    pub beam_width: usize,
    pub min_blank_margin: f32,
    pub hotword_boost: f32,
    pub hotwords: Vec<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            max_tokens_per_step: 10,
            beam_width: 1,
            min_blank_margin: 0.0,
            hotword_boost: 0.0,
            hotwords: Vec::new(),
        }
    }
}

impl InferenceConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env_overrides("ASR_");
        config
    }

    pub fn streaming_from_env() -> Self {
        let mut config = Self::streaming_defaults();
        config.apply_env_overrides("ASR_");
        config.apply_env_overrides("STREAM_ASR_");
        config
    }

    fn streaming_defaults() -> Self {
        Self {
            max_tokens_per_step: 8,
            ..Default::default()
        }
    }

    fn apply_env_overrides(&mut self, prefix: &str) {
        let parse_env = |suffix: &str| std::env::var(format!("{prefix}{suffix}")).ok();
        let apply = |suffix: &str, target: &mut f32| {
            if let Some(v) = parse_env(suffix).and_then(|s| s.parse().ok()) {
                *target = v;
            }
        };

        apply("TEMPERATURE", &mut self.temperature);
        apply("MIN_BLANK_MARGIN", &mut self.min_blank_margin);
        apply("HOTWORD_BOOST", &mut self.hotword_boost);

        if let Some(v) = parse_env("MAX_TOKENS_PER_STEP").and_then(|s| s.parse().ok()) {
            self.max_tokens_per_step = v;
        }
        if let Some(v) = parse_env("BEAM_WIDTH").and_then(|s| s.parse::<usize>().ok()) {
            self.beam_width = v.max(1);
        }
        if let Some(v) = parse_env("HOTWORDS") {
            self.hotwords = v
                .split(',')
                .map(|w| w.trim())
                .filter(|w| !w.is_empty())
                .map(String::from)
                .collect();
        }
    }
}

pub fn build_hotword_mask(model: &AsrModel, config: &InferenceConfig) -> Option<Vec<bool>> {
    if config.hotword_boost <= 0.0 || config.hotwords.is_empty() {
        return None;
    }

    let hotwords: HashSet<_> = config
        .hotwords
        .iter()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();

    if hotwords.is_empty() {
        return None;
    }

    let mut matched = 0;
    let mask: Vec<bool> = model
        .vocab
        .iter()
        .map(|token| {
            let is_match = hotwords.contains(&token.trim().to_lowercase());
            if is_match {
                matched += 1;
            }
            is_match
        })
        .collect();

    if matched == 0 {
        None
    } else {
        Some(mask)
    }
}
