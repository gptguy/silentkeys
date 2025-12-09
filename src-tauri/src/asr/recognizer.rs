use std::fs;
use std::path::Path;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayD, ArrayViewD, IxDyn};
use num_cpus::get_physical;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::asr::decoder::DecoderWorkspace;

const THREAD_ENV: &str = "ORT_THREADS";

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub timestamps: Vec<f32>,
    pub tokens: Vec<String>,
}

#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("ORT error")]
    Ort(#[from] ort::Error),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error")]
    Shape(#[from] ndarray::ShapeError),
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

fn resolve_thread_count() -> usize {
    if let Ok(value) = std::env::var(THREAD_ENV) {
        match value.parse::<usize>() {
            Ok(parsed) => {
                log::info!("Using ORT_THREADS override: {} threads", parsed);
                return parsed;
            }
            Err(err) => {
                log::warn!("Ignoring invalid ORT_THREADS value '{}': {}", value, err);
            }
        }
    }

    let physical = get_physical();
    log::info!(
        "ORT_THREADS not set; defaulting to {} physical cores",
        physical
    );
    physical
}

pub struct AsrModel {
    pub(super) encoder: Session,
    pub(super) decoder_joint: Session,
    pub(super) preprocessor: Session,

    pub(super) vocab: Vec<String>,
    pub(super) blank_idx: i32,
    pub(super) vocab_size: usize,
    pub(super) decoder_workspace: DecoderWorkspace,
}

impl Drop for AsrModel {
    fn drop(&mut self) {
        log::debug!("Dropping ASR model with {} vocab tokens", self.vocab.len());
    }
}

impl AsrModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, quantized: bool) -> Result<Self, AsrError> {
        let start = Instant::now();
        let threads = resolve_thread_count();
        let encoder = Self::init_session(&model_dir, "encoder-model", threads, quantized)?;
        let decoder_joint =
            Self::init_session(&model_dir, "decoder_joint-model", threads, quantized)?;
        let preprocessor = Self::init_session(&model_dir, "nemo128", threads, false)?;
        let decoder_workspace = DecoderWorkspace::new(&decoder_joint)?;

        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;
        let vocab_size = vocab.len();

        log::info!(
            "Loaded vocabulary with {} tokens, blank_idx={}",
            vocab_size,
            blank_idx
        );

        log::info!(
            "ASR model initialized (quantized={}) in {:?}",
            quantized,
            start.elapsed()
        );

        Ok(Self {
            encoder,
            decoder_joint,
            preprocessor,
            vocab,
            blank_idx,
            vocab_size,
            decoder_workspace,
        })
    }

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        model_name: &str,
        intra_threads: usize,
        try_quantized: bool,
    ) -> Result<Session, AsrError> {
        let providers = vec![CPUExecutionProvider::default().build()];

        let model_filename = if try_quantized {
            let quantized_name = format!("{}.int8.onnx", model_name);
            let quantized_path = model_dir.as_ref().join(&quantized_name);
            if quantized_path.exists() {
                log::info!("Loading quantized model from {}...", quantized_name);
                quantized_name
            } else {
                let regular_name = format!("{}.onnx", model_name);
                log::info!(
                    "Quantized model not found, loading regular model from {}...",
                    regular_name
                );
                regular_name
            }
        } else {
            let regular_name = format!("{}.onnx", model_name);
            log::info!("Loading model from {}...", regular_name);
            regular_name
        };

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?;

        builder = builder
            .with_intra_threads(intra_threads)?
            .with_inter_threads(intra_threads)?;

        let session = builder.commit_from_file(model_dir.as_ref().join(&model_filename))?;

        for input in &session.inputs {
            log::info!(
                "Model '{}' input: name={}, type={:?}",
                model_filename,
                input.name,
                input.input_type
            );
        }

        Ok(session)
    }

    fn load_vocab<P: AsRef<Path>>(model_dir: P) -> Result<(Vec<String>, i32), AsrError> {
        let vocab_path = model_dir.as_ref().join("vocab.txt");
        let content = fs::read_to_string(vocab_path)?;

        let mut entries: Vec<(String, usize)> = Vec::new();
        let mut blank_idx: Option<usize> = None;

        for line in content.lines() {
            let mut parts = line.split_whitespace();
            let token = match parts.next() {
                Some(token) => token,
                None => continue,
            };
            let id = match parts.next().and_then(|id| id.parse::<usize>().ok()) {
                Some(id) => id,
                None => continue,
            };

            if token == "<blk>" {
                blank_idx = Some(id);
            }

            entries.push((token.replace('\u{2581}', " "), id));
        }

        let max_id = entries.iter().map(|(_, id)| *id).max().unwrap_or(0);
        let mut vocab = vec![String::new(); max_id + 1];
        for (token, id) in entries {
            if let Some(slot) = vocab.get_mut(id) {
                *slot = token;
            }
        }

        let blank_idx = blank_idx.ok_or_else(|| {
            AsrError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing <blk> token in vocabulary",
            ))
        })? as i32;

        Ok((vocab, blank_idx))
    }

    fn preprocess(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_lens: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), AsrError> {
        log::trace!("Running preprocessor inference...");
        let inputs = inputs![
            "waveforms" => TensorRef::from_array_view(waveforms.view())?,
            "waveforms_lens" => TensorRef::from_array_view(waveforms_lens.view())?,
        ];
        let start = Instant::now();
        let outputs = self.preprocessor.run(inputs)?;
        log::debug!("Preprocessor inference completed in {:?}", start.elapsed());

        let features = outputs
            .get("features")
            .ok_or_else(|| AsrError::OutputNotFound("features".to_string()))?
            .try_extract_array()?;
        let features_lens = outputs
            .get("features_lens")
            .ok_or_else(|| AsrError::OutputNotFound("features_lens".to_string()))?
            .try_extract_array()?;

        Ok((features.to_owned(), features_lens.to_owned()))
    }

    fn encode(
        &mut self,
        audio_signal: &ArrayViewD<f32>,
        length: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), AsrError> {
        log::trace!("Running encoder inference...");
        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ];
        let start = Instant::now();
        let outputs = self.encoder.run(inputs)?;
        log::debug!("Encoder inference completed in {:?}", start.elapsed());

        let encoder_output = outputs
            .get("outputs")
            .ok_or_else(|| AsrError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        let encoded_lengths = outputs
            .get("encoded_lengths")
            .ok_or_else(|| AsrError::OutputNotFound("encoded_lengths".to_string()))?
            .try_extract_array()?;

        let encoder_output = encoder_output.permuted_axes(IxDyn(&[0, 2, 1]));

        Ok((encoder_output.to_owned(), encoded_lengths.to_owned()))
    }

    fn recognize_batch(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_len: &ArrayViewD<i64>,
    ) -> Result<Vec<Transcript>, AsrError> {
        let recognize_start = Instant::now();

        let (features, features_lens) = self.preprocess(waveforms, waveforms_len)?;
        let (encoder_out, encoder_out_lens) =
            self.encode(&features.view(), &features_lens.view())?;

        let mut results = Vec::new();
        for (encodings, &encodings_len) in encoder_out.outer_iter().zip(encoder_out_lens.iter()) {
            let (tokens, timestamps) =
                self.decode_sequence(&encodings.view(), encodings_len as usize)?;
            let result = self.decode_tokens(tokens, timestamps);
            results.push(result);
        }

        log::debug!(
            "recognize_batch completed for {} item(s) in {:?}",
            results.len(),
            recognize_start.elapsed()
        );

        Ok(results)
    }

    pub fn transcribe_samples(&mut self, samples: Vec<f32>) -> Result<Transcript, AsrError> {
        let batch_size = 1;
        let samples_len = samples.len();

        let audio = Array2::from_shape_vec((batch_size, samples_len), samples)?.into_dyn();

        let audio_lengths = Array1::from_vec(vec![samples_len as i64]).into_dyn();

        let results = self.recognize_batch(&audio.view(), &audio_lengths.view())?;

        let timestamped_result = results.into_iter().next().ok_or_else(|| {
            AsrError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No transcription result returned",
            ))
        })?;

        Ok(timestamped_result)
    }
}
