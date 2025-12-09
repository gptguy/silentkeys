use std::path::Path;

use ndarray::{Array1, ArrayD, ArrayView2};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VadError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Invalid chunk size: expected {expected}, got {actual}")]
    InvalidChunkSize { expected: usize, actual: usize },
    #[error("VAD model not found at {0}")]
    ModelNotFound(String),
}

impl VadError {
    pub fn user_message(&self) -> &'static str {
        match self {
            Self::Ort(_) => "Voice detection failed. Try restarting the app.",
            Self::Io(_) => "Could not read VAD model files.",
            Self::Shape(_) => "Voice detection encountered an internal error.",
            Self::InvalidChunkSize { .. } => "Audio chunk size mismatch in voice detection.",
            Self::ModelNotFound(_) => "Voice detection model not found. Please download it.",
        }
    }
}

pub const VAD_CHUNK_SIZE: usize = 480;
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;

pub struct VadModel {
    session: Mutex<Session>,
    sample_rate: i64,
}

impl VadModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, VadError> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(VadError::ModelNotFound(model_path.display().to_string()));
        }

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        log::info!("Silero VAD model loaded from {}", model_path.display());

        Ok(Self {
            session: Mutex::new(session),
            sample_rate: 16000,
        })
    }

    pub fn create_state(&self) -> VadState {
        VadState::new()
    }
}

pub struct VadState {
    inner: ArrayD<f32>,
}

impl Default for VadState {
    fn default() -> Self {
        Self {
            inner: ArrayD::<f32>::zeros(vec![2, 1, 128]),
        }
    }
}

impl VadState {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn reset(&mut self) {
        self.inner = ArrayD::<f32>::zeros(vec![2, 1, 128]);
    }
}

pub struct SileroVad {
    model: Arc<VadModel>,
    state: VadState,
    sr_array: ArrayD<i64>,
}

impl SileroVad {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, VadError> {
        let model = Arc::new(VadModel::new(model_path)?);
        let sr_array = Array1::from_vec(vec![model.sample_rate]).into_dyn();
        Ok(Self {
            state: model.create_state(),
            sr_array,
            model,
        })
    }

    pub fn new_with_model(model: Arc<VadModel>) -> Self {
        let sr_array = Array1::from_vec(vec![model.sample_rate]).into_dyn();
        Self {
            state: model.create_state(),
            sr_array,
            model,
        }
    }

    #[inline]
    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<f32, VadError> {
        if samples.len() != VAD_CHUNK_SIZE {
            return Err(VadError::InvalidChunkSize {
                expected: VAD_CHUNK_SIZE,
                actual: samples.len(),
            });
        }

        let frame = ArrayView2::from_shape((1, VAD_CHUNK_SIZE), samples)?;
        let frame_dyn = frame.into_dyn();
        let inputs = inputs![
            "input" => TensorRef::from_array_view(frame_dyn.view())?,
            "state" => TensorRef::from_array_view(self.state.inner.view())?,
            "sr" => TensorRef::from_array_view(self.sr_array.view())?,
        ];

        let mut session_guard = self.model.session.lock().unwrap();
        let outputs = session_guard.run(inputs)?;

        if let Some(state_out) = outputs.get("stateN") {
            self.state.inner = state_out.try_extract_array::<f32>()?.to_owned();
        }

        let prob = if let Some(output) = outputs.get("output") {
            let prob_array = output.try_extract_array::<f32>()?;
            *prob_array.iter().next().unwrap_or(&0.0)
        } else {
            0.0
        };

        Ok(prob)
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub const fn chunk_size(&self) -> usize {
        VAD_CHUNK_SIZE
    }
}
