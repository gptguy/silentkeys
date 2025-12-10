use ndarray::{Array1, ArrayD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValueTypeMarker, Value};
use std::collections::VecDeque;
use std::path::Path;
use thiserror::Error;

pub const VAD_CHUNK_SIZE: usize = 480;
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.5;

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

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub threshold: f32,
    pub prefill_frames: usize,
    pub hangover_frames: usize,
    pub onset_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefill_frames: 10,
            hangover_frames: 16,
            onset_frames: 3,
        }
    }
}

pub struct VadSession {
    session: Session,
    state_tensor: Option<Value<DynValueTypeMarker>>,

    input_buffer: ArrayD<f32>,
    state_buffer: ArrayD<f32>,

    config: VadConfig,

    frame_buffer: VecDeque<f32>,
    hangover_counter: usize,
    onset_counter: usize,
    in_speech: bool,
    prefill_ready: bool,
}

impl VadSession {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        threshold: f32,
        prefill_frames: usize,
        hangover_frames: usize,
        onset_frames: usize,
    ) -> Result<Self, VadError> {
        let config = VadConfig {
            threshold,
            prefill_frames,
            hangover_frames,
            onset_frames,
        };
        Self::with_config(model_path, config)
    }

    pub fn with_config<P: AsRef<Path>>(model_path: P, config: VadConfig) -> Result<Self, VadError> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(VadError::ModelNotFound(model_path.display().to_string()));
        }

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        let state_arr = ArrayD::<f32>::zeros(vec![2, 1, 128]);
        let input_buffer = ArrayD::<f32>::zeros(vec![1, VAD_CHUNK_SIZE]);
        let state_buffer = ArrayD::<f32>::zeros(vec![2, 1, 128]);

        let state_tensor = Value::from_array(state_arr)?.into_dyn();

        Ok(Self {
            session,
            state_tensor: Some(state_tensor),
            input_buffer,
            state_buffer,
            frame_buffer: VecDeque::with_capacity(config.prefill_frames * VAD_CHUNK_SIZE),
            config,
            hangover_counter: 0,
            onset_counter: 0,
            in_speech: false,
            prefill_ready: false,
        })
    }

    #[inline]
    pub fn is_speech(&self) -> bool {
        self.in_speech
    }

    #[inline]
    pub fn has_prefill(&self) -> bool {
        self.prefill_ready
    }

    pub fn take_prefill(&mut self) -> Vec<f32> {
        self.prefill_ready = false;
        self.frame_buffer.iter().copied().collect()
    }

    pub fn process_frame(&mut self, frame: &[f32]) -> Result<bool, VadError> {
        if frame.len() != VAD_CHUNK_SIZE {
            return Err(VadError::InvalidChunkSize {
                expected: VAD_CHUNK_SIZE,
                actual: frame.len(),
            });
        }

        self.frame_buffer.extend(frame.iter());
        let max_samples = self.config.prefill_frames * VAD_CHUNK_SIZE;
        if self.frame_buffer.len() > max_samples {
            let remove = self.frame_buffer.len() - max_samples;
            self.frame_buffer.drain(0..remove);
        }

        let prob = self.run_inference(frame)?;
        let is_voice_raw = prob >= self.config.threshold;

        match (self.in_speech, is_voice_raw) {
            (false, true) => {
                self.onset_counter += 1;
                if self.onset_counter >= self.config.onset_frames {
                    self.in_speech = true;
                    self.hangover_counter = self.config.hangover_frames;
                    self.onset_counter = 0;
                    self.prefill_ready = true;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            (true, true) => {
                self.hangover_counter = self.config.hangover_frames;
                Ok(true)
            }
            (true, false) => {
                if self.hangover_counter > 0 {
                    self.hangover_counter -= 1;
                    Ok(true)
                } else {
                    self.in_speech = false;
                    Ok(false)
                }
            }
            (false, false) => {
                self.onset_counter = 0;
                Ok(false)
            }
        }
    }

    fn run_inference(&mut self, samples: &[f32]) -> Result<f32, VadError> {
        for (i, &sample) in samples.iter().enumerate() {
            self.input_buffer[[0, i]] = sample;
        }

        if self.state_tensor.is_none() {
            #[cfg(debug_assertions)]
            log::warn!("VAD state tensor missing, resetting to zeros");
            self.state_buffer.fill(0.0);
            self.state_tensor = Some(Value::from_array(self.state_buffer.clone())?.into_dyn());
        }

        let state_val = self
            .state_tensor
            .take()
            .ok_or(VadError::ModelNotFound("State tensor missing".to_string()))?;

        let input_val = Value::from_array(self.input_buffer.clone())?;
        let sr_val = Value::from_array(Array1::from_vec(vec![16000i64]).into_dyn())?.into_dyn();

        let inputs = ort::inputs![
            "input" => input_val.into_dyn(),
            "state" => state_val,
            "sr" => sr_val,
        ];

        let outputs = self.session.run(inputs)?;

        let output_tensor = outputs
            .get("output")
            .ok_or(VadError::ModelNotFound("output tensor missing".to_string()))?;
        let state_n_tensor = outputs
            .get("stateN")
            .ok_or(VadError::ModelNotFound("stateN tensor missing".to_string()))?;

        let (state_shape, state_data) = state_n_tensor.try_extract_tensor::<f32>()?;
        let new_state_arr = ArrayD::from_shape_vec(
            state_shape.iter().map(|x| *x as usize).collect::<Vec<_>>(),
            state_data.to_vec(),
        )?;
        self.state_tensor = Some(Value::from_array(new_state_arr)?.into_dyn());

        let (_prob_shape, prob_data) = output_tensor.try_extract_tensor::<f32>()?;
        let prob = *prob_data.first().unwrap_or(&0.0);

        Ok(prob)
    }

    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.hangover_counter = 0;
        self.onset_counter = 0;
        self.in_speech = false;
        self.prefill_ready = false;

        self.state_buffer.fill(0.0);
        if let Ok(zeros) = Value::from_array(self.state_buffer.clone()) {
            self.state_tensor = Some(zeros.into_dyn());
        }
    }
}
