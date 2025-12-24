#![allow(clippy::manual_is_multiple_of)]
use ndarray::{Array1, ArrayD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{TensorRef, Value};
use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;
use thiserror::Error;

pub const VAD_CHUNK_SIZE: usize = 480;
const VAD_DC_ALPHA: f32 = 0.001;
const VAD_RMS_TARGET: f32 = 0.08;
const VAD_MIN_GAIN: f32 = 0.6;
const VAD_MAX_GAIN: f32 = 4.0;
const VAD_NOISE_GATE_RMS: f32 = 0.004;
const VAD_METRICS_EVERY_FRAMES: u64 = 200;
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
    pub dynamic_threshold: bool,
    pub noise_floor_alpha: f32,
    pub noise_floor_margin: f32,
    pub max_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct SpeechStartData {
    pub samples: Vec<f32>,
    pub prefill_len: usize,
}

#[derive(Debug, Clone)]
pub enum VadEvent {
    SpeechStart(SpeechStartData),
    Speech(Vec<f32>),
    SpeechEnd,
}

pub struct VadSegmenter {
    vad: VadSession,
    buffer: VecDeque<f32>,
    frame_scratch: Vec<f32>,
    vad_scratch: Vec<f32>,
    dc_estimate: f32,
    vad_metrics_frames: u64,
    vad_metrics_speech_frames: u64,
    vad_metrics_starts: u64,
    vad_metrics_ends: u64,
    last_vad_log: Instant,
    was_speech: bool,
}

impl VadSegmenter {
    pub fn new(model_path: &Path) -> Result<Self, VadError> {
        let config = VadConfig::from_env();
        let vad = VadSession::with_config(model_path, config)?;
        Ok(Self {
            vad,
            buffer: VecDeque::with_capacity(VAD_CHUNK_SIZE * 4),
            frame_scratch: Vec::with_capacity(VAD_CHUNK_SIZE),
            vad_scratch: Vec::with_capacity(VAD_CHUNK_SIZE),
            dc_estimate: 0.0,
            vad_metrics_frames: 0,
            vad_metrics_speech_frames: 0,
            vad_metrics_starts: 0,
            vad_metrics_ends: 0,
            last_vad_log: Instant::now(),
            was_speech: false,
        })
    }

    pub fn process_audio(
        &mut self,
        samples: &[f32],
        mut emit: impl FnMut(VadEvent),
    ) -> Result<(), VadError> {
        self.buffer.extend(samples.iter().copied());
        while self.buffer.len() >= VAD_CHUNK_SIZE {
            let mut frame = std::mem::take(&mut self.frame_scratch);
            frame.clear();
            let (front, back) = self.buffer.as_slices();
            let front_take = front.len().min(VAD_CHUNK_SIZE);
            frame.extend_from_slice(&front[..front_take]);
            if front_take < VAD_CHUNK_SIZE && !back.is_empty() {
                let back_take = VAD_CHUNK_SIZE - front_take;
                frame.extend_from_slice(&back[..back_take.min(back.len())]);
            }
            self.buffer.drain(..VAD_CHUNK_SIZE);
            let result = self.process_frame(&frame, &mut emit);
            self.frame_scratch = frame;
            result?;
        }

        if self.buffer.capacity() > VAD_CHUNK_SIZE * 16 && self.buffer.len() < VAD_CHUNK_SIZE {
            self.buffer.shrink_to(VAD_CHUNK_SIZE * 4);
        }

        Ok(())
    }

    pub fn flush(&mut self, mut emit: impl FnMut(VadEvent)) -> Result<(), VadError> {
        if self.was_speech {
            if !self.buffer.is_empty() {
                let mut tail = Vec::with_capacity(self.buffer.len());
                for sample in self.buffer.drain(..) {
                    tail.push(sample);
                }
                emit(VadEvent::Speech(tail));
            }
            emit(VadEvent::SpeechEnd);
            self.was_speech = false;
        } else {
            self.buffer.clear();
        }
        Ok(())
    }

    fn preprocess_for_vad_into(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(samples.len());
        let mut sum_sq = 0.0;
        for &sample in samples {
            self.dc_estimate += VAD_DC_ALPHA * (sample - self.dc_estimate);
            let val = sample - self.dc_estimate;
            out.push(val);
            sum_sq += val * val;
        }

        if out.is_empty() {
            return;
        }

        let rms = (sum_sq / out.len() as f32).sqrt();
        if rms < VAD_NOISE_GATE_RMS {
            for val in out {
                *val = 0.0;
            }
            return;
        }

        let gain = (VAD_RMS_TARGET / rms).clamp(VAD_MIN_GAIN, VAD_MAX_GAIN);
        if (gain - 1.0).abs() > 0.01 {
            for val in out {
                *val *= gain;
            }
        }
    }

    fn process_frame(
        &mut self,
        samples: &[f32],
        emit: &mut impl FnMut(VadEvent),
    ) -> Result<(), VadError> {
        let mut vad_samples = std::mem::take(&mut self.vad_scratch);
        self.preprocess_for_vad_into(samples, &mut vad_samples);
        let is_speech = self.vad.process_frame(samples, &vad_samples)?;
        self.vad_scratch = vad_samples;

        self.vad_metrics_frames += 1;
        if is_speech {
            self.vad_metrics_speech_frames += 1;
        }
        if is_speech && !self.was_speech {
            self.vad_metrics_starts += 1;
        }
        if !is_speech && self.was_speech {
            self.vad_metrics_ends += 1;
        }
        if self.vad_metrics_frames % VAD_METRICS_EVERY_FRAMES == 0
            || self.last_vad_log.elapsed().as_secs() >= 5
        {
            let speech_ratio = if self.vad_metrics_frames == 0 {
                0.0
            } else {
                self.vad_metrics_speech_frames as f32 / self.vad_metrics_frames as f32
            };
            log::debug!(
                "VAD stats: frames={}, speech_ratio={:.2}, starts={}, ends={}",
                self.vad_metrics_frames,
                speech_ratio,
                self.vad_metrics_starts,
                self.vad_metrics_ends
            );
            self.last_vad_log = Instant::now();
        }

        if is_speech {
            if !self.was_speech {
                log::debug!("VAD: Speech START");
                if self.vad.has_prefill() {
                    let prefill = self.vad.take_prefill();
                    let prefill_len = prefill.len();
                    let mut joined = Vec::with_capacity(prefill_len + samples.len());
                    joined.extend_from_slice(&prefill);
                    joined.extend_from_slice(samples);
                    emit(VadEvent::SpeechStart(SpeechStartData {
                        samples: joined,
                        prefill_len,
                    }));
                } else {
                    emit(VadEvent::SpeechStart(SpeechStartData {
                        samples: samples.to_vec(),
                        prefill_len: 0,
                    }));
                }
            } else {
                emit(VadEvent::Speech(samples.to_vec()));
            }
        } else if self.was_speech {
            log::debug!("VAD: Speech END");
            emit(VadEvent::SpeechEnd);
        }

        self.was_speech = is_speech;
        Ok(())
    }
}
impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_SPEECH_THRESHOLD,
            prefill_frames: 10,
            hangover_frames: 16,
            onset_frames: 3,
            dynamic_threshold: true,
            noise_floor_alpha: 0.05,
            noise_floor_margin: 0.15,
            max_threshold: 0.9,
        }
    }
}

impl VadConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(value) = std::env::var("VAD_THRESHOLD") {
            if let Ok(parsed) = value.parse::<f32>() {
                config.threshold = parsed;
            }
        }
        if let Ok(value) = std::env::var("VAD_PREFILL_FRAMES") {
            if let Ok(parsed) = value.parse::<usize>() {
                config.prefill_frames = parsed;
            }
        }
        if let Ok(value) = std::env::var("VAD_HANGOVER_FRAMES") {
            if let Ok(parsed) = value.parse::<usize>() {
                config.hangover_frames = parsed;
            }
        }
        if let Ok(value) = std::env::var("VAD_ONSET_FRAMES") {
            if let Ok(parsed) = value.parse::<usize>() {
                config.onset_frames = parsed;
            }
        }
        if let Ok(value) = std::env::var("VAD_DYNAMIC_THRESHOLD") {
            if let Ok(parsed) = value.parse::<u8>() {
                config.dynamic_threshold = parsed != 0;
            }
        }
        if let Ok(value) = std::env::var("VAD_NOISE_ALPHA") {
            if let Ok(parsed) = value.parse::<f32>() {
                config.noise_floor_alpha = parsed;
            }
        }
        if let Ok(value) = std::env::var("VAD_NOISE_MARGIN") {
            if let Ok(parsed) = value.parse::<f32>() {
                config.noise_floor_margin = parsed;
            }
        }
        if let Ok(value) = std::env::var("VAD_MAX_THRESHOLD") {
            if let Ok(parsed) = value.parse::<f32>() {
                config.max_threshold = parsed;
            }
        }
        config
    }
}

struct PrefillBuffer {
    data: Vec<f32>,
    head: usize,
    len: usize,
}

impl PrefillBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            len: 0,
        }
    }

    fn push_samples(&mut self, samples: &[f32]) {
        if self.data.is_empty() {
            return;
        }
        for &sample in samples {
            let idx = (self.head + self.len) % self.data.len();
            if self.len < self.data.len() {
                self.data[idx] = sample;
                self.len += 1;
            } else {
                self.data[idx] = sample;
                self.head = (self.head + 1) % self.data.len();
            }
        }
    }

    fn to_vec(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        if self.data.is_empty() {
            return out;
        }
        for i in 0..self.len {
            out.push(self.data[(self.head + i) % self.data.len()]);
        }
        out
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

pub struct VadSession {
    session: Session,
    input_buffer: ArrayD<f32>,
    state_buffer: ArrayD<f32>,

    config: VadConfig,

    prefill_buffer: PrefillBuffer,
    hangover_counter: usize,
    onset_counter: usize,
    in_speech: bool,
    prefill_ready: bool,
    noise_floor: f32,
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
            ..VadConfig::default()
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

        let input_buffer = ArrayD::<f32>::zeros(vec![1, VAD_CHUNK_SIZE]);
        let state_buffer = ArrayD::<f32>::zeros(vec![2, 1, 128]);

        Ok(Self {
            session,
            input_buffer,
            state_buffer,
            prefill_buffer: PrefillBuffer::new(config.prefill_frames * VAD_CHUNK_SIZE),
            config,
            hangover_counter: 0,
            onset_counter: 0,
            in_speech: false,
            prefill_ready: false,
            noise_floor: 0.0,
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
        self.prefill_buffer.to_vec()
    }

    pub fn process_frame(
        &mut self,
        raw_frame: &[f32],
        vad_frame: &[f32],
    ) -> Result<bool, VadError> {
        if raw_frame.len() != VAD_CHUNK_SIZE {
            return Err(VadError::InvalidChunkSize {
                expected: VAD_CHUNK_SIZE,
                actual: raw_frame.len(),
            });
        }
        if vad_frame.len() != VAD_CHUNK_SIZE {
            return Err(VadError::InvalidChunkSize {
                expected: VAD_CHUNK_SIZE,
                actual: vad_frame.len(),
            });
        }

        self.prefill_buffer.push_samples(raw_frame);

        let prob = self.run_inference(vad_frame)?;
        let threshold = if self.config.dynamic_threshold {
            if !self.in_speech {
                let alpha = self.config.noise_floor_alpha.clamp(0.0, 1.0);
                self.noise_floor = (1.0 - alpha) * self.noise_floor + alpha * prob;
            }
            (self.noise_floor + self.config.noise_floor_margin)
                .max(self.config.threshold)
                .min(self.config.max_threshold)
        } else {
            self.config.threshold
        };
        let is_voice_raw = prob >= threshold;

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

        let input_val = TensorRef::from_array_view(self.input_buffer.view())?;
        let state_val = TensorRef::from_array_view(self.state_buffer.view())?;
        let sr_val = Value::from_array(Array1::from_vec(vec![16000i64]).into_dyn())?.into_dyn();

        let inputs = ort::inputs![
            "input" => input_val,
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
        let shape: Vec<usize> = state_shape.iter().map(|x| *x as usize).collect();
        if self.state_buffer.shape() != shape.as_slice() {
            self.state_buffer = ArrayD::zeros(shape.clone());
        }
        if let Some(buf) = self.state_buffer.as_slice_mut() {
            if buf.len() == state_data.len() {
                buf.copy_from_slice(state_data);
            } else {
                self.state_buffer = ArrayD::from_shape_vec(shape, state_data.to_vec())?;
            }
        } else {
            self.state_buffer = ArrayD::from_shape_vec(shape, state_data.to_vec())?;
        }

        let (_prob_shape, prob_data) = output_tensor.try_extract_tensor::<f32>()?;
        let prob = *prob_data.first().unwrap_or(&0.0);

        Ok(prob)
    }

    pub fn reset(&mut self) {
        self.prefill_buffer.clear();
        self.hangover_counter = 0;
        self.onset_counter = 0;
        self.in_speech = false;
        self.prefill_ready = false;
        self.noise_floor = 0.0;

        self.state_buffer.fill(0.0);
    }
}
