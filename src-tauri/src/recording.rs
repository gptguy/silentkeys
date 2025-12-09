use std::mem;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use thiserror::Error;

use crate::asr::{resample_linear, TARGET_SAMPLE_RATE};

#[derive(Error, Debug)]
pub enum RecordingError {
    #[error("Recording is already in progress")]
    AlreadyRecording,
    #[error("No recording in progress")]
    NotRecording,
    #[error("No default input device available")]
    NoInputDevice,
    #[error("Unsupported sample format")]
    UnsupportedFormat,
    #[error("No audio captured")]
    NoAudioCaptured,
    #[error("Device error: {0}")]
    Device(String),
    #[error("Mutex lock failed")]
    LockFailed,
}

impl RecordingError {
    pub fn user_message(&self) -> &'static str {
        match self {
            Self::AlreadyRecording => "Recording is already in progress.",
            Self::NotRecording => "No recording in progress.",
            Self::NoInputDevice => "No microphone found. Please check your audio settings.",
            Self::UnsupportedFormat => "Your microphone's audio format is not supported.",
            Self::NoAudioCaptured => "No audio was captured. Please try again.",
            Self::Device(_) => "A microphone error occurred. Please check your audio settings.",
            Self::LockFailed => "The recorder is busy. Please try again.",
        }
    }
}

pub trait ToNormalizedSample: Copy {
    fn to_normalized(self) -> f32;
}

impl ToNormalizedSample for i8 {
    #[inline]
    fn to_normalized(self) -> f32 {
        self as f32 / i8::MAX as f32
    }
}

impl ToNormalizedSample for i16 {
    #[inline]
    fn to_normalized(self) -> f32 {
        self as f32 / i16::MAX as f32
    }
}

impl ToNormalizedSample for i32 {
    #[inline]
    fn to_normalized(self) -> f32 {
        self as f32 / i32::MAX as f32
    }
}

impl ToNormalizedSample for f32 {
    #[inline]
    fn to_normalized(self) -> f32 {
        self
    }
}

fn normalize_samples<T: ToNormalizedSample>(samples: &[T]) -> impl Iterator<Item = f32> + '_ {
    samples
        .iter()
        .copied()
        .map(ToNormalizedSample::to_normalized)
}

struct SafeStream {
    _stream: cpal::Stream,
}

unsafe impl Send for SafeStream {}
unsafe impl Sync for SafeStream {}

pub struct Recorder {
    is_recording: AtomicBool,
    samples: Mutex<Vec<f32>>,
    sample_rate: Mutex<Option<u32>>,
    stream: Mutex<Option<SafeStream>>,
}

impl Recorder {
    const fn new() -> Self {
        Self {
            is_recording: AtomicBool::new(false),
            samples: Mutex::new(Vec::new()),
            sample_rate: Mutex::new(None),
            stream: Mutex::new(None),
        }
    }

    pub fn global() -> &'static Self {
        static RECORDER: OnceLock<Recorder> = OnceLock::new();
        RECORDER.get_or_init(Self::new)
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    pub fn start(&self) -> Result<(), RecordingError> {
        if self
            .is_recording
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(RecordingError::AlreadyRecording);
        }

        let result = (|| {
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .ok_or(RecordingError::NoInputDevice)?;

            let default_config = device
                .default_input_config()
                .map_err(|e| RecordingError::Device(e.to_string()))?;

            let config = device
                .supported_input_configs()
                .map_err(|e| RecordingError::Device(e.to_string()))?
                .find_map(|c| {
                    let min = c.min_sample_rate().0;
                    let max = c.max_sample_rate().0;
                    if min <= 16000 && max >= 16000 {
                        Some(c.with_sample_rate(cpal::SampleRate(16000)))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| {
                    log::warn!(
                        "Native 16kHz not supported, falling back to default: {:?}",
                        default_config
                    );
                    default_config
                });

            log::info!(
                "Recording started using Input Device: {}",
                device.name().unwrap_or_else(|_| "Unknown".to_string())
            );
            log::info!("Input Config: {:?}", config);

            {
                let mut samples = self
                    .samples
                    .lock()
                    .map_err(|_| RecordingError::LockFailed)?;
                samples.clear();
            }
            {
                let mut rate = self
                    .sample_rate
                    .lock()
                    .map_err(|_| RecordingError::LockFailed)?;
                *rate = Some(config.sample_rate().0);
            }

            let err_fn = move |err: cpal::StreamError| {
                log::error!("mic recording stream error: {err}");
            };

            let recorder = Recorder::global();
            let stream_result = match config.sample_format() {
                cpal::SampleFormat::I8 => device.build_input_stream(
                    &config.clone().into(),
                    move |data: &[i8], _: &_| recorder.append_normalized_samples(data),
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::I16 => device.build_input_stream(
                    &config.clone().into(),
                    move |data: &[i16], _: &_| recorder.append_normalized_samples(data),
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::I32 => device.build_input_stream(
                    &config.clone().into(),
                    move |data: &[i32], _: &_| recorder.append_normalized_samples(data),
                    err_fn,
                    None,
                ),
                cpal::SampleFormat::F32 => device.build_input_stream(
                    &config.clone().into(),
                    move |data: &[f32], _: &_| recorder.append_normalized_samples(data),
                    err_fn,
                    None,
                ),
                _ => return Err(RecordingError::UnsupportedFormat),
            };

            let stream = stream_result.map_err(|e| RecordingError::Device(e.to_string()))?;
            stream
                .play()
                .map_err(|e| RecordingError::Device(e.to_string()))?;

            {
                let mut guard = self.stream.lock().map_err(|_| RecordingError::LockFailed)?;
                *guard = Some(SafeStream { _stream: stream });
            }

            Ok(())
        })();

        if result.is_err() {
            self.is_recording.store(false, Ordering::SeqCst);
        }

        result
    }

    pub fn stop(&self) -> Result<Vec<f32>, RecordingError> {
        if self
            .is_recording
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(RecordingError::NotRecording);
        }

        if let Ok(mut guard) = self.stream.lock() {
            if let Some(stream) = guard.take() {
                drop(stream);
            }
        }

        let mut samples_guard = self
            .samples
            .lock()
            .map_err(|_| RecordingError::LockFailed)?;
        let raw_samples = mem::take(&mut *samples_guard);
        log::info!("Recorder stopped with {} raw samples", raw_samples.len());

        if raw_samples.is_empty() {
            return Err(RecordingError::NoAudioCaptured);
        }

        let sample_rate = self
            .sample_rate
            .lock()
            .map_err(|_| RecordingError::LockFailed)?
            .take()
            .unwrap_or(TARGET_SAMPLE_RATE);

        let samples = if sample_rate == TARGET_SAMPLE_RATE {
            raw_samples
        } else {
            log::info!(
                "Resampling captured audio from {} Hz to {} Hz",
                sample_rate,
                TARGET_SAMPLE_RATE
            );
            resample_linear(&raw_samples, sample_rate, TARGET_SAMPLE_RATE)
        };

        Ok(samples)
    }

    fn append_normalized_samples<T: ToNormalizedSample>(&self, input: &[T]) {
        if let Ok(mut guard) = self.samples.try_lock() {
            guard.reserve(input.len());
            guard.extend(normalize_samples(input));
        }
    }

    pub fn drain(&self) -> Result<Vec<f32>, RecordingError> {
        if !self.is_recording() {
            return Err(RecordingError::NotRecording);
        }

        let (raw_samples, sample_rate) = {
            let mut samples_guard = match self.samples.try_lock() {
                Ok(guard) => guard,
                Err(_) => return Ok(Vec::new()),
            };

            if samples_guard.is_empty() {
                return Ok(Vec::new());
            }

            let raw = std::mem::take(&mut *samples_guard);

            drop(samples_guard);

            let sr = self
                .sample_rate
                .lock()
                .map_err(|_| RecordingError::LockFailed)?
                .unwrap_or(TARGET_SAMPLE_RATE);

            (raw, sr)
        };

        if sample_rate == TARGET_SAMPLE_RATE {
            Ok(raw_samples)
        } else {
            Ok(resample_linear(
                &raw_samples,
                sample_rate,
                TARGET_SAMPLE_RATE,
            ))
        }
    }
}
