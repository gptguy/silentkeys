//! Microphone audio capture using cpal.

use std::mem;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use thiserror::Error;

use crate::asr::{resample_linear, TARGET_SAMPLE_RATE};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during audio recording.
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
    /// Returns a user-friendly error message suitable for display in the UI.
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

// ============================================================================
// Sample Conversion
// ============================================================================

/// Converts audio samples to normalized f32 in the range [-1.0, 1.0].
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

/// Converts a slice of samples to normalized f32 values.
fn normalize_samples<T: ToNormalizedSample>(samples: &[T]) -> impl Iterator<Item = f32> + '_ {
    samples
        .iter()
        .copied()
        .map(ToNormalizedSample::to_normalized)
}

// ============================================================================
// Recorder
// ============================================================================

struct SafeStream {
    _stream: cpal::Stream,
}

unsafe impl Send for SafeStream {}
unsafe impl Sync for SafeStream {}

/// Thread-safe audio recorder for capturing microphone input.
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

    /// Returns the global recorder instance.
    pub fn global() -> &'static Self {
        static RECORDER: OnceLock<Recorder> = OnceLock::new();
        RECORDER.get_or_init(Self::new)
    }

    /// Returns `true` if recording is currently in progress.
    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    /// Starts recording audio from the default input device.
    pub fn start(&self) -> Result<(), RecordingError> {
        if self
            .is_recording
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(RecordingError::AlreadyRecording);
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(RecordingError::NoInputDevice)?;

        let config: cpal::SupportedStreamConfig = device
            .default_input_config()
            .map_err(|e| RecordingError::Device(e.to_string()))?;

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
    }

    /// Stops recording and returns the captured audio samples at 16kHz.
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
            guard.extend(normalize_samples(input));
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i8_normalization() {
        assert!((i8::MAX.to_normalized() - 1.0).abs() < f32::EPSILON);
        assert!((0i8.to_normalized()).abs() < f32::EPSILON);
        assert!(((-64i8).to_normalized() - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn i16_normalization() {
        assert!((i16::MAX.to_normalized() - 1.0).abs() < f32::EPSILON);
        assert!((0i16.to_normalized()).abs() < f32::EPSILON);
    }

    #[test]
    fn i32_normalization() {
        assert!((i32::MAX.to_normalized() - 1.0).abs() < f32::EPSILON);
        assert!((0i32.to_normalized()).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_passthrough() {
        assert!((0.5f32.to_normalized() - 0.5).abs() < f32::EPSILON);
        assert!((-1.0f32.to_normalized() - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn recording_error_user_messages() {
        assert!(!RecordingError::AlreadyRecording.user_message().is_empty());
        assert!(!RecordingError::NotRecording.user_message().is_empty());
        assert!(!RecordingError::NoInputDevice.user_message().is_empty());
        assert!(!RecordingError::UnsupportedFormat.user_message().is_empty());
        assert!(!RecordingError::NoAudioCaptured.user_message().is_empty());
        assert!(!RecordingError::Device("test".to_string())
            .user_message()
            .is_empty());
        assert!(!RecordingError::LockFailed.user_message().is_empty());
    }
}
