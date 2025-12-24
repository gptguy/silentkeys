mod audio_thread;

use std::mem;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{self, Sender},
    Arc, Mutex, OnceLock,
};
use std::thread;
use std::time::Duration;

use thiserror::Error;

use crate::audio_processing::AudioFrame;

#[derive(Error, Debug, Clone)]
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
    #[error("Failed to communicate with audio thread")]
    ChannelError,
    #[error("Audio thread panicked or failed to start")]
    ThreadError,
    #[error("Audio processing error: {0}")]
    AudioProcessingError(String),
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
            Self::ChannelError | Self::ThreadError | Self::AudioProcessingError(_) => {
                "Internal audio error. Please restart the app."
            }
        }
    }
}

pub(super) enum AudioCmd {
    Stop,
}

pub struct Recorder {
    is_recording: AtomicBool,
    processed_samples: Arc<Mutex<Vec<f32>>>,
    streamed_any: AtomicBool,
    cmd_tx: Mutex<Option<Sender<AudioCmd>>>,
    worker_handle: Mutex<Option<thread::JoinHandle<()>>>,
    pub overrun_count: Arc<AtomicUsize>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            is_recording: AtomicBool::new(false),
            processed_samples: Arc::new(Mutex::new(Vec::new())),
            streamed_any: AtomicBool::new(false),
            cmd_tx: Mutex::new(None),
            worker_handle: Mutex::new(None),
            overrun_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn global() -> &'static Self {
        static RECORDER: OnceLock<Recorder> = OnceLock::new();
        RECORDER.get_or_init(Self::new)
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    pub fn start(&self, streaming_tx: Option<Sender<AudioFrame>>) -> Result<(), RecordingError> {
        if self.is_recording.swap(true, Ordering::SeqCst) {
            return Err(RecordingError::AlreadyRecording);
        }

        if let Ok(mut guard) = self.processed_samples.lock() {
            guard.clear();
        }
        self.streamed_any.store(false, Ordering::Relaxed);
        self.overrun_count.store(0, Ordering::Relaxed);

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::channel();

        let samples_clone = self.processed_samples.clone();
        let overrun_clone = self.overrun_count.clone();
        let handle = thread::spawn(move || {
            if let Err(e) = audio_thread::init_and_run_audio_thread(
                cmd_rx,
                samples_clone,
                init_tx,
                streaming_tx,
                overrun_clone,
            ) {
                log::error!("Audio thread failed: {e}");
            }
        });

        match init_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(())) => {
                *self.cmd_tx.lock().unwrap() = Some(cmd_tx);
                *self.worker_handle.lock().unwrap() = Some(handle);
                Ok(())
            }
            res => {
                self.is_recording.store(false, Ordering::SeqCst);
                let _ = handle.join();
                match res {
                    Ok(Err(e)) => Err(e),
                    _ => Err(RecordingError::ThreadError),
                }
            }
        }
    }

    pub fn stop(&self) -> Result<Vec<f32>, RecordingError> {
        if self
            .is_recording
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(RecordingError::NotRecording);
        }

        let cmd_tx_opt = self
            .cmd_tx
            .lock()
            .map_err(|_| RecordingError::LockFailed)?
            .take();
        if let Some(tx) = cmd_tx_opt {
            let _ = tx.send(AudioCmd::Stop);
        }

        let handle_opt = self
            .worker_handle
            .lock()
            .map_err(|_| RecordingError::LockFailed)?
            .take();
        if let Some(handle) = handle_opt {
            let _ = handle.join();
        }

        let mut samples_guard = self
            .processed_samples
            .lock()
            .map_err(|_| RecordingError::LockFailed)?;
        let samples = mem::take(&mut *samples_guard);

        log::info!(
            "Recorder stopped. Total samples captured: {}",
            samples.len()
        );

        if samples.is_empty() {
            log::warn!("Recorder error: 0 samples in buffer.");
            return Err(RecordingError::NoAudioCaptured);
        }

        Ok(samples)
    }

    pub fn mark_streamed_any(&self) {
        self.streamed_any.store(true, Ordering::Relaxed);
    }

    pub fn streamed_any(&self) -> bool {
        self.streamed_any.load(Ordering::Relaxed)
    }
}
