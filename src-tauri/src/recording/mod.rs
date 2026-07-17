mod audio_thread;

use std::mem;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{self, Sender},
    Arc, Condvar, Mutex, OnceLock,
};
use std::thread;
use std::time::Duration;

use thiserror::Error;

use crate::activity::{self, ActivityError, ActivityGuard, AppActivity};
use crate::audio_processing::AudioFrame;
use crate::errors::UserFacing;

#[derive(Error, Debug)]
pub enum RecordingError {
    #[error("Recording is already in progress")]
    AlreadyRecording,
    #[error("No recording in progress")]
    NotRecording,
    #[error("No default input device available")]
    NoInputDevice,
    #[error("No audio captured")]
    NoAudioCaptured,
    #[error("Device error: {0}")]
    Device(String),
    #[error("Mutex lock failed")]
    LockFailed,
    #[error("Audio thread panicked or failed to start")]
    ThreadError,
    #[error("Failed to start audio thread: {0}")]
    ThreadStart(#[source] std::io::Error),
    #[error("Audio processing error: {0}")]
    AudioProcessingError(String),
    #[error("Audio input overran by {0} samples")]
    AudioOverrun(usize),
    #[error("An app update is being installed")]
    UpdateInProgress,
    #[error("Speech settings are being changed")]
    SettingsInProgress,
}

impl UserFacing for RecordingError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::AlreadyRecording => "Recording is already in progress.",
            Self::NotRecording => "No recording in progress.",
            Self::NoInputDevice => "No microphone found. Please check your audio settings.",
            Self::NoAudioCaptured => "No audio was captured. Please try again.",
            Self::Device(_) => "A microphone error occurred. Please check your audio settings.",
            Self::LockFailed => "The recorder is busy. Please try again.",
            Self::UpdateInProgress => {
                "An app update is being installed. Recording will be available after restart."
            }
            Self::SettingsInProgress => {
                "Speech settings are being changed. Please try recording again."
            }
            Self::ThreadError | Self::ThreadStart(_) | Self::AudioProcessingError(_) => {
                "Internal audio error. Please restart the app."
            }
            Self::AudioOverrun(_) => {
                "Audio could not be captured fast enough. Close demanding apps and try again."
            }
        }
    }
}

pub(super) enum AudioCmd {
    Stop,
}

struct RecordingSession {
    cmd_tx: Sender<AudioCmd>,
    worker_handle: thread::JoinHandle<Result<(), RecordingError>>,
    activity_guard: ActivityGuard,
}

/// Resets the recorder's `starting` flag when the reservation ends without a
/// session, so `is_recording` cannot stay stuck on an abandoned reservation.
struct StartingGuard;

impl Drop for StartingGuard {
    fn drop(&mut self) {
        let recorder = Recorder::global();
        recorder.starting.store(false, Ordering::Release);
        recorder.session_ready.notify_all();
    }
}

pub struct RecordingReservation {
    activity_guard: ActivityGuard,
    starting: StartingGuard,
}

pub struct RecordedAudio {
    samples: Vec<f32>,
    _activity_guard: ActivityGuard,
}

impl RecordedAudio {
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

pub struct Recorder {
    processed_samples: Arc<Mutex<Vec<f32>>>,
    session: Mutex<Option<RecordingSession>>,
    session_ready: Condvar,
    starting: AtomicBool,
    overrun_count: Arc<AtomicUsize>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            processed_samples: Arc::new(Mutex::new(Vec::new())),
            session: Mutex::new(None),
            session_ready: Condvar::new(),
            starting: AtomicBool::new(false),
            overrun_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn global() -> &'static Self {
        static RECORDER: OnceLock<Recorder> = OnceLock::new();
        RECORDER.get_or_init(Self::new)
    }

    pub fn is_recording(&self) -> bool {
        if self.starting.load(Ordering::Acquire) {
            return true;
        }

        self.session
            .lock()
            .map(|session| session.is_some())
            .unwrap_or(false)
    }

    pub fn reserve(&self) -> Result<RecordingReservation, RecordingError> {
        let activity_guard = match activity::try_begin(AppActivity::Recording) {
            Ok(guard) => guard,
            Err(ActivityError::Busy(AppActivity::Recording)) => {
                return Err(RecordingError::AlreadyRecording)
            }
            Err(ActivityError::Busy(AppActivity::Updating)) => {
                return Err(RecordingError::UpdateInProgress)
            }
            Err(ActivityError::Busy(AppActivity::Configuring)) => {
                return Err(RecordingError::SettingsInProgress)
            }
            Err(ActivityError::LockFailed) => return Err(RecordingError::LockFailed),
        };

        if self
            .session
            .lock()
            .map_err(|_| RecordingError::LockFailed)?
            .is_some()
        {
            return Err(RecordingError::AlreadyRecording);
        }

        self.starting.store(true, Ordering::Release);
        Ok(RecordingReservation {
            activity_guard,
            starting: StartingGuard,
        })
    }

    pub fn start(
        &self,
        reservation: RecordingReservation,
        streaming_tx: Option<Sender<AudioFrame>>,
    ) -> Result<(), RecordingError> {
        let RecordingReservation {
            activity_guard,
            starting,
        } = reservation;
        self.processed_samples
            .lock()
            .map_err(|_| RecordingError::LockFailed)?
            .clear();
        self.overrun_count.store(0, Ordering::Relaxed);

        // The session owns the stop sender and joins its sole audio worker;
        // the one-shot init channel cannot outlive startup.
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::channel();

        let samples_clone = self.processed_samples.clone();
        let overrun_clone = self.overrun_count.clone();
        let init_error_tx = init_tx.clone();
        let handle = thread::Builder::new()
            .name("audio-capture".to_string())
            .spawn(move || {
                let result = audio_thread::init_and_run_audio_thread(
                    cmd_rx,
                    samples_clone,
                    init_tx,
                    streaming_tx,
                    overrun_clone,
                );
                if result.is_err() {
                    let _ = init_error_tx.send(Err(()));
                }
                result
            })
            .map_err(RecordingError::ThreadStart)?;

        match init_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(())) => {
                let mut session = match self.session.lock() {
                    Ok(session) => session,
                    Err(_) => {
                        let _ = cmd_tx.send(AudioCmd::Stop);
                        let _ = handle.join();
                        return Err(RecordingError::LockFailed);
                    }
                };
                *session = Some(RecordingSession {
                    cmd_tx,
                    worker_handle: handle,
                    activity_guard,
                });
                drop(session);
                drop(starting);
                Ok(())
            }
            init_result => {
                let _ = cmd_tx.send(AudioCmd::Stop);
                let worker_result = handle.join();
                match (init_result, worker_result) {
                    (_, Ok(Err(err))) => Err(err),
                    _ => Err(RecordingError::ThreadError),
                }
            }
        }
    }

    pub fn stop(&self) -> Result<RecordedAudio, RecordingError> {
        let mut session_guard = self
            .session
            .lock()
            .map_err(|_| RecordingError::LockFailed)?;
        while session_guard.is_none() && self.starting.load(Ordering::Acquire) {
            session_guard = self
                .session_ready
                .wait(session_guard)
                .map_err(|_| RecordingError::LockFailed)?;
        }
        let session = session_guard.take().ok_or(RecordingError::NotRecording)?;
        drop(session_guard);

        let RecordingSession {
            cmd_tx,
            worker_handle,
            activity_guard,
        } = session;
        let _ = cmd_tx.send(AudioCmd::Stop);
        match worker_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(_) => return Err(RecordingError::ThreadError),
        }

        let overrun_count = self.overrun_count.swap(0, Ordering::Relaxed);
        if overrun_count > 0 {
            return Err(RecordingError::AudioOverrun(overrun_count));
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

        Ok(RecordedAudio {
            samples,
            _activity_guard: activity_guard,
        })
    }
}
