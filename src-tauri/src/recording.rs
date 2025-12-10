use std::mem;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{self, Receiver, Sender},
    Arc, Mutex, OnceLock,
};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SizedSample};
use rtrb::{Producer, RingBuffer};
use thiserror::Error;

use crate::asr::TARGET_SAMPLE_RATE;
use crate::audio_processing::{AudioProcessor, ProcessingEvent};

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
    #[error("VAD error: {0}")]
    VadError(String),
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
            Self::ChannelError | Self::ThreadError => {
                "Internal audio error. Please restart the app."
            }
            Self::VadError(_) => "Voice detection error. Please restart.",
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

impl ToNormalizedSample for u16 {
    #[inline]
    fn to_normalized(self) -> f32 {
        cpal::Sample::to_sample::<f32>(self)
    }
}

enum AudioCmd {
    Stop,
}

pub struct Recorder {
    is_recording: AtomicBool,
    processed_samples: Arc<Mutex<Vec<f32>>>,
    cmd_tx: Mutex<Option<Sender<AudioCmd>>>,
    worker_handle: Mutex<Option<thread::JoinHandle<()>>>,
    pub overrun_count: Arc<AtomicUsize>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            is_recording: AtomicBool::new(false),
            processed_samples: Arc::new(Mutex::new(Vec::new())),
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

    pub fn start(
        &self,
        vad_model_path: &std::path::Path,
        streaming_tx: Option<Sender<ProcessingEvent>>,
    ) -> Result<(), RecordingError> {
        if self
            .is_recording
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(RecordingError::AlreadyRecording);
        }

        if let Ok(mut guard) = self.processed_samples.lock() {
            guard.clear();
        }
        self.overrun_count.store(0, Ordering::Relaxed);

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::channel();

        let processed_samples_clone = self.processed_samples.clone();
        let overrun_clone = self.overrun_count.clone();
        let model_path = vad_model_path.to_path_buf();

        let handle = thread::spawn(move || {
            let res = init_and_run_audio_thread(
                cmd_rx,
                processed_samples_clone,
                init_tx,
                &model_path,
                streaming_tx,
                overrun_clone,
            );
            if let Err(e) = res {
                log::error!("Audio thread failed: {e}");
            }
        });

        match init_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(())) => {
                *self.cmd_tx.lock().map_err(|_| RecordingError::LockFailed)? = Some(cmd_tx);
                *self
                    .worker_handle
                    .lock()
                    .map_err(|_| RecordingError::LockFailed)? = Some(handle);
                Ok(())
            }
            Ok(Err(e)) => {
                self.is_recording.store(false, Ordering::SeqCst);
                let _ = handle.join();
                Err(e)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                log::error!("Audio thread initialization timed out");
                self.is_recording.store(false, Ordering::SeqCst);
                Err(RecordingError::ThreadError)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.is_recording.store(false, Ordering::SeqCst);
                Err(RecordingError::ThreadError)
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
}

fn init_and_run_audio_thread(
    cmd_rx: Receiver<AudioCmd>,
    processed_samples: Arc<Mutex<Vec<f32>>>,
    init_tx: Sender<Result<(), RecordingError>>,
    vad_model_path: &std::path::Path,
    streaming_tx: Option<Sender<ProcessingEvent>>,
    overrun_count: Arc<AtomicUsize>,
) -> Result<(), RecordingError> {
    fn send_init_err(
        tx: &Sender<Result<(), RecordingError>>,
        err: RecordingError,
    ) -> RecordingError {
        let _ = tx.send(Err(err.clone()));
        err
    }

    #[cfg(target_os = "linux")]
    let host = cpal::host_from_id(cpal::HostId::Alsa).unwrap_or_else(|_| cpal::default_host());
    #[cfg(not(target_os = "linux"))]
    let host = cpal::default_host();

    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            return Err(send_init_err(&init_tx, RecordingError::NoInputDevice));
        }
    };

    let device_name = device
        .name()
        .unwrap_or_else(|_| "Unknown input device".to_string());
    log::info!("Audio input: host={:?}, device={}", host.id(), device_name);

    let default_config = match device
        .default_input_config()
        .map_err(|e| RecordingError::Device(e.to_string()))
    {
        Ok(c) => c,
        Err(e) => return Err(send_init_err(&init_tx, e)),
    };

    log::info!("Default input config: {:?}", default_config);

    let stream_config = default_config;

    let sample_rate = stream_config.sample_rate().0;
    let channels = stream_config.channels() as usize;

    log::info!("Audio Config: {} Hz, {} channels", sample_rate, channels);

    let buffer_size = sample_rate as usize;
    let (producer, mut consumer) = RingBuffer::<f32>::new(buffer_size);

    let err_fn = move |err| log::error!("Stream error: {}", err);

    let stream = match stream_config.sample_format() {
        cpal::SampleFormat::F32 => build_stream::<f32>(
            &device,
            &stream_config.into(),
            producer,
            channels,
            err_fn,
            overrun_count.clone(),
        ),
        cpal::SampleFormat::I16 => build_stream::<i16>(
            &device,
            &stream_config.into(),
            producer,
            channels,
            err_fn,
            overrun_count.clone(),
        ),
        cpal::SampleFormat::U16 => build_stream::<u16>(
            &device,
            &stream_config.into(),
            producer,
            channels,
            err_fn,
            overrun_count.clone(),
        ),
        _ => return Err(send_init_err(&init_tx, RecordingError::UnsupportedFormat)),
    }
    .map_err(|e| send_init_err(&init_tx, RecordingError::Device(e.to_string())))?;

    let mut stream = Some(stream);
    if let Some(s) = &stream {
        s.play()
            .map_err(|e| send_init_err(&init_tx, RecordingError::Device(e.to_string())))?;
    }

    init_tx
        .send(Ok(()))
        .map_err(|_| RecordingError::ChannelError)?;

    let mut processor = AudioProcessor::new(
        sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        vad_model_path,
    )
    .map_err(|e| RecordingError::VadError(e.to_string()))?;

    const PROCESS_CHUNK_SIZE: usize = 480;
    let mut scratch_buf = vec![0.0f32; PROCESS_CHUNK_SIZE * 2];
    let mut stopping = false;

    let mut dispatch_event = |event: ProcessingEvent| match event {
        ProcessingEvent::SpeechStart(chunk) => {
            if let Ok(mut guard) = processed_samples.lock() {
                guard.extend_from_slice(&chunk);
            }
            if let Some(tx) = &streaming_tx {
                let _ = tx.send(ProcessingEvent::SpeechStart(chunk));
            }
        }
        ProcessingEvent::Speech(chunk) => {
            if let Ok(mut guard) = processed_samples.lock() {
                guard.extend_from_slice(&chunk);
            }
            if let Some(tx) = &streaming_tx {
                let _ = tx.send(ProcessingEvent::Speech(chunk));
            }
        }
        ProcessingEvent::SpeechEnd => {
            if let Some(tx) = &streaming_tx {
                let _ = tx.send(ProcessingEvent::SpeechEnd);
            }
        }
    };

    'consumer_loop: loop {
        if !stopping {
            if let Ok(AudioCmd::Stop) = cmd_rx.try_recv() {
                stopping = true;
                if let Some(s) = stream.take() {
                    drop(s);
                }
            }
        }

        let available = consumer.slots();

        if available >= PROCESS_CHUNK_SIZE {
            let to_read = (available / PROCESS_CHUNK_SIZE) * PROCESS_CHUNK_SIZE;
            let to_read = to_read.min(scratch_buf.len());

            if let Ok(chunk) = consumer.read_chunk(to_read) {
                let (first, second) = chunk.as_slices();

                let first_len = first.len();
                scratch_buf[..first_len].copy_from_slice(first);
                if !second.is_empty() {
                    scratch_buf[first_len..first_len + second.len()].copy_from_slice(second);
                }
                let total_read = first_len + second.len();
                chunk.commit_all();

                let data = &scratch_buf[..total_read];

                if let Err(e) = processor.process(data, &mut dispatch_event) {
                    log::error!("Processing error: {:?}", e);
                }
            }
        } else if available > 0 {
            if let Ok(chunk) = consumer.read_chunk(available) {
                let (first, second) = chunk.as_slices();
                let first_len = first.len();
                scratch_buf[..first_len].copy_from_slice(first);
                if !second.is_empty() {
                    scratch_buf[first_len..first_len + second.len()].copy_from_slice(second);
                }
                let total_read = first_len + second.len();
                chunk.commit_all();

                let data = &scratch_buf[..total_read];
                if let Err(e) = processor.process(data, &mut dispatch_event) {
                    log::error!("Processing error: {:?}", e);
                }
            }
        } else if stopping {
            break 'consumer_loop;
        } else {
            thread::sleep(Duration::from_millis(2));
        }
    }

    if let Err(e) = processor.flush(&mut dispatch_event) {
        log::error!("Processing flush error: {:?}", e);
    }

    let overruns = overrun_count.load(Ordering::Relaxed);
    if overruns > 0 {
        log::warn!("Audio thread exiting with {} buffer overruns", overruns);
    } else {
        log::debug!("Audio thread exiting cleanly with no overruns");
    }

    Ok(())
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: Producer<f32>,
    channels: usize,
    err_fn: impl Fn(cpal::StreamError) + Send + 'static,
    overrun_count: Arc<AtomicUsize>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: Sample + SizedSample + ToNormalizedSample + Send + 'static,
{
    let has_logged = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            if !has_logged.load(std::sync::atomic::Ordering::Relaxed) {
                log::debug!(
                    "CPAL: First chunk of {} samples (format specific)",
                    data.len()
                );
                has_logged.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            if channels == 1 {
                for &sample in data {
                    if producer.push(sample.to_normalized()).is_err() {
                        overrun_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            } else {
                for frame in data.chunks(channels) {
                    let mut sum = 0.0;
                    for &sample in frame {
                        sum += sample.to_normalized();
                    }
                    if producer.push(sum / channels as f32).is_err() {
                        overrun_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        },
        err_fn,
        None,
    )
}
