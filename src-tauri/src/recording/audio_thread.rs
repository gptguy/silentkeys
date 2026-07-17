use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{Receiver, Sender},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SizedSample};
use rtrb::{Producer, RingBuffer};

use crate::asr::TARGET_SAMPLE_RATE;
use crate::audio_processing::{AudioFrame, AudioProcessor, PROCESS_CHUNK_SIZE};

use super::{AudioCmd, RecordingError};

pub(super) fn init_and_run_audio_thread(
    cmd_rx: Receiver<AudioCmd>,
    processed_samples: Arc<Mutex<Vec<f32>>>,
    init_tx: Sender<Result<(), ()>>,
    streaming_tx: Option<Sender<AudioFrame>>,
    overrun_count: Arc<AtomicUsize>,
) -> Result<(), RecordingError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(RecordingError::NoInputDevice)?;

    let stream_config = device.default_input_config().map_err(|error| {
        RecordingError::Device(format!("read default input configuration: {error}"))
    })?;

    let sample_rate = stream_config.sample_rate();
    let channels = stream_config.channels() as usize;

    log::info!(
        "Audio: {} Hz, {} channels, device={:?}",
        sample_rate,
        channels,
        device
            .description()
            .map(|description| description.to_string())
            .unwrap_or_else(|_| "unknown input device".to_string())
    );

    let (producer, mut consumer) = RingBuffer::<f32>::new(sample_rate as usize);
    let stream_failed = Arc::new(AtomicBool::new(false));
    let callback_failed = stream_failed.clone();
    let err_fn = move |_| callback_failed.store(true, Ordering::Release);
    let sample_format = stream_config.sample_format();
    let stream_config = stream_config.into();
    let mut processor = AudioProcessor::new(sample_rate as usize, TARGET_SAMPLE_RATE as usize)
        .map_err(|e| RecordingError::AudioProcessingError(e.to_string()))?;

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(
            &device,
            stream_config,
            producer,
            channels,
            err_fn,
            overrun_count.clone(),
        ),
        cpal::SampleFormat::I16 => build_stream::<i16>(
            &device,
            stream_config,
            producer,
            channels,
            err_fn,
            overrun_count.clone(),
        ),
        cpal::SampleFormat::U16 => build_stream::<u16>(
            &device,
            stream_config,
            producer,
            channels,
            err_fn,
            overrun_count.clone(),
        ),
        _ => Err(cpal::Error::new(cpal::ErrorKind::UnsupportedConfig)),
    }
    .map_err(|error| RecordingError::Device(format!("build input stream: {error}")))?;

    stream
        .play()
        .map_err(|error| RecordingError::Device(format!("start input stream: {error}")))?;
    let mut stream = Some(stream);

    let _ = init_tx.send(Ok(()));

    let mut processed_local = Vec::new();
    let mut stopping = false;

    loop {
        if stream_failed.load(Ordering::Acquire) {
            return Err(RecordingError::Device(
                "input stream failed during capture".to_string(),
            ));
        }
        if !stopping && matches!(cmd_rx.try_recv(), Ok(AudioCmd::Stop)) {
            stopping = true;
            stream.take();
        }

        let available = consumer.slots();
        if available >= PROCESS_CHUNK_SIZE || (stopping && available > 0) {
            if let Ok(chunk) = consumer.read_chunk(available.min(PROCESS_CHUNK_SIZE * 8)) {
                let (f, s) = chunk.as_slices();
                let mut dispatch = |frame: AudioFrame| {
                    processed_local.extend_from_slice(&frame.samples);
                    if let Some(tx) = &streaming_tx {
                        let _ = tx.send(frame);
                    }
                };
                processor
                    .process(f, &mut dispatch)
                    .map_err(|e| RecordingError::AudioProcessingError(e.to_string()))?;
                if !s.is_empty() {
                    processor
                        .process(s, &mut dispatch)
                        .map_err(|e| RecordingError::AudioProcessingError(e.to_string()))?;
                }
                chunk.commit_all();
            }
        } else if stopping {
            break;
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }

    processor
        .flush(&mut |frame: AudioFrame| {
            processed_local.extend_from_slice(&frame.samples);
            if let Some(tx) = &streaming_tx {
                let _ = tx.send(frame);
            }
        })
        .map_err(|e| RecordingError::AudioProcessingError(e.to_string()))?;

    if let Ok(mut guard) = processed_samples.lock() {
        *guard = processed_local;
    }

    Ok(())
}

fn build_stream<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut producer: Producer<f32>,
    channels: usize,
    err_fn: impl FnMut(cpal::Error) + Send + 'static,
    overrun_count: Arc<AtomicUsize>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: Sample + SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            if channels == 1 {
                for &sample in data {
                    if producer.push(sample.to_sample::<f32>()).is_err() {
                        overrun_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            } else {
                for frame in data.chunks(channels) {
                    let mut sum = 0.0;
                    for &sample in frame {
                        sum += sample.to_sample::<f32>();
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
