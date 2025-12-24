use std::sync::{
    atomic::{AtomicUsize, Ordering},
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
    init_tx: Sender<Result<(), RecordingError>>,
    streaming_tx: Option<Sender<AudioFrame>>,
    overrun_count: Arc<AtomicUsize>,
) -> Result<(), RecordingError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(RecordingError::NoInputDevice)?;

    let stream_config = device
        .default_input_config()
        .map_err(|e| RecordingError::Device(e.to_string()))?;

    let sample_rate = stream_config.sample_rate();
    let channels = stream_config.channels() as usize;

    log::info!(
        "Audio: {} Hz, {} channels, device={:?}",
        sample_rate,
        channels,
        device.description().unwrap()
    );

    let (producer, mut consumer) = RingBuffer::<f32>::new(sample_rate as usize);
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
        _ => Err(cpal::BuildStreamError::DeviceNotAvailable),
    }
    .map_err(|e| RecordingError::Device(e.to_string()))?;

    let mut stream = Some(stream);
    stream
        .as_ref()
        .unwrap()
        .play()
        .map_err(|e| RecordingError::Device(e.to_string()))?;

    let _ = init_tx.send(Ok(()));

    let mut processor = AudioProcessor::new(sample_rate as usize, TARGET_SAMPLE_RATE as usize)
        .map_err(|e| RecordingError::AudioProcessingError(e.to_string()))?;

    let mut processed_local = Vec::new();
    let mut stopping = false;

    loop {
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
                let _ = processor.process(f, &mut dispatch);
                if !s.is_empty() {
                    let _ = processor.process(s, &mut dispatch);
                }
                chunk.commit_all();
            }
        } else if stopping {
            break;
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }

    let _ = processor.flush(&mut |frame: AudioFrame| {
        processed_local.extend_from_slice(&frame.samples);
        if let Some(tx) = &streaming_tx {
            let _ = tx.send(frame);
        }
    });

    if let Ok(mut guard) = processed_samples.lock() {
        *guard = processed_local;
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
    T: Sample + SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    let has_logged = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            if !has_logged.load(Ordering::Relaxed) {
                log::debug!("CPAL: First chunk of {} samples", data.len());
                has_logged.store(true, Ordering::Relaxed);
            }

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
