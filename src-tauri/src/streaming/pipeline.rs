use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::audio_processing::AudioFrame;

use crate::asr::{InferenceConfig, TARGET_SAMPLE_RATE};
use crate::streaming::decoding::DecodingSession;
use crate::streaming::word_aggregation::tokens_to_words;
use crate::streaming::word_hypothesis::{WordHypothesisConfig, WordHypothesisManager};
use rtrb::RingBuffer;

pub struct StreamingPipeline {
    running: Arc<AtomicBool>,
}

impl Default for StreamingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingPipeline {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts the streaming pipeline. Binds to the given AudioFrame receiver.
    ///
    /// `on_update`: Callback invoked whenever the hypothesis changes (draft or commit).
    /// `model`: The shared ASR model.
    pub fn start<F>(
        &self,
        rx: std::sync::mpsc::Receiver<AudioFrame>,
        model: Arc<std::sync::RwLock<Option<crate::asr::AsrModel>>>,
        on_update: F,
    ) where
        F: Fn(crate::streaming::TranscriptionPatch) + Send + 'static,
    {
        self.running.store(true, Ordering::SeqCst);
        let running_flag = self.running.clone();
        let buffer_samples = 10 * TARGET_SAMPLE_RATE as usize;
        let (mut producer, mut consumer) = RingBuffer::<f32>::new(buffer_samples);

        let running_ingest = running_flag.clone();
        thread::spawn(move || {
            log::info!("Streaming ingest thread started");
            let mut frame_count = 0usize;
            let mut sample_count = 0usize;
            let mut dropped_samples = 0usize;
            let mut warned = false;
            while running_ingest.load(Ordering::Relaxed) {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(frame) => {
                        frame_count += 1;
                        sample_count += frame.samples.len();
                        if frame_count == 1 || frame_count.is_multiple_of(10) {
                            log::debug!(
                                "Ingest: Received frame #{}, total samples: {}",
                                frame_count,
                                sample_count
                            );
                        }
                        for &sample in &frame.samples {
                            if producer.push(sample).is_err() {
                                dropped_samples += 1;
                                if !warned {
                                    log::warn!("Streaming buffer full; dropping samples");
                                    warned = true;
                                }
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => {
                        log::info!("Ingest: Channel disconnected after {} frames", frame_count);
                        // Signal decode thread to stop
                        running_ingest.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
            log::info!(
                "Streaming ingest thread exiting (frames: {}, samples: {})",
                frame_count,
                sample_count
            );
            if dropped_samples > 0 {
                log::warn!("Streaming ingest dropped {} samples", dropped_samples);
            }
        });

        thread::spawn(move || {
            log::info!("Streaming decode thread starting");

            let model_arc = model;

            let mut session = DecodingSession::new(InferenceConfig::streaming_from_env());
            let config = WordHypothesisConfig::default();
            let mut hypothesis = WordHypothesisManager::new(config);

            let tick_rate = Duration::from_millis(50);

            let mut total_samples = 0;
            let mut tick_count = 0usize;
            let mut samples = Vec::new();

            while running_flag.load(Ordering::Relaxed) {
                let start = Instant::now();
                tick_count += 1;

                samples.clear();
                let mut available = consumer.slots();
                while available > 0 {
                    if let Ok(chunk) = consumer.read_chunk(available) {
                        let (first, second) = chunk.as_slices();
                        samples.extend_from_slice(first);
                        samples.extend_from_slice(second);
                        chunk.commit_all();
                    } else {
                        log::warn!("Streaming decode failed to read audio chunk");
                        break;
                    }
                    available = consumer.slots();
                }

                if tick_count <= 5 || tick_count.is_multiple_of(20) {
                    log::debug!("Decode tick #{}: {} samples", tick_count, samples.len());
                }

                if !samples.is_empty() {
                    total_samples += samples.len();
                    let current_audio_ms =
                        (total_samples as i64 * 1000) / TARGET_SAMPLE_RATE as i64;

                    let result = {
                        let mut guard = match model_arc.write() {
                            Ok(g) => g,
                            Err(e) => {
                                log::error!("Model lock poisoned: {}", e);
                                break;
                            }
                        };

                        if let Some(model) = guard.as_mut() {
                            session.advance_segment(model, &samples)
                        } else {
                            Err("Model not loaded".to_string())
                        }
                    };

                    match result {
                        Ok(new_tokens) => {
                            if !new_tokens.is_empty() {
                                log::debug!(
                                    "Streaming: Decoded {} new tokens",
                                    new_tokens.len()
                                );

                                let new_words = tokens_to_words(new_tokens);

                                let commit_result =
                                    hypothesis.update_draft(new_words, current_audio_ms);

                                if let Some((commit_frame, commit_token)) = commit_result {
                                    log::debug!(
                                        "Advancing decoder cursor to frame={}, token={}",
                                        commit_frame,
                                        commit_token
                                    );
                                    session.commit_to(commit_frame, commit_token);
                                }

                                if let Some(new_committed) = hypothesis.take_newly_committed() {
                                    log::debug!(
                                        "Streaming: Emitting stable text len={}",
                                        new_committed.len()
                                    );
                                    let stable_patch = crate::streaming::TranscriptionPatch {
                                        start: 0,
                                        end: 0,
                                        text: new_committed,
                                        stable: true,
                                    };
                                    on_update(stable_patch);
                                }

                                let full_text = hypothesis.get_full_text();
                                let ui_patch = crate::streaming::TranscriptionPatch {
                                    start: 0,
                                    end: u32::MAX as usize,
                                    text: full_text,
                                    stable: false,
                                };
                                on_update(ui_patch);
                            }
                        }
                        Err(e) => {
                            log::error!("Streaming decode error: {}", e);
                        }
                    }
                }

                let elapsed = start.elapsed();
                if elapsed < tick_rate {
                    thread::sleep(tick_rate - elapsed);
                }
            }

            log::info!("Decode loop ending - flushing remaining draft");
            let remaining_text = hypothesis.get_draft_only_text();
            if !remaining_text.is_empty() {
                log::info!(
                    "Final flush: emitting {} chars of draft-only text",
                    remaining_text.len()
                );
                let final_patch = crate::streaming::TranscriptionPatch {
                    start: 0,
                    end: 0,
                    text: format!(" {}", remaining_text.trim_start()),
                    stable: true,
                };
                on_update(final_patch);
            }

            log::info!("Streaming decode loop exiting");
        });
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}
