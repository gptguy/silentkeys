use crate::asr::{InferenceConfig, TARGET_SAMPLE_RATE};
use crate::audio_processing::AudioFrame;
use crate::streaming::{
    decoding::DecodingSession,
    word_hypothesis::{WordHypothesisConfig, WordHypothesisManager},
    words::tokens_to_words,
    TranscriptionPatch,
};
use rtrb::RingBuffer;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

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

    pub fn start<F>(
        &self,
        rx: std::sync::mpsc::Receiver<AudioFrame>,
        model: Arc<std::sync::RwLock<Option<crate::asr::AsrModel>>>,
        on_update: F,
    ) where
        F: Fn(TranscriptionPatch) + Send + 'static,
    {
        self.running.store(true, Ordering::SeqCst);
        let run = self.running.clone();
        let (mut prod, mut cons) = RingBuffer::<f32>::new(60 * TARGET_SAMPLE_RATE as usize);
        let ri = run.clone();
        thread::spawn(move || Self::run_ingest(ri, rx, &mut prod));
        thread::spawn(move || Self::run_decode(run, model, &mut cons, on_update));
    }

    fn run_ingest(
        running: Arc<AtomicBool>,
        rx: std::sync::mpsc::Receiver<AudioFrame>,
        prod: &mut rtrb::Producer<f32>,
    ) {
        log::info!("Streaming ingest thread started");
        let mut frames = 0usize;
        while running.load(Ordering::Relaxed) {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(f) => {
                    frames += 1;
                    if frames == 1 || frames.is_multiple_of(100) {
                        log::debug!("Ingest: Frame #{}, len: {}", frames, f.samples.len());
                    }
                    for &s in &f.samples {
                        if prod.push(s).is_err() {
                            log::warn!("Streaming buffer full");
                            break;
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => {
                    running.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }
    }

    fn run_decode<F>(
        running: Arc<AtomicBool>,
        model_arc: Arc<std::sync::RwLock<Option<crate::asr::AsrModel>>>,
        cons: &mut rtrb::Consumer<f32>,
        on_update: F,
    ) where
        F: Fn(TranscriptionPatch) + Send + 'static,
    {
        log::info!("Streaming decode thread starting");
        let mut session = DecodingSession::new(InferenceConfig::streaming_from_env());
        let mut hyps = WordHypothesisManager::new(WordHypothesisConfig::default());
        let (mut total_samples, mut samples) = (0, Vec::new());

        while running.load(Ordering::Relaxed) {
            let start = Instant::now();
            samples.clear();
            if let Ok(chunk) = cons.read_chunk(cons.slots()) {
                let (f, s) = chunk.as_slices();
                samples.extend_from_slice(f);
                samples.extend_from_slice(s);
                chunk.commit_all();
            }

            if !samples.is_empty() {
                total_samples += samples.len();
                let audio_ms = (total_samples as i64 * 1000) / TARGET_SAMPLE_RATE as i64;
                let res = model_arc
                    .write()
                    .map_err(|e| e.to_string())
                    .and_then(|mut g| {
                        g.as_mut()
                            .map(|m| session.advance_segment(m, &samples))
                            .unwrap_or_else(|| Err("Model not loaded".to_string()))
                    });

                if let Ok(tokens) = res {
                    if !tokens.is_empty() {
                        if let Some((f, t)) = hyps.update_draft(tokens_to_words(tokens), audio_ms) {
                            session.commit_to(f, t);
                        }
                        if let Some((start, text)) = hyps.take_newly_committed() {
                            on_update(TranscriptionPatch {
                                start,
                                end: start,
                                text,
                                stable: true,
                            });
                        }
                        on_update(TranscriptionPatch {
                            start: 0,
                            end: usize::MAX,
                            text: hyps.get_full_text(),
                            stable: false,
                        });
                    }
                } else if let Err(e) = res {
                    log::error!("Streaming decode error: {}", e);
                }
            }
            thread::sleep(Duration::from_millis(50).saturating_sub(start.elapsed()));
        }

        let rem = hyps.get_draft_only_text();
        if !rem.is_empty() {
            let start = hyps.stable_char_len;
            on_update(TranscriptionPatch {
                start,
                end: start,
                text: format!(" {}", rem.trim_start()),
                stable: true,
            });
        }
        log::info!("Streaming decode loop exiting");
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}
