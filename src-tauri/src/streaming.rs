use std::time::Instant;

use crate::vad::{SileroVad, VadError, VadModel, VAD_CHUNK_SIZE};

pub trait VadPredictor: Send {
    fn process_chunk(&mut self, samples: &[f32]) -> Result<f32, VadError>;
    fn reset(&mut self);
}

impl VadPredictor for SileroVad {
    fn process_chunk(&mut self, samples: &[f32]) -> Result<f32, VadError> {
        self.process_chunk(samples)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    SpeechStart,

    PhraseComplete(Vec<f32>),

    ForcedCommit(Vec<f32>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Idle,

    Speaking,

    TrailingSilence,
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub speech_threshold: f32,

    pub min_speech_ms: u32,

    pub min_silence_ms: u32,

    pub max_phrase_sec: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.3,
            min_speech_ms: 100,
            min_silence_ms: 300,
            max_phrase_sec: 15.0,
        }
    }
}

pub struct StreamingTranscriber {
    vad: Box<dyn VadPredictor>,
    config: StreamConfig,
    state: StreamState,

    phrase_buffer: Vec<f32>,

    pending_samples: std::collections::VecDeque<f32>,

    chunk_buffer: Vec<f32>,

    speech_start: Option<Instant>,

    silence_start: Option<Instant>,
}

impl StreamingTranscriber {
    pub fn new(vad_model: std::sync::Arc<VadModel>, config: StreamConfig) -> Self {
        let vad = SileroVad::new_with_model(vad_model);

        Self {
            vad: Box::new(vad),
            config,
            state: StreamState::Idle,
            phrase_buffer: Vec::with_capacity(16000 * 30),
            pending_samples: std::collections::VecDeque::with_capacity(VAD_CHUNK_SIZE * 2),
            chunk_buffer: Vec::with_capacity(VAD_CHUNK_SIZE),
            speech_start: None,
            silence_start: None,
        }
    }

    #[doc(hidden)]
    pub fn new_with_predictor(vad: Box<dyn VadPredictor>, config: StreamConfig) -> Self {
        Self {
            vad,
            config,
            state: StreamState::Idle,
            phrase_buffer: Vec::with_capacity(16000 * 30),
            pending_samples: std::collections::VecDeque::with_capacity(VAD_CHUNK_SIZE * 2),
            chunk_buffer: Vec::with_capacity(VAD_CHUNK_SIZE),
            speech_start: None,
            silence_start: None,
        }
    }

    pub fn feed_samples(&mut self, samples: &[f32]) -> Vec<StreamEvent> {
        let mut events = Vec::with_capacity(2);

        self.pending_samples.extend(samples.iter().copied());
        if self.state != StreamState::Idle {
            self.phrase_buffer.extend_from_slice(samples);
        }

        while self.pending_samples.len() >= VAD_CHUNK_SIZE {
            self.chunk_buffer.clear();
            self.chunk_buffer
                .extend(self.pending_samples.drain(..VAD_CHUNK_SIZE));

            match self.vad.process_chunk(&self.chunk_buffer) {
                Ok(prob) => {
                    log::debug!("VAD probability: {:.3}, state: {}", prob, self.state());
                    if let Some(event) = self.update_state_with_chunk(prob) {
                        events.push(event);
                    }
                }
                Err(e) => log::error!("VAD processing error: {}", e),
            }
        }

        if let Some(start) = self.speech_start {
            let duration = start.elapsed().as_secs_f32();
            if duration > self.config.max_phrase_sec {
                log::info!(
                    "Forced commit: phrase duration {:.1}s > max {:.1}s",
                    duration,
                    self.config.max_phrase_sec
                );
                events.push(StreamEvent::ForcedCommit(self.take_phrase_buffer()));
                self.reset_state();
            }
        }

        events
    }

    #[inline]
    fn update_state_with_chunk(&mut self, prob: f32) -> Option<StreamEvent> {
        let is_speech = prob >= self.config.speech_threshold;

        match self.state {
            StreamState::Idle => {
                if is_speech {
                    self.state = StreamState::Speaking;
                    self.speech_start = Some(Instant::now());
                    self.silence_start = None;

                    self.phrase_buffer.extend_from_slice(&self.chunk_buffer);
                    log::debug!(
                        "Speech started (prob={:.2}), added {} samples to phrase buffer",
                        prob,
                        self.chunk_buffer.len()
                    );
                    return Some(StreamEvent::SpeechStart);
                }
            }
            StreamState::Speaking => {
                if !is_speech {
                    self.state = StreamState::TrailingSilence;
                    self.silence_start = Some(Instant::now());
                    log::debug!("Entering trailing silence (prob={:.2})", prob);
                }
            }
            StreamState::TrailingSilence => {
                if is_speech {
                    self.state = StreamState::Speaking;
                    self.silence_start = None;
                    log::debug!("Speech resumed (prob={:.2})", prob);
                } else if let Some(silence_start) = self.silence_start {
                    let silence_ms = silence_start.elapsed().as_millis() as u32;
                    if silence_ms >= self.config.min_silence_ms {
                        if let Some(speech_start) = self.speech_start {
                            let speech_ms = speech_start.elapsed().as_millis() as u32;
                            if speech_ms >= self.config.min_speech_ms {
                                log::info!(
                                    "Phrase complete: {:.1}s speech, {:.0}ms silence",
                                    speech_ms as f32 / 1000.0,
                                    silence_ms
                                );
                                let buffer = self.take_phrase_buffer();
                                self.reset_state();
                                return Some(StreamEvent::PhraseComplete(buffer));
                            }
                        }

                        log::debug!("Discarding short utterance");
                        self.reset_state();
                    }
                }
            }
        }

        None
    }

    #[inline]
    fn take_phrase_buffer(&mut self) -> Vec<f32> {
        let buffer = std::mem::take(&mut self.phrase_buffer);
        self.phrase_buffer.reserve(16000 * 30);
        buffer
    }

    fn reset_state(&mut self) {
        self.state = StreamState::Idle;
        self.phrase_buffer.clear();
        self.speech_start = None;
        self.silence_start = None;
        self.vad.reset();
    }

    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if self.phrase_buffer.is_empty() {
            log::warn!("Flush: phrase_buffer is empty (VAD never detected speech)");
            self.reset_state();
            return None;
        }

        if let Some(start) = self.speech_start {
            let speech_ms = start.elapsed().as_millis() as u32;
            if speech_ms >= self.config.min_speech_ms {
                log::info!(
                    "Flush: returning {:.1}s of audio",
                    speech_ms as f32 / 1000.0
                );
                let buffer = self.take_phrase_buffer();
                self.reset_state();
                return Some(buffer);
            } else {
                log::warn!(
                    "Flush: speech too short ({}ms < {}ms required)",
                    speech_ms,
                    self.config.min_speech_ms
                );
            }
        } else {
            log::warn!("Flush: speech_start is None (VAD detected speech but state was reset?)");
        }

        log::debug!("Flush: discarding insufficient audio");
        self.reset_state();
        None
    }

    pub fn state(&self) -> &'static str {
        match self.state {
            StreamState::Idle => "idle",
            StreamState::Speaking => "speaking",
            StreamState::TrailingSilence => "trailing_silence",
        }
    }

    pub fn phrase_buffer_len(&self) -> usize {
        self.phrase_buffer.len()
    }

    pub fn phrase_duration_sec(&self) -> f32 {
        self.phrase_buffer.len() as f32 / 16000.0
    }
}

pub fn run_streaming<F>(vad_model: std::sync::Arc<VadModel>, mut on_phrase: F)
where
    F: FnMut(Vec<f32>),
{
    let mut transcriber = StreamingTranscriber::new(vad_model, StreamConfig::default());

    log::info!("VAD streaming loop started");
    let recorder = crate::recording::Recorder::global();

    loop {
        std::thread::sleep(std::time::Duration::from_millis(32));

        if !recorder.is_recording() {
            if let Some(samples) = transcriber.flush() {
                on_phrase(samples);
            }
            break;
        }

        if let Ok(samples) = recorder.drain() {
            if !samples.is_empty() {
                for event in transcriber.feed_samples(&samples) {
                    match event {
                        StreamEvent::PhraseComplete(phrase) | StreamEvent::ForcedCommit(phrase) => {
                            on_phrase(phrase);
                        }
                        StreamEvent::SpeechStart => {}
                    }
                }
            }
        }
    }
    log::info!("VAD streaming loop ended");
}
