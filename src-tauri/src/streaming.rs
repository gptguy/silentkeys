use std::sync::mpsc::Receiver;

use crate::audio_processing::ProcessingEvent;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    SpeechStart,
    PhraseComplete(Vec<f32>),
    ForcedCommit(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub min_speech_ms: u32,
    pub max_phrase_sec: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            min_speech_ms: 100,
            max_phrase_sec: 15.0,
        }
    }
}

pub struct StreamingTranscriber {
    config: StreamConfig,
    phrase_buffer: Vec<f32>,
    is_speaking: bool,
}

impl StreamingTranscriber {
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            phrase_buffer: Vec::with_capacity(16000 * 30),
            is_speaking: false,
        }
    }

    pub fn handle_event(&mut self, event: ProcessingEvent) -> Vec<StreamEvent> {
        let mut streams = Vec::new();

        match event {
            ProcessingEvent::SpeechStart(samples) => {
                log::debug!(
                    "Streaming: speech start (prefill {} samples, total {})",
                    samples.len(),
                    self.phrase_buffer.len() + samples.len()
                );
                if !self.is_speaking {
                    self.is_speaking = true;
                    streams.push(StreamEvent::SpeechStart);
                }
                self.phrase_buffer.extend_from_slice(&samples);
                self.check_forced_commit(&mut streams);
            }
            ProcessingEvent::Speech(samples) => {
                if self.is_speaking {
                    self.phrase_buffer.extend_from_slice(&samples);
                    self.check_forced_commit(&mut streams);
                } else {
                    self.is_speaking = true;
                    self.phrase_buffer.extend_from_slice(&samples);
                    self.check_forced_commit(&mut streams);
                }
            }
            ProcessingEvent::SpeechEnd => {
                if self.is_speaking {
                    self.is_speaking = false;

                    if !self.phrase_buffer.is_empty() {
                        let duration_sec = self.phrase_buffer.len() as f32 / 16000.0;
                        if duration_sec >= (self.config.min_speech_ms as f32 / 1000.0) {
                            log::info!(
                                "Streaming phrase complete: {:.2}s ({} samples)",
                                duration_sec,
                                self.phrase_buffer.len()
                            );
                            streams.push(StreamEvent::PhraseComplete(self.take_phrase_buffer()));
                        } else {
                            log::debug!(
                                "Streaming: discarding short phrase ({:.2}s, {} samples)",
                                duration_sec,
                                self.phrase_buffer.len()
                            );
                            self.phrase_buffer.clear();
                        }
                    }
                }
            }
        }

        streams
    }

    fn check_forced_commit(&mut self, events: &mut Vec<StreamEvent>) {
        if self.is_speaking {
            let duration_sec = self.phrase_buffer.len() as f32 / 16000.0;
            if duration_sec > self.config.max_phrase_sec {
                log::info!(
                    "Streaming forced commit at {:.2}s ({} samples)",
                    duration_sec,
                    self.phrase_buffer.len()
                );
                if !self.phrase_buffer.is_empty() {
                    events.push(StreamEvent::ForcedCommit(self.take_phrase_buffer()));
                }
            }
        }
    }

    fn take_phrase_buffer(&mut self) -> Vec<f32> {
        let buffer = std::mem::take(&mut self.phrase_buffer);
        self.phrase_buffer.reserve(16000 * 15);
        buffer
    }

    pub fn flush(&mut self) -> Option<Vec<f32>> {
        if !self.phrase_buffer.is_empty() {
            let duration_sec = self.phrase_buffer.len() as f32 / 16000.0;
            if duration_sec > 0.1 {
                return Some(self.take_phrase_buffer());
            }
        }
        self.is_speaking = false;
        None
    }
}

pub fn run_streaming<F>(rx: Receiver<ProcessingEvent>, mut on_phrase: F)
where
    F: FnMut(Vec<f32>, bool),
{
    let mut transcriber = StreamingTranscriber::new(StreamConfig::default());

    log::info!("Streaming loop started (ProcessingEvent driven)");

    loop {
        match rx.recv() {
            Ok(event) => {
                for stream_event in transcriber.handle_event(event) {
                    match stream_event {
                        StreamEvent::PhraseComplete(phrase) => {
                            on_phrase(phrase, false);
                        }
                        StreamEvent::ForcedCommit(phrase) => {
                            on_phrase(phrase, true);
                        }
                        StreamEvent::SpeechStart => {}
                    }
                }
            }
            Err(_) => {
                log::info!("Streaming source disconnected");
                break;
            }
        }
    }

    if let Some(samples) = transcriber.flush() {
        on_phrase(samples, false);
    }

    log::info!("Streaming loop ended");
}

pub fn process_streaming_loop<F>(
    rx: Receiver<ProcessingEvent>,
    engine: &crate::engine::SpeechEngine,
    mut on_transcription: F,
) where
    F: FnMut(String),
{
    run_streaming(rx, |samples, is_forced| {
        log::debug!(
            "Streaming chunk received: {} samples (forced={})",
            samples.len(),
            is_forced
        );

        match engine.transcribe_samples(samples, is_forced) {
            Ok(text) if !text.is_empty() => {
                on_transcription(text);
            }
            Ok(_) => {
                log::debug!("Transcription returned empty text (forced={})", is_forced);
            }
            Err(e) => log::error!("Transcription error: {e}"),
        }
    });
}
