use silent_keys_lib::streaming::*;
use silent_keys_lib::vad::{VadError, VAD_CHUNK_SIZE};

#[test]
fn default_config_has_sane_values() {
    let config = StreamConfig::default();
    assert!((config.speech_threshold - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.min_speech_ms, 250);
    assert_eq!(config.min_silence_ms, 400);
    assert!((config.max_phrase_sec - 15.0).abs() < f32::EPSILON);
}

#[test]
fn stream_state_transitions() {
    let state = StreamState::Idle;
    assert_eq!(state, StreamState::Idle);
}

struct MockVad {
    probabilities: std::collections::VecDeque<f32>,
}

impl MockVad {
    fn new(probs: Vec<f32>) -> Self {
        Self {
            probabilities: probs.into(),
        }
    }
}

impl VadPredictor for MockVad {
    fn process_chunk(&mut self, _samples: &[f32]) -> Result<f32, VadError> {
        Ok(self.probabilities.pop_front().unwrap_or(0.0))
    }

    fn reset(&mut self) {
        // No-op for mock
    }
}

#[test]
fn sequence_detection_with_mock() {
    let config = StreamConfig {
        min_speech_ms: 0,
        min_silence_ms: 0,
        ..Default::default()
    };

    // 0.0 (Idle) -> 0.8 (Speak) -> 0.8 (Speak) -> 0.0 (Silence) -> 0.0 (Silence) -> Complete
    let vad = MockVad::new(vec![
        0.0, 0.0, // Idle
        0.8, 0.8, 0.8, 0.8, 0.8, // Speech
        0.0, 0.0, 0.0, 0.0, 0.0, // Silence
    ]);

    let mut transcriber = StreamingTranscriber::new_with_predictor(Box::new(vad), config);

    let chunk = vec![0.0; VAD_CHUNK_SIZE];
    let mut event_count = 0;
    let mut phrases = 0;

    // Feed samples
    for _ in 0..15 {
        let events = transcriber.feed_samples(&chunk);
        for event in events {
            match event {
                StreamEvent::SpeechStart => event_count += 1,
                StreamEvent::PhraseComplete(_) => phrases += 1,
                _ => {}
            }
        }
    }

    assert_eq!(event_count, 1, "Should have one speech start");
    assert_eq!(phrases, 1, "Should have one phrase complete");
}

#[test]
fn flush_with_insufficient_speech() {
    let config = StreamConfig {
        min_speech_ms: 500, // 500ms required
        ..Default::default()
    };

    // Short speech: 0.8, 0.8 (60ms) then stop
    let vad = MockVad::new(vec![0.8, 0.8]);
    let mut transcriber = StreamingTranscriber::new_with_predictor(Box::new(vad), config);

    let chunk = vec![0.0; VAD_CHUNK_SIZE];
    transcriber.feed_samples(&chunk);
    transcriber.feed_samples(&chunk);

    let flush_result = transcriber.flush();
    assert!(
        flush_result.is_none(),
        "Should discard short speech on flush"
    );
}

#[test]
fn forced_commit_at_max_duration() {
    let config = StreamConfig {
        max_phrase_sec: 0.0, // Force commit immediately
        ..Default::default()
    };

    // Continuous speech
    let probs = vec![0.9; 100];

    let vad = MockVad::new(probs);
    let mut transcriber = StreamingTranscriber::new_with_predictor(Box::new(vad), config);

    let chunk = vec![0.0; VAD_CHUNK_SIZE];
    let mut forced_commits = 0;

    for _ in 0..10 {
        std::thread::sleep(std::time::Duration::from_millis(1));
        for event in transcriber.feed_samples(&chunk) {
            if let StreamEvent::ForcedCommit(_) = event {
                forced_commits += 1;
            }
        }
    }

    assert!(forced_commits >= 1, "Should trigger forced commit");
}
