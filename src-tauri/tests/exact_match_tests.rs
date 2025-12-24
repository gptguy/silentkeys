use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use silent_keys_lib::asr::{fallback_model_root, resolve_model_dir, AsrModel, InferenceConfig};
use silent_keys_lib::audio_processing::AudioFrame;
use silent_keys_lib::streaming::StreamingPipeline;

fn get_wav_path() -> PathBuf {
    PathBuf::from("tests/samples/jfk.wav")
}

fn get_transcript_path() -> PathBuf {
    PathBuf::from("tests/samples/transcript.txt")
}

fn load_samples(path: &PathBuf) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "Expected mono audio");
    assert_eq!(spec.sample_rate, 16000, "Expected 16kHz audio");

    reader
        .samples::<i16>()
        .map(|s| s.expect("Failed to read sample") as f32 / 32768.0)
        .collect()
}

fn load_ground_truth() -> String {
    std::fs::read_to_string(get_transcript_path())
        .expect("Failed to read transcript")
        .trim()
        .to_string()
}

fn normalize_text(text: &str) -> String {
    let mut s = text.to_lowercase();
    s.retain(|c| !c.is_ascii_punctuation());
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn test_exact_match_non_streaming() {
    std::env::set_var("ASR_MAX_TOKENS_PER_STEP", "100");
    let model_root = std::env::var("SILENT_KEYS_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback_model_root());
    let model_dir = resolve_model_dir(&model_root).expect("Model dir not resolved");

    let mut model = AsrModel::new(&model_dir, true).expect("Failed to load model");
    let samples = load_samples(&get_wav_path());

    let transcript = model
        .transcribe_samples(samples, false, Some(InferenceConfig::from_env()))
        .expect("Transcription failed");

    let expected = normalize_text(&load_ground_truth());
    let actual = normalize_text(&transcript.text);

    assert_eq!(actual, expected, "Non-streaming transcription mismatch");
}

#[test]
fn test_exact_match_streaming() {
    std::env::set_var("ASR_MAX_TOKENS_PER_STEP", "100");
    let _ = env_logger::builder().is_test(true).try_init();

    let model_root = std::env::var("SILENT_KEYS_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback_model_root());
    let model_dir = resolve_model_dir(&model_root).expect("Model dir not resolved");

    // Load model shared
    let model = AsrModel::new(&model_dir, true).expect("Failed to load model");
    let model_arc = Arc::new(RwLock::new(Some(model)));

    let pipeline = Arc::new(StreamingPipeline::new());
    let (tx, rx) = std::sync::mpsc::channel();

    let accumulated_text = Arc::new(Mutex::new(String::new()));
    let latest_full_text = Arc::new(Mutex::new(String::new()));
    let acc_clone = accumulated_text.clone();
    let full_clone = latest_full_text.clone();

    pipeline.start(rx, model_arc.clone(), move |patch| {
        if patch.stable && !patch.text.is_empty() {
            let mut guard = acc_clone.lock().unwrap();
            if !guard.is_empty() && !guard.ends_with(' ') && !patch.text.starts_with(' ') {
                guard.push(' ');
            }
            guard.push_str(&patch.text);
            log::info!("Test accumulator: '{}'", *guard);
            return;
        }

        if !patch.stable {
            if let Ok(mut guard) = full_clone.lock() {
                *guard = patch.text;
            }
        }
    });

    let samples = load_samples(&get_wav_path());
    let chunk_size = 3200; // 200ms
    let expected = load_ground_truth();
    let expected_norm = normalize_text(&expected);
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(10);

    for chunk in samples.chunks(chunk_size) {
        tx.send(AudioFrame {
            samples: chunk.to_vec(),
        })
        .expect("Failed to send chunk");

        std::thread::sleep(Duration::from_millis(5));
    }

    loop {
        if start.elapsed() > timeout {
            // Force stop to dump status
            pipeline.stop();
            break;
        }

        {
            let full_guard = latest_full_text.lock().unwrap();
            let current = normalize_text(&full_guard);
            if !current.is_empty() && current == expected_norm {
                log::info!("Match found!");
                return; // Success
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    pipeline.stop();
    std::thread::sleep(Duration::from_millis(250));

    let full_text = latest_full_text.lock().unwrap().clone();
    let fallback_text = accumulated_text.lock().unwrap().clone();
    let normalized = if full_text.trim().is_empty() {
        normalize_text(&fallback_text)
    } else {
        normalize_text(&full_text)
    };

    assert_eq!(
        normalized, expected_norm,
        "Streaming transcription mismatch after timeout"
    );
}
