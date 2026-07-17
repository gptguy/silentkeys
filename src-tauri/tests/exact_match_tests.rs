use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use silent_keys_lib::asr::{fallback_model_root, resolve_model_dir, AsrModel};
use silent_keys_lib::audio_processing::AudioFrame;
use silent_keys_lib::streaming::StreamingPipeline;

fn get_wav_path() -> PathBuf {
    PathBuf::from("tests/samples/jfk.wav")
}

fn model_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
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

fn report_performance(mode: &str, sample_count: usize, elapsed: Duration) {
    let audio_seconds = sample_count as f64 / 16_000.0;
    eprintln!(
        "{mode}: {:.2}s audio in {:.2}s (RTF {:.3}, {:.1}x real time)",
        audio_seconds,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / audio_seconds,
        audio_seconds / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "requires the downloaded Nemotron model"]
fn test_exact_match_non_streaming() {
    let _guard = model_test_lock();
    let model_root = std::env::var("SILENT_KEYS_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback_model_root());
    let model_dir = resolve_model_dir(&model_root).expect("Model dir not resolved");

    let mut model = AsrModel::new(&model_dir, "en-US").expect("Failed to load model");
    let samples = load_samples(&get_wav_path());

    let started = Instant::now();
    let transcript = model
        .transcribe_samples(&samples)
        .expect("Transcription failed");
    report_performance("offline", samples.len(), started.elapsed());

    let expected = normalize_text(&load_ground_truth());
    let actual = normalize_text(&transcript);

    assert_eq!(actual, expected, "Non-streaming transcription mismatch");
}

#[test]
#[ignore = "requires the downloaded Nemotron model"]
fn test_exact_match_streaming() {
    let _guard = model_test_lock();
    let _ = env_logger::builder().is_test(true).try_init();

    let model_root = std::env::var("SILENT_KEYS_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback_model_root());
    let model_dir = resolve_model_dir(&model_root).expect("Model dir not resolved");

    let model = AsrModel::new(&model_dir, "en-US").expect("Failed to load model");
    let model_arc = Arc::new(RwLock::new(Some(model)));

    let pipeline = Arc::new(StreamingPipeline::new());
    let (tx, rx) = std::sync::mpsc::channel();

    let accumulated_text = Arc::new(Mutex::new(String::new()));
    let acc_clone = accumulated_text.clone();

    pipeline
        .start(rx, model_arc.clone(), move |update| {
            let mut guard = acc_clone.lock().unwrap();
            match update {
                silent_keys_lib::streaming::TranscriptionUpdate::Append(text) => {
                    guard.push_str(&text)
                }
                silent_keys_lib::streaming::TranscriptionUpdate::Replace(text) => *guard = text,
            }
            Ok(())
        })
        .expect("streaming pipeline should start");

    let samples = load_samples(&get_wav_path());
    let chunk_size = 3200;
    let expected = load_ground_truth();
    let expected_norm = normalize_text(&expected);
    let start = Instant::now();
    let timeout = Duration::from_secs(10);

    for chunk in samples.chunks(chunk_size) {
        tx.send(AudioFrame {
            samples: chunk.to_vec(),
        })
        .expect("Failed to send chunk");

        std::thread::sleep(Duration::from_millis(5));
    }
    drop(tx);

    loop {
        if start.elapsed() > timeout {
            break;
        }

        {
            let accumulated = accumulated_text.lock().unwrap();
            let current = normalize_text(&accumulated);
            if !current.is_empty() && current == expected_norm {
                log::info!("Match found!");
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    pipeline.finish().expect("pipeline should finish");

    let text = accumulated_text.lock().unwrap().clone();
    let normalized = normalize_text(&text);
    report_performance("streaming partials", samples.len(), start.elapsed());

    assert_eq!(
        normalized, expected_norm,
        "Streaming transcription mismatch after timeout"
    );

    let final_text = model_arc
        .write()
        .expect("model lock should be available")
        .as_mut()
        .expect("model should remain loaded")
        .transcribe_samples(&samples)
        .expect("Final transcription failed");
    report_performance(
        "streaming mode through final correction",
        samples.len(),
        start.elapsed(),
    );
    assert_eq!(
        normalize_text(&final_text),
        expected_norm,
        "Final streaming-mode transcription mismatch"
    );
}
