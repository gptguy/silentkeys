use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use silent_keys_lib::asr::{fallback_model_root, resolve_model_dir, AsrModel};
use unicode_segmentation::UnicodeSegmentation;

const CORPUS_MANIFEST_ENV: &str = "SILENT_KEYS_CORPUS_MANIFEST";

#[derive(Deserialize)]
struct Corpus {
    corpus_version: u32,
    max_wer: f64,
    clips: Vec<Clip>,
}

#[derive(Deserialize)]
struct Clip {
    name: String,
    audio: String,
    transcript: String,
    language: String,
}

fn load_samples(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("failed to open corpus WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "corpus audio must be mono");
    assert_eq!(spec.sample_rate, 16_000, "corpus audio must be 16 kHz");
    reader
        .samples::<i16>()
        .map(|s| s.expect("failed to read sample") as f32 / 32768.0)
        .collect()
}

fn normalize_words(text: &str) -> Vec<String> {
    text.unicode_words().map(str::to_lowercase).collect()
}

fn corpus_version_from_name(name: &str) -> Option<u32> {
    name.strip_prefix('v')?.strip_suffix(".json")?.parse().ok()
}

fn newest_manifest_name<'a>(names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    names
        .filter_map(|name| corpus_version_from_name(name).map(|version| (version, name)))
        .max_by_key(|(version, _)| *version)
        .map(|(_, name)| name)
}

fn corpus_manifest_path() -> PathBuf {
    if let Some(path) = std::env::var_os(CORPUS_MANIFEST_ENV).filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }

    let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let names: Vec<String> = std::fs::read_dir(&corpus_dir)
        .expect("corpus directory should exist")
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    let name = newest_manifest_name(names.iter().map(String::as_str))
        .expect("corpus directory should contain a versioned manifest");
    corpus_dir.join(name)
}

fn word_edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0; hypothesis.len() + 1];
    for (i, reference_word) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, hypothesis_word) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(reference_word != hypothesis_word);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

#[test]
fn word_edit_distance_counts_substitutions_insertions_deletions() {
    let reference = normalize_words("ask not what your country can do");
    assert_eq!(word_edit_distance(&reference, &reference), 0);
    assert_eq!(
        word_edit_distance(
            &reference,
            &normalize_words("ask not what her country can do")
        ),
        1
    );
    assert_eq!(
        word_edit_distance(&reference, &normalize_words("ask what your country can do")),
        1
    );
    assert_eq!(
        word_edit_distance(
            &reference,
            &normalize_words("so ask not what your country can do")
        ),
        1
    );
}

#[test]
fn normalization_ignores_case_and_punctuation() {
    assert_eq!(
        normalize_words("Ask not, what your Country can do!"),
        normalize_words("ask not what your country can do")
    );
}

#[test]
fn normalization_handles_unicode_words_and_punctuation() {
    assert_eq!(
        normalize_words("Bonjour, L’ÉTÉ — déjà! 你好，世界。"),
        normalize_words("bonjour l’été déjà 你 好 世 界")
    );
}

#[test]
fn newest_versioned_manifest_is_selected() {
    let names = ["README.md", "v1.json", "v12.json", "v2.json"];
    assert_eq!(newest_manifest_name(names.into_iter()), Some("v12.json"));
}

#[test]
#[ignore = "requires the downloaded Nemotron model"]
fn corpus_word_error_rate_stays_within_ceiling() {
    let manifest_path = corpus_manifest_path();
    let manifest = std::fs::read_to_string(&manifest_path).expect("corpus manifest should exist");
    let corpus: Corpus = serde_json::from_str(&manifest).expect("corpus manifest should parse");
    assert!(!corpus.clips.is_empty(), "corpus must contain clips");
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let model_root = std::env::var("SILENT_KEYS_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback_model_root());
    let model_dir = resolve_model_dir(&model_root).expect("model dir should resolve");
    let mut model =
        AsrModel::new(&model_dir, &corpus.clips[0].language).expect("model should load");

    let mut total_errors = 0usize;
    let mut total_words = 0usize;
    for clip in &corpus.clips {
        model
            .set_language(&clip.language)
            .expect("corpus clip language must be supported by the model");
        let samples = load_samples(&project_root.join(&clip.audio));
        let reference_text = std::fs::read_to_string(project_root.join(&clip.transcript))
            .expect("corpus transcript should exist");

        let started = Instant::now();
        let hypothesis_text = model
            .transcribe_samples(&samples)
            .expect("transcription should succeed");
        let elapsed = started.elapsed();

        let reference = normalize_words(&reference_text);
        let hypothesis = normalize_words(&hypothesis_text);
        assert!(
            !reference.is_empty(),
            "corpus transcript {} must contain words",
            clip.transcript
        );
        let errors = word_edit_distance(&reference, &hypothesis);
        total_errors += errors;
        total_words += reference.len();

        let audio_seconds = samples.len() as f64 / 16_000.0;
        eprintln!(
            "{}: WER {:.4} ({errors}/{} words, {:.2}s audio, RTF {:.3})",
            clip.name,
            errors as f64 / reference.len() as f64,
            reference.len(),
            audio_seconds,
            elapsed.as_secs_f64() / audio_seconds
        );
    }

    let corpus_wer = total_errors as f64 / total_words as f64;
    eprintln!(
        "corpus v{}: WER {:.4} ({total_errors}/{total_words} words over {} clips)",
        corpus.corpus_version,
        corpus_wer,
        corpus.clips.len()
    );
    assert!(
        corpus_wer <= corpus.max_wer,
        "corpus v{} WER {corpus_wer:.4} exceeds ceiling {:.4}",
        corpus.corpus_version,
        corpus.max_wer
    );
}
