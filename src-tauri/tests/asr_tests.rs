use silent_keys_lib::asr::audio_io::resample_linear;
use silent_keys_lib::asr::download_progress::{
    current_download_progress, mark_finished, record_failure, set_file_index, start_tracking,
};
use silent_keys_lib::asr::{AsrError, Transcript};

#[test]
fn download_progress_lifecycle_is_coherent() {
    start_tracking(3);
    let p = current_download_progress().expect("progress should be initialized");
    assert_eq!(p.file_count, 3);
    assert_eq!(p.file_index, 0);
    assert!(!p.done);
    assert_eq!(p.downloaded_bytes, 0);
    assert_eq!(p.total_bytes, 0);
    assert!(p.error.is_none());

    set_file_index(1);
    let p = current_download_progress().unwrap();
    assert_eq!(p.file_index, 1);

    mark_finished();
    let p = current_download_progress().unwrap();
    assert!(p.done);
    assert_eq!(p.file_index, 3);
    assert!(p.error.is_none());

    record_failure("network error".to_string());
    let p = current_download_progress().unwrap();
    assert!(p.done);
    assert_eq!(p.error.as_deref(), Some("network error"));
}

// ============================================================================
// AsrError Tests
// ============================================================================

#[test]
fn asr_error_user_message_download() {
    let err = AsrError::Download("connection failed".to_string());
    let msg = err.user_message();
    assert!(msg.contains("download") || msg.contains("internet"));
}

#[test]
fn asr_error_user_message_snapshot_not_found() {
    let err = AsrError::SnapshotNotFound("/path/to/model".to_string());
    let msg = err.user_message();
    assert!(msg.contains("missing") || msg.contains("corrupted"));
}

#[test]
fn asr_error_user_message_audio() {
    let err = AsrError::Audio("invalid format".to_string());
    let msg = err.user_message();
    assert!(msg.contains("decode") || msg.contains("recording"));
}

#[test]
fn asr_error_user_message_sample_rate() {
    let err = AsrError::SampleRate(44100);
    let msg = err.user_message();
    assert!(msg.contains("decode") || msg.contains("recording"));
}

#[test]
fn asr_error_user_message_io() {
    let err = AsrError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "file not found",
    ));
    let msg = err.user_message();
    assert!(msg.contains("read") || msg.contains("write") || msg.contains("files"));
}

#[test]
fn asr_error_user_message_runtime_errors() {
    // InputNotFound
    let err = AsrError::InputNotFound("encoder".to_string());
    assert!(!err.user_message().is_empty());

    // OutputNotFound
    let err = AsrError::OutputNotFound("logits".to_string());
    assert!(!err.user_message().is_empty());

    // TensorShape
    let err = AsrError::TensorShape("input_states".to_string());
    assert!(!err.user_message().is_empty());
}

#[test]
fn asr_error_display_includes_details() {
    let err = AsrError::Download("timeout".to_string());
    let display = format!("{}", err);
    assert!(display.contains("download") || display.contains("failed"));
}

// ============================================================================
// Transcript Tests
// ============================================================================

#[test]
fn transcript_clone() {
    let t = Transcript {
        text: "hello world".to_string(),
        timestamps: vec![0.0, 0.5],
        tokens: vec!["hello".to_string(), "world".to_string()],
    };
    let cloned = t.clone();
    assert_eq!(cloned.text, t.text);
    assert_eq!(cloned.timestamps, t.timestamps);
    assert_eq!(cloned.tokens, t.tokens);
}

#[test]
fn transcript_debug() {
    let t = Transcript {
        text: "test".to_string(),
        timestamps: vec![0.0],
        tokens: vec!["test".to_string()],
    };
    let debug = format!("{:?}", t);
    assert!(debug.contains("Transcript"));
    assert!(debug.contains("test"));
}

// ============================================================================
// Audio IO Tests (Resampling)
// ============================================================================

#[test]
fn resample_empty_input() {
    let result = resample_linear(&[], 44100, 16000);
    assert!(result.is_empty());
}

#[test]
fn resample_zero_rates() {
    let input = vec![1.0, 2.0, 3.0];
    assert!(resample_linear(&input, 0, 16000).is_empty());
    assert!(resample_linear(&input, 44100, 0).is_empty());
}

#[test]
fn resample_same_rate() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = resample_linear(&input, 16000, 16000);
    assert_eq!(result.len(), input.len());
    for (a, b) in result.iter().zip(input.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn resample_downsample() {
    // 4 samples at 32kHz -> 2 samples at 16kHz
    let input = vec![0.0, 0.5, 1.0, 0.5];
    let result = resample_linear(&input, 32000, 16000);
    assert_eq!(result.len(), 2);
}

#[test]
fn resample_upsample() {
    // 2 samples at 8kHz -> 4 samples at 16kHz
    let input = vec![0.0, 1.0];
    let result = resample_linear(&input, 8000, 16000);
    assert_eq!(result.len(), 4);
    // First sample should be 0.0
    assert!((result[0] - 0.0).abs() < f32::EPSILON);
}

#[test]
fn resample_single_sample() {
    let input = vec![0.5];
    let result = resample_linear(&input, 44100, 16000);
    assert!(!result.is_empty());
    assert!((result[0] - 0.5).abs() < f32::EPSILON);
}

#[test]
fn resample_interpolation_quality() {
    // Linear interpolation should produce smooth output
    let input = vec![0.0, 1.0, 0.0];
    let result = resample_linear(&input, 8000, 16000);
    // Check that we get smooth interpolation
    assert!(result.len() >= 3);
    // Middle values should be interpolated
    for sample in &result {
        assert!(*sample >= 0.0 && *sample <= 1.0);
    }
}
