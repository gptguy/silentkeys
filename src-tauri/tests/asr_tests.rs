use silent_keys_lib::asr::{
    current_download_progress, mark_finished, record_failure, set_file_index, start_tracking,
    AsrError, Transcript,
};

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
