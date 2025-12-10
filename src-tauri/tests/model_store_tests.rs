use std::time::{SystemTime, UNIX_EPOCH};

use silent_keys_lib::asr::missing_model_files_for_tests;

#[test]
fn missing_model_files_detects_incomplete_snapshot() {
    let temp_dir = std::env::temp_dir().join(format!(
        "asr_missing_snapshot_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    std::fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");

    let initial_missing = missing_model_files_for_tests(&temp_dir);
    assert!(
        initial_missing.contains(&"encoder-model.int8.onnx".to_string()),
        "expected a known model file to be missing in an empty snapshot"
    );

    for file in &initial_missing {
        std::fs::write(temp_dir.join(file), b"ok").expect("write should succeed");
    }

    let missing_after = missing_model_files_for_tests(&temp_dir);
    assert!(missing_after.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}
