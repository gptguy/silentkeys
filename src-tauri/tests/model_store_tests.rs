use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use silent_keys_lib::asr::{
    fallback_model_root, invalid_model_files_for_tests, model_file_matches_for_tests,
    resolve_model_dir, verification_receipt_matches_for_tests,
    write_verification_receipt_for_tests,
};

#[test]
fn invalid_model_files_detects_incomplete_snapshot() {
    let temp_dir = std::env::temp_dir().join(format!(
        "asr_missing_snapshot_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    std::fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");

    let initial_missing =
        invalid_model_files_for_tests(&temp_dir).expect("validation should succeed");
    assert!(
        initial_missing.contains(&"encoder.onnx".to_string()),
        "expected a known model file to be missing in an empty snapshot"
    );

    std::fs::write(temp_dir.join("encoder.onnx"), b"present but invalid")
        .expect("write should succeed");
    let invalid_after =
        invalid_model_files_for_tests(&temp_dir).expect("validation should succeed");
    assert!(invalid_after.contains(&"encoder.onnx".to_string()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn model_file_validation_rejects_same_size_corruption() {
    let temp_dir = std::env::temp_dir().join(format!(
        "asr_corrupt_asset_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
    let asset = temp_dir.join("asset.bin");

    std::fs::write(&asset, b"valid").expect("fixture should be writable");
    assert!(model_file_matches_for_tests(
        &asset,
        5,
        "ec654fac9599f62e79e2706abef23dfb7c07c08185aa86db4d8695f0b718d1b3"
    ));

    std::fs::write(&asset, b"xxxxx").expect("fixture should be replaceable");
    assert!(!model_file_matches_for_tests(
        &asset,
        5,
        "ec654fac9599f62e79e2706abef23dfb7c07c08185aa86db4d8695f0b718d1b3"
    ));

    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
fn verification_receipt_tracks_snapshot_metadata_and_manifest() {
    let temp_dir = std::env::temp_dir().join(format!(
        "asr_verification_receipt_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos()
    ));
    let snapshot = temp_dir.join("snapshot");
    std::fs::create_dir_all(&snapshot).expect("snapshot directory should be creatable");
    std::fs::write(snapshot.join("asset.bin"), b"model").expect("fixture should be writable");

    let assets = [("asset.bin", 5, "expected-sha256")];
    assert!(!verification_receipt_matches_for_tests(
        &snapshot,
        "revision-a",
        &assets,
    ));

    write_verification_receipt_for_tests(&snapshot, "revision-a", &assets)
        .expect("verification receipt should be writable");
    assert!(verification_receipt_matches_for_tests(
        &snapshot,
        "revision-a",
        &assets,
    ));

    assert!(!verification_receipt_matches_for_tests(
        &snapshot,
        "revision-b",
        &assets,
    ));
    assert!(!verification_receipt_matches_for_tests(
        &snapshot,
        "revision-a",
        &[("asset.bin", 5, "replacement-sha256")],
    ));

    let asset_path = snapshot.join("asset.bin");
    std::fs::write(&asset_path, b"other").expect("fixture should be replaceable");
    std::fs::File::open(&asset_path)
        .expect("fixture should be readable")
        .set_modified(UNIX_EPOCH + Duration::from_secs(1))
        .expect("fixture modification time should be adjustable");
    assert_eq!(
        std::fs::metadata(&asset_path)
            .expect("fixture metadata should be readable")
            .len(),
        5,
    );
    assert!(!verification_receipt_matches_for_tests(
        &snapshot,
        "revision-a",
        &assets,
    ));

    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "requires network access; repairs the local model cache in place"]
fn corrupt_cached_asset_is_repaired() {
    let root = std::env::var("SILENT_KEYS_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback_model_root());
    let snapshot = resolve_model_dir(&root).expect("model should resolve");

    std::fs::write(snapshot.join("config.json"), b"corrupted")
        .expect("cached asset should be writable");
    let invalid = invalid_model_files_for_tests(&snapshot).expect("validation should succeed");
    assert!(
        invalid.contains(&"config.json".to_string()),
        "corruption should be detected before repair"
    );

    let repaired = resolve_model_dir(&root).expect("repair should succeed");
    assert_eq!(repaired, snapshot);
    assert!(
        invalid_model_files_for_tests(&snapshot)
            .expect("validation should succeed")
            .is_empty(),
        "all assets should match the manifest after repair"
    );
}
