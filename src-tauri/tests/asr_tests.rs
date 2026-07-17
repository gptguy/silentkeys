use silent_keys_lib::asr::{language_candidates_for_tests, language_options_for_tests, AsrError};
use silent_keys_lib::errors::UserFacing;
use silent_keys_lib::settings::{Settings, DEFAULT_ASR_LANGUAGE};

#[test]
fn system_locale_candidates_fall_back_from_region_to_language() {
    assert_eq!(
        language_candidates_for_tests("en_CA.UTF-8"),
        ["en-CA", "en", "auto"]
    );
    assert_eq!(
        language_candidates_for_tests("fr_CA"),
        ["fr-CA", "fr", "auto"]
    );
}

#[test]
fn model_language_options_use_the_first_declared_code_per_prompt() {
    let languages = language_options_for_tests(&[
        ("en-US", 0),
        ("en", 0),
        ("en-GB", 1),
        ("enGB", 1),
        ("hi-IN", 6),
        ("hi-HI", 6),
        ("auto", 101),
    ]);
    assert_eq!(languages, ["en-GB", "en-US", "hi-IN"]);
}

#[test]
fn english_us_is_the_default_language() {
    assert_eq!(Settings::default().asr_language, DEFAULT_ASR_LANGUAGE);
}

#[test]
fn asr_error_user_message_download() {
    let err = AsrError::Download("connection failed".to_string());
    let msg = err.user_message();
    assert!(msg.contains("download") || msg.contains("internet"));
}

#[test]
fn asr_error_user_message_io() {
    let err = AsrError::Io {
        context: "read model config".to_string(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
    };
    let msg = err.user_message();
    assert!(msg.contains("read") || msg.contains("write") || msg.contains("files"));
}

#[test]
fn asr_error_display_includes_details() {
    let err = AsrError::Download("timeout".to_string());
    let display = format!("{}", err);
    assert!(display.contains("download") || display.contains("failed"));
}
