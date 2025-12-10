use silent_keys_lib::streaming::*;

#[test]
fn default_config_has_sane_values() {
    let config = StreamConfig::default();
    assert_eq!(config.min_speech_ms, 100);
    assert!((config.max_phrase_sec - 15.0).abs() < f32::EPSILON);
}
