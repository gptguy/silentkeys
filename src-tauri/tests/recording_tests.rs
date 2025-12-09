use silent_keys_lib::recording::*;

#[test]
fn i8_normalization() {
    assert!((i8::MAX.to_normalized() - 1.0).abs() < f32::EPSILON);
    assert!((0i8.to_normalized()).abs() < f32::EPSILON);
    assert!(((-64i8).to_normalized() - (-0.5)).abs() < 0.01);
}

#[test]
fn i16_normalization() {
    assert!((i16::MAX.to_normalized() - 1.0).abs() < f32::EPSILON);
    assert!((0i16.to_normalized()).abs() < f32::EPSILON);
}

#[test]
fn i32_normalization() {
    assert!((i32::MAX.to_normalized() - 1.0).abs() < f32::EPSILON);
    assert!((0i32.to_normalized()).abs() < f32::EPSILON);
}

#[test]
fn f32_passthrough() {
    assert!((0.5f32.to_normalized() - 0.5).abs() < f32::EPSILON);
    assert!((-1.0f32.to_normalized() - (-1.0)).abs() < f32::EPSILON);
}

#[test]
fn recording_error_user_messages() {
    assert!(!RecordingError::AlreadyRecording.user_message().is_empty());
    assert!(!RecordingError::NotRecording.user_message().is_empty());
    assert!(!RecordingError::NoInputDevice.user_message().is_empty());
    assert!(!RecordingError::UnsupportedFormat.user_message().is_empty());
    assert!(!RecordingError::NoAudioCaptured.user_message().is_empty());
    assert!(!RecordingError::Device("test".to_string())
        .user_message()
        .is_empty());
    assert!(!RecordingError::LockFailed.user_message().is_empty());
}
