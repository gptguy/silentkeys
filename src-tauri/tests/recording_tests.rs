use cpal::Sample;
use silent_keys_lib::errors::UserFacing;
use silent_keys_lib::recording::{Recorder, RecordingError};

#[test]
fn i8_normalization() {
    assert!((Sample::to_sample::<f32>(i8::MAX) - 0.9921875).abs() < f32::EPSILON);
    assert!((Sample::to_sample::<f32>(0i8)).abs() < f32::EPSILON);
    assert!((Sample::to_sample::<f32>(-64i8) - (-0.5)).abs() < f32::EPSILON);
}

#[test]
fn i16_normalization() {
    let val = Sample::to_sample::<f32>(i16::MAX);
    assert!((val - (32767.0 / 32768.0)).abs() < f32::EPSILON);
    assert!((Sample::to_sample::<f32>(0i16)).abs() < f32::EPSILON);
}

#[test]
fn i32_normalization() {
    let val = Sample::to_sample::<f32>(i32::MAX);
    assert!((val - (i32::MAX as f32 / 2147483648.0)).abs() < f32::EPSILON);
    assert!((Sample::to_sample::<f32>(0i32)).abs() < f32::EPSILON);
}

#[test]
fn f32_passthrough() {
    assert!((Sample::to_sample::<f32>(0.5f32) - 0.5).abs() < f32::EPSILON);
    assert!((Sample::to_sample::<f32>(-1.0f32) - (-1.0)).abs() < f32::EPSILON);
}

#[test]
fn recording_error_user_messages() {
    assert!(!RecordingError::AlreadyRecording.user_message().is_empty());
    assert!(!RecordingError::NotRecording.user_message().is_empty());
    assert!(!RecordingError::NoInputDevice.user_message().is_empty());
    assert!(!RecordingError::NoAudioCaptured.user_message().is_empty());
    assert!(!RecordingError::Device("test".to_string())
        .user_message()
        .is_empty());
    assert!(!RecordingError::LockFailed.user_message().is_empty());
    assert!(!RecordingError::ThreadError.user_message().is_empty());
    assert!(!RecordingError::ThreadStart(std::io::Error::other("test"))
        .user_message()
        .is_empty());
    assert!(!RecordingError::AudioProcessingError("test".to_string())
        .user_message()
        .is_empty());
    assert!(!RecordingError::UpdateInProgress.user_message().is_empty());
    assert!(!RecordingError::SettingsInProgress.user_message().is_empty());
}

#[test]
fn audio_overrun_has_actionable_user_message() {
    let error = RecordingError::AudioOverrun(42);
    assert_eq!(
        error.user_message(),
        "Audio could not be captured fast enough. Close demanding apps and try again."
    );
}

#[test]
fn recording_reservation_is_visible_and_exclusive() {
    let recorder = Recorder::global();
    let reservation = recorder.reserve().expect("reservation should succeed");

    assert!(recorder.is_recording());
    assert!(matches!(
        recorder.reserve(),
        Err(RecordingError::AlreadyRecording)
    ));

    drop(reservation);
    assert!(!recorder.is_recording());
}
