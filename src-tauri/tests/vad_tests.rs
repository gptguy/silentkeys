use silent_keys_lib::vad::{VadError, VAD_CHUNK_SIZE};

#[test]
fn vad_error_display_messages_are_not_empty() {
    let errors = [
        VadError::Ort(ort::Error::new("test error")),
        VadError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test")),
        VadError::InvalidChunkSize {
            expected: 480,
            actual: 256,
        },
        VadError::ModelNotFound("/path/to/model".to_string()),
        VadError::Shape(ndarray::ShapeError::from_kind(
            ndarray::ErrorKind::IncompatibleShape,
        )),
    ];

    for err in errors {
        assert!(!err.to_string().is_empty());
    }
}

#[test]
fn vad_chunk_size_is_480() {
    assert_eq!(VAD_CHUNK_SIZE, 480);
}
