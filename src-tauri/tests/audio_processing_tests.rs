use silent_keys_lib::audio_processing::AudioProcessor;

#[test]
fn matching_sample_rate_preserves_all_samples() {
    let input: Vec<f32> = (0..1_001).map(|index| index as f32 / 1_001.0).collect();
    let mut output = Vec::new();
    let mut processor = AudioProcessor::new(16_000, 16_000).expect("processor should initialize");

    processor
        .process(&input, |frame| output.extend(frame.samples))
        .expect("processing should succeed");
    processor
        .flush(|frame| output.extend(frame.samples))
        .expect("flush should succeed");

    assert_eq!(output, input);
}

#[test]
fn resampling_one_second_preserves_duration() {
    let input: Vec<f32> = (0..48_000)
        .map(|index| ((index as f32 / 48_000.0) * std::f32::consts::TAU * 440.0).sin())
        .collect();
    let mut output = Vec::new();
    let mut processor = AudioProcessor::new(48_000, 16_000).expect("processor should initialize");

    for chunk in input.chunks(480) {
        processor
            .process(chunk, |frame| output.extend(frame.samples))
            .expect("processing should succeed");
    }
    processor
        .flush(|frame| output.extend(frame.samples))
        .expect("flush should succeed");

    assert!(output.iter().all(|sample| sample.is_finite()));
    assert!(
        (15_520..=16_480).contains(&output.len()),
        "expected about one second at 16 kHz, got {} samples",
        output.len()
    );
}

#[test]
fn two_minute_resampling_preserves_duration() {
    const SECONDS: usize = 120;
    const INPUT_RATE: usize = 48_000;
    const CHUNK_SIZE: usize = 480;

    let chunk = [0.0_f32; CHUNK_SIZE];
    let mut emitted = 0;
    let mut processor =
        AudioProcessor::new(INPUT_RATE, 16_000).expect("processor should initialize");

    for _ in 0..(SECONDS * INPUT_RATE / CHUNK_SIZE) {
        processor
            .process(&chunk, |frame| emitted += frame.samples.len())
            .expect("processing should succeed");
    }
    processor
        .flush(|frame| emitted += frame.samples.len())
        .expect("flush should succeed");

    let expected = SECONDS * 16_000;
    let tolerance = expected / 200;
    assert!(
        (expected - tolerance..=expected + tolerance).contains(&emitted),
        "expected about {expected} samples after a {SECONDS}s resampled session, got {emitted}"
    );
}

#[test]
fn twenty_minute_passthrough_drops_no_samples() {
    const TOTAL_SAMPLES: usize = 20 * 60 * 16_000;
    const CHUNK_SIZE: usize = 1_024;

    let mut emitted = 0;
    let mut processed = 0;
    let chunk = [0.0; CHUNK_SIZE];
    let mut processor = AudioProcessor::new(16_000, 16_000).expect("processor should initialize");

    while processed < TOTAL_SAMPLES {
        let count = CHUNK_SIZE.min(TOTAL_SAMPLES - processed);
        processor
            .process(&chunk[..count], |frame| emitted += frame.samples.len())
            .expect("processing should succeed");
        processed += count;
    }
    processor
        .flush(|frame| emitted += frame.samples.len())
        .expect("flush should succeed");

    assert_eq!(emitted, TOTAL_SAMPLES);
}
