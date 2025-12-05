use std::path::Path;
use std::time::Instant;

use crate::asr::recognizer::AsrError;

pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;

pub fn load_audio(path: &Path) -> Result<Vec<f32>, AsrError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let io_start = Instant::now();
    let src = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();

    let fmt_opts: FormatOptions = Default::default();
    let meta_opts: MetadataOptions = Default::default();
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .map_err(|e| AsrError::Audio(format!("failed to probe audio {}: {e}", path.display())))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| AsrError::Audio("missing default track".to_string()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AsrError::Audio(format!("decoder init failed: {e}")))?;

    let track_id = track.id;
    let mut pcm = track
        .codec_params
        .n_frames
        .and_then(|n| usize::try_from(n).ok())
        .map(Vec::with_capacity)
        .unwrap_or_else(Vec::new);
    let mut sample_rate = track.codec_params.sample_rate;

    let decode_start = Instant::now();

    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                sample_rate = sample_rate.or(Some(spec.rate));
                let channels = spec.channels.count();
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                buf.copy_interleaved_ref(decoded);

                if channels == 1 {
                    pcm.extend_from_slice(buf.samples());
                } else {
                    for frame in buf.samples().chunks_exact(channels) {
                        let sum: f32 = frame.iter().copied().sum();
                        pcm.push(sum / channels as f32);
                    }
                }
            }
            Err(err) if matches!(err, SymphoniaError::DecodeError(_)) => {
                log::warn!("skipping corrupt packet: {err}");
                continue;
            }
            Err(err) => {
                return Err(AsrError::Audio(format!(
                    "decode error for {}: {err}",
                    path.display()
                )));
            }
        }
    }

    let decode_elapsed = decode_start.elapsed();

    let sr = sample_rate
        .ok_or_else(|| AsrError::Audio(format!("missing sample rate for {}", path.display())))?;

    let (pcm, resample_elapsed) = if sr == TARGET_SAMPLE_RATE {
        (pcm, None)
    } else {
        log::warn!(
            "Resampling from {} Hz to {} Hz for {}",
            sr,
            TARGET_SAMPLE_RATE,
            path.display()
        );
        let resample_start = Instant::now();
        let output = resample_linear(&pcm, sr, TARGET_SAMPLE_RATE);
        (output, Some(resample_start.elapsed()))
    };

    log::info!(
        "Loaded audio {} samples from {} at {} Hz (I/O+probe: {:?}, decode: {:?}, resample: {:?})",
        pcm.len(),
        path.display(),
        sr,
        io_start.elapsed(),
        decode_elapsed,
        resample_elapsed.unwrap_or_default()
    );

    Ok(pcm)
}

pub(crate) fn resample_linear(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == 0 || to_sr == 0 || input.is_empty() {
        return Vec::new();
    }

    let out_len = ((input.len() as f64) * (to_sr as f64) / (from_sr as f64))
        .ceil()
        .max(1.0) as usize;
    let step = from_sr as f64 / to_sr as f64;

    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = (i as f64) * step;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;

        let current = input.get(idx).copied().unwrap_or_default();
        let next = input.get(idx + 1).copied().unwrap_or(current);
        let sample = current + (next - current) * (frac as f32);
        output.push(sample);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
