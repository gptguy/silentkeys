pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;

pub fn resample_linear(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == 0 || to_sr == 0 || input.is_empty() {
        return Vec::new();
    }

    if from_sr == to_sr {
        return input.to_vec();
    }

    let out_len = ((input.len() as f64) * (to_sr as f64) / (from_sr as f64))
        .ceil()
        .max(1.0) as usize;
    let step = from_sr as f64 / to_sr as f64;

    let mut output = Vec::with_capacity(out_len);
    let input_len = input.len();

    for i in 0..out_len {
        let pos = (i as f64) * step;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;

        unsafe {
            let current = *input.get_unchecked(idx.min(input_len - 1));
            let next_idx = (idx + 1).min(input_len - 1);
            let next = *input.get_unchecked(next_idx);
            output.push(current + (next - current) * frac);
        }
    }

    output
}
