use crate::asr::recognizer::{AsrError, InferenceConfig};
use ndarray::{Array, Array1, Array2, Array3, ArrayView1};

pub type DecoderOutputState = (Array3<f32>, Array3<f32>);

pub struct StepScores {
    pub blank_score: f32,
    pub top_tokens: Vec<(usize, f32)>,
    pub output_state: DecoderOutputState,
}

#[derive(Clone)]
pub struct Hypothesis {
    pub tokens: Vec<i32>,
    pub timestamps: Vec<usize>,
    pub score: f32,
    pub state: DecoderOutputState,
    pub last_token: i32,
}

#[derive(Debug)]
pub struct DecoderWorkspace {
    pub(super) encoder_step: Array3<f32>,
    pub(super) targets: Array2<i32>,
    pub(super) target_length: Array1<i32>,
    pub(super) state: DecoderOutputState,
}

impl DecoderWorkspace {
    pub fn new(session: &ort::session::Session) -> Result<Self, AsrError> {
        let encoder_dim = session
            .inputs
            .iter()
            .find(|i| i.name == "encoder_outputs")
            .and_then(|i| i.input_type.tensor_shape())
            .and_then(|s| s.get(1).copied())
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(1024);

        let s1_shape = session
            .inputs
            .iter()
            .find(|i| i.name == "input_states_1")
            .and_then(|i| i.input_type.tensor_shape())
            .ok_or_else(|| AsrError::InputNotFound("input_states_1".to_string()))?;
        let s2_shape = session
            .inputs
            .iter()
            .find(|i| i.name == "input_states_2")
            .and_then(|i| i.input_type.tensor_shape())
            .ok_or_else(|| AsrError::InputNotFound("input_states_2".to_string()))?;

        Ok(Self {
            encoder_step: Array::zeros((1, encoder_dim, 1)),
            targets: Array2::zeros((1, 1)),
            target_length: Array1::from_vec(vec![1]),
            state: (
                Array::zeros((s1_shape[0] as usize, 1, s1_shape[2] as usize)),
                Array::zeros((s2_shape[0] as usize, 1, s2_shape[2] as usize)),
            ),
        })
    }

    pub fn set_encoder_step(&mut self, frame: &ArrayView1<f32>) {
        self.encoder_step
            .index_axis_mut(ndarray::Axis(2), 0)
            .index_axis_mut(ndarray::Axis(0), 0)
            .assign(frame);
    }

    pub fn set_target(&mut self, token: i32) {
        self.targets[[0, 0]] = token;
    }
}

pub fn normalized_temperature(config: &InferenceConfig) -> f32 {
    if config.temperature <= 0.0 {
        1.0
    } else {
        config.temperature
    }
}

pub fn extract_top_tokens(
    vocab_logits: &[f32],
    blank_score: f32,
    temp: f32,
    config: &InferenceConfig,
    beam_width: usize,
    blank_idx: usize,
    mask: Option<&Vec<bool>>,
    boost: f32,
) -> Vec<(usize, f32)> {
    let mut candidates: Vec<_> = vocab_logits
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != blank_idx)
        .map(|(i, &l)| {
            let mut s = l / temp;
            if let Some(m) = mask.and_then(|m| m.get(i)) {
                if *m {
                    s += boost;
                }
            }
            (i, s)
        })
        .filter(|(_, s)| {
            s.is_finite()
                && (config.min_blank_margin <= 0.0
                    || !blank_score.is_finite()
                    || (s - blank_score) >= config.min_blank_margin)
        })
        .collect();

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    candidates.truncate(beam_width.max(1));
    candidates
}
