use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::LazyLock;
use std::time::Instant;

use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayViewD};
use ort::inputs;
use ort::value::TensorRef;
use regex::Regex;

use crate::asr::recognizer::{AsrError, AsrModel, InferenceConfig, Transcript};
use crate::asr::FRAME_DURATION_SEC;

type DecoderState = (Array3<f32>, Array3<f32>);

static DECODE_SPACE_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

#[derive(Clone)]
struct Hypothesis {
    tokens: Vec<i32>,
    timestamps: Vec<usize>,
    score: f32,
    state: DecoderState,
    last_token: i32,
}

struct StepScores {
    blank_score: f32,
    top_tokens: Vec<(usize, f32)>,
    output_state: DecoderState,
}

fn retain_top_k(hyps: &mut Vec<Hypothesis>, k: usize) {
    if k == 0 || hyps.len() <= k {
        return;
    }
    hyps.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    hyps.truncate(k);
}

pub struct DecoderSession {
    workspace: DecoderWorkspace,
    last_token: i32,
    hotword_mask: Option<Vec<bool>>,
    hotword_boost: f32,
    vocab_size: usize,
    blank_idx: i32,
}

impl DecoderSession {
    pub(crate) fn new(
        model: &AsrModel,
        workspace: Option<DecoderWorkspace>,
        last_token: i32,
        config: &InferenceConfig,
    ) -> Result<Self, AsrError> {
        let workspace = if let Some(ws) = workspace {
            log::debug!("Reusing cached decoder workspace");
            ws
        } else {
            log::debug!("Initializing new decoder workspace");
            DecoderWorkspace::new(&model.decoder_joint)?
        };

        let hotword_mask = build_hotword_mask(model, config);

        log::debug!("Decoder session initialized with last_token={}", last_token);

        Ok(Self {
            workspace,
            last_token,
            hotword_mask,
            hotword_boost: config.hotword_boost,
            vocab_size: model.vocab_size,
            blank_idx: model.blank_idx,
        })
    }

    pub(crate) fn into_parts(self) -> (DecoderWorkspace, i32) {
        (self.workspace, self.last_token)
    }

    pub(crate) fn decode_sequence(
        &mut self,
        model: &mut AsrModel,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
        config: &InferenceConfig,
    ) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
        let beam_width = config.beam_width.max(1);
        if beam_width <= 1 {
            log::debug!("Decoding (Greedy) frames_len={}", encodings_len);
            return self.decode_sequence_greedy(model, encodings, encodings_len, config);
        }
        log::debug!(
            "Decoding (Beam={}) frames_len={}",
            beam_width,
            encodings_len
        );
        self.decode_sequence_beam(model, encodings, encodings_len, config, beam_width)
    }

    fn decode_sequence_greedy(
        &mut self,
        model: &mut AsrModel,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
        config: &InferenceConfig,
    ) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
        let decode_start = Instant::now();
        let mut tokens = Vec::with_capacity(std::cmp::max(1, encodings_len / 2));
        let mut timestamps = Vec::with_capacity(std::cmp::max(1, encodings_len / 2));

        let max_tokens_per_step = config.max_tokens_per_step.max(1);
        let temperature = Self::normalized_temperature(config);

        let mut t = 0;
        let mut emitted_tokens = 0;

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray::s![t, ..]);
            self.workspace.set_encoder_step(&encoder_step);

            let target_token = if let Some(last) = tokens.last() {
                *last
            } else {
                self.last_token
            };
            self.workspace.set_target(target_token);

            let inputs = inputs![
                "encoder_outputs" => TensorRef::from_array_view(self.workspace.encoder_step.view())?,
                "targets" => TensorRef::from_array_view(self.workspace.targets.view())?,
                "target_length" => TensorRef::from_array_view(self.workspace.target_length.view())?,
                "input_states_1" => TensorRef::from_array_view(self.workspace.state.0.view())?,
                "input_states_2" => TensorRef::from_array_view(self.workspace.state.1.view())?,
            ];

            let outputs = model.decoder_joint.run(inputs)?;

            let logits = outputs
                .get("outputs")
                .ok_or_else(|| AsrError::OutputNotFound("outputs".to_string()))?
                .try_extract_array()?;

            let logits = logits.remove_axis(ndarray::Axis(0));

            let vocab_logits_slice: &[f32] = logits.as_slice().ok_or_else(|| {
                AsrError::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                ))
            })?;

            let vocab_logits = if logits.len() > self.vocab_size {
                &vocab_logits_slice[..self.vocab_size]
            } else {
                vocab_logits_slice
            };

            let mut best_idx = self.blank_idx as usize;
            let mut best_score = f32::NEG_INFINITY;
            let mut blank_score = f32::NEG_INFINITY;

            for (idx, &logit) in vocab_logits.iter().enumerate() {
                let mut score = logit;
                if temperature != 1.0 {
                    score /= temperature;
                }
                if let Some(mask) = self.hotword_mask.as_ref() {
                    if idx < mask.len() && mask[idx] {
                        score += self.hotword_boost;
                    }
                }
                if idx == self.blank_idx as usize {
                    blank_score = score;
                }
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }

            let mut token = best_idx as i32;
            if token != self.blank_idx
                && config.min_blank_margin > 0.0
                && blank_score.is_finite()
                && (best_score - blank_score) < config.min_blank_margin
            {
                token = self.blank_idx;
            }

            if token != self.blank_idx {
                let state1 = outputs
                    .get("output_states_1")
                    .ok_or_else(|| AsrError::OutputNotFound("output_states_1".to_string()))?
                    .try_extract_array::<f32>()?;
                let state2 = outputs
                    .get("output_states_2")
                    .ok_or_else(|| AsrError::OutputNotFound("output_states_2".to_string()))?
                    .try_extract_array::<f32>()?;

                self.workspace.assign_state(state1, state2);

                tokens.push(token);
                timestamps.push(t);
                emitted_tokens += 1;
                self.last_token = token;
            }

            if token == self.blank_idx || emitted_tokens == max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        log::debug!(
            "decode_sequence completed in {:?} (frames: {}, tokens: {})",
            decode_start.elapsed(),
            encodings_len,
            tokens.len()
        );

        Ok((tokens, timestamps))
    }

    fn decode_sequence_beam(
        &mut self,
        model: &mut AsrModel,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
        config: &InferenceConfig,
        beam_width: usize,
    ) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
        let decode_start = Instant::now();
        let max_tokens_per_step = config.max_tokens_per_step.max(1);

        let mut beam = vec![Hypothesis {
            tokens: Vec::with_capacity(std::cmp::max(1, encodings_len / 2)),
            timestamps: Vec::with_capacity(std::cmp::max(1, encodings_len / 2)),
            score: 0.0,
            state: self.workspace.state.clone(),
            last_token: self.last_token,
        }];

        for t in 0..encodings_len {
            let encoder_step = encodings.slice(ndarray::s![t, ..]);
            let mut blank_candidates = Vec::new();
            let mut active = beam;

            let mut step = 0;
            while step < max_tokens_per_step && !active.is_empty() {
                let mut next_active = Vec::new();
                for hyp in &active {
                    let scores = self.step_scores(
                        model,
                        &encoder_step,
                        hyp.last_token,
                        &hyp.state,
                        config,
                        beam_width,
                    )?;

                    let mut blank_hyp = hyp.clone();
                    blank_hyp.score += scores.blank_score;
                    blank_candidates.push(blank_hyp);

                    for (idx, score) in &scores.top_tokens {
                        let mut next = hyp.clone();
                        next.score += *score;
                        next.tokens.push(*idx as i32);
                        next.timestamps.push(t);
                        next.state = scores.output_state.clone();
                        next.last_token = *idx as i32;
                        next_active.push(next);
                    }
                }

                retain_top_k(&mut next_active, beam_width);
                active = next_active;
                step += 1;
            }

            if blank_candidates.is_empty() {
                blank_candidates = active;
            }

            retain_top_k(&mut blank_candidates, beam_width);
            beam = blank_candidates;
        }

        retain_top_k(&mut beam, 1);
        let Some(best) = beam.pop() else {
            return Ok((Vec::new(), Vec::new()));
        };

        let (state1, state2) = best.state;
        self.workspace.state.0 = state1;
        self.workspace.state.1 = state2;
        self.last_token = best.last_token;

        log::debug!(
            "decode_sequence completed in {:?} (frames: {}, tokens: {})",
            decode_start.elapsed(),
            encodings_len,
            best.tokens.len()
        );

        Ok((best.tokens, best.timestamps))
    }

    fn step_scores(
        &mut self,
        model: &mut AsrModel,
        encoder_step: &ArrayView1<f32>,
        target_token: i32,
        state: &DecoderState,
        config: &InferenceConfig,
        beam_width: usize,
    ) -> Result<StepScores, AsrError> {
        self.workspace.set_encoder_step(encoder_step);
        self.workspace.set_target(target_token);

        self.workspace.state.0 = state.0.clone();
        self.workspace.state.1 = state.1.clone();

        let inputs = inputs![
            "encoder_outputs" => TensorRef::from_array_view(self.workspace.encoder_step.view())?,
            "targets" => TensorRef::from_array_view(self.workspace.targets.view())?,
            "target_length" => TensorRef::from_array_view(self.workspace.target_length.view())?,
            "input_states_1" => TensorRef::from_array_view(self.workspace.state.0.view())?,
            "input_states_2" => TensorRef::from_array_view(self.workspace.state.1.view())?,
        ];

        let outputs = model.decoder_joint.run(inputs)?;

        let logits = outputs
            .get("outputs")
            .ok_or_else(|| AsrError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;

        let logits = logits.remove_axis(ndarray::Axis(0));

        let vocab_logits_slice: &[f32] = logits.as_slice().ok_or_else(|| {
            AsrError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            ))
        })?;

        let vocab_logits = if logits.len() > self.vocab_size {
            &vocab_logits_slice[..self.vocab_size]
        } else {
            vocab_logits_slice
        };

        let temperature = Self::normalized_temperature(config);
        let blank_idx = self.blank_idx as usize;
        let mut blank_score = f32::NEG_INFINITY;

        for (idx, &logit) in vocab_logits.iter().enumerate() {
            if idx != blank_idx {
                continue;
            }
            let mut score = logit;
            if temperature != 1.0 {
                score /= temperature;
            }
            blank_score = score;
            break;
        }

        let mut top_tokens = Vec::new();
        let max_tokens = beam_width.max(1);
        for (idx, &logit) in vocab_logits.iter().enumerate() {
            if idx == blank_idx {
                continue;
            }

            let mut score = logit;
            if temperature != 1.0 {
                score /= temperature;
            }
            if let Some(mask) = self.hotword_mask.as_ref() {
                if idx < mask.len() && mask[idx] {
                    score += self.hotword_boost;
                }
            }
            if config.min_blank_margin > 0.0
                && blank_score.is_finite()
                && (score - blank_score) < config.min_blank_margin
            {
                continue;
            }

            Self::push_top_token(&mut top_tokens, idx, score, max_tokens);
        }

        let state1 = outputs
            .get("output_states_1")
            .ok_or_else(|| AsrError::OutputNotFound("output_states_1".to_string()))?
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(AsrError::Shape)?
            .to_owned();
        let state2 = outputs
            .get("output_states_2")
            .ok_or_else(|| AsrError::OutputNotFound("output_states_2".to_string()))?
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(AsrError::Shape)?
            .to_owned();

        Ok(StepScores {
            blank_score,
            top_tokens,
            output_state: (state1, state2),
        })
    }

    fn push_top_token(top: &mut Vec<(usize, f32)>, idx: usize, score: f32, k: usize) {
        if !score.is_finite() {
            return;
        }
        if top.len() < k {
            top.push((idx, score));
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            return;
        }
        let replace = match top.last() {
            Some((_, tail_score)) => score > *tail_score,
            None => true,
        };
        if replace {
            top.pop();
            top.push((idx, score));
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        }
    }

    fn normalized_temperature(config: &InferenceConfig) -> f32 {
        if config.temperature <= 0.0 {
            1.0
        } else {
            config.temperature
        }
    }
}

#[derive(Debug)]
pub(crate) struct DecoderWorkspace {
    encoder_step: Array3<f32>,
    targets: Array2<i32>,
    target_length: Array1<i32>,
    state: DecoderState,
}

impl DecoderWorkspace {
    pub(crate) fn new(session: &ort::session::Session) -> Result<Self, AsrError> {
        let encoder_dim = session
            .inputs
            .iter()
            .find(|input| input.name == "encoder_outputs")
            .and_then(|input| input.input_type.tensor_shape())
            .and_then(|shape| shape.get(1).copied())
            .and_then(|d| usize::try_from(d).ok());

        let encoder_dim = match encoder_dim {
            Some(dim) => dim,
            None => {
                log::warn!("Could not determine encoder_dim from model, falling back to 1024");
                1024
            }
        };

        let state1_shape = session
            .inputs
            .iter()
            .find(|input| input.name == "input_states_1")
            .ok_or_else(|| AsrError::InputNotFound("input_states_1".to_string()))?
            .input_type
            .tensor_shape()
            .ok_or_else(|| AsrError::TensorShape("input_states_1".to_string()))?;

        let state2_shape = session
            .inputs
            .iter()
            .find(|input| input.name == "input_states_2")
            .ok_or_else(|| AsrError::InputNotFound("input_states_2".to_string()))?
            .input_type
            .tensor_shape()
            .ok_or_else(|| AsrError::TensorShape("input_states_2".to_string()))?;

        let state1 = Array::zeros((state1_shape[0] as usize, 1, state1_shape[2] as usize));
        let state2 = Array::zeros((state2_shape[0] as usize, 1, state2_shape[2] as usize));

        Ok(Self {
            encoder_step: Array::zeros((1, encoder_dim, 1)),
            targets: Array2::zeros((1, 1)),
            target_length: Array1::from_vec(vec![1]),
            state: (state1, state2),
        })
    }

    pub(crate) fn set_encoder_step(&mut self, frame: &ArrayView1<f32>) {
        let mut view = self.encoder_step.index_axis_mut(ndarray::Axis(2), 0);
        let mut view = view.index_axis_mut(ndarray::Axis(0), 0);
        view.assign(frame);
    }

    pub(crate) fn set_target(&mut self, token: i32) {
        self.targets[[0, 0]] = token;
    }

    pub(crate) fn assign_state(
        &mut self,
        state1: ndarray::ArrayViewD<f32>,
        state2: ndarray::ArrayViewD<f32>,
    ) {
        if let Ok(s1) = state1.into_dimensionality::<ndarray::Ix3>() {
            if self.state.0.shape() == s1.shape() {
                self.state.0.assign(&s1);
            } else {
                self.state.0 = s1.to_owned();
            }
        }
        if let Ok(s2) = state2.into_dimensionality::<ndarray::Ix3>() {
            if self.state.1.shape() == s2.shape() {
                self.state.1.assign(&s2);
            } else {
                self.state.1 = s2.to_owned();
            }
        }
    }
}

impl AsrModel {
    pub(crate) fn decode_tokens(&self, ids: Vec<i32>, timestamps: Vec<usize>) -> Transcript {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx < self.vocab.len() {
                    Some(self.vocab[idx].clone())
                } else {
                    None
                }
            })
            .collect();

        let text = match &*DECODE_SPACE_RE {
            Ok(regex) => regex
                .replace_all(&tokens.join(""), |caps: &regex::Captures| {
                    if caps.get(1).is_some() {
                        " "
                    } else {
                        ""
                    }
                })
                .to_string(),
            Err(_) => tokens.join(""),
        };

        let float_timestamps: Vec<f32> = timestamps
            .iter()
            .map(|&t| FRAME_DURATION_SEC * t as f32)
            .collect();

        Transcript {
            text,
            timestamps: float_timestamps,
            tokens,
        }
    }
}

fn build_hotword_mask(model: &AsrModel, config: &InferenceConfig) -> Option<Vec<bool>> {
    if config.hotword_boost <= 0.0 || config.hotwords.is_empty() {
        return None;
    }

    let mut hotwords = HashSet::new();
    for word in &config.hotwords {
        let trimmed = word.trim();
        if !trimmed.is_empty() {
            hotwords.insert(trimmed.to_lowercase());
        }
    }
    if hotwords.is_empty() {
        return None;
    }

    let mut mask = vec![false; model.vocab.len()];
    let mut matched = 0;
    for (idx, token) in model.vocab.iter().enumerate() {
        let normalized = token.trim().to_lowercase();
        if hotwords.contains(&normalized) {
            mask[idx] = true;
            matched += 1;
        }
    }

    if matched == 0 {
        None
    } else {
        Some(mask)
    }
}
