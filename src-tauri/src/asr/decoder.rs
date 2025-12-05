use std::sync::LazyLock;
use std::time::Instant;

use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayViewD};
use ort::inputs;
use ort::value::TensorRef;
use regex::Regex;

use crate::asr::recognizer::{AsrError, AsrModel, Transcript};

type DecoderState = (Array3<f32>, Array3<f32>);

const SUBSAMPLING_FACTOR: usize = 8;
const WINDOW_SIZE: f32 = 0.01;
const MAX_TOKENS_PER_STEP: usize = 10;

static DECODE_SPACE_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

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
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(1024);

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

    pub(crate) fn reset_state(&mut self) {
        self.state.0.fill(0.0);
        self.state.1.fill(0.0);
    }

    pub(crate) fn set_encoder_step(&mut self, frame: &ArrayView1<f32>) {
        // encoder_step shape: [1, feature_dim, 1]
        let mut view = self.encoder_step.index_axis_mut(ndarray::Axis(2), 0);
        let mut view = view.index_axis_mut(ndarray::Axis(0), 0);
        view.assign(frame);
    }

    pub(crate) fn set_target(&mut self, token: i32) {
        self.targets[[0, 0]] = token;
    }
}

impl AsrModel {
    pub(crate) fn decode_sequence(
        &mut self,
        encodings: &ArrayViewD<f32>, // [time_steps, 1024]
        encodings_len: usize,
    ) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
        let decode_start = Instant::now();
        let mut tokens = Vec::with_capacity(encodings_len / 2 + 4);
        let mut timestamps = Vec::with_capacity(encodings_len / 2 + 4);

        // Reuse decoder workspace to avoid per-step allocations.
        let workspace = &mut self.decoder_workspace;
        workspace.reset_state();

        let mut t = 0;
        let mut emitted_tokens = 0;

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray::s![t, ..]);
            workspace.set_encoder_step(&encoder_step);

            let target_token = tokens.last().copied().unwrap_or(self.blank_idx);
            workspace.set_target(target_token);

            let inputs = inputs![
                "encoder_outputs" => TensorRef::from_array_view(workspace.encoder_step.view())?,
                "targets" => TensorRef::from_array_view(workspace.targets.view())?,
                "target_length" => TensorRef::from_array_view(workspace.target_length.view())?,
                "input_states_1" => TensorRef::from_array_view(workspace.state.0.view())?,
                "input_states_2" => TensorRef::from_array_view(workspace.state.1.view())?,
            ];

            let outputs = self.decoder_joint.run(inputs)?;

            let logits = outputs
                .get("outputs")
                .ok_or_else(|| AsrError::OutputNotFound("outputs".to_string()))?
                .try_extract_array()?;

            // Squeeze outputs like Python (remove batch dimension)
            let logits = logits.remove_axis(ndarray::Axis(0));

            // For TDT models, split output into vocab logits and duration logits
            // output[:vocab_size] = vocabulary logits
            // output[vocab_size:] = duration logits
            let vocab_logits_slice: &[f32] = logits.as_slice().ok_or_else(|| {
                AsrError::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                ))
            })?;

            let vocab_logits = if logits.len() > self.vocab_size {
                // TDT model - extract only vocabulary logits
                log::trace!(
                    "TDT model detected: splitting {} logits into vocab({}) + duration",
                    logits.len(),
                    self.vocab_size
                );
                &vocab_logits_slice[..self.vocab_size]
            } else {
                // Regular RNN-T model
                vocab_logits_slice
            };

            // Get argmax token from vocabulary logits only
            let token = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i32)
                .unwrap_or(self.blank_idx);

            if token != self.blank_idx {
                let state1 = outputs
                    .get("output_states_1")
                    .ok_or_else(|| AsrError::OutputNotFound("output_states_1".to_string()))?
                    .try_extract_array()?;
                let state2 = outputs
                    .get("output_states_2")
                    .ok_or_else(|| AsrError::OutputNotFound("output_states_2".to_string()))?
                    .try_extract_array()?;

                let state1_3d = state1.to_owned().into_dimensionality::<ndarray::Ix3>()?;
                let state2_3d = state2.to_owned().into_dimensionality::<ndarray::Ix3>()?;

                // Reuse buffers when shapes match; otherwise replace.
                if workspace.state.0.shape() == state1_3d.shape() {
                    workspace.state.0.assign(&state1_3d);
                } else {
                    workspace.state.0 = state1_3d;
                }
                if workspace.state.1.shape() == state2_3d.shape() {
                    workspace.state.1.assign(&state2_3d);
                } else {
                    workspace.state.1 = state2_3d;
                }

                tokens.push(token);
                timestamps.push(t);
                emitted_tokens += 1;
            }

            // Step logic from Python - simplified since step is always -1
            if token == self.blank_idx || emitted_tokens == MAX_TOKENS_PER_STEP {
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
            Err(_) => tokens.join(""), // Fallback if regex failed to compile
        };

        let float_timestamps: Vec<f32> = timestamps
            .iter()
            .map(|&t| WINDOW_SIZE * SUBSAMPLING_FACTOR as f32 * t as f32)
            .collect();

        Transcript {
            text,
            timestamps: float_timestamps,
            tokens,
        }
    }
}
