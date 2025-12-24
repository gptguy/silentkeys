use ndarray::{Array3, ArrayView1};
use ort::value::TensorRef;

use crate::asr::recognizer::{AsrError, AsrModel, InferenceConfig};

use super::state::{
    extract_top_tokens, normalized_temperature, DecoderOutputState, DecoderWorkspace, StepScores,
};

pub struct DecoderSession {
    pub(crate) workspace: DecoderWorkspace,
    pub(crate) last_token: i32,
    pub(crate) hotword_mask: Option<Vec<bool>>,
    pub(crate) hotword_boost: f32,
    pub(crate) vocab_size: usize,
    pub(crate) blank_idx: i32,
}

impl DecoderSession {
    pub(crate) fn new(
        model: &AsrModel,
        workspace: Option<DecoderWorkspace>,
        last_token: i32,
        config: &InferenceConfig,
    ) -> Result<Self, AsrError> {
        let workspace = if let Some(ws) = workspace {
            ws
        } else {
            DecoderWorkspace::new(model.decoder_session())?
        };

        let hotword_mask = crate::asr::recognizer::config::build_hotword_mask(model, config);

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

    pub(crate) fn step_scores(
        &mut self,
        model: &mut AsrModel,
        encoder_step: &ArrayView1<f32>,
        target_token: i32,
        state: &DecoderOutputState,
        config: &InferenceConfig,
        beam_width: usize,
    ) -> Result<StepScores, AsrError> {
        self.workspace.set_encoder_step(encoder_step);
        self.workspace.set_target(target_token);
        self.workspace.state = state.clone();

        let outs = model.decoder_session_mut().run(ort::inputs![
            "encoder_outputs" => TensorRef::from_array_view(self.workspace.encoder_step.view())?,
            "targets" => TensorRef::from_array_view(self.workspace.targets.view())?,
            "target_length" => TensorRef::from_array_view(self.workspace.target_length.view())?,
            "input_states_1" => TensorRef::from_array_view(self.workspace.state.0.view())?,
            "input_states_2" => TensorRef::from_array_view(self.workspace.state.1.view())?,
        ])?;

        let logits = outs
            .get("outputs")
            .ok_or_else(|| AsrError::OutputNotFound("outputs".into()))?
            .try_extract_array()?
            .remove_axis(ndarray::Axis(0));
        let slice = logits.as_slice().ok_or_else(|| {
            AsrError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            ))
        })?;
        let slice = if slice.len() > self.vocab_size {
            &slice[..self.vocab_size]
        } else {
            slice
        };

        let t = normalized_temperature(config);
        let bs = slice[self.blank_idx as usize] / t;
        Ok(StepScores {
            blank_score: bs,
            top_tokens: extract_top_tokens(
                slice,
                bs,
                t,
                config,
                beam_width,
                self.blank_idx as usize,
                self.hotword_mask.as_ref(),
                self.hotword_boost,
            ),
            output_state: (
                Self::extract_state(outs.get("output_states_1"), "s1")?,
                Self::extract_state(outs.get("output_states_2"), "s2")?,
            ),
        })
    }

    fn extract_state(
        out: Option<&ort::value::DynValue>,
        name: &str,
    ) -> Result<Array3<f32>, AsrError> {
        Ok(out
            .ok_or_else(|| AsrError::OutputNotFound(name.into()))?
            .try_extract_array()?
            .into_dimensionality()?
            .to_owned())
    }
}
