use crate::asr::recognizer::{AsrError, AsrModel, InferenceConfig};

pub(crate) mod search;
pub(crate) mod session;
pub(crate) mod state;

pub use session::DecoderSession;
pub use state::*;

impl DecoderSession {
    pub(crate) fn decode_sequence(
        &mut self,
        model: &mut AsrModel,
        encodings: &ndarray::ArrayViewD<f32>,
        encodings_len: usize,
        config: &InferenceConfig,
    ) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
        let beam_width = config.beam_width.max(1);
        if beam_width <= 1 {
            return search::decode_sequence_greedy(self, model, encodings, encodings_len, config);
        }
        search::decode_sequence_beam(self, model, encodings, encodings_len, config, beam_width)
    }
}
