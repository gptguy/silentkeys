use crate::asr::decoder::DecoderSession as InnerDecoder;
use crate::asr::{AsrModel, InferenceConfig};
use crate::streaming::words::TokenWithTime;

pub struct DecodingSession {
    config: InferenceConfig,
    encoder_cursor: usize,
    buffer_start_frame: usize,
    audio_buffer: Vec<f32>,
    last_token: i32,
}

impl DecodingSession {
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            encoder_cursor: 0,
            buffer_start_frame: 0,
            audio_buffer: Vec::new(),
            last_token: -1,
        }
    }

    pub fn advance_segment(
        &mut self,
        model: &mut AsrModel,
        new_samples: &[f32],
    ) -> Result<Vec<TokenWithTime>, String> {
        if self.last_token == -1 {
            self.last_token = model.blank_idx;
        }

        if !new_samples.is_empty() {
            self.audio_buffer.extend_from_slice(new_samples);
        }

        const MIN_BUFFER_SIZE: usize = 3200;

        if self.audio_buffer.len() < MIN_BUFFER_SIZE {
            return Ok(Vec::new());
        }

        let batch_size = 1;
        let samples_len = self.audio_buffer.len();
        let audio = ndarray::ArrayView2::from_shape((batch_size, samples_len), &self.audio_buffer)
            .map_err(|e| e.to_string())?
            .into_dyn();
        let audio_lengths = ndarray::Array1::from_vec(vec![samples_len as i64]).into_dyn();

        let (features, features_lens) = model
            .preprocess(&audio, &audio_lengths.view())
            .map_err(|e| e.to_string())?;

        let (encoder_out, _encoder_out_lens) = model
            .encode(&features.view(), &features_lens.view())
            .map_err(|e| e.to_string())?;

        let total_valid_frames = encoder_out.shape()[1];

        let skip_frames = self.encoder_cursor.saturating_sub(self.buffer_start_frame);

        const RIGHT_CONTEXT_DELAY: usize = 6;
        let delayed_frame_count = total_valid_frames.saturating_sub(RIGHT_CONTEXT_DELAY);

        let start_frame_idx = skip_frames;
        let end_frame_idx = delayed_frame_count;

        if end_frame_idx <= start_frame_idx {
            return Ok(Vec::new());
        }

        let time_axis = ndarray::Axis(1);

        let new_encodings: ndarray::ArrayViewD<f32> = encoder_out.slice_axis(
            time_axis,
            ndarray::Slice::from(start_frame_idx..end_frame_idx),
        );
        let frames_count = new_encodings.shape()[1];

        let mut decoder = InnerDecoder::new(model, None, self.last_token, &self.config)
            .map_err(|e| e.to_string())?;

        let single_batch_encodings = new_encodings.index_axis(ndarray::Axis(0), 0);
        let (token_ids, timestamps) = decoder
            .decode_sequence(model, &single_batch_encodings, frames_count, &self.config)
            .map_err(|e| e.to_string())?;

        let mut draft_tokens = Vec::with_capacity(token_ids.len());
        let vocab = &model.vocab;

        for (i, &token_id) in token_ids.iter().enumerate() {
            let idx = token_id as usize;
            let text = if idx < vocab.len() {
                vocab[idx].clone()
            } else {
                String::new()
            };
            let relative_frame = timestamps[i];
            let abs_frame = self.encoder_cursor + relative_frame;

            draft_tokens.push(TokenWithTime {
                token_id,
                text,
                start_frame: abs_frame,
                end_frame: abs_frame,
            });
        }

        Ok(draft_tokens)
    }

    pub fn commit_to(&mut self, frame_limit: usize, last_token: i32) {
        if frame_limit > self.encoder_cursor {
            self.encoder_cursor = frame_limit;
            self.last_token = last_token;

            const SAMPLES_PER_FRAME: usize = 1280;
            const CONTEXT_FRAMES: usize = 50;

            let target_start_frame = self.encoder_cursor.saturating_sub(CONTEXT_FRAMES);

            if target_start_frame > self.buffer_start_frame {
                let frames_to_drop = target_start_frame - self.buffer_start_frame;
                let samples_to_remove = frames_to_drop * SAMPLES_PER_FRAME;

                if samples_to_remove < self.audio_buffer.len() {
                    self.audio_buffer.drain(0..samples_to_remove);
                    self.buffer_start_frame += frames_to_drop;

                    log::debug!(
                        "Trimmed audio buffer: removed {} frames, context start={}, {} samples remain",
                        frames_to_drop,
                        self.buffer_start_frame,
                        self.audio_buffer.len()
                    );
                } else {
                    self.audio_buffer.clear();
                    self.buffer_start_frame = self.encoder_cursor;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.encoder_cursor = 0;
        self.buffer_start_frame = 0;
        self.audio_buffer.clear();
        self.last_token = -1;
    }
}
