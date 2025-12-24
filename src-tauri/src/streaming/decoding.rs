use crate::asr::decoder::DecoderSession as InnerDecoder;
use crate::asr::{AsrModel, InferenceConfig};
use crate::streaming::hypothesis::TokenWithTime;

pub struct DecodingSession {
    config: InferenceConfig,
    /// Total frames processed by the encoder since session start.
    encoder_cursor: usize,
    /// The frame index corresponding to the start of `audio_buffer`.
    buffer_start_frame: usize,
    /// Accumulates audio until we have enough for a valid chunk.
    audio_buffer: Vec<f32>,
    /// Last committed token (to initialize decoder if reset)
    last_token: i32,
}

impl DecodingSession {
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            encoder_cursor: 0,
            buffer_start_frame: 0,
            audio_buffer: Vec::new(),
            last_token: -1, // Will be replaced by blank_idx on first run
        }
    }

    /// Feeds new audio to the session.
    /// Returns tokens emitted from the accumulated audio so far.
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

        // Minimum audio (reduced for responsiveness)
        const MIN_BUFFER_SIZE: usize = 3200;

        if self.audio_buffer.len() < MIN_BUFFER_SIZE {
            return Ok(Vec::new());
        }

        // 1. Preprocess & Encode (Full Buffer: Context + New)
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

        // Calculate where the "new" (uncommitted) content starts in the encoder output
        // buffer_start_frame points to the start of audio_buffer
        // encoder_cursor points to the start of uncommitted content
        // skip_frames is the offset within the buffer
        let skip_frames = self.encoder_cursor.saturating_sub(self.buffer_start_frame);

        // 2. Identify Pending Frames
        // Right Context Delay: Wait for 6 frames (~480ms) of future context
        const RIGHT_CONTEXT_DELAY: usize = 6;
        let delayed_frame_count = total_valid_frames.saturating_sub(RIGHT_CONTEXT_DELAY);

        // We decode from encoder_cursor (via skip_frames) to delayed_frame_count
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

        // 3. Decode Draft (Stateless)
        // We use a FRESH decoder every time to avoid state drift.
        // Initialize with `last_token` (from the end of committed text).
        let mut decoder = InnerDecoder::new(model, None, self.last_token, &self.config)
            .map_err(|e| e.to_string())?;

        let single_batch_encodings = new_encodings.index_axis(ndarray::Axis(0), 0);
        let (token_ids, timestamps) = decoder
            .decode_sequence(model, &single_batch_encodings, frames_count, &self.config)
            .map_err(|e| e.to_string())?;

        // 4. Convert to Tokens
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
            // Abs frame is relative to encoder_cursor (start of this slice)
            let abs_frame = self.encoder_cursor + relative_frame;

            draft_tokens.push(TokenWithTime {
                token_id,
                text,
                start_frame: abs_frame,
                end_frame: abs_frame + 2,
            });
        }

        Ok(draft_tokens)
    }

    /// Commit the transcript up to `frame_limit`.
    /// This advances the internal cursor and updates `last_token` for the next draft start.
    pub fn commit_to(&mut self, frame_limit: usize, last_token: i32) {
        if frame_limit > self.encoder_cursor {
            self.encoder_cursor = frame_limit;
            self.last_token = last_token;

            // CRITICAL: Manage audio buffer with Context Retention
            // Each frame is 80ms of audio at 16kHz = 1280 samples
            const SAMPLES_PER_FRAME: usize = 1280;
            // Keep 20 frames (1.6s) of committed audio as left context for Conformer stability
            const CONTEXT_FRAMES: usize = 20;

            // We want buffer to start at max(0, encoder_cursor - CONTEXT_FRAMES)
            let target_start_frame = self.encoder_cursor.saturating_sub(CONTEXT_FRAMES);

            // If current buffer starts before target, we can drop some frames
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
                    // Should theoretically not happen if logic is sound, but safety first
                    self.audio_buffer.clear();
                    self.buffer_start_frame = self.encoder_cursor; // Reset/Catchup
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
