use std::time::Instant;

use ndarray::{Array1, ArrayD, ArrayView2, ArrayViewD, IxDyn};
use ort::inputs;
use ort::value::TensorRef;

use super::config::{AsrError, InferenceConfig, Transcript};
use super::model::AsrModel;

impl AsrModel {
    pub fn preprocess(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_lens: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), AsrError> {
        let wv = waveforms.as_standard_layout();
        let wl = waveforms_lens.as_standard_layout();
        let start = Instant::now();
        let outputs = self.preprocessor.run(inputs![
            "waveforms" => TensorRef::from_array_view(wv.view())?,
            "waveforms_lens" => TensorRef::from_array_view(wl.view())?,
        ])?;
        log::debug!("Preprocessor inference completed in {:?}", start.elapsed());

        Ok((
            outputs
                .get("features")
                .ok_or_else(|| AsrError::OutputNotFound("features".into()))?
                .try_extract_array()?
                .to_owned(),
            outputs
                .get("features_lens")
                .ok_or_else(|| AsrError::OutputNotFound("features_lens".into()))?
                .try_extract_array()?
                .to_owned(),
        ))
    }

    pub fn encode(
        &mut self,
        audio_signal: &ArrayViewD<f32>,
        length: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), AsrError> {
        let sig = audio_signal.as_standard_layout();
        let len = length.as_standard_layout();
        let start = Instant::now();
        let outputs = self.encoder.run(inputs![
            "audio_signal" => TensorRef::from_array_view(sig.view())?,
            "length" => TensorRef::from_array_view(len.view())?,
        ])?;
        log::debug!("Encoder inference completed in {:?}", start.elapsed());

        let out = outputs
            .get("outputs")
            .ok_or_else(|| AsrError::OutputNotFound("outputs".into()))?
            .try_extract_array()?
            .permuted_axes(IxDyn(&[0, 2, 1]))
            .to_owned();
        let lens = outputs
            .get("encoded_lengths")
            .ok_or_else(|| AsrError::OutputNotFound("encoded_lengths".into()))?
            .try_extract_array()?
            .to_owned();

        Ok((out, lens))
    }

    fn recognize_batch(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_len: &ArrayViewD<i64>,
        config: &InferenceConfig,
    ) -> Result<Vec<Transcript>, AsrError> {
        let recognize_start = Instant::now();
        let (features, features_lens) = self.preprocess(waveforms, waveforms_len)?;
        let (encoder_out, encoder_out_lens) =
            self.encode(&features.view(), &features_lens.view())?;

        let mut results = Vec::new();
        let ws = self.cached_workspace.take();
        let mut session =
            crate::asr::decoder::DecoderSession::new(self, ws, self.cached_last_token, config)?;

        let mut raw_results = Vec::new();
        for (enc, &enc_len) in encoder_out.outer_iter().zip(encoder_out_lens.iter()) {
            raw_results.push(session.decode_sequence(
                self,
                &enc.view(),
                enc_len as usize,
                config,
            )?);
        }

        let (ws, last_token) = session.into_parts();
        self.cached_workspace = Some(ws);
        self.cached_last_token = last_token;

        for (tokens, timestamps) in raw_results {
            results.push(self.decode_tokens(tokens, timestamps));
        }

        log::debug!(
            "recognize_batch completed for {} item(s) in {:?}",
            results.len(),
            recognize_start.elapsed()
        );
        Ok(results)
    }

    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        reuse_workspace: bool,
        config: Option<InferenceConfig>,
    ) -> Result<Transcript, AsrError> {
        self.transcribe_samples_ref(&samples, reuse_workspace, config)
    }

    pub fn transcribe_samples_ref(
        &mut self,
        samples: &[f32],
        reuse_workspace: bool,
        config: Option<InferenceConfig>,
    ) -> Result<Transcript, AsrError> {
        let samples_len = samples.len();
        log::debug!("Transcription (len: {})", samples_len);

        let audio = ArrayView2::from_shape((1, samples_len), samples)?.into_dyn();
        let audio_lengths = Array1::from_vec(vec![samples_len as i64]).into_dyn();

        if !reuse_workspace {
            self.reset_state();
        }

        let results =
            self.recognize_batch(&audio, &audio_lengths.view(), &config.unwrap_or_default())?;
        results.into_iter().next().ok_or_else(|| {
            AsrError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No result",
            ))
        })
    }

    pub fn reset_state(&mut self) {
        self.cached_workspace = None;
        self.cached_last_token = self.blank_idx;
    }
}
