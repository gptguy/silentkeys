use crate::error::AudioError;
use crate::vad::{VadSession, VAD_CHUNK_SIZE};
use rubato::{
    Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::collections::VecDeque;

const RESAMPLER_CHUNK_OUT: usize = VAD_CHUNK_SIZE; // 480

#[derive(Debug)]
pub enum ProcessingEvent {
    Speech(Vec<f32>),
    SpeechStart(Vec<f32>),
    SpeechEnd,
}

pub struct AudioProcessor {
    resampler: Option<SincFixedOut<f32>>,
    vad: VadSession,
    buffer: VecDeque<f32>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<f32>,
    was_speech: bool,
}

impl AudioProcessor {
    pub fn new(
        in_sample_rate: usize,
        out_sample_rate: usize,
        vad_model_path: &std::path::Path,
    ) -> Result<Self, AudioError> {
        let resampler = if in_sample_rate != out_sample_rate {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };

            log::info!(
                "Configuring resampler: {} Hz -> {} Hz",
                in_sample_rate,
                out_sample_rate
            );

            Some(
                SincFixedOut::<f32>::new(
                    out_sample_rate as f64 / in_sample_rate as f64,
                    2.0,
                    params,
                    RESAMPLER_CHUNK_OUT,
                    1,
                )
                .map_err(|e| AudioError::ResamplerCreation(e.to_string()))?,
            )
        } else {
            log::debug!(
                "Resampler not needed ({} Hz input matches target)",
                in_sample_rate
            );
            None
        };

        let vad = VadSession::new(
            vad_model_path,
            0.5,
            10, // 300ms prefill
            16, // 500ms hangover
            3,  // 90ms onset
        )
        .map_err(|e| AudioError::VadError(e.to_string()))?;

        Ok(Self {
            resampler,
            vad,
            buffer: VecDeque::with_capacity(4096),
            scratch_in: Vec::with_capacity(2048),
            scratch_out: Vec::with_capacity(VAD_CHUNK_SIZE),
            was_speech: false,
        })
    }

    pub fn process(
        &mut self,
        data: &[f32],
        mut emit: impl FnMut(ProcessingEvent),
    ) -> Result<(), AudioError> {
        self.buffer.extend(data.iter());

        loop {
            if let Some(resampler) = &mut self.resampler {
                let needed = resampler.input_frames_next();
                if self.buffer.len() >= needed {
                    self.scratch_in.clear();
                    let (front, back) = self.buffer.as_slices();
                    let front_take = front.len().min(needed);
                    self.scratch_in.extend_from_slice(&front[..front_take]);
                    if front_take < needed && !back.is_empty() {
                        let back_take = (needed - front_take).min(back.len());
                        self.scratch_in.extend_from_slice(&back[..back_take]);
                    }
                    self.buffer.drain(..needed);

                    let resampled = resampler
                        .process(&[&self.scratch_in], None)
                        .map_err(|e| AudioError::ResamplerProcessing(e.to_string()))?;

                    self.process_vad_chunk(&resampled[0], &mut emit)?;
                } else {
                    break;
                }
            } else if self.buffer.len() >= VAD_CHUNK_SIZE {
                self.scratch_out.clear();
                let (front, back) = self.buffer.as_slices();
                let front_take = front.len().min(VAD_CHUNK_SIZE);
                self.scratch_out.extend_from_slice(&front[..front_take]);
                if front_take < VAD_CHUNK_SIZE && !back.is_empty() {
                    let back_take = (VAD_CHUNK_SIZE - front_take).min(back.len());
                    self.scratch_out.extend_from_slice(&back[..back_take]);
                }
                self.buffer.drain(..VAD_CHUNK_SIZE);

                let chunk: Vec<f32> = self.scratch_out.clone();
                self.process_vad_chunk(&chunk, &mut emit)?;
            } else {
                break;
            }
        }

        if self.buffer.capacity() > 16384 && self.buffer.len() < 1024 {
            self.buffer.shrink_to_fit();
        }

        Ok(())
    }

    pub fn flush(&mut self, mut emit: impl FnMut(ProcessingEvent)) -> Result<(), AudioError> {
        if let Some(resampler) = &mut self.resampler {
            if !self.buffer.is_empty() {
                let needed = resampler.input_frames_next();
                let mut tail: Vec<f32> = self.buffer.drain(..).collect();
                if tail.len() < needed {
                    tail.resize(needed, 0.0);
                }

                let resampled = resampler
                    .process(&[&tail], None)
                    .map_err(|e| AudioError::ResamplerProcessing(e.to_string()))?;
                for chunk in resampled {
                    self.process_vad_chunk(&chunk, &mut emit)?;
                }
            }
        } else if !self.buffer.is_empty() {
            let mut tail: Vec<f32> = self.buffer.drain(..).collect();
            if tail.len() < VAD_CHUNK_SIZE {
                tail.resize(VAD_CHUNK_SIZE, 0.0);
            }
            self.process_vad_chunk(&tail, &mut emit)?;
        }

        if self.was_speech {
            emit(ProcessingEvent::SpeechEnd);
            self.was_speech = false;
        }

        Ok(())
    }

    fn process_vad_chunk(
        &mut self,
        samples: &[f32],
        emit: &mut impl FnMut(ProcessingEvent),
    ) -> Result<(), AudioError> {
        let is_speech = self
            .vad
            .process_frame(samples)
            .map_err(|e| AudioError::VadError(e.to_string()))?;

        if is_speech {
            if !self.was_speech {
                log::debug!("VAD: Speech START");
                if self.vad.has_prefill() {
                    let mut prefill = self.vad.take_prefill();
                    prefill.extend_from_slice(samples);
                    emit(ProcessingEvent::SpeechStart(prefill));
                } else {
                    emit(ProcessingEvent::SpeechStart(samples.to_vec()));
                }
            } else {
                emit(ProcessingEvent::Speech(samples.to_vec()));
            }
        } else if self.was_speech {
            log::debug!("VAD: Speech END");
            emit(ProcessingEvent::SpeechEnd);
        }

        self.was_speech = is_speech;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.vad.reset();
        self.buffer.clear();
        self.was_speech = false;
    }
}
