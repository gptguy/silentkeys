use rubato::{
    Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::collections::VecDeque;

pub const PROCESS_CHUNK_SIZE: usize = 480;
const RESAMPLER_CHUNK_OUT: usize = PROCESS_CHUNK_SIZE;

#[derive(thiserror::Error, Debug)]
pub enum AudioError {
    #[error("Failed to create resampler: {0}")]
    ResamplerCreation(String),
    #[error("Resampler processing failed: {0}")]
    ResamplerProcessing(String),
}

#[derive(Debug)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
}

pub struct AudioProcessor {
    resampler: Option<SincFixedOut<f32>>,
    buffer: VecDeque<f32>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<f32>,
}

impl AudioProcessor {
    pub fn new(in_sample_rate: usize, out_sample_rate: usize) -> Result<Self, AudioError> {
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

        Ok(Self {
            resampler,
            buffer: VecDeque::with_capacity(4096),
            scratch_in: Vec::with_capacity(2048),
            scratch_out: Vec::with_capacity(PROCESS_CHUNK_SIZE),
        })
    }

    pub fn process(
        &mut self,
        data: &[f32],
        mut emit: impl FnMut(AudioFrame),
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

                    for chunk in resampled {
                        if !chunk.is_empty() {
                            emit(AudioFrame { samples: chunk });
                        }
                    }
                } else {
                    break;
                }
            } else if self.buffer.len() >= PROCESS_CHUNK_SIZE {
                let mut scratch = std::mem::take(&mut self.scratch_out);
                scratch.clear();
                let (front, back) = self.buffer.as_slices();
                let front_take = front.len().min(PROCESS_CHUNK_SIZE);
                scratch.extend_from_slice(&front[..front_take]);
                if front_take < PROCESS_CHUNK_SIZE && !back.is_empty() {
                    let back_take = (PROCESS_CHUNK_SIZE - front_take).min(back.len());
                    scratch.extend_from_slice(&back[..back_take]);
                }
                self.buffer.drain(..PROCESS_CHUNK_SIZE);

                let frame = scratch.clone();
                if !frame.is_empty() {
                    emit(AudioFrame { samples: frame });
                }
                self.scratch_out = scratch;
            } else {
                break;
            }
        }

        if self.buffer.capacity() > 16384 && self.buffer.len() < 1024 {
            self.buffer.shrink_to(4096);
        }

        Ok(())
    }

    pub fn flush(&mut self, mut emit: impl FnMut(AudioFrame)) -> Result<(), AudioError> {
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
                    if !chunk.is_empty() {
                        emit(AudioFrame { samples: chunk });
                    }
                }
            }
        } else if !self.buffer.is_empty() {
            let tail: Vec<f32> = self.buffer.drain(..).collect();
            if !tail.is_empty() {
                emit(AudioFrame { samples: tail });
            }
        }

        Ok(())
    }
}
