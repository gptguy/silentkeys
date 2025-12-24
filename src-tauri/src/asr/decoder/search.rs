use crate::asr::recognizer::{AsrError, AsrModel, InferenceConfig};

use super::session::DecoderSession;
use super::state::Hypothesis;

pub(crate) fn decode_sequence_greedy(
    session: &mut DecoderSession,
    model: &mut AsrModel,
    encodings: &ndarray::ArrayViewD<f32>,
    encodings_len: usize,
    config: &InferenceConfig,
) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
    let mut tokens = Vec::with_capacity(encodings_len / 2);
    let mut timestamps = Vec::with_capacity(encodings_len / 2);
    let max_tokens = config.max_tokens_per_step.max(1);

    for t in 0..encodings_len {
        let step = encodings.slice(ndarray::s![t, ..]);
        for _ in 0..max_tokens {
            let s = session.step_scores(
                model,
                &step,
                *tokens.last().unwrap_or(&session.last_token),
                &session.workspace.state.clone(),
                config,
                1,
            )?;
            let token = s
                .top_tokens
                .first()
                .filter(|(_, sc)| *sc > s.blank_score)
                .map(|(i, _)| *i as i32)
                .unwrap_or(session.blank_idx);
            if token == session.blank_idx {
                break;
            }
            session.workspace.state = s.output_state;
            session.last_token = token;
            tokens.push(token);
            timestamps.push(t);
        }
    }
    Ok((tokens, timestamps))
}

pub(crate) fn decode_sequence_beam(
    session: &mut DecoderSession,
    model: &mut AsrModel,
    encodings: &ndarray::ArrayViewD<f32>,
    encodings_len: usize,
    config: &InferenceConfig,
    beam_width: usize,
) -> Result<(Vec<i32>, Vec<usize>), AsrError> {
    let mut beam = vec![Hypothesis {
        tokens: Vec::with_capacity(encodings_len / 2),
        timestamps: Vec::with_capacity(encodings_len / 2),
        score: 0.0,
        state: session.workspace.state.clone(),
        last_token: session.last_token,
    }];

    for t in 0..encodings_len {
        let step = encodings.slice(ndarray::s![t, ..]);
        let mut candidates = Vec::new();
        let mut active = beam;

        for _ in 0..config.max_tokens_per_step.max(1) {
            if active.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for h in &active {
                let s = session.step_scores(
                    model,
                    &step,
                    h.last_token,
                    &h.state,
                    config,
                    beam_width,
                )?;
                candidates.push({
                    let mut bh = h.clone();
                    bh.score += s.blank_score;
                    bh
                });
                for (idx, sc) in &s.top_tokens {
                    let mut nh = h.clone();
                    nh.score += *sc;
                    nh.tokens.push(*idx as i32);
                    nh.timestamps.push(t);
                    nh.state = s.output_state.clone();
                    nh.last_token = *idx as i32;
                    next.push(nh);
                }
            }
            next.sort_by(|a, b| b.score.total_cmp(&a.score));
            next.truncate(beam_width);
            active = next;
        }
        candidates.extend(active);
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        candidates.truncate(beam_width);
        beam = candidates;
    }

    beam.sort_by(|a, b| b.score.total_cmp(&a.score));
    let best = beam.remove(0);
    session.workspace.state = best.state;
    session.last_token = best.last_token;
    Ok((best.tokens, best.timestamps))
}
