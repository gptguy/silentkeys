use std::collections::VecDeque;

use crate::streaming::words::{longest_stable_prefix, words_to_text, WordWithTime};

pub struct WordHypothesisConfig {
    pub history_size: usize,
    pub commit_lag_ms: i64,
    pub time_bucket_ms: i64,
    pub safety_margin_words: usize,
    pub max_uncommitted_duration_ms: i64,
}

impl Default for WordHypothesisConfig {
    fn default() -> Self {
        Self {
            history_size: 3,
            commit_lag_ms: 50,
            time_bucket_ms: 100,
            safety_margin_words: 0,
            max_uncommitted_duration_ms: 1500,
        }
    }
}

pub struct WordHypothesisManager {
    config: WordHypothesisConfig,
    committed: Vec<WordWithTime>,
    current_draft: Vec<WordWithTime>,
    history: VecDeque<Vec<WordWithTime>>,
    pub stable_char_len: usize,
    pub total_char_len: usize,
}

impl WordHypothesisManager {
    pub fn new(config: WordHypothesisConfig) -> Self {
        Self {
            config,
            committed: Vec::new(),
            current_draft: Vec::new(),
            history: VecDeque::new(),
            stable_char_len: 0,
            total_char_len: 0,
        }
    }

    pub fn update_draft(
        &mut self,
        new_words: Vec<WordWithTime>,
        current_audio_ms: i64,
    ) -> Option<(usize, i32)> {
        // Filter out words that have already been committed based on time
        let last_committed_t1 = self.committed.last().map(|w| w.t1_ms).unwrap_or(-1);
        self.current_draft = new_words
            .into_iter()
            .filter(|w| w.t0_ms >= last_committed_t1)
            .collect();

        log::debug!("Draft after update: {} words", self.current_draft.len());

        self.history.push_back(self.current_draft.clone());
        if self.history.len() > self.config.history_size {
            self.history.pop_front();
        }

        let stable_prefix = if self.history.len() >= self.config.history_size {
            longest_stable_prefix(&self.history, self.config.time_bucket_ms)
        } else {
            Vec::new()
        };

        log::debug!("LCP stable prefix: {} words", stable_prefix.len());

        let words_to_commit = self
            .check_and_commit(stable_prefix, current_audio_ms)
            .or_else(|| self.check_force_commit(current_audio_ms));

        if let Some(to_commit) = words_to_commit {
            let last = to_commit.last()?;
            let commit_boundary_ms = last.t1_ms;
            let commit_frame = last.end_frame;
            let commit_token = last.last_token_id;

            log::info!(
                "Committing {} words up to {}ms (frame={}, token={})",
                to_commit.len(),
                commit_boundary_ms,
                commit_frame,
                commit_token
            );

            self.committed.extend(to_commit);

            self.current_draft.retain(|w| w.t0_ms > commit_boundary_ms);
            for draft in &mut self.history {
                draft.retain(|w| w.t0_ms > commit_boundary_ms);
            }

            self.total_char_len = self.get_full_text().chars().count();
            return Some((commit_frame, commit_token));
        }

        None
    }

    fn check_and_commit(
        &self,
        stable_prefix: Vec<WordWithTime>,
        current_audio_ms: i64,
    ) -> Option<Vec<WordWithTime>> {
        if stable_prefix.is_empty() {
            return None;
        }

        let mut to_commit = Vec::new();

        for (i, word) in stable_prefix.iter().enumerate() {
            if i >= stable_prefix
                .len()
                .saturating_sub(self.config.safety_margin_words)
            {
                break;
            }

            if word.t1_ms <= current_audio_ms - self.config.commit_lag_ms {
                to_commit.push(word.clone());
            } else {
                break;
            }
        }

        if to_commit.is_empty() {
            None
        } else {
            Some(to_commit)
        }
    }

    fn check_force_commit(&self, current_audio_ms: i64) -> Option<Vec<WordWithTime>> {
        if self.current_draft.is_empty() {
            return None;
        }

        let first = &self.current_draft[0];
        if current_audio_ms - first.t0_ms > self.config.max_uncommitted_duration_ms {
            log::info!(
                "Force committing due to lag ({}ms > {}ms)",
                current_audio_ms - first.t0_ms,
                self.config.max_uncommitted_duration_ms
            );
            return Some(vec![first.clone()]);
        }

        None
    }

    pub fn get_full_text(&self) -> String {
        words_to_text(&self.committed, &self.current_draft)
    }

    pub fn take_newly_committed(&mut self) -> Option<(usize, String)> {
        let committed_text = words_to_text(&self.committed, &[]);
        let committed_chars: Vec<char> = committed_text.chars().collect();

        if committed_chars.len() > self.stable_char_len {
            let new_text: String = committed_chars[self.stable_char_len..].iter().collect();
            let start = self.stable_char_len;
            self.stable_char_len = committed_chars.len();
            return Some((start, new_text));
        }
        None
    }

    pub fn get_draft_only_text(&self) -> String {
        words_to_text(&[], &self.current_draft)
    }
}
