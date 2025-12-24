use std::collections::VecDeque;

use crate::streaming::hypothesis::WordWithTime;

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
            history_size: 4,
            commit_lag_ms: 50,
            time_bucket_ms: 100,
            safety_margin_words: 0,
            max_uncommitted_duration_ms: 4000,
        }
    }
}

pub struct WordHypothesisManager {
    config: WordHypothesisConfig,
    committed: Vec<WordWithTime>,
    current_draft: Vec<WordWithTime>,
    history: VecDeque<Vec<WordWithTime>>,
    emitted_committed_len: usize,
}

impl WordHypothesisManager {
    pub fn new(config: WordHypothesisConfig) -> Self {
        Self {
            config,
            committed: Vec::new(),
            current_draft: Vec::new(),
            history: VecDeque::new(),
            emitted_committed_len: 0,
        }
    }

    pub fn update_draft(
        &mut self,
        new_words: Vec<WordWithTime>,
        current_audio_ms: i64,
    ) -> Option<(usize, i32)> {
        self.current_draft = new_words;

        log::debug!("Draft after update: {} words", self.current_draft.len());

        self.history.push_back(self.current_draft.clone());
        if self.history.len() > self.config.history_size {
            self.history.pop_front();
        }

        let stable_prefix = if self.history.len() >= 3 {
            longest_stable_prefix(&self.history, self.config.time_bucket_ms)
        } else {
            Vec::new()
        };

        log::debug!("LCP stable prefix: {} words", stable_prefix.len());

        let words_to_commit = self
            .check_and_commit(stable_prefix, current_audio_ms)
            .or_else(|| self.check_force_commit(current_audio_ms));

        if let Some(to_commit) = words_to_commit {
            let last = match to_commit.last() {
                Some(last) => last,
                None => return None,
            };
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

    pub fn take_newly_committed(&mut self) -> Option<String> {
        let committed_text = words_to_text(&self.committed, &[]);
        let committed_chars: Vec<char> = committed_text.chars().collect();

        if committed_chars.len() > self.emitted_committed_len {
            let new_text: String = committed_chars[self.emitted_committed_len..]
                .iter()
                .collect();
            self.emitted_committed_len = committed_chars.len();
            if !new_text.trim().is_empty() {
                return Some(new_text);
            }
        }
        None
    }

    pub fn get_draft_only_text(&self) -> String {
        words_to_text(&[], &self.current_draft)
    }
}

fn longest_stable_prefix(
    history: &VecDeque<Vec<WordWithTime>>,
    bucket_ms: i64,
) -> Vec<WordWithTime> {
    if history.is_empty() {
        return Vec::new();
    }

    let first = &history[0];
    let mut stable = Vec::new();

    'outer: for (i, word) in first.iter().enumerate() {
        for draft in history.iter().skip(1) {
            if i >= draft.len() {
                break 'outer;
            }

            let other = &draft[i];

            if !word.matches_in_time_bucket(other, bucket_ms) {
                break 'outer;
            }
        }

        stable.push(word.clone());
    }

    stable
}

fn words_to_text(committed: &[WordWithTime], draft: &[WordWithTime]) -> String {
    let mut result = String::new();
    let mut first = true;

    for word in committed.iter().chain(draft.iter()) {
        if !first {
            result.push(' ');
        }
        result.push_str(&word.text);
        first = false;
    }

    result
}
