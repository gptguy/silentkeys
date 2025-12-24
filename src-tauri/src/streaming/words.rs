use std::collections::VecDeque;

use crate::asr::FRAME_DURATION_MS;

#[derive(Debug, Clone, PartialEq)]
pub struct TokenWithTime {
    pub token_id: i32,
    pub text: String,
    pub start_frame: usize,
    pub end_frame: usize,
}

impl TokenWithTime {
    pub fn start_ms(&self) -> i64 {
        (self.start_frame as i64) * FRAME_DURATION_MS
    }

    pub fn end_ms(&self) -> i64 {
        (self.end_frame as i64) * FRAME_DURATION_MS
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordWithTime {
    pub text: String,
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub end_frame: usize,
    pub last_token_id: i32,
}

impl WordWithTime {
    pub fn center_ms(&self) -> i64 {
        (self.t0_ms + self.t1_ms) / 2
    }

    pub fn matches_in_time_bucket(&self, other: &WordWithTime, bucket_ms: i64) -> bool {
        if self.text != other.text {
            return false;
        }
        let self_bucket = self.center_ms() / bucket_ms;
        let other_bucket = other.center_ms() / bucket_ms;
        self_bucket == other_bucket
    }
}

pub fn tokens_to_words(tokens: Vec<TokenWithTime>) -> Vec<WordWithTime> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut current_word_tokens = Vec::new();

    for token in tokens {
        let starts_word = token.text.starts_with(' ') || current_word_tokens.is_empty();

        if starts_word && !current_word_tokens.is_empty() {
            if let Some(word) = finalize_word(&current_word_tokens) {
                words.push(word);
            }
            current_word_tokens.clear();
        }

        current_word_tokens.push(token);
    }

    if !current_word_tokens.is_empty() {
        if let Some(word) = finalize_word(&current_word_tokens) {
            words.push(word);
        }
    }

    words
}

fn finalize_word(tokens: &[TokenWithTime]) -> Option<WordWithTime> {
    let first = tokens.first()?;
    let last = tokens.last()?;
    let raw_text = tokens.iter().map(|t| t.text.as_str()).collect::<String>();
    let text = raw_text.trim();
    if text.is_empty() {
        return None;
    }

    Some(WordWithTime {
        text: text.to_string(),
        t0_ms: first.start_ms(),
        t1_ms: last.end_ms(),
        end_frame: last.end_frame,
        last_token_id: last.token_id,
    })
}

pub(crate) fn longest_stable_prefix(
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

pub(crate) fn words_to_text(committed: &[WordWithTime], draft: &[WordWithTime]) -> String {
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
