use crate::streaming::hypothesis::{TokenWithTime, WordWithTime};

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
