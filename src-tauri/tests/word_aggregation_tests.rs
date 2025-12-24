use silent_keys_lib::streaming::hypothesis::TokenWithTime;
use silent_keys_lib::streaming::word_aggregation::tokens_to_words;

fn make_token(text: &str, start_frame: usize, end_frame: usize) -> TokenWithTime {
    TokenWithTime {
        token_id: 0,
        text: text.to_string(),
        start_frame,
        end_frame,
    }
}

#[test]
fn test_simple_words() {
    let tokens = vec![make_token("hello", 0, 2), make_token(" world", 2, 4)];

    let words = tokens_to_words(tokens);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "hello");
    assert_eq!(words[1].text, "world");
}

#[test]
fn test_subword_aggregation() {
    let tokens = vec![
        make_token("hello", 0, 2),
        make_token(" sh", 2, 4),
        make_token("it", 4, 6),
    ];

    let words = tokens_to_words(tokens);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "hello");
    assert_eq!(words[1].text, "shit");
    assert_eq!(words[1].t0_ms, 160);
    assert_eq!(words[1].t1_ms, 480);
}

#[test]
fn test_punctuation_preserved() {
    let tokens = vec![make_token("hello", 0, 2), make_token(".", 2, 3)];

    let words = tokens_to_words(tokens);
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].text, "hello.");
}

#[test]
fn test_time_conversion() {
    let tokens = vec![make_token("test", 10, 15)];

    let words = tokens_to_words(tokens);
    assert_eq!(words[0].t0_ms, 800);
    assert_eq!(words[0].t1_ms, 1200);
}
