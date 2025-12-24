use silent_keys_lib::streaming::word_hypothesis::{WordHypothesisConfig, WordHypothesisManager};
use silent_keys_lib::streaming::words::WordWithTime;

fn make_word(
    text: &str,
    t0_ms: i64,
    t1_ms: i64,
    end_frame: usize,
    last_token_id: i32,
) -> WordWithTime {
    WordWithTime {
        text: text.to_string(),
        t0_ms,
        t1_ms,
        end_frame,
        last_token_id,
    }
}

#[test]
fn test_update_draft_requires_consensus() {
    let config = WordHypothesisConfig {
        commit_lag_ms: 0,
        ..Default::default()
    };
    let mut manager = WordHypothesisManager::new(config);
    let words = vec![make_word("hello", 0, 200, 2, 10)];

    assert!(manager.update_draft(words.clone(), 300).is_none());
    assert!(manager.update_draft(words.clone(), 300).is_none());
    assert!(manager.update_draft(words, 300).is_some());
    assert_eq!(manager.get_full_text(), "hello");
}

#[test]
fn test_update_draft_breaks_on_change() {
    let config = WordHypothesisConfig {
        commit_lag_ms: 0,
        ..Default::default()
    };
    let mut manager = WordHypothesisManager::new(config);
    let words = vec![
        make_word("hello", 0, 200, 2, 10),
        make_word("it", 201, 400, 4, 11),
    ];
    let changed = vec![
        make_word("hello", 0, 200, 2, 10),
        make_word("there", 201, 400, 4, 12),
    ];

    assert!(manager.update_draft(words.clone(), 500).is_none());
    assert!(manager.update_draft(words, 500).is_none());
    assert!(manager.update_draft(changed, 500).is_some());
    assert_eq!(
        manager.take_newly_committed().map(|(_, t)| t),
        Some("hello".to_string())
    );
}

#[test]
fn test_commit_lag_respected() {
    let config = WordHypothesisConfig {
        commit_lag_ms: 600,
        ..Default::default()
    };
    let mut manager = WordHypothesisManager::new(config);
    let words = vec![make_word("hello", 0, 200, 2, 10)];

    assert!(manager.update_draft(words.clone(), 500).is_none());
    assert!(manager.update_draft(words.clone(), 500).is_none());
    assert!(manager.update_draft(words, 900).is_some());
}

#[test]
fn test_take_newly_committed_returns_proper_indices() {
    let config = WordHypothesisConfig {
        commit_lag_ms: 0,
        ..Default::default()
    };
    let mut manager = WordHypothesisManager::new(config);

    let words1 = vec![make_word("hello", 0, 200, 2, 10)];
    manager.update_draft(words1.clone(), 300);
    manager.update_draft(words1.clone(), 300);
    manager.update_draft(words1, 300);

    let (start, text) = manager.take_newly_committed().expect("should commit");
    assert_eq!(start, 0, "First commit should start at index 0");
    assert_eq!(text, "hello");

    let words2 = vec![
        make_word("hello", 0, 200, 2, 10),
        make_word("world", 201, 400, 4, 11),
    ];
    manager.update_draft(words2.clone(), 500);
    manager.update_draft(words2.clone(), 500);
    manager.update_draft(words2, 500);

    let (start, text) = manager
        .take_newly_committed()
        .expect("should commit second word");
    assert_eq!(
        start, 5,
        "Second commit should start at index 5 (after 'hello')"
    );
    assert_eq!(text, " world", "Should include leading space");
}

#[test]
fn test_take_newly_committed_allows_whitespace() {
    let config = WordHypothesisConfig {
        commit_lag_ms: 0,
        ..Default::default()
    };
    let mut manager = WordHypothesisManager::new(config);

    let words = vec![make_word("a", 0, 100, 1, 5)];
    manager.update_draft(words.clone(), 200);
    manager.update_draft(words.clone(), 200);
    manager.update_draft(words, 200);

    manager.take_newly_committed();

    let words2 = vec![make_word("a", 0, 100, 1, 5), make_word("b", 101, 200, 2, 6)];
    manager.update_draft(words2.clone(), 300);
    manager.update_draft(words2.clone(), 300);
    manager.update_draft(words2, 300);

    let result = manager.take_newly_committed();
    assert!(
        result.is_some(),
        "Should emit even if text starts with space"
    );
    let (_, text) = result.unwrap();
    assert_eq!(text, " b", "Should include the leading space");
}

#[test]
fn test_stable_char_len_tracks_emissions() {
    let config = WordHypothesisConfig {
        commit_lag_ms: 0,
        ..Default::default()
    };
    let mut manager = WordHypothesisManager::new(config);

    assert_eq!(manager.stable_char_len, 0);
    assert_eq!(manager.total_char_len, 0);

    let words = vec![make_word("hello", 0, 200, 2, 10)];
    manager.update_draft(words.clone(), 300);
    manager.update_draft(words.clone(), 300);
    manager.update_draft(words, 300);

    assert_eq!(
        manager.total_char_len, 5,
        "total_char_len should be updated"
    );

    let result = manager.take_newly_committed();
    assert!(result.is_some());
    let (start, _) = result.unwrap();
    assert_eq!(start, 0, "First emission should start at 0");
    assert_eq!(
        manager.stable_char_len, 5,
        "stable_char_len should track committed text"
    );
}
