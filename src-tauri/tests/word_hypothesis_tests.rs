use silent_keys_lib::streaming::hypothesis::WordWithTime;
use silent_keys_lib::streaming::word_hypothesis::{WordHypothesisConfig, WordHypothesisManager};

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
    assert_eq!(manager.take_newly_committed().as_deref(), Some("hello"));
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
    assert!(manager.update_draft(words.clone(), 500).is_none());
    assert!(manager.update_draft(words, 900).is_some());
}
