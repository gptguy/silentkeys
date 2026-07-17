use silent_keys_lib::desktop::{
    append_for_tests, deliver_for_tests, plan_final_delivery, FinalDelivery,
};
use silent_keys_lib::streaming::TranscriptionUpdate;

#[test]
fn final_delivery_appends_only_a_true_suffix() {
    assert_eq!(
        plan_final_delivery("hello", "hello world"),
        FinalDelivery::Append(" world".to_string())
    );
}

#[test]
fn final_delivery_replaces_divergent_streaming_text() {
    assert_eq!(
        plan_final_delivery("recognize this wrong", "recognize this correctly"),
        FinalDelivery::Replace {
            previous_chars: 20,
            text: "recognize this correctly".to_string(),
        }
    );
}

#[test]
fn text_commits_only_after_typing_is_acknowledged() {
    let mut buffer = "hello".to_string();
    let mut buffer_at_submit = None;

    deliver_for_tests(&mut buffer, "hello world".to_string(), |delivery| {
        assert_eq!(delivery, FinalDelivery::Append(" world".to_string()));
        buffer_at_submit = Some("hello".to_string());
        Ok(())
    })
    .expect("delivery should succeed");

    assert_eq!(buffer_at_submit.as_deref(), Some("hello"));
    assert_eq!(buffer, "hello world");
}

#[test]
fn buffer_stays_unchanged_when_typing_fails() {
    let mut buffer = "hello".to_string();

    let result = deliver_for_tests(&mut buffer, "goodbye".to_string(), |_| {
        Err("keyboard busy".to_string())
    });

    assert_eq!(result, Err("keyboard busy".to_string()));
    assert_eq!(buffer, "hello");
}

#[test]
fn streaming_append_commits_only_the_new_chunk_after_acknowledgement() {
    let mut buffer = "hello".to_string();
    let mut submitted = None;

    append_for_tests(&mut buffer, " world".to_string(), |delivery| {
        submitted = Some(delivery);
        Ok(())
    })
    .expect("append should succeed");

    assert_eq!(submitted, Some(FinalDelivery::Append(" world".to_string())));
    assert_eq!(buffer, "hello world");
}

#[test]
fn empty_final_transcript_clears_divergent_streamed_text() {
    let mut buffer = "wrong words".to_string();
    let mut submitted = None;

    deliver_for_tests(&mut buffer, String::new(), |delivery| {
        submitted = Some(delivery);
        Ok(())
    })
    .expect("delivery should succeed");

    assert_eq!(
        submitted,
        Some(FinalDelivery::Replace {
            previous_chars: 11,
            text: String::new(),
        })
    );
    assert!(buffer.is_empty());
}

#[test]
fn transcription_updates_have_stable_wire_shapes() {
    let append = serde_json::to_value(TranscriptionUpdate::Append("hello".to_string()))
        .expect("append update should serialize");
    let replace = serde_json::to_value(TranscriptionUpdate::Replace("world".to_string()))
        .expect("replace update should serialize");

    assert_eq!(
        append,
        serde_json::json!({ "kind": "append", "text": "hello" })
    );
    assert_eq!(
        replace,
        serde_json::json!({ "kind": "replace", "text": "world" })
    );
}
