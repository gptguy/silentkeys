use std::sync::{Mutex, OnceLock};
use std::thread;

use crate::streaming::TranscriptionPatch;
use enigo::{Enigo, Keyboard, Settings};
use rtrb::RingBuffer;

static TRANSCRIPTION_BUFFER: OnceLock<Mutex<String>> = OnceLock::new();
static TYPING_PRODUCER: OnceLock<Mutex<rtrb::Producer<String>>> = OnceLock::new();

fn transcription_buffer() -> &'static Mutex<String> {
    TRANSCRIPTION_BUFFER.get_or_init(|| Mutex::new(String::new()))
}

fn typing_producer() -> &'static Mutex<rtrb::Producer<String>> {
    TYPING_PRODUCER.get_or_init(|| {
        log::info!("Initializing typing queue and worker thread");
        let (prod, mut cons) = RingBuffer::<String>::new(256);
        thread::spawn(move || {
            log::info!("Typing worker thread started");
            typing_worker(&mut cons);
        });
        Mutex::new(prod)
    })
}

fn typing_worker(cons: &mut rtrb::Consumer<String>) {
    let mut enigo = match Enigo::new(&Settings::default()) {
        Ok(e) => {
            log::info!("Enigo initialized successfully in worker thread");
            e
        }
        Err(err) => {
            log::error!("Failed to initialize Enigo for typing worker: {err}");
            return;
        }
    };

    log::info!("Typing worker entering main loop");
    let mut chunks_processed = 0;
    loop {
        match cons.read_chunk(cons.slots()) {
            Ok(chunk) => {
                let (first, second) = chunk.as_slices();
                let count = first.len() + second.len();
                if count > 0 {
                    chunks_processed += 1;
                    log::info!(
                        "Worker processing chunk #{}, {} items",
                        chunks_processed,
                        count
                    );
                    for text in first.iter().chain(second.iter()) {
                        log::info!("About to type: {:?}", text);
                        let start = std::time::Instant::now();
                        if let Err(err) = enigo.text(text) {
                            log::error!("Failed to type text: {err}");
                        } else {
                            log::info!("Typed {:?} in {:?}", text, start.elapsed());
                        }
                    }
                    chunk.commit_all();
                }
            }
            Err(_) => {
                thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

pub(super) fn reset_buffer() {
    if let Ok(mut state) = transcription_buffer().lock() {
        state.clear();
    }
}

fn queue_text(text: String) {
    log::info!("Queueing for typing: {:?}", text);
    if let Ok(mut prod) = typing_producer().lock() {
        match prod.push(text.clone()) {
            Ok(_) => log::info!("Successfully queued text"),
            Err(_) => log::error!("Typing queue full, text dropped: {:?}", text),
        }
    } else {
        log::error!("Failed to lock typing producer");
    }
}

pub(super) fn append_and_type(text: String) {
    if text.is_empty() {
        return;
    }

    if let Ok(mut state) = transcription_buffer().lock() {
        state.push_str(&text);
        queue_text(text);
    }
}

fn current_transcription() -> Option<String> {
    transcription_buffer()
        .lock()
        .ok()
        .map(|state| state.clone())
}

fn suffix_after_prefix(prefix: &str, full: &str) -> Option<String> {
    let prefix_trimmed = prefix.trim_end();
    let full_trimmed = full.trim_end();
    if full_trimmed.len() < prefix_trimmed.len() {
        return None;
    }
    if !full_trimmed.starts_with(prefix_trimmed) {
        return None;
    }
    let suffix = &full_trimmed[prefix_trimmed.len()..];
    if suffix.trim().is_empty() {
        return None;
    }
    Some(suffix.to_string())
}

pub(super) fn append_offline_suffix(text: String) -> bool {
    let current = match current_transcription() {
        Some(value) => value,
        None => return false,
    };
    if current.trim().is_empty() {
        append_and_type(text);
        return true;
    }
    if let Some(suffix) = suffix_after_prefix(&current, &text) {
        append_and_type(suffix);
        return true;
    }
    false
}

pub(super) fn apply_patch_for_typing(patch: TranscriptionPatch) {
    if !patch.stable || patch.text.is_empty() {
        return;
    }
    log::info!("Enigo patch: start={}, text={:?}", patch.start, patch.text);
    append_and_type(patch.text);
}
