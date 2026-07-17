use std::sync::{mpsc, Mutex, OnceLock};
use std::thread;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};

use crate::errors::UserFacing;

static TRANSCRIPTION_BUFFER: OnceLock<Mutex<String>> = OnceLock::new();
static TYPING_SENDER: OnceLock<Result<mpsc::Sender<TypingRequest>, String>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalDelivery {
    None,
    Append(String),
    Replace { previous_chars: usize, text: String },
}

#[derive(thiserror::Error, Debug)]
pub enum TypingError {
    #[error("typing state is unavailable")]
    State,
    #[error("typing worker is unavailable: {0}")]
    Worker(String),
    #[error("virtual keyboard failed: {0}")]
    Keyboard(String),
}

impl UserFacing for TypingError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::Keyboard(_) => "Could not type into the focused app. Check input permissions.",
            Self::State | Self::Worker(_) => "Text output is unavailable. Please restart the app.",
        }
    }
}

struct TypingRequest {
    delivery: FinalDelivery,
    completion: mpsc::Sender<Result<(), String>>,
}

fn transcription_buffer() -> &'static Mutex<String> {
    TRANSCRIPTION_BUFFER.get_or_init(|| Mutex::new(String::new()))
}

fn typing_sender() -> Result<&'static mpsc::Sender<TypingRequest>, TypingError> {
    TYPING_SENDER
        .get_or_init(|| {
            let (sender, receiver) = mpsc::channel();
            thread::Builder::new()
                .name("desktop-typing".to_string())
                .spawn(move || typing_worker(receiver))
                .map(|_| sender)
                .map_err(|error| format!("start desktop typing thread: {error}"))
        })
        .as_ref()
        .map_err(|error| TypingError::Worker(error.clone()))
}

/// The receiver owns `Enigo` for its entire lifetime. Every request carries an
/// acknowledgement, so transcript state advances only after the keyboard API
/// reports success.
fn typing_worker(receiver: mpsc::Receiver<TypingRequest>) {
    let mut keyboard = Enigo::new(&Settings::default()).map_err(|error| error.to_string());
    while let Ok(request) = receiver.recv() {
        let result = match &mut keyboard {
            Ok(keyboard) => perform_delivery(keyboard, &request.delivery),
            Err(error) => Err(error.clone()),
        };
        let _ = request.completion.send(result);
    }
}

fn perform_delivery(keyboard: &mut Enigo, delivery: &FinalDelivery) -> Result<(), String> {
    match delivery {
        FinalDelivery::None => Ok(()),
        FinalDelivery::Append(text) => keyboard.text(text).map_err(|error| error.to_string()),
        FinalDelivery::Replace {
            previous_chars,
            text,
        } => {
            for _ in 0..*previous_chars {
                keyboard
                    .key(Key::Backspace, Direction::Click)
                    .map_err(|error| error.to_string())?;
            }
            if text.is_empty() {
                return Ok(());
            }
            keyboard.text(text).map_err(|error| error.to_string())
        }
    }
}

fn submit(delivery: FinalDelivery) -> Result<(), TypingError> {
    if delivery == FinalDelivery::None {
        return Ok(());
    }
    let (completion, result) = mpsc::channel();
    typing_sender()?
        .send(TypingRequest {
            delivery,
            completion,
        })
        .map_err(|_| TypingError::Worker("request channel closed".to_string()))?;
    result
        .recv()
        .map_err(|_| TypingError::Worker("acknowledgement channel closed".to_string()))?
        .map_err(TypingError::Keyboard)
}

pub fn plan_final_delivery(current: &str, final_text: &str) -> FinalDelivery {
    if current == final_text {
        return FinalDelivery::None;
    }
    if let Some(suffix) = final_text.strip_prefix(current) {
        return FinalDelivery::Append(suffix.to_string());
    }
    FinalDelivery::Replace {
        previous_chars: current.chars().count(),
        text: final_text.to_string(),
    }
}

pub(super) fn reset_buffer() -> Result<(), TypingError> {
    transcription_buffer()
        .lock()
        .map_err(|_| TypingError::State)?
        .clear();
    Ok(())
}

/// Advances the transcript buffer to `target` only after the submitter
/// acknowledges the keyboard operation.
fn deliver<E>(
    current: &mut String,
    target: String,
    submit: impl FnOnce(FinalDelivery) -> Result<(), E>,
) -> Result<(), E> {
    submit(plan_final_delivery(current, &target))?;
    *current = target;
    Ok(())
}

fn append<E>(
    current: &mut String,
    text: String,
    submit: impl FnOnce(FinalDelivery) -> Result<(), E>,
) -> Result<(), E> {
    submit(FinalDelivery::Append(text.clone()))?;
    current.push_str(&text);
    Ok(())
}

#[doc(hidden)]
pub fn deliver_for_tests(
    current: &mut String,
    target: String,
    submit: impl FnOnce(FinalDelivery) -> Result<(), String>,
) -> Result<(), String> {
    deliver(current, target, submit)
}

#[doc(hidden)]
pub fn append_for_tests(
    current: &mut String,
    text: String,
    submit: impl FnOnce(FinalDelivery) -> Result<(), String>,
) -> Result<(), String> {
    append(current, text, submit)
}

pub(super) fn append_streaming_text(text: String) -> Result<(), TypingError> {
    let mut current = transcription_buffer()
        .lock()
        .map_err(|_| TypingError::State)?;
    append(&mut current, text, submit)
}

pub(super) fn deliver_final_text(text: String) -> Result<(), TypingError> {
    let mut current = transcription_buffer()
        .lock()
        .map_err(|_| TypingError::State)?;
    deliver(&mut current, text, submit)
}
