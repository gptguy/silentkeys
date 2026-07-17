use std::sync::{Mutex, OnceLock};

use tauri::{async_runtime, AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutEvent, ShortcutState,
};

use super::typing::{append_streaming_text, deliver_final_text, reset_buffer};
use crate::engine::SpeechEngine;
use crate::errors::UserFacing;
use crate::recording::{RecordingError, RecordingReservation};

const SHORTCUT_STORE: &str = "settings.json";
const SHORTCUT_STORE_KEY: &str = "record_shortcut";

static ACTIVE_SHORTCUT: OnceLock<Mutex<Option<Shortcut>>> = OnceLock::new();

fn active_shortcut() -> &'static Mutex<Option<Shortcut>> {
    ACTIVE_SHORTCUT.get_or_init(|| Mutex::new(None))
}

#[doc(hidden)]
pub fn default_shortcut() -> Shortcut {
    Shortcut::new(Some(Modifiers::ALT), Code::KeyZ)
}

#[doc(hidden)]
pub fn parse_shortcut_str(s: &str) -> Result<Shortcut, String> {
    s.parse::<Shortcut>()
        .map_err(|e| format!("Invalid shortcut: {e}"))
}

fn persist_shortcut(app: &AppHandle, shortcut: &Shortcut) -> Result<(), String> {
    use tauri_plugin_store::StoreExt;
    let store = app
        .store(SHORTCUT_STORE)
        .map_err(|e| format!("Could not open settings store: {e}"))?;
    store.set(
        SHORTCUT_STORE_KEY.to_string(),
        serde_json::json!(shortcut.into_string()),
    );
    store
        .save()
        .map_err(|e| format!("Could not persist shortcut: {e}"))
}

fn load_persisted_shortcut(app: &AppHandle) -> Option<String> {
    use tauri_plugin_store::StoreExt;
    app.store(SHORTCUT_STORE)
        .ok()
        .and_then(|store| store.get(SHORTCUT_STORE_KEY))
        .and_then(|value| value.as_str().map(ToString::to_string))
}

/// Logs the full error and shows the user-facing message in the app window,
/// since shortcut-driven dictation has no other visible failure surface.
fn report_failure(app: &AppHandle, context: &str, error: &(impl UserFacing + std::fmt::Display)) {
    log::error!("{context}: {error}");
    if let Err(emit_error) = app.emit("dictation_error", error.user_message()) {
        log::warn!("Could not report dictation error to UI: {emit_error}");
    }
}

fn make_handler() -> impl Fn(&AppHandle, &Shortcut, ShortcutEvent) + Send + Sync {
    move |app: &AppHandle, _, event| {
        let engine = app.state::<SpeechEngine>();
        match event.state() {
            ShortcutState::Pressed if !engine.is_dictating() => {
                log::info!("Shortcut PRESSED -> Starting recording");
                match engine.reserve_dictation() {
                    Ok(reservation) => start_recording_async(app, reservation),
                    Err(err) => report_failure(app, "Failed to reserve recording", &err),
                }
            }
            ShortcutState::Released if engine.is_dictating() => {
                log::info!("Shortcut RELEASED -> Stopping recording");
                stop_recording_async(app);
            }
            _ => {}
        }
    }
}

fn start_recording_async(app: &AppHandle, reservation: RecordingReservation) {
    let app = app.clone();

    async_runtime::spawn_blocking(move || {
        if let Err(error) = reset_buffer() {
            report_failure(&app, "Failed to reset typing state", &error);
            return;
        }
        let engine = app.state::<SpeechEngine>();
        let result = engine.start_dictation(reservation, |update| {
            if let crate::streaming::TranscriptionUpdate::Append(text) = update {
                append_streaming_text(text).map_err(|error| error.to_string())?;
            }
            Ok(())
        });
        if let Err(err) = result {
            report_failure(&app, "Failed to start recording", &err);
        }
    });
}

fn stop_recording_async(app: &AppHandle) {
    let report_app = app.clone();
    let worker_app = app.clone();
    let result = std::thread::Builder::new()
        .name("shortcut-stop".to_string())
        .spawn(move || {
            let engine = worker_app.state::<SpeechEngine>();
            let result =
                engine.finish_dictation(|text| deliver_final_text(text).map_err(|e| e.to_string()));
            if let Err(err) = result {
                report_failure(&worker_app, "Failed to finish dictation", &err);
            }
        });
    if let Err(error) = result {
        report_failure(
            &report_app,
            "Failed to start shortcut stop worker",
            &RecordingError::ThreadStart(error),
        );
    }
}

fn register_record_shortcut(app: &AppHandle, shortcut: Shortcut) -> Result<String, String> {
    let mut active = active_shortcut().lock().map_err(|e| e.to_string())?;
    if *active == Some(shortcut) {
        persist_shortcut(app, &shortcut)?;
        return Ok(shortcut.into_string());
    }

    app.global_shortcut()
        .on_shortcut(shortcut, make_handler())
        .map_err(|e| e.to_string())?;
    let previous = *active;
    if let Some(previous) = previous {
        if let Err(error) = app.global_shortcut().unregister(previous) {
            let _ = app.global_shortcut().unregister(shortcut);
            return Err(error.to_string());
        }
    }
    if let Err(error) = persist_shortcut(app, &shortcut) {
        let _ = app.global_shortcut().unregister(shortcut);
        if let Some(previous) = previous {
            if let Err(rollback_error) = app.global_shortcut().on_shortcut(previous, make_handler())
            {
                return Err(format!(
                    "{error}; shortcut rollback failed: {rollback_error}"
                ));
            }
        }
        return Err(error);
    }
    let s = shortcut.into_string();
    *active = Some(shortcut);
    Ok(s)
}

pub fn update_record_shortcut(app: AppHandle, s: String) -> Result<String, String> {
    register_record_shortcut(&app, parse_shortcut_str(&s)?)
}

pub fn get_record_shortcut(app: AppHandle) -> Option<String> {
    active_shortcut()
        .lock()
        .ok()
        .and_then(|a| a.as_ref().map(|s| s.into_string()))
        .or_else(|| load_persisted_shortcut(&app))
}

pub fn default_record_shortcut() -> String {
    default_shortcut().into_string()
}

fn resolve_shortcut(app: &AppHandle) -> Shortcut {
    load_persisted_shortcut(app)
        .and_then(|s| parse_shortcut_str(&s).ok())
        .unwrap_or_else(default_shortcut)
}

#[doc(hidden)]
#[cfg(desktop)]
pub(super) fn init_shortcuts(app: &AppHandle) -> tauri::Result<()> {
    app.plugin(tauri_plugin_global_shortcut::Builder::new().build())?;
    let shortcut = resolve_shortcut(app);
    if let Err(e) = register_record_shortcut(app, shortcut) {
        log::warn!("Global shortcut failed: {e}");
    }
    let _ = app
        .global_shortcut()
        .on_shortcut(Shortcut::new(Some(Modifiers::FN), Code::Fn), make_handler());
    Ok(())
}
