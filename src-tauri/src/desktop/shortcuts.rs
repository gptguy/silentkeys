use std::sync::{mpsc::Receiver, Arc, Mutex, OnceLock};

use tauri::{async_runtime, AppHandle, Manager};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutEvent, ShortcutState,
};

use super::typing::{append_and_type, append_offline_suffix, apply_patch_for_typing, reset_buffer};
use crate::audio_processing::AudioFrame;
use crate::engine::SpeechEngine;
use crate::recording::Recorder;
use crate::streaming::StreamingPipeline;

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

fn make_handler() -> impl Fn(&AppHandle, &Shortcut, ShortcutEvent) + Send + Sync {
    move |app: &AppHandle, _, event| {
        let recorder = Recorder::global();
        match event.state() {
            ShortcutState::Pressed if !recorder.is_recording() => {
                log::info!("Shortcut PRESSED -> Starting recording");
                start_recording_async(app);
            }
            ShortcutState::Released if recorder.is_recording() => {
                log::info!("Shortcut RELEASED -> Stopping recording");
                stop_recording_async(app);
            }
            _ => {}
        }
    }
}

fn start_recording_async(app: &AppHandle) {
    let app = app.clone();
    reset_buffer();
    app.state::<SpeechEngine>().reset_model_state();

    async_runtime::spawn(async move {
        let streaming = crate::settings::get_settings(&app).streaming_enabled;
        let (tx, rx) = if streaming {
            let (tx, rx) = std::sync::mpsc::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        if let Err(e) = Recorder::global().start(tx) {
            log::error!("Failed to start recording: {e}");
        } else if let Some(rx) = rx {
            start_streaming_pipeline(&app, rx);
        }
    });
}

fn start_streaming_pipeline(app: &AppHandle, rx: Receiver<AudioFrame>) {
    let root = crate::asr::default_model_root(app);
    if crate::asr::resolve_model_dir(&root).is_err() {
        return;
    }

    Arc::new(StreamingPipeline::new()).start(
        rx,
        app.state::<SpeechEngine>().get_model(),
        move |p| {
            if !p.text.is_empty() {
                Recorder::global().mark_streamed_any();
                if p.stable {
                    apply_patch_for_typing(p.clone());
                }
            }
        },
    );
}

fn stop_recording_async(app: &AppHandle) {
    let app = app.clone();
    std::thread::spawn(move || {
        if let Ok(samples) = Recorder::global().stop() {
            let streaming = crate::settings::get_settings(&app).streaming_enabled;
            match app
                .state::<SpeechEngine>()
                .transcribe_samples(samples, false)
            {
                Ok(text) if !text.is_empty() => {
                    if streaming {
                        append_offline_suffix(text);
                    } else {
                        append_and_type(text);
                    }
                }
                Err(e) => log::error!("Transcription error: {e}"),
                _ => {}
            }
        }
    });
}

fn register_record_shortcut(app: &AppHandle, shortcut: Shortcut) -> Result<String, String> {
    let mut active = active_shortcut().lock().map_err(|e| e.to_string())?;
    if let Some(prev) = *active {
        let _ = app.global_shortcut().unregister(prev);
    }
    app.global_shortcut()
        .on_shortcut(shortcut, make_handler())
        .map_err(|e| e.to_string())?;
    persist_shortcut(app, &shortcut)?;
    let s = shortcut.into_string();
    *active = parse_shortcut_str(&s).ok();
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
