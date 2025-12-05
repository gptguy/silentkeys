//! Desktop-specific functionality: tray icon, global shortcuts, and recording.

use std::sync::{Mutex, OnceLock};

use enigo::{Enigo, Keyboard, Settings};
use serde_json::json;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{async_runtime, AppHandle, Manager};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutEvent, ShortcutState,
};

use crate::engine::SpeechEngine;
use crate::recording::Recorder;

// ============================================================================
// Constants
// ============================================================================

const SHORTCUT_STORE: &str = "settings.json";
const SHORTCUT_STORE_KEY: &str = "record_shortcut";

// ============================================================================
// State
// ============================================================================

static ACTIVE_SHORTCUT: OnceLock<Mutex<Option<Shortcut>>> = OnceLock::new();

fn active_shortcut() -> &'static Mutex<Option<Shortcut>> {
    ACTIVE_SHORTCUT.get_or_init(|| Mutex::new(None))
}

// ============================================================================
// Setup
// ============================================================================

#[cfg(desktop)]
pub fn setup_desktop(app: &mut tauri::App) -> tauri::Result<()> {
    let handle = app.handle();
    init_tray(handle)?;
    init_shortcuts(handle)?;
    Ok(())
}

// ============================================================================
// Tray Icon
// ============================================================================

#[cfg(desktop)]
fn init_tray(app: &AppHandle) -> tauri::Result<()> {
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&quit])?;

    let mut tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "quit" => {
                log::info!("Quit menu item clicked");
                app.exit(0);
            }
            _ => log::debug!("Unhandled menu item: {:?}", event.id),
        });

    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    } else {
        log::warn!("Default window icon missing; using system tray default.");
    }

    tray.build(app)?;
    Ok(())
}

// ============================================================================
// Shortcuts
// ============================================================================

fn default_shortcut() -> Shortcut {
    Shortcut::new(Some(Modifiers::ALT), Code::KeyZ)
}

fn parse_shortcut_str(s: &str) -> Result<Shortcut, String> {
    s.parse::<Shortcut>()
        .map_err(|e| format!("Invalid shortcut: {e}"))
}

fn type_text(text: String) {
    match Enigo::new(&Settings::default()) {
        Ok(mut enigo) => {
            if let Err(err) = enigo.text(&text) {
                log::error!("Failed to type transcription: {err}");
            }
        }
        Err(err) => log::error!("Could not initialize Enigo for typing: {err}"),
    }
}

fn persist_shortcut(app: &AppHandle, shortcut: &Shortcut) -> Result<(), String> {
    use tauri_plugin_store::StoreExt;
    let store = app
        .store(SHORTCUT_STORE)
        .map_err(|e| format!("Could not open settings store: {e}"))?;
    store.set(
        SHORTCUT_STORE_KEY.to_string(),
        json!(shortcut.into_string()),
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

fn start_recording_async() {
    let recorder = Recorder::global();
    async_runtime::spawn(async move {
        log::info!("Global shortcut pressed; starting recording");
        match recorder.start() {
            Ok(()) => log::info!("Recording started (shortcut held)"),
            Err(err) => log::error!("Failed to start recording: {err}"),
        }
    });
}

fn stop_recording_async(app: &AppHandle) {
    let recorder = Recorder::global();
    let app_handle = app.clone();
    async_runtime::spawn(async move {
        log::info!("Global shortcut released; stopping recording");
        let result = recorder.stop();

        match result {
            Ok(samples) => {
                let app_handle = app_handle.clone();
                match async_runtime::spawn_blocking(move || {
                    let state = app_handle.state::<SpeechEngine>();
                    state.transcribe_samples(samples)
                })
                .await
                {
                    Ok(Ok(text)) => {
                        async_runtime::spawn_blocking(move || type_text(text));
                    }
                    Ok(Err(err)) => log::error!("Transcription failed: {err}"),
                    Err(err) => log::error!("Transcription task join error: {err}"),
                };
            }
            Err(err) => log::error!("Failed to stop recording: {err}"),
        }
    });
}

fn make_handler() -> impl Fn(&AppHandle, &Shortcut, ShortcutEvent) + Send + Sync {
    move |app: &AppHandle, _shortcut: &Shortcut, event| {
        let recorder = Recorder::global();
        match event.state() {
            ShortcutState::Pressed if !recorder.is_recording() => start_recording_async(),
            ShortcutState::Released if recorder.is_recording() => stop_recording_async(app),
            _ => log::debug!("Shortcut event ignored in state {:?}", event.state()),
        }
    }
}

fn register_record_shortcut(app: &AppHandle, shortcut: Shortcut) -> Result<String, String> {
    let mut active = active_shortcut()
        .lock()
        .map_err(|_| "Shortcut state lock poisoned".to_string())?;

    if let Some(prev) = *active {
        let _ = app.global_shortcut().unregister(prev);
    }

    app.global_shortcut()
        .on_shortcut(shortcut, make_handler())
        .map_err(|e| format!("Failed to register shortcut: {e}"))?;

    persist_shortcut(app, &shortcut)?;
    let shortcut_string = shortcut.into_string();

    *active = parse_shortcut_str(&shortcut_string).ok();
    Ok(shortcut_string)
}

pub fn update_record_shortcut(app: AppHandle, shortcut: String) -> Result<String, String> {
    let shortcut = parse_shortcut_str(&shortcut)?;
    register_record_shortcut(&app, shortcut)
}

pub fn get_record_shortcut(app: AppHandle) -> Option<String> {
    if let Ok(active) = active_shortcut().lock() {
        if let Some(shortcut) = *active {
            return Some(shortcut.into_string());
        }
    }
    load_persisted_shortcut(&app)
}

pub fn default_record_shortcut() -> String {
    default_shortcut().into_string()
}

fn resolve_shortcut(app: &AppHandle) -> Shortcut {
    load_persisted_shortcut(app)
        .and_then(|s| parse_shortcut_str(&s).ok())
        .unwrap_or_else(default_shortcut)
}

#[cfg(desktop)]
fn init_shortcuts(app: &AppHandle) -> tauri::Result<()> {
    app.plugin(tauri_plugin_global_shortcut::Builder::new().build())?;

    let shortcut_to_use = resolve_shortcut(app);

    match register_record_shortcut(app, shortcut_to_use) {
        Ok(s) => log::info!("Global shortcut ready: {}", s),
        Err(err) => log::warn!("No global shortcut registered: {err}"),
    }

    // Try to register Fn key as an alternative shortcut (may not work on all platforms)
    let fn_shortcut = Shortcut::new(Some(Modifiers::FN), Code::Fn);
    if let Err(err) = app
        .global_shortcut()
        .on_shortcut(fn_shortcut, make_handler())
    {
        log::debug!("Fn key shortcut not supported: {err}");
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_shortcut_is_valid() {
        let shortcut = default_shortcut();
        let shortcut_str = shortcut.into_string();
        assert!(shortcut_str.contains("alt") || shortcut_str.contains("Alt"));
        assert!(shortcut_str.contains("KeyZ"));
    }

    #[test]
    fn parse_valid_shortcut() {
        let result = parse_shortcut_str("Alt+KeyZ");
        assert!(result.is_ok());
    }

    #[test]
    fn parse_shortcut_with_modifier() {
        let result = parse_shortcut_str("Control+Shift+KeyS");
        assert!(result.is_ok());
    }

    #[test]
    fn parse_invalid_shortcut() {
        let result = parse_shortcut_str("InvalidShortcut");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid"));
    }

    #[test]
    fn parse_empty_shortcut() {
        let result = parse_shortcut_str("");
        assert!(result.is_err());
    }

    #[test]
    fn default_record_shortcut_returns_string() {
        let shortcut = default_record_shortcut();
        assert!(!shortcut.is_empty());
        assert!(shortcut.contains("Alt") || shortcut.contains("Key"));
    }
}
