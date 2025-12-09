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

const SHORTCUT_STORE: &str = "settings.json";
const SHORTCUT_STORE_KEY: &str = "record_shortcut";

static ACTIVE_SHORTCUT: OnceLock<Mutex<Option<Shortcut>>> = OnceLock::new();
static TRANSCRIPTION_BUFFER: OnceLock<Mutex<String>> = OnceLock::new();

fn active_shortcut() -> &'static Mutex<Option<Shortcut>> {
    ACTIVE_SHORTCUT.get_or_init(|| Mutex::new(None))
}

fn transcription_buffer() -> &'static Mutex<String> {
    TRANSCRIPTION_BUFFER.get_or_init(|| Mutex::new(String::new()))
}

#[cfg(desktop)]
pub fn setup_desktop(app: &mut tauri::App) -> tauri::Result<()> {
    let handle = app.handle();
    init_tray(handle)?;
    init_shortcuts(handle)?;
    Ok(())
}

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

#[doc(hidden)]
pub fn default_shortcut() -> Shortcut {
    Shortcut::new(Some(Modifiers::ALT), Code::KeyZ)
}

#[doc(hidden)]
pub fn parse_shortcut_str(s: &str) -> Result<Shortcut, String> {
    s.parse::<Shortcut>()
        .map_err(|e| format!("Invalid shortcut: {e}"))
}

fn type_text(text: String) {
    match Enigo::new(&Settings::default()) {
        Ok(mut enigo) => {
            if let Err(err) = enigo.text(&text) {
                log::error!("Failed to type text: {err}");
            }
        }
        Err(err) => log::error!("Could not initialize Enigo for typing: {err}"),
    }
}

fn append_and_type(phrase_text: String) {
    if phrase_text.is_empty() {
        return;
    }

    let mut state = match transcription_buffer().lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };

    if !state.is_empty() {
        state.push(' ');
        type_text(" ".to_string());
    }
    state.push_str(&phrase_text);
    type_text(phrase_text);
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

fn start_recording_async(app: &AppHandle) {
    let recorder = Recorder::global();
    let app_clone = app.clone();

    if let Ok(mut state) = transcription_buffer().lock() {
        state.clear();
    }

    async_runtime::spawn(async move {
        log::info!("Global shortcut pressed; starting recording");
        match recorder.start() {
            Ok(()) => {
                log::info!("Recording started (shortcut held)");

                let streaming = crate::settings::get_settings(&app_clone).streaming_enabled;
                if streaming {
                    let app_for_loop = app_clone.clone();
                    std::thread::spawn(move || {
                        run_vad_streaming_loop(&app_for_loop);
                    });
                }
            }
            Err(err) => log::error!("Failed to start recording: {err}"),
        }
    });
}

fn run_vad_streaming_loop(app: &AppHandle) {
    let model = crate::asr::get_or_init_vad_model(app);

    crate::streaming::run_streaming(model, |samples| {
        let engine = app.state::<SpeechEngine>();
        match engine.transcribe_samples(samples) {
            Ok(text) if !text.is_empty() => append_and_type(text),
            Ok(_) => {}
            Err(e) => log::error!("Transcription error: {e}"),
        }
    });
}

fn stop_recording_async(app: &AppHandle) {
    let recorder = Recorder::global();
    let app_handle = app.clone();
    std::thread::spawn(move || {
        log::info!("Global shortcut released; stopping recording");
        match recorder.stop() {
            Ok(samples) => {
                let streaming = crate::settings::get_settings(&app_handle).streaming_enabled;
                if !streaming {
                    let engine = app_handle.state::<SpeechEngine>();
                    match engine.transcribe_samples(samples) {
                        Ok(text) if !text.is_empty() => append_and_type(text),
                        Ok(_) => {}
                        Err(e) => log::error!("Transcription error: {e}"),
                    }
                }
            }
            Err(err) => {
                log::error!("Failed to stop recording: {err}");
            }
        }
    });
}

fn make_handler() -> impl Fn(&AppHandle, &Shortcut, ShortcutEvent) + Send + Sync {
    move |app: &AppHandle, _shortcut: &Shortcut, event| {
        let recorder = Recorder::global();
        match event.state() {
            ShortcutState::Pressed if !recorder.is_recording() => start_recording_async(app),
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

#[doc(hidden)]
#[cfg(desktop)]
fn init_shortcuts(app: &AppHandle) -> tauri::Result<()> {
    app.plugin(tauri_plugin_global_shortcut::Builder::new().build())?;

    let shortcut_to_use = resolve_shortcut(app);

    match register_record_shortcut(app, shortcut_to_use) {
        Ok(s) => log::info!("Global shortcut ready: {}", s),
        Err(err) => log::warn!("No global shortcut registered: {err}"),
    }

    let fn_shortcut = Shortcut::new(Some(Modifiers::FN), Code::Fn);
    if let Err(err) = app
        .global_shortcut()
        .on_shortcut(fn_shortcut, make_handler())
    {
        log::debug!("Fn key shortcut not supported: {err}");
    }

    Ok(())
}
