use std::sync::{Mutex, OnceLock};

use crate::engine::SpeechEngine;
use crate::recording::Recorder;
use crate::streaming::TranscriptionPatch;
use enigo::{Enigo, Keyboard, Settings};
use serde_json::json;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{async_runtime, AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutEvent, ShortcutState,
};

const SHORTCUT_STORE: &str = "settings.json";
const SHORTCUT_STORE_KEY: &str = "record_shortcut";
const MENU_ITEM_QUIT: &str = "quit";
const MENU_ITEM_VIEW_LOGS: &str = "view_logs";

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
    let quit = MenuItem::with_id(app, MENU_ITEM_QUIT, "Quit", true, None::<&str>)?;
    let view_logs = MenuItem::with_id(
        app,
        MENU_ITEM_VIEW_LOGS,
        "View Log File",
        true,
        None::<&str>,
    )?;
    let menu = Menu::with_items(app, &[&view_logs, &quit])?;

    let mut tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| match event.id.as_ref() {
            MENU_ITEM_QUIT => {
                log::info!("Quit menu item clicked");
                app.exit(0);
            }
            MENU_ITEM_VIEW_LOGS => {
                log::info!("View Log File menu item clicked");
                if let Err(e) = open_log_file(app) {
                    log::error!("Failed to open log file: {}", e);
                    let _ = app
                        .dialog()
                        .message(format!("Failed to open log file: {}", e))
                        .kind(MessageDialogKind::Error)
                        .title("Warning")
                        .blocking_show();
                }
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

fn open_log_file(app: &AppHandle) -> Result<(), String> {
    let log_dir = app.path().app_log_dir().map_err(|e| e.to_string())?;

    if !log_dir.exists() {
        return Err("Log directory does not exist".to_string());
    }

    let log_file = std::fs::read_dir(&log_dir)
        .ok()
        .and_then(|entries| {
            let mut logs: Vec<_> = entries
                .flatten()
                .filter(|entry| {
                    entry
                        .path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| ext.eq_ignore_ascii_case("log"))
                        .unwrap_or(false)
                })
                .collect();

            logs.sort_by_key(|entry| {
                entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            });
            logs.last().map(|entry| entry.path())
        })
        .unwrap_or(log_dir);

    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";

    std::process::Command::new(cmd)
        .arg(log_file)
        .spawn()
        .map_err(|e| e.to_string())?;

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

fn should_insert_space(buffer: &str, new_text: &str) -> bool {
    if buffer.is_empty() || new_text.is_empty() {
        return false;
    }
    let last_char = buffer.chars().last();
    let first_char = new_text.chars().next();

    match (last_char, first_char) {
        (Some(last), Some(first)) => !last.is_whitespace() && !first.is_whitespace(),
        _ => false,
    }
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

fn append_and_type(text: String) {
    if text.trim().is_empty() {
        return;
    }

    let mut state = match transcription_buffer().lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };

    let mut output = String::new();
    if should_insert_space(&state, &text) {
        state.push(' ');
        output.push(' ');
    }
    state.push_str(&text);
    output.push_str(&text);
    if !output.is_empty() {
        type_text(output);
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

fn append_offline_suffix(text: String) -> bool {
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

fn apply_patch_for_typing(patch: TranscriptionPatch) {
    // Only type stable (committed) text
    if !patch.stable || patch.text.is_empty() {
        return;
    }

    let mut state = match transcription_buffer().lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };

    // Stable patches from the streaming pipeline are append-only (start=0, end=0)
    // We simply append and type the new committed text
    let text_to_type = patch.text.clone();

    // Add space if needed between existing buffer and new text
    let mut output = String::new();
    if should_insert_space(&state, &text_to_type) {
        state.push(' ');
        output.push(' ');
    }

    state.push_str(&text_to_type);
    output.push_str(&text_to_type);

    if !output.is_empty() {
        type_text(output);
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

fn start_recording_async(app: &AppHandle) {
    let recorder = Recorder::global();
    let app_clone = app.clone();

    if let Ok(mut state) = transcription_buffer().lock() {
        state.clear();
    }

    let engine = app.state::<SpeechEngine>();
    engine.reset_model_state();

    async_runtime::spawn(async move {
        log::info!("Global shortcut pressed; starting recording");

        let streaming = crate::settings::get_settings(&app_clone).streaming_enabled;
        // let vad_model = ... // Removed
        let (streaming_tx, streaming_rx) = if streaming {
            let (tx, rx) = std::sync::mpsc::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        match recorder.start(streaming_tx) {
            Ok(()) => {
                log::info!("Recording started (shortcut held)");

                log::debug!("Global shortcut streaming mode: {}", streaming);
                if let Some(rx) = streaming_rx {
                    let app_for_loop = app_clone.clone();
                    // We need to resolve model path again?
                    // `vad_model` here is `VadSession` loaded?
                    // No, `crate::asr::get_or_init_vad_model` returns `PathBuf`.

                    // But `pipeline.start` expects `model_path` for ASR, not VAD.
                    // The old code used separate VAD model?
                    // `get_or_init_vad_model` returns the path to clean silero vad?
                    // My new pipeline does NOT use silero vad yet?
                    // My arch.md said `streaming.rs` uses `VadSegmenter`.
                    // But `pipeline.rs` I wrote DOES NOT use VAD yet.
                    // It just decodes everything. The feedback said "Separate decode clock from VAD".
                    // It said "VAD only controls when to display and when to finalize".

                    // So I should pass the ASR model path, NOT the VAD model path.
                    // `get_or_init_vad_model` was used in `start_recording` legacy.
                    // Desktop shortcut uses `vad_model`.
                    // We should use the MAIN ASR model.

                    // We check if model exists using resolve_model_dir to ensure we don't start stream if no model.
                    let model_root = crate::asr::default_model_root(&app_clone);
                    if crate::asr::resolve_model_dir(&model_root).is_ok() {
                        let pipeline =
                            std::sync::Arc::new(crate::streaming::StreamingPipeline::new());
                        let engine = app_for_loop.state::<SpeechEngine>();

                        pipeline.start(rx, engine.get_model(), move |patch| {
                            if !patch.stable && patch.text.is_empty() {
                                return;
                            }

                            // Desktop logic for typing (Enigo)
                            // We ONLY process stable (committed) patches for typing.
                            // We do NOT emit to frontend here (keeping isolation).
                            if patch.stable {
                                Recorder::global().mark_streamed_any();
                                log::info!("Shortcut transcription update: '{}'", patch.text);
                                apply_patch_for_typing(patch);
                            }
                        });
                    }
                }
            }
            Err(err) => log::error!("Failed to start recording: {err}"),
        }
    });
}

// run_vad_streaming_loop removed

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
                    match engine.transcribe_samples(samples, false) {
                        Ok(text) if !text.is_empty() => {
                            log::info!("Shortcut transcribed (final): '{}'", text);
                            append_and_type(text);
                        }
                        Ok(_) => {
                            log::info!("Shortcut transcribed (final) was empty");
                        }
                        Err(e) => log::error!("Transcription error: {e}"),
                    }
                } else {
                    let engine = app_handle.state::<SpeechEngine>();
                    match engine.transcribe_samples(samples, false) {
                        Ok(text) if !text.is_empty() => {
                            let appended = append_offline_suffix(text);
                            if appended {
                                log::info!("Shortcut finalization appended offline suffix");
                            } else {
                                log::debug!("Shortcut finalization skipped: no appendable suffix");
                            }
                        }
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
        log::trace!(
            "Shortcut event: {:?}, state: {:?}",
            event.state(),
            event.id()
        );
        match event.state() {
            ShortcutState::Pressed if !recorder.is_recording() => {
                log::info!("Shortcut PRESSED -> Starting recording");
                start_recording_async(app);
            }
            ShortcutState::Released if recorder.is_recording() => {
                log::info!("Shortcut RELEASED -> Stopping recording");
                stop_recording_async(app);
            }
            _ => log::trace!(
                "Shortcut event ignored in state {:?} (is_recording={})",
                event.state(),
                recorder.is_recording()
            ),
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
