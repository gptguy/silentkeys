use tauri::{AppHandle, Emitter, State};

use crate::asr::DownloadProgress;
#[cfg(desktop)]
use crate::desktop;
use crate::engine::SpeechEngine;
use crate::recording::Recorder;

#[tauri::command]
pub fn is_model_ready(state: State<'_, SpeechEngine>) -> bool {
    state.is_ready()
}

#[tauri::command]
pub fn model_download_progress(state: State<'_, SpeechEngine>) -> Option<DownloadProgress> {
    state.download_progress()
}

#[tauri::command]
pub fn get_model_path(app: AppHandle) -> String {
    crate::asr::default_model_root(&app)
        .to_string_lossy()
        .to_string()
}

#[tauri::command]
pub fn set_model_path(app: AppHandle, path: String) -> Result<(), String> {
    let p = std::path::PathBuf::from(&path);
    if !p.exists() || !p.is_dir() {
        return Err("Path does not exist or is not a directory".to_string());
    }

    let mut settings = crate::settings::get_settings(&app);
    settings.model_path = Some(path);
    crate::settings::save_settings(&app, &settings)
}

#[tauri::command]
pub fn get_use_streaming(app: AppHandle) -> bool {
    crate::settings::get_settings(&app).streaming_enabled
}

#[tauri::command]
pub fn set_use_streaming(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = crate::settings::get_settings(&app);
    settings.streaming_enabled = enabled;
    crate::settings::save_settings(&app, &settings)
}

#[tauri::command]
pub async fn pick_model_folder(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let result =
        tauri::async_runtime::spawn_blocking(move || app.dialog().file().blocking_pick_folder())
            .await
            .map_err(|e| format!("Dialog task failed: {e}"))?;

    Ok(result.map(|p| p.to_string()))
}

#[tauri::command]
pub fn retry_model_download(state: State<'_, SpeechEngine>) -> Result<(), String> {
    state.retry_model_download()
}

#[tauri::command]
pub fn start_recording(app: AppHandle, state: State<'_, SpeechEngine>) -> Result<(), String> {
    log::info!("Tauri command start_recording invoked");
    state.reset_model_state();

    let streaming = crate::settings::get_settings(&app).streaming_enabled;
    let streaming_tx = if streaming {
        Some(state.start_streaming()?)
    } else {
        None
    };

    Recorder::global()
        .start(streaming_tx)
        .map_err(|e| e.user_message().to_string())?;

    log::info!("Recorder started (Streaming={})", streaming);
    Ok(())
}

#[tauri::command]
pub fn stop_recording(app: AppHandle, state: State<'_, SpeechEngine>) -> Result<String, String> {
    log::info!("Tauri command stop_recording invoked");
    let samples = Recorder::global()
        .stop()
        .map_err(|e| e.user_message().to_string())?;

    if crate::settings::get_settings(&app).streaming_enabled {
        if !Recorder::global().streamed_any() {
            let text = state.transcribe_samples(samples, false)?;
            if !text.trim().is_empty() {
                let _ = app.emit(
                    "transcription_update",
                    crate::streaming::TranscriptionPatch {
                        start: 0,
                        end: usize::MAX,
                        text,
                        stable: true,
                    },
                );
            }
        }
        return Ok(String::new());
    }

    state.transcribe_samples(samples, false)
}

#[cfg(desktop)]
#[tauri::command]
pub fn update_record_shortcut(app: AppHandle, shortcut: String) -> Result<String, String> {
    desktop::update_record_shortcut(app, shortcut)
}

#[cfg(desktop)]
#[tauri::command]
pub fn get_record_shortcut(app: AppHandle) -> Option<String> {
    desktop::get_record_shortcut(app)
}

#[cfg(desktop)]
#[tauri::command]
pub fn default_record_shortcut() -> String {
    desktop::default_record_shortcut()
}

#[tauri::command]
pub fn reset_settings(app: AppHandle) -> Result<(), String> {
    use crate::settings::{save_settings, Settings};

    let settings = Settings::default();
    save_settings(&app, &settings)?;

    #[cfg(desktop)]
    {
        let default = desktop::default_record_shortcut();
        let _ = desktop::update_record_shortcut(app, default);
    }

    Ok(())
}
