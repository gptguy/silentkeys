use tauri::{AppHandle, Emitter, Manager, State};

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
    use crate::settings::{save_settings, Settings};

    let p = std::path::PathBuf::from(&path);
    if !p.exists() || !p.is_dir() {
        return Err("Path does not exist or is not a directory".to_string());
    }

    let settings = Settings {
        model_path: Some(path),
        ..crate::settings::get_settings(&app)
    };
    save_settings(&app, &settings)
}

#[tauri::command]
pub fn get_use_streaming(app: AppHandle) -> bool {
    crate::settings::get_settings(&app).streaming_enabled
}

#[tauri::command]
pub fn set_use_streaming(app: AppHandle, enabled: bool) -> Result<(), String> {
    use crate::settings::{get_settings, save_settings};
    let mut settings = get_settings(&app);
    log::info!("Command set_use_streaming invoked: enabled={}", enabled);
    settings.streaming_enabled = enabled;
    save_settings(&app, &settings)
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
pub fn start_recording(app: AppHandle) -> Result<(), String> {
    log::info!("Tauri command start_recording invoked");
    Recorder::global()
        .start()
        .map_err(|e| e.user_message().to_string())?;

    let streaming = crate::settings::get_settings(&app).streaming_enabled;
    log::info!(
        "Command start_recording: Streaming mode enabled? {}",
        streaming
    );
    if streaming {
        let app_handle = app.clone();
        std::thread::spawn(move || {
            let model = crate::asr::get_or_init_vad_model(&app_handle);
            crate::streaming::run_streaming(model, |samples| {
                let state = app_handle.state::<SpeechEngine>();
                if let Ok(text) = state.transcribe_samples(samples) {
                    log::info!("UI streaming: emitting transcription (len={})", text.len());
                    let _ = app_handle.emit("transcription_update", text);
                }
            });
        });
    }
    Ok(())
}

#[tauri::command]
pub fn stop_recording(app: AppHandle, state: State<'_, SpeechEngine>) -> Result<String, String> {
    log::info!("Tauri command stop_recording invoked");
    let samples = Recorder::global()
        .stop()
        .map_err(|e| e.user_message().to_string())?;

    let streaming = crate::settings::get_settings(&app).streaming_enabled;
    if streaming {
        log::info!(
            "Streaming mode: {} samples remaining (drained during recording)",
            samples.len()
        );
        return Ok(String::new());
    }

    state.transcribe_samples(samples)
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
