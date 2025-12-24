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

    let streaming = crate::settings::get_settings(&app).streaming_enabled;
    let engine = app.state::<SpeechEngine>();
    engine.reset_model_state();

    let streaming_tx = if streaming {
        let (tx, rx) = std::sync::mpsc::channel();
        let app_handle = app.clone();

        // We get the model path from settings or default
        let model_root = crate::asr::default_model_root(&app);
        let model_path = crate::asr::resolve_model_dir(&model_root).map_err(|e| e.to_string())?;

        log::info!("Starting streaming pipeline with model: {:?}", model_path);

        let pipeline = std::sync::Arc::new(crate::streaming::StreamingPipeline::new());
        let emit_handle = app_handle.clone();
        let engine_state = app_handle.state::<SpeechEngine>().clone();

        pipeline.start(rx, engine_state.get_model(), move |patch| {
            if !patch.text.is_empty() {
                Recorder::global().mark_streamed_any();
            }

            if !patch.stable {
                log::debug!("Sending UI event, len={}", patch.text.len());
                if let Err(e) = emit_handle.emit("transcription_update", patch) {
                    log::error!("Failed to emit transcription_update: {}", e);
                }
            }
        });

        Some(tx)
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

    let streaming = crate::settings::get_settings(&app).streaming_enabled;
    if streaming {
        log::info!(
            "Streaming mode: {} samples remaining (drained during recording)",
            samples.len()
        );
        if !Recorder::global().streamed_any() {
            let text = state.transcribe_samples(samples, false)?;
            log::debug!(
                "Stop recording (streaming): leftover text len={}",
                text.len()
            );
            if !text.trim().is_empty() {
                let patch = crate::streaming::TranscriptionPatch {
                    start: 0,
                    end: u32::MAX as usize,
                    text,
                    stable: true,
                };
                let _ = app.emit("transcription_update", patch);
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
