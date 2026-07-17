use tauri::{AppHandle, Emitter, State};

#[cfg(desktop)]
use crate::desktop;
use crate::engine::{EngineState, SpeechEngine};
use crate::errors::UserFacing;
use crate::updater::AppUpdateInfo;

fn user_error(err: impl UserFacing) -> String {
    err.user_message().to_string()
}

fn command_error(context: &str, err: impl UserFacing + std::fmt::Display) -> String {
    log::error!("{context}: {err}");
    user_error(err)
}

async fn run_blocking<T, F>(task: &'static str, work: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|error| {
            log::error!("{task} worker failed: {error}");
            format!("{task} could not complete. Please try again.")
        })?
}

#[tauri::command]
pub fn engine_state(state: State<'_, SpeechEngine>) -> EngineState {
    state.state()
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

    crate::settings::set_model_path(&app, path)
        .map_err(|error| command_error("Could not set model path", error))
}

#[tauri::command]
pub fn get_use_streaming(app: AppHandle) -> bool {
    crate::settings::get_settings(&app).streaming_enabled
}

#[tauri::command]
pub fn set_use_streaming(app: AppHandle, enabled: bool) -> Result<(), String> {
    crate::settings::set_streaming_enabled(&app, enabled)
        .map_err(|error| command_error("Could not set streaming preference", error))
}

#[tauri::command]
pub fn get_asr_language(app: AppHandle) -> String {
    crate::settings::get_settings(&app).asr_language
}

#[tauri::command]
pub fn get_asr_languages(state: State<'_, SpeechEngine>) -> Result<Vec<String>, String> {
    state
        .languages()
        .map_err(|error| command_error("Could not list speech languages", error))
}

#[tauri::command]
pub async fn set_asr_language(
    app: AppHandle,
    state: State<'_, SpeechEngine>,
    language: String,
) -> Result<(), String> {
    let engine = state.inner().clone();
    run_blocking("Speech language", move || {
        crate::settings::set_asr_language(&app, &engine, language)
            .map_err(|error| command_error("Could not change speech language", error))
    })
    .await
}

#[tauri::command]
pub async fn pick_model_folder(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let result = run_blocking("Dialog", move || {
        Ok(app.dialog().file().blocking_pick_folder())
    })
    .await?;

    Ok(result.map(|p| p.to_string()))
}

#[tauri::command]
pub async fn retry_model_download(state: State<'_, SpeechEngine>) -> Result<(), String> {
    let engine = state.inner().clone();
    run_blocking("Model download", move || {
        engine
            .retry_model_download()
            .map_err(|error| command_error("Model download failed", error))
    })
    .await
}

#[tauri::command]
pub async fn start_recording(app: AppHandle, state: State<'_, SpeechEngine>) -> Result<(), String> {
    let engine = state.inner().clone();
    run_blocking("Recording", move || start_recording_blocking(app, engine)).await
}

fn start_recording_blocking(app: AppHandle, state: SpeechEngine) -> Result<(), String> {
    let reservation = state
        .reserve_dictation()
        .map_err(|error| command_error("Could not reserve dictation", error))?;
    state
        .start_dictation(reservation, move |update| {
            app.emit("transcription_update", update)
                .map_err(|error| error.to_string())
        })
        .map_err(|error| command_error("Could not start dictation", error))
}

#[tauri::command]
pub async fn stop_recording(app: AppHandle, state: State<'_, SpeechEngine>) -> Result<(), String> {
    let engine = state.inner().clone();
    run_blocking("Recording", move || stop_recording_blocking(app, engine)).await
}

fn stop_recording_blocking(app: AppHandle, state: SpeechEngine) -> Result<(), String> {
    state
        .finish_dictation(move |text| {
            app.emit(
                "transcription_update",
                crate::streaming::TranscriptionUpdate::Replace(text),
            )
            .map_err(|error| error.to_string())
        })
        .map_err(|error| command_error("Could not finish dictation", error))
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
pub async fn reset_settings(app: AppHandle, state: State<'_, SpeechEngine>) -> Result<(), String> {
    let engine = state.inner().clone();
    run_blocking("Settings reset", move || {
        crate::settings::reset_settings(&app, &engine)
            .map_err(|error| command_error("Could not reset settings", error))
    })
    .await
}

#[tauri::command]
pub async fn check_for_app_update(app: AppHandle) -> Result<Option<AppUpdateInfo>, String> {
    crate::updater::check_for_update(app).await.map_err(|err| {
        log::warn!("Update check failed: {err}");
        user_error(err)
    })
}

#[tauri::command]
pub async fn install_app_update(app: AppHandle) -> Result<bool, String> {
    crate::updater::install_update(app).await.map_err(|err| {
        log::warn!("Update installation failed: {err}");
        user_error(err)
    })
}
