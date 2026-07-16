use tauri::{AppHandle, Emitter, State};

use crate::activity::{self, ActivityError, ActivityGuard, AppActivity};
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

fn reserve_settings_change() -> Result<ActivityGuard, String> {
    match activity::try_begin(AppActivity::Configuring) {
        Ok(guard) => Ok(guard),
        Err(ActivityError::Busy(AppActivity::Recording)) => {
            Err("Finish the current recording before changing settings.".to_string())
        }
        Err(ActivityError::Busy(AppActivity::Updating)) => {
            Err("Wait for the app update to finish before changing settings.".to_string())
        }
        Err(ActivityError::Busy(AppActivity::Configuring)) => {
            Err("A settings change is already in progress.".to_string())
        }
        Err(ActivityError::LockFailed) => Err("Settings are temporarily unavailable.".to_string()),
    }
}

async fn run_blocking<T, F>(task: &'static str, work: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|err| format!("{task} task failed: {err}"))?
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
        set_asr_language_blocking(app, engine, language)
    })
    .await
}

fn set_asr_language_blocking(
    app: AppHandle,
    engine: SpeechEngine,
    language: String,
) -> Result<(), String> {
    let _activity = reserve_settings_change()?;
    engine
        .validate_language(&language)
        .map_err(|error| command_error("Invalid speech language", error))?;

    let mut settings = crate::settings::get_settings(&app);
    let previous = settings.asr_language.clone();
    settings.asr_language = language.clone();
    crate::settings::save_settings(&app, &settings).map_err(|error| {
        log::error!("Could not persist speech language: {error}");
        "Could not save the speech language.".to_string()
    })?;

    if let Err(error) = engine.set_language(&language) {
        settings.asr_language = previous;
        if let Err(rollback_error) = crate::settings::save_settings(&app, &settings) {
            log::error!("Could not roll back speech language setting: {rollback_error}");
        }
        return Err(command_error("Could not apply speech language", error));
    }
    Ok(())
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
        reset_settings_blocking(app, engine)
    })
    .await
}

fn reset_settings_blocking(app: AppHandle, engine: SpeechEngine) -> Result<(), String> {
    use crate::settings::{save_settings, Settings};

    let _activity = reserve_settings_change()?;
    if engine.state() == EngineState::Loading {
        return Err(
            "Wait for the speech model to finish loading before resetting settings.".into(),
        );
    }

    let previous = crate::settings::get_settings(&app);
    let settings = Settings::default();
    if engine.is_ready() {
        engine
            .validate_language(&settings.asr_language)
            .map_err(|error| command_error("Invalid default speech language", error))?;
    }

    #[cfg(desktop)]
    let previous_shortcut = desktop::get_record_shortcut(app.clone());
    #[cfg(desktop)]
    desktop::update_record_shortcut(app.clone(), desktop::default_record_shortcut()).map_err(
        |error| {
            log::error!("Could not reset record shortcut: {error}");
            "Could not reset the record shortcut.".to_string()
        },
    )?;

    if let Err(error) = save_settings(&app, &settings) {
        #[cfg(desktop)]
        rollback_shortcut(&app, previous_shortcut.as_deref());
        return Err(error);
    }
    if engine.is_ready() {
        if let Err(error) = engine.set_language(&settings.asr_language) {
            if let Err(rollback_error) = save_settings(&app, &previous) {
                log::error!("Could not roll back settings reset: {rollback_error}");
            }
            #[cfg(desktop)]
            rollback_shortcut(&app, previous_shortcut.as_deref());
            return Err(command_error("Could not reset speech language", error));
        }
    }

    Ok(())
}

#[cfg(desktop)]
fn rollback_shortcut(app: &AppHandle, previous: Option<&str>) {
    if let Some(previous) = previous {
        if let Err(error) = desktop::update_record_shortcut(app.clone(), previous.to_string()) {
            log::error!("Could not roll back record shortcut: {error}");
        }
    }
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
