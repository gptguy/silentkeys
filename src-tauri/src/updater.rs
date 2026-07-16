use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Serialize;
use tauri::AppHandle;
use tauri_plugin_updater::{Update, UpdaterExt};

use crate::activity::{self, ActivityError, AppActivity};
use crate::errors::UserFacing;

const UPDATE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

static UPDATE_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, Serialize)]
pub struct AppUpdateInfo {
    pub current_version: String,
    pub version: String,
    pub date: Option<String>,
    pub body: Option<String>,
}

#[derive(thiserror::Error, Debug)]
pub enum AppUpdateError {
    #[error("could not initialize updater: {0}")]
    Initialize(#[source] tauri_plugin_updater::Error),
    #[error("could not check for updates: {0}")]
    Check(#[source] tauri_plugin_updater::Error),
    #[error("could not install update: {0}")]
    Install(#[source] tauri_plugin_updater::Error),
    #[error("finish recording before installing an update")]
    RecordingInProgress,
    #[error("an update installation is already running")]
    InstallInProgress,
    #[error("speech settings are being changed")]
    SettingsInProgress,
    #[error("app activity state is unavailable")]
    ActivityState,
}

impl UserFacing for AppUpdateError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::RecordingInProgress => "Finish recording before installing an update.",
            Self::InstallInProgress => "An update installation is already running.",
            Self::SettingsInProgress => {
                "Speech settings are being changed. Please try the update again."
            }
            Self::Initialize(_) | Self::Check(_) => {
                "Could not check for updates. Check your connection and try again."
            }
            Self::Install(_) | Self::ActivityState => {
                "Could not install the update. Please try again."
            }
        }
    }
}

struct InFlightGuard;

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        UPDATE_IN_FLIGHT.store(false, Ordering::Release);
    }
}

fn begin_exclusive_update() -> Result<InFlightGuard, AppUpdateError> {
    if UPDATE_IN_FLIGHT.swap(true, Ordering::AcqRel) {
        return Err(AppUpdateError::InstallInProgress);
    }
    Ok(InFlightGuard)
}

#[doc(hidden)]
pub fn begin_exclusive_update_for_tests() -> Result<impl Drop, AppUpdateError> {
    begin_exclusive_update()
}

async fn available_update(app: &AppHandle) -> Result<Option<Update>, AppUpdateError> {
    let updater = app
        .updater_builder()
        .timeout(UPDATE_TIMEOUT)
        .build()
        .map_err(AppUpdateError::Initialize)?;

    updater.check().await.map_err(AppUpdateError::Check)
}

pub async fn check_for_update(app: AppHandle) -> Result<Option<AppUpdateInfo>, AppUpdateError> {
    let update = available_update(&app).await?;

    Ok(update.map(|update| AppUpdateInfo {
        current_version: update.current_version,
        version: update.version,
        date: update.date.map(|date| date.to_string()),
        body: update.body,
    }))
}

/// Downloads without excluding dictation; only the install step takes the
/// `Updating` activity so recording stays available while bytes transfer.
pub async fn install_update(app: AppHandle) -> Result<bool, AppUpdateError> {
    let _in_flight = begin_exclusive_update()?;

    let Some(update) = available_update(&app).await? else {
        return Ok(false);
    };

    let bytes = update
        .download(|_, _| {}, || log::info!("Update download finished"))
        .await
        .map_err(AppUpdateError::Install)?;

    let _activity_guard = match activity::try_begin(AppActivity::Updating) {
        Ok(guard) => guard,
        Err(ActivityError::Busy(AppActivity::Recording)) => {
            return Err(AppUpdateError::RecordingInProgress)
        }
        Err(ActivityError::Busy(AppActivity::Updating)) => {
            return Err(AppUpdateError::InstallInProgress)
        }
        Err(ActivityError::Busy(AppActivity::Configuring)) => {
            return Err(AppUpdateError::SettingsInProgress)
        }
        Err(ActivityError::LockFailed) => return Err(AppUpdateError::ActivityState),
    };

    update.install(bytes).map_err(AppUpdateError::Install)?;
    app.request_restart();
    Ok(true)
}
