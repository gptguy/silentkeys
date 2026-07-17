use tauri::AppHandle;

use crate::activity::{self, ActivityError, ActivityGuard, AppActivity};
#[cfg(desktop)]
use crate::desktop;
use crate::engine::{EngineError, EngineState, SpeechEngine};
use crate::errors::UserFacing;

use super::transaction::{
    self, EngineReadiness, SettingsAction, SettingsTransactionBackend, TransactionFailure,
};
use super::{get_settings, save_settings, Settings, SettingsStoreError};

#[derive(thiserror::Error, Debug)]
pub(crate) enum SettingsServiceError {
    #[error(transparent)]
    Activity(#[from] ActivityError),
    #[error("{action}: {source}")]
    Storage {
        action: SettingsAction,
        #[source]
        source: SettingsStoreError,
    },
    #[error("{action}: {source}")]
    Engine {
        action: SettingsAction,
        #[source]
        source: EngineError,
    },
    #[cfg(desktop)]
    #[error("{action}: {detail}")]
    Shortcut {
        action: SettingsAction,
        detail: String,
    },
    #[error("speech model is still loading")]
    ModelLoading,
    #[error("{action}; original error: {primary}; rollback error: {rollback}")]
    Rollback {
        action: SettingsAction,
        #[source]
        primary: Box<SettingsServiceError>,
        rollback: Box<SettingsServiceError>,
    },
}

impl UserFacing for SettingsServiceError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::Activity(ActivityError::Busy(AppActivity::Recording)) => {
                "Finish the current recording before changing settings."
            }
            Self::Activity(ActivityError::Busy(AppActivity::Updating)) => {
                "Wait for the app update to finish before changing settings."
            }
            Self::Activity(ActivityError::Busy(AppActivity::Configuring)) => {
                "A settings change is already in progress."
            }
            Self::Activity(ActivityError::LockFailed) => "Settings are temporarily unavailable.",
            Self::Storage { .. } => "Could not save settings. Please try again.",
            Self::Engine { source, .. } => source.user_message(),
            #[cfg(desktop)]
            Self::Shortcut { .. } => "Could not update the record shortcut.",
            Self::ModelLoading => {
                "Wait for the speech model to finish loading before resetting settings."
            }
            Self::Rollback { .. } => {
                "Could not apply settings safely. Please restart the app and try again."
            }
        }
    }
}

pub(crate) fn set_model_path(app: &AppHandle, path: String) -> Result<(), SettingsServiceError> {
    let mut settings = get_settings(app);
    settings.model_path = Some(path);
    persist(app, &settings, SettingsAction::PersistModelPath)
}

pub(crate) fn set_streaming_enabled(
    app: &AppHandle,
    enabled: bool,
) -> Result<(), SettingsServiceError> {
    let mut settings = get_settings(app);
    settings.streaming_enabled = enabled;
    persist(app, &settings, SettingsAction::PersistStreamingPreference)
}

pub(crate) fn set_asr_language(
    app: &AppHandle,
    engine: &SpeechEngine,
    language: String,
) -> Result<(), SettingsServiceError> {
    let _activity = reserve_change()?;
    let mut backend = AppSettingsBackend { app, engine };
    transaction::set_asr_language_transaction(&mut backend, &language)
        .map_err(|failure| transaction_error(SettingsAction::SpeechLanguageUpdate, failure))
}

pub(crate) fn reset_settings(
    app: &AppHandle,
    engine: &SpeechEngine,
) -> Result<(), SettingsServiceError> {
    let _activity = reserve_change()?;
    let mut backend = AppSettingsBackend { app, engine };
    transaction::reset_settings_transaction(&mut backend)
        .map_err(|failure| transaction_error(SettingsAction::SettingsReset, failure))
}

struct AppSettingsBackend<'a> {
    app: &'a AppHandle,
    engine: &'a SpeechEngine,
}

impl SettingsTransactionBackend for AppSettingsBackend<'_> {
    type Error = SettingsServiceError;

    fn current_settings(&self) -> Settings {
        get_settings(self.app)
    }

    fn persist_settings(
        &mut self,
        settings: &Settings,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        persist(self.app, settings, action)
    }

    fn engine_readiness(&self) -> EngineReadiness {
        match self.engine.state() {
            EngineState::Loading => EngineReadiness::Loading,
            EngineState::Loaded => EngineReadiness::Ready,
            EngineState::Unloaded | EngineState::Failed(_) => EngineReadiness::Unavailable,
        }
    }

    fn validate_language(
        &mut self,
        language: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        self.engine
            .validate_language(language)
            .map_err(|source| SettingsServiceError::Engine { action, source })
    }

    fn apply_language(
        &mut self,
        language: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        self.engine
            .set_language(language)
            .map(|_| ())
            .map_err(|source| SettingsServiceError::Engine { action, source })
    }

    fn current_shortcut(&self) -> Option<String> {
        #[cfg(desktop)]
        {
            desktop::get_record_shortcut(self.app.clone())
        }
        #[cfg(not(desktop))]
        {
            None
        }
    }

    fn default_shortcut(&self) -> Option<String> {
        #[cfg(desktop)]
        {
            Some(desktop::default_record_shortcut())
        }
        #[cfg(not(desktop))]
        {
            None
        }
    }

    fn update_shortcut(
        &mut self,
        shortcut: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        #[cfg(desktop)]
        {
            desktop::update_record_shortcut(self.app.clone(), shortcut.to_string())
                .map(|_| ())
                .map_err(|detail| SettingsServiceError::Shortcut { action, detail })
        }
        #[cfg(not(desktop))]
        {
            let _ = (shortcut, action);
            Ok(())
        }
    }
}

fn reserve_change() -> Result<ActivityGuard, SettingsServiceError> {
    activity::try_begin(AppActivity::Configuring).map_err(Into::into)
}

fn persist(
    app: &AppHandle,
    settings: &Settings,
    action: SettingsAction,
) -> Result<(), SettingsServiceError> {
    save_settings(app, settings).map_err(|source| SettingsServiceError::Storage { action, source })
}

fn transaction_error(
    action: SettingsAction,
    failure: TransactionFailure<SettingsServiceError>,
) -> SettingsServiceError {
    match failure {
        TransactionFailure::Operation(error) => error,
        TransactionFailure::ModelLoading => SettingsServiceError::ModelLoading,
        TransactionFailure::Rollback { primary, rollback } => SettingsServiceError::Rollback {
            action,
            primary: Box::new(transaction_error(action, *primary)),
            rollback: Box::new(transaction_error(action, *rollback)),
        },
    }
}
