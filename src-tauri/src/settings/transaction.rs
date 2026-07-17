use std::fmt;

use super::Settings;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineReadiness {
    Loading,
    Ready,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettingsAction {
    PersistModelPath,
    PersistStreamingPreference,
    ValidateSpeechLanguage,
    PersistSpeechLanguage,
    ApplySpeechLanguage,
    RestoreSpeechLanguage,
    ValidateDefaultSpeechLanguage,
    ResetRecordShortcut,
    PersistDefaultSettings,
    ApplyDefaultSpeechLanguage,
    RestoreSettings,
    RestoreRecordShortcut,
    SpeechLanguageUpdate,
    SettingsReset,
}

impl fmt::Display for SettingsAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::PersistModelPath => "persist model path",
            Self::PersistStreamingPreference => "persist streaming preference",
            Self::ValidateSpeechLanguage => "validate speech language",
            Self::PersistSpeechLanguage => "persist speech language",
            Self::ApplySpeechLanguage => "apply speech language",
            Self::RestoreSpeechLanguage => "restore speech language",
            Self::ValidateDefaultSpeechLanguage => "validate default speech language",
            Self::ResetRecordShortcut => "reset record shortcut",
            Self::PersistDefaultSettings => "persist default settings",
            Self::ApplyDefaultSpeechLanguage => "apply default speech language",
            Self::RestoreSettings => "restore settings",
            Self::RestoreRecordShortcut => "restore record shortcut",
            Self::SpeechLanguageUpdate => "speech language update",
            Self::SettingsReset => "settings reset",
        };
        formatter.write_str(description)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum TransactionFailure<E> {
    Operation(E),
    ModelLoading,
    Rollback {
        primary: Box<TransactionFailure<E>>,
        rollback: Box<TransactionFailure<E>>,
    },
}

pub trait SettingsTransactionBackend {
    type Error;

    fn current_settings(&self) -> Settings;
    fn persist_settings(
        &mut self,
        settings: &Settings,
        action: SettingsAction,
    ) -> Result<(), Self::Error>;
    fn engine_readiness(&self) -> EngineReadiness;
    fn validate_language(
        &mut self,
        language: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error>;
    fn apply_language(&mut self, language: &str, action: SettingsAction)
        -> Result<(), Self::Error>;
    fn current_shortcut(&self) -> Option<String>;
    fn default_shortcut(&self) -> Option<String>;
    fn update_shortcut(
        &mut self,
        shortcut: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error>;
}

pub fn set_asr_language_transaction<B: SettingsTransactionBackend>(
    backend: &mut B,
    language: &str,
) -> Result<(), TransactionFailure<B::Error>> {
    operation(backend.validate_language(language, SettingsAction::ValidateSpeechLanguage))?;

    let mut settings = backend.current_settings();
    let previous = settings.asr_language.clone();
    settings.asr_language = language.to_string();
    operation(backend.persist_settings(&settings, SettingsAction::PersistSpeechLanguage))?;

    let apply = operation(backend.apply_language(language, SettingsAction::ApplySpeechLanguage));
    rollback_on_failure(apply, || {
        settings.asr_language = previous;
        operation(backend.persist_settings(&settings, SettingsAction::RestoreSpeechLanguage))
    })
}

pub fn reset_settings_transaction<B: SettingsTransactionBackend>(
    backend: &mut B,
) -> Result<(), TransactionFailure<B::Error>> {
    let readiness = backend.engine_readiness();
    if readiness == EngineReadiness::Loading {
        return Err(TransactionFailure::ModelLoading);
    }

    let previous_settings = backend.current_settings();
    let defaults = Settings::default();
    if readiness == EngineReadiness::Ready {
        operation(backend.validate_language(
            &defaults.asr_language,
            SettingsAction::ValidateDefaultSpeechLanguage,
        ))?;
    }

    let previous_shortcut = backend.current_shortcut();
    if let Some(default_shortcut) = backend.default_shortcut() {
        operation(backend.update_shortcut(&default_shortcut, SettingsAction::ResetRecordShortcut))?;
    }

    let persist =
        operation(backend.persist_settings(&defaults, SettingsAction::PersistDefaultSettings));
    rollback_on_failure(persist, || {
        restore_shortcut(backend, previous_shortcut.as_deref())
    })?;

    if readiness != EngineReadiness::Ready {
        return Ok(());
    }

    let apply = operation(backend.apply_language(
        &defaults.asr_language,
        SettingsAction::ApplyDefaultSpeechLanguage,
    ));
    rollback_on_failure(apply, || {
        let settings = operation(
            backend.persist_settings(&previous_settings, SettingsAction::RestoreSettings),
        );
        let shortcut = restore_shortcut(backend, previous_shortcut.as_deref());
        combine_rollbacks(settings, shortcut)
    })
}

fn restore_shortcut<B: SettingsTransactionBackend>(
    backend: &mut B,
    previous: Option<&str>,
) -> Result<(), TransactionFailure<B::Error>> {
    match previous {
        Some(shortcut) => {
            operation(backend.update_shortcut(shortcut, SettingsAction::RestoreRecordShortcut))
        }
        None => Ok(()),
    }
}

fn operation<E>(result: Result<(), E>) -> Result<(), TransactionFailure<E>> {
    result.map_err(TransactionFailure::Operation)
}

fn rollback_on_failure<E>(
    result: Result<(), TransactionFailure<E>>,
    rollback: impl FnOnce() -> Result<(), TransactionFailure<E>>,
) -> Result<(), TransactionFailure<E>> {
    match result {
        Ok(()) => Ok(()),
        Err(primary) => match rollback() {
            Ok(()) => Err(primary),
            Err(rollback) => Err(TransactionFailure::Rollback {
                primary: Box::new(primary),
                rollback: Box::new(rollback),
            }),
        },
    }
}

fn combine_rollbacks<E>(
    first: Result<(), TransactionFailure<E>>,
    second: Result<(), TransactionFailure<E>>,
) -> Result<(), TransactionFailure<E>> {
    match (first, second) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(primary), Err(rollback)) => Err(TransactionFailure::Rollback {
            primary: Box::new(primary),
            rollback: Box::new(rollback),
        }),
    }
}
