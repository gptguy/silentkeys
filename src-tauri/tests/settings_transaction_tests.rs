use silent_keys_lib::settings::{
    reset_settings_transaction, set_asr_language_transaction, EngineReadiness, Settings,
    SettingsAction, SettingsTransactionBackend, TransactionFailure,
};

struct FakeSettingsBackend {
    settings: Settings,
    readiness: EngineReadiness,
    shortcut: Option<String>,
    failures: Vec<(SettingsAction, &'static str)>,
    calls: Vec<SettingsAction>,
}

impl FakeSettingsBackend {
    fn new() -> Self {
        Self {
            settings: Settings {
                model_path: Some("/models/custom".to_string()),
                streaming_enabled: true,
                asr_language: "en-US".to_string(),
            },
            readiness: EngineReadiness::Ready,
            shortcut: Some("Alt+X".to_string()),
            failures: Vec::new(),
            calls: Vec::new(),
        }
    }

    fn fail(&mut self, action: SettingsAction, message: &'static str) {
        self.failures.push((action, message));
    }

    fn operation(&mut self, action: SettingsAction) -> Result<(), &'static str> {
        self.calls.push(action);
        self.failures
            .iter()
            .find_map(|(candidate, message)| (*candidate == action).then_some(*message))
            .map_or(Ok(()), Err)
    }
}

impl SettingsTransactionBackend for FakeSettingsBackend {
    type Error = &'static str;

    fn current_settings(&self) -> Settings {
        self.settings.clone()
    }

    fn persist_settings(
        &mut self,
        settings: &Settings,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        self.operation(action)?;
        self.settings = settings.clone();
        Ok(())
    }

    fn engine_readiness(&self) -> EngineReadiness {
        self.readiness
    }

    fn validate_language(
        &mut self,
        _language: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        self.operation(action)
    }

    fn apply_language(
        &mut self,
        _language: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        self.operation(action)
    }

    fn current_shortcut(&self) -> Option<String> {
        self.shortcut.clone()
    }

    fn default_shortcut(&self) -> Option<String> {
        Some("Alt+Z".to_string())
    }

    fn update_shortcut(
        &mut self,
        shortcut: &str,
        action: SettingsAction,
    ) -> Result<(), Self::Error> {
        self.operation(action)?;
        self.shortcut = Some(shortcut.to_string());
        Ok(())
    }
}

#[test]
fn language_transaction_persists_and_applies_selection() {
    let mut backend = FakeSettingsBackend::new();

    set_asr_language_transaction(&mut backend, "fr-FR").expect("transaction should succeed");

    assert_eq!(backend.settings.asr_language, "fr-FR");
    assert_eq!(
        backend.calls,
        [
            SettingsAction::ValidateSpeechLanguage,
            SettingsAction::PersistSpeechLanguage,
            SettingsAction::ApplySpeechLanguage,
        ]
    );
}

#[test]
fn language_transaction_restores_persisted_selection_after_apply_failure() {
    let mut backend = FakeSettingsBackend::new();
    backend.fail(SettingsAction::ApplySpeechLanguage, "apply failed");

    let error = set_asr_language_transaction(&mut backend, "fr-FR")
        .expect_err("apply failure should be returned");

    assert_eq!(error, TransactionFailure::Operation("apply failed"));
    assert_eq!(backend.settings.asr_language, "en-US");
    assert!(backend
        .calls
        .contains(&SettingsAction::RestoreSpeechLanguage));
}

#[test]
fn language_transaction_preserves_apply_and_rollback_failures() {
    let mut backend = FakeSettingsBackend::new();
    backend.fail(SettingsAction::ApplySpeechLanguage, "apply failed");
    backend.fail(SettingsAction::RestoreSpeechLanguage, "restore failed");

    let error = set_asr_language_transaction(&mut backend, "fr-FR")
        .expect_err("both failures should be returned");

    assert_eq!(
        error,
        TransactionFailure::Rollback {
            primary: Box::new(TransactionFailure::Operation("apply failed")),
            rollback: Box::new(TransactionFailure::Operation("restore failed")),
        }
    );
}

#[test]
fn reset_restores_shortcut_when_settings_persistence_fails() {
    let mut backend = FakeSettingsBackend::new();
    let previous_settings = backend.settings.clone();
    backend.fail(SettingsAction::PersistDefaultSettings, "persist failed");

    let error = reset_settings_transaction(&mut backend)
        .expect_err("settings persistence failure should be returned");

    assert_eq!(error, TransactionFailure::Operation("persist failed"));
    assert_eq!(backend.settings, previous_settings);
    assert_eq!(backend.shortcut.as_deref(), Some("Alt+X"));
}

#[test]
fn reset_restores_settings_and_shortcut_when_language_apply_fails() {
    let mut backend = FakeSettingsBackend::new();
    let previous_settings = backend.settings.clone();
    backend.fail(SettingsAction::ApplyDefaultSpeechLanguage, "apply failed");

    let error = reset_settings_transaction(&mut backend)
        .expect_err("language apply failure should be returned");

    assert_eq!(error, TransactionFailure::Operation("apply failed"));
    assert_eq!(backend.settings, previous_settings);
    assert_eq!(backend.shortcut.as_deref(), Some("Alt+X"));
}

#[test]
fn reset_preserves_each_rollback_failure() {
    let mut backend = FakeSettingsBackend::new();
    backend.fail(SettingsAction::ApplyDefaultSpeechLanguage, "apply failed");
    backend.fail(SettingsAction::RestoreSettings, "settings restore failed");
    backend.fail(
        SettingsAction::RestoreRecordShortcut,
        "shortcut restore failed",
    );

    let error = reset_settings_transaction(&mut backend)
        .expect_err("all rollback failures should be returned");

    assert_eq!(
        error,
        TransactionFailure::Rollback {
            primary: Box::new(TransactionFailure::Operation("apply failed")),
            rollback: Box::new(TransactionFailure::Rollback {
                primary: Box::new(TransactionFailure::Operation("settings restore failed")),
                rollback: Box::new(TransactionFailure::Operation("shortcut restore failed")),
            }),
        }
    );
}

#[test]
fn reset_waits_for_loading_model_before_mutating_settings() {
    let mut backend = FakeSettingsBackend::new();
    let previous_settings = backend.settings.clone();
    backend.readiness = EngineReadiness::Loading;

    let error =
        reset_settings_transaction(&mut backend).expect_err("loading model should block the reset");

    assert_eq!(error, TransactionFailure::ModelLoading);
    assert_eq!(backend.settings, previous_settings);
    assert!(backend.calls.is_empty());
}
