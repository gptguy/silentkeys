use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

mod service;
mod transaction;

pub(crate) use service::{reset_settings, set_asr_language, set_model_path, set_streaming_enabled};
#[doc(hidden)]
pub use transaction::{
    reset_settings_transaction, set_asr_language_transaction, EngineReadiness, SettingsAction,
    SettingsTransactionBackend, TransactionFailure,
};

pub const DEFAULT_ASR_LANGUAGE: &str = "en-US";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Settings {
    pub model_path: Option<String>,
    pub streaming_enabled: bool,
    pub asr_language: String,
}

const STORE_PATH: &str = "settings.json";

#[derive(thiserror::Error, Debug)]
pub enum SettingsStoreError {
    #[error("open settings store: {0}")]
    Open(#[source] tauri_plugin_store::Error),
    #[error("save settings store: {0}")]
    Save(#[source] tauri_plugin_store::Error),
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_path: None,
            streaming_enabled: false,
            asr_language: DEFAULT_ASR_LANGUAGE.to_string(),
        }
    }
}

pub fn get_settings(app: &AppHandle) -> Settings {
    match app.store(STORE_PATH) {
        Ok(store) => {
            let model_path = store
                .get("model_path")
                .and_then(|v| v.as_str().map(|s| s.to_string()));
            let streaming_enabled = store
                .get("streaming_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let asr_language = store
                .get("asr_language")
                .and_then(|value| value.as_str().map(str::to_owned))
                .unwrap_or_else(|| DEFAULT_ASR_LANGUAGE.to_string());
            Settings {
                model_path,
                streaming_enabled,
                asr_language,
            }
        }
        Err(e) => {
            log::warn!("Failed to load settings store: {e}");
            Settings::default()
        }
    }
}

pub fn save_settings(app: &AppHandle, settings: &Settings) -> Result<(), SettingsStoreError> {
    let store = app.store(STORE_PATH).map_err(SettingsStoreError::Open)?;

    if let Some(path) = &settings.model_path {
        store.set("model_path", serde_json::json!(path));
    } else {
        store.delete("model_path");
    }

    store.set(
        "streaming_enabled",
        serde_json::json!(settings.streaming_enabled),
    );
    store.set("asr_language", serde_json::json!(settings.asr_language));

    log::info!(
        "Saving settings: streaming_enabled={}",
        settings.streaming_enabled
    );

    store.save().map_err(SettingsStoreError::Save)
}

pub fn get_custom_model_path(app: &AppHandle) -> Option<PathBuf> {
    get_settings(app).model_path.map(PathBuf::from)
}
