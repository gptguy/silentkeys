use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

#[derive(Serialize, Deserialize, Default)]
pub struct Settings {
    pub model_path: Option<String>,
}

const STORE_PATH: &str = "settings.json";

pub fn get_settings(app: &AppHandle) -> Settings {
    match app.store(STORE_PATH) {
        Ok(store) => {
            let model_path = store
                .get("model_path")
                .and_then(|v| v.as_str().map(|s| s.to_string()));
            Settings { model_path }
        }
        Err(e) => {
            log::warn!("Failed to load settings store: {e}");
            Settings::default()
        }
    }
}

pub fn save_settings(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let store = app
        .store(STORE_PATH)
        .map_err(|e| format!("Failed to open settings store: {e}"))?;

    if let Some(path) = &settings.model_path {
        store.set("model_path", serde_json::json!(path));
    } else {
        store.delete("model_path");
    }

    store.save().map_err(|e| e.to_string())
}

pub fn get_custom_model_path(app: &AppHandle) -> Option<PathBuf> {
    get_settings(app).model_path.map(PathBuf::from)
}
