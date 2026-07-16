use serde::{Deserialize, Serialize};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], catch)]
    async fn invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"], catch)]
    pub async fn listen(
        event: &str,
        handler: &Closure<dyn FnMut(JsValue)>,
    ) -> Result<JsValue, JsValue>;
}

#[derive(Serialize, Deserialize)]
pub struct UpdateShortcutArgs<'a> {
    pub shortcut: &'a str,
}

#[derive(Deserialize, Debug)]
pub struct ModelDownloadProgressDto {
    pub file_index: usize,
    pub file_count: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub done: bool,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "state", content = "message", rename_all = "snake_case")]
pub enum EngineStateDto {
    Unloaded,
    Loading,
    Loaded,
    Failed(String),
}

#[derive(Clone, Deserialize, Debug)]
pub struct AppUpdateInfoDto {
    pub current_version: String,
    pub version: String,
    pub date: Option<String>,
    pub body: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
pub enum TranscriptionUpdateDto {
    Append(String),
    Replace(String),
}

#[derive(Serialize)]
struct SetModelPathArgs {
    path: String,
}

#[derive(Serialize)]
struct SetStreamingArgs {
    enabled: bool,
}

#[derive(Serialize)]
struct SetAsrLanguageArgs {
    language: String,
}

async fn invoke_no_args(cmd: &str) -> Result<JsValue, String> {
    invoke(cmd, JsValue::NULL).await.map_err(extract_error)
}

pub async fn fetch_engine_state() -> Result<EngineStateDto, String> {
    let value = invoke_no_args("engine_state").await?;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn retry_model_download_cmd() -> Result<(), String> {
    invoke_no_args("retry_model_download").await.map(|_| ())
}

pub async fn start_recording_cmd() -> Result<(), String> {
    invoke_no_args("start_recording").await.map(|_| ())
}

pub async fn stop_recording_cmd() -> Result<(), String> {
    invoke_no_args("stop_recording").await.map(|_| ())
}

pub async fn fetch_current_shortcut() -> Result<Option<String>, String> {
    let value = invoke_no_args("get_record_shortcut").await?;
    Ok(value.as_string())
}

pub async fn fetch_model_path() -> Result<String, String> {
    let value = invoke_no_args("get_model_path").await?;
    value
        .as_string()
        .ok_or_else(|| "Invalid response format".to_string())
}

pub async fn fetch_default_shortcut() -> Result<Option<String>, String> {
    let value = invoke_no_args("default_record_shortcut").await?;
    Ok(value.as_string())
}

pub async fn save_model_path(path: String) -> Result<(), String> {
    let args =
        serde_wasm_bindgen::to_value(&SetModelPathArgs { path }).map_err(|err| err.to_string())?;
    invoke("set_model_path", args)
        .await
        .map(|_| ())
        .map_err(extract_error)
}

pub async fn pick_model_folder_cmd() -> Result<Option<String>, String> {
    let value = invoke_no_args("pick_model_folder").await?;
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_string()
        .ok_or_else(|| "Invalid response".to_string())
        .map(Some)
}

pub async fn reset_settings_cmd() -> Result<(), String> {
    invoke_no_args("reset_settings").await.map(|_| ())
}

pub async fn save_shortcut(shortcut: &str) -> Result<String, String> {
    let args = serde_wasm_bindgen::to_value(&UpdateShortcutArgs { shortcut })
        .map_err(|err| err.to_string())?;

    let value = invoke("update_record_shortcut", args)
        .await
        .map_err(extract_error)?;

    value
        .as_string()
        .ok_or_else(|| "Shortcut saved, but response was empty".to_string())
}

pub async fn fetch_streaming_enabled() -> Result<bool, String> {
    let value = invoke_no_args("get_use_streaming").await?;
    Ok(value.as_bool().unwrap_or(false))
}

pub async fn save_streaming_enabled(enabled: bool) -> Result<(), String> {
    let args = serde_wasm_bindgen::to_value(&SetStreamingArgs { enabled })
        .map_err(|err| err.to_string())?;
    invoke("set_use_streaming", args)
        .await
        .map(|_| ())
        .map_err(extract_error)
}

pub async fn fetch_asr_language() -> Result<String, String> {
    let value = invoke_no_args("get_asr_language").await?;
    value
        .as_string()
        .ok_or_else(|| "Speech language response was invalid".to_string())
}

pub async fn fetch_asr_languages() -> Result<Vec<String>, String> {
    let value = invoke_no_args("get_asr_languages").await?;
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn save_asr_language(language: String) -> Result<(), String> {
    let args = serde_wasm_bindgen::to_value(&SetAsrLanguageArgs { language })
        .map_err(|err| err.to_string())?;
    invoke("set_asr_language", args)
        .await
        .map(|_| ())
        .map_err(extract_error)
}

pub async fn check_for_app_update_cmd() -> Result<Option<AppUpdateInfoDto>, String> {
    let value = invoke_no_args("check_for_app_update").await?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn install_app_update_cmd() -> Result<bool, String> {
    let value = invoke_no_args("install_app_update").await?;
    value
        .as_bool()
        .ok_or_else(|| "Updater returned an invalid response".to_string())
}

pub fn extract_error(err: JsValue) -> String {
    err.as_string()
        .or_else(|| {
            js_sys::Reflect::get(&err, &"message".into())
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| "Unknown error".to_string())
}
