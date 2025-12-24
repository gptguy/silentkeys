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
    pub error: Option<String>,
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct TranscriptionPatchDto {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub stable: bool,
}

#[derive(Serialize)]
struct SetModelPathArgs {
    path: String,
}

#[derive(Serialize)]
struct SetStreamingArgs {
    enabled: bool,
}

async fn invoke_no_args(cmd: &str) -> Result<JsValue, String> {
    invoke(cmd, JsValue::NULL).await.map_err(extract_error)
}

pub async fn fetch_download_progress() -> Result<Option<ModelDownloadProgressDto>, String> {
    let value = invoke_no_args("model_download_progress").await?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

pub async fn check_model_ready_flag() -> Result<bool, String> {
    let value = invoke_no_args("is_model_ready").await?;
    Ok(value.as_bool().unwrap_or(false))
}

pub async fn retry_model_download_cmd() -> Result<(), String> {
    invoke_no_args("retry_model_download").await.map(|_| ())
}

pub async fn start_recording_cmd() -> Result<(), String> {
    invoke_no_args("start_recording").await.map(|_| ())
}

pub async fn stop_recording_cmd() -> Result<String, String> {
    let value = invoke_no_args("stop_recording").await?;
    value
        .as_string()
        .ok_or_else(|| "Invalid response format".to_string())
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

pub fn extract_error(err: JsValue) -> String {
    err.as_string()
        .or_else(|| {
            js_sys::Reflect::get(&err, &"message".into())
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| "Unknown error".to_string())
}
