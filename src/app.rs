use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos::web_sys;
use leptos::web_sys::HtmlInputElement;
use serde::{Deserialize, Serialize};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{prelude::*, JsCast};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], catch)]
    async fn invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;
}

#[derive(Serialize, Deserialize)]
struct UpdateShortcutArgs<'a> {
    shortcut: &'a str,
}

#[derive(Deserialize)]
struct ModelDownloadProgressDto {
    file_index: usize,
    file_count: usize,
    downloaded_bytes: u64,
    total_bytes: u64,
    done: bool,
    error: Option<String>,
}

async fn invoke_no_args(cmd: &str) -> Result<JsValue, String> {
    invoke(cmd, JsValue::NULL).await.map_err(extract_error)
}

async fn fetch_download_progress() -> Result<Option<ModelDownloadProgressDto>, String> {
    let value = invoke_no_args("model_download_progress").await?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    serde_wasm_bindgen::from_value(value).map_err(|err| err.to_string())
}

async fn check_model_ready_flag() -> Result<bool, String> {
    let value = invoke_no_args("is_model_ready").await?;
    Ok(value.as_bool().unwrap_or(false))
}

async fn retry_model_download_cmd() -> Result<(), String> {
    invoke_no_args("retry_model_download").await.map(|_| ())
}

async fn start_recording_cmd() -> Result<(), String> {
    invoke_no_args("start_recording").await.map(|_| ())
}

async fn stop_recording_cmd() -> Result<String, String> {
    let value = invoke_no_args("stop_recording").await?;
    value
        .as_string()
        .ok_or_else(|| "Invalid response format".to_string())
}

async fn fetch_current_shortcut() -> Result<Option<String>, String> {
    let value = invoke_no_args("get_record_shortcut").await?;
    Ok(value.as_string())
}

async fn fetch_model_path() -> Result<String, String> {
    let value = invoke_no_args("get_model_path").await?;
    value
        .as_string()
        .ok_or_else(|| "Invalid response format".to_string())
}

async fn fetch_default_shortcut() -> Result<Option<String>, String> {
    let value = invoke_no_args("default_record_shortcut").await?;
    Ok(value.as_string())
}

#[derive(Serialize)]
struct SetModelPathArgs {
    path: String,
}

async fn save_model_path(path: String) -> Result<(), String> {
    let args =
        serde_wasm_bindgen::to_value(&SetModelPathArgs { path }).map_err(|err| err.to_string())?;
    invoke("set_model_path", args)
        .await
        .map(|_| ())
        .map_err(extract_error)
}

async fn pick_model_folder_cmd() -> Result<Option<String>, String> {
    let value = invoke_no_args("pick_model_folder").await?;
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_string()
        .ok_or_else(|| "Invalid response".to_string())
        .map(Some)
}

async fn reset_settings_cmd() -> Result<(), String> {
    invoke_no_args("reset_settings").await.map(|_| ())
}

async fn save_shortcut(shortcut: &str) -> Result<String, String> {
    let args = serde_wasm_bindgen::to_value(&UpdateShortcutArgs { shortcut })
        .map_err(|err| err.to_string())?;

    let value = invoke("update_record_shortcut", args)
        .await
        .map_err(extract_error)?;

    value
        .as_string()
        .ok_or_else(|| "Shortcut saved, but response was empty".to_string())
}

fn extract_error(err: JsValue) -> String {
    err.as_string()
        .or_else(|| {
            js_sys::Reflect::get(&err, &"message".into())
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| "Unknown error".to_string())
}

fn input_value(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
        .map(|input| input.value())
        .unwrap_or_default()
}

fn start_model_progress_polling(
    is_recording: ReadSignal<bool>,
    transcribing: ReadSignal<bool>,
    status: ReadSignal<String>,
    set_status: WriteSignal<String>,
    set_model_ready: WriteSignal<bool>,
    set_model_error: WriteSignal<Option<String>>,
) {
    let callback = Closure::wrap(Box::new(move || {
        if is_recording.get_untracked() || transcribing.get_untracked() {
            return;
        }

        let current_status = status.get_untracked();
        let is_model_message = current_status.is_empty()
            || current_status.starts_with("Preparing speech model")
            || current_status.starts_with("Downloading")
            || current_status.starts_with("Model download failed");
        if !is_model_message {
            return;
        }

        spawn_local(async move {
            match fetch_download_progress().await {
                Ok(Some(progress)) => {
                    if let Some(error) = progress.error {
                        set_model_ready.set(false);
                        match fetch_model_path().await {
                            Ok(path) => {
                                set_model_error.set(Some(format!(
                                    "Download failed: {}. Manual download required to: {}",
                                    error, path
                                )));
                            }
                            Err(_) => {
                                set_model_error.set(Some(error.clone()));
                            }
                        }
                        set_status.set("Model download failed.".to_string());
                    } else if progress.done {
                        set_model_ready.set(true);
                        set_model_error.set(None);
                        set_status.set("Tap to start recording.".to_string());
                    } else if progress.file_count > 0 {
                        set_model_ready.set(false);
                        set_model_error.set(None);
                        let index = progress.file_index.min(progress.file_count);

                        let progress_text = if progress.total_bytes > 0 {
                            let mb_downloaded = progress.downloaded_bytes as f64 / 1024.0 / 1024.0;
                            let mb_total = progress.total_bytes as f64 / 1024.0 / 1024.0;
                            format!(
                                "Downloading ({:.1}/{:.1} MB, file {}/{})",
                                mb_downloaded, mb_total, index, progress.file_count
                            )
                        } else {
                            format!("Downloading (file {}/{})...", index, progress.file_count)
                        };
                        set_status.set(progress_text);
                    } else {
                        set_model_ready.set(false);
                        set_model_error.set(None);
                        set_status.set("Preparing speech model for first use...".to_string());
                    }
                }
                Ok(None) => match check_model_ready_flag().await {
                    Ok(true) => {
                        set_model_ready.set(true);
                        set_model_error.set(None);
                        set_status.set("Tap to start recording.".to_string());
                    }
                    Ok(false) => {}
                    Err(err) => {
                        set_model_ready.set(false);
                        set_model_error.set(Some(err.clone()));
                        set_status.set(format!("Preparing speech model: {}", err));
                    }
                },
                Err(err) => {
                    set_model_ready.set(false);
                    set_model_error.set(Some(err.clone()));
                    set_status.set(format!("Preparing speech model: {}", err));
                }
            }
        });
    }) as Box<dyn FnMut()>);

    if let Some(window) = web_sys::window() {
        let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            1_000,
        );
    }

    // Leak the callback so the interval keeps running for the lifetime of the app.
    callback.forget();
}

#[component]
pub fn App() -> impl IntoView {
    let (transcription, set_transcription) = signal(String::new());
    let (transcribing, set_transcribing) = signal(false);
    let (is_recording, set_is_recording) = signal(false);
    let (status, set_status) = signal(String::new());
    let (model_ready, set_model_ready) = signal(false);
    let (model_error, set_model_error) = signal::<Option<String>>(None);
    let (shortcut, set_shortcut) = signal(String::new());
    let (shortcut_status, set_shortcut_status) = signal(String::new());

    let toggle_recording = move |_| {
        if !model_ready.get() {
            if model_error.get().is_some() {
                set_status.set("Retrying speech model download...".to_string());
                let set_status_retry = set_status;
                let set_model_error_retry = set_model_error;
                spawn_local(async move {
                    match retry_model_download_cmd().await {
                        Ok(_) => set_model_error_retry.set(None),
                        Err(err) => {
                            set_model_error_retry.set(Some(err.clone()));
                            set_status_retry.set(format!("Could not download model: {}", err));
                        }
                    }
                });
            }
            return;
        }

        if transcribing.get() {
            return;
        }

        if !is_recording.get() {
            set_status.set("Starting recording...".to_string());
            set_is_recording.set(true);
            spawn_local(async move {
                match start_recording_cmd().await {
                    Ok(_) => set_status.set("Recording... tap to stop.".to_string()),
                    Err(message) => {
                        set_is_recording.set(false);
                        set_status.set(format!("Could not start recording: {}", message));
                    }
                }
            });
            return;
        }

        spawn_local(async move {
            set_status.set("Stopping recording...".to_string());
            set_transcribing.set(true);
            set_transcription.set(String::new());

            match stop_recording_cmd().await {
                Ok(text) => {
                    set_is_recording.set(false);
                    set_transcription.set(text);
                    set_status.set("Finished.".to_string());
                }
                Err(err) => {
                    set_is_recording.set(false);
                    set_status.set(format!("Could not stop recording: {}", err));
                }
            }

            set_transcribing.set(false);
        });
    };

    // Check model readiness and load persisted shortcut or default on startup
    {
        start_model_progress_polling(
            is_recording,
            transcribing,
            status,
            set_status,
            set_model_ready,
            set_model_error,
        );
        spawn_local(async move {
            match check_model_ready_flag().await {
                Ok(true) => {
                    set_model_ready.set(true);
                    set_model_error.set(None);
                    set_status.set("Tap to start recording.".to_string());
                }
                Ok(false) => {
                    set_model_ready.set(false);
                    set_status.set("Preparing speech model for first use...".to_string());
                }
                Err(err) => {
                    set_model_ready.set(false);
                    set_model_error.set(Some(err.clone()));
                    set_status.set(format!("Preparing speech model for first use: {}", err));
                }
            }
        });

        spawn_local(async move {
            match fetch_current_shortcut().await {
                Ok(Some(shortcut)) => set_shortcut.set(shortcut),
                Ok(None) => match fetch_default_shortcut().await {
                    Ok(Some(shortcut)) => set_shortcut.set(shortcut),
                    Ok(None) => {
                        set_shortcut_status.set("No shortcut available".to_string());
                    }
                    Err(err) => {
                        set_shortcut_status.set(format!("Could not load default shortcut: {}", err))
                    }
                },
                Err(err) => {
                    set_shortcut_status.set(format!("Could not load shortcut: {}", err));
                }
            }
        });
    }

    let (model_path, set_model_path) = signal(String::new());

    let change_path_action = {
        move |_| {
            spawn_local(async move {
                match pick_model_folder_cmd().await {
                    Ok(Some(path)) => match save_model_path(path.clone()).await {
                        Ok(_) => {
                            set_model_path.set(path);
                            set_status.set("Model path updated.".to_string());
                        }
                        Err(e) => set_status.set(format!("Failed to save path: {}", e)),
                    },
                    Ok(None) => {} // User cancelled
                    Err(e) => set_status.set(format!("Failed to pick folder: {}", e)),
                }
            });
        }
    };

    // Fetch initial model path
    {
        spawn_local(async move {
            if let Ok(path) = fetch_model_path().await {
                set_model_path.set(path);
            }
        });
    }

    let save_shortcut_action = {
        move |_| {
            let shortcut_value = shortcut.get();
            set_shortcut_status.set("Saving shortcut...".to_string());
            spawn_local(async move {
                match save_shortcut(&shortcut_value).await {
                    Ok(saved) => set_shortcut_status.set(format!("Shortcut saved: {}", saved)),
                    Err(err) => {
                        set_shortcut_status.set(format!("Failed to save shortcut: {}", err));
                    }
                }
            });
        }
    };

    let reset_settings_action = {
        move |_| {
            spawn_local(async move {
                match reset_settings_cmd().await {
                    Ok(_) => {
                        // Refresh UI with new defaults
                        if let Ok(path) = fetch_model_path().await {
                            set_model_path.set(path);
                        }
                        if let Ok(Some(shortcut)) = fetch_default_shortcut().await {
                            set_shortcut.set(shortcut);
                        }
                        set_status.set("Settings reset.".to_string());
                    }
                    Err(e) => set_status.set(format!("Reset failed: {}", e)),
                }
            });
        }
    };

    view! {
        <main class="shell">
            <header class="hero">
                <p class="eyebrow">"Local-first capture"</p>
                <h1>"Silent Keys"</h1>
                <p class="lede">
                    "Hands-free mic capture that stays local, transcribes fast, and types for you."
                </p>
            </header>

            <section class="card control-card">
                <div class="card-header">
                    <div>
                        <p class="eyebrow">"Recorder"</p>
                        <h2>"One tap to capture, one tap to finish"</h2>
                        <p class="hint">
                            "Click start or press your shortcut to record. We wrap up and transcribe when you stop."
                        </p>
                    </div>
                    <span
                        class="pill"
                        class:live=move || is_recording.get()
                        class:glow=move || transcribing.get()
                        class:idle=move || !is_recording.get() && !transcribing.get()
                    >
                        {move || {
                            if is_recording.get() {
                                "Listening"
                            } else if transcribing.get() {
                                "Transcribing"
                            } else {
                                "Idle"
                            }
                        }}
                    </span>
                </div>

                <div class="control-row">
                    <button
                        on:click=toggle_recording
                        disabled=move || transcribing.get() || (!model_ready.get() && model_error.get().is_none())
                    >
                        {move || {
                            if !model_ready.get() {
                                if model_error.get().is_some() {
                                    "Retry model download"
                                } else {
                                    "Preparing speech model..."
                                }
                            } else if transcribing.get() {
                                "Working..."
                            } else if is_recording.get() {
                                "Finish recording"
                            } else {
                                "Start recording"
                            }
                        }}
                    </button>
                    <div class="status-container">
                        <p class="inline-status">{ move || status.get() }</p>
                        {move || model_error.get().map(|err| view! {
                             <div class="error-details">
                                <p class="error-msg">{err}</p>
                                <p class="help-text">"You can download the model manually using:"</p>
                                <code class="cmd-block">"hf download istupakov/parakeet-tdt-0.6b-v3-onnx"</code>
                             </div>
                        })}
                    </div>
                </div>
            </section>

            <section class="grid">
                <div class="card">
                    <div class="card-header">
                        <p class="eyebrow">"Transcript"</p>
                    </div>
                    <div class="transcription-body">
                        <p class="muted-label">"Preview"</p>
                        <p class="result-text">
                            {move || {
                                if transcription.get().is_empty() {
                                    "Your transcription will appear here.".to_string()
                                } else {
                                    transcription.get()
                                }
                            }}
                        </p>
                    </div>
                </div>

                <div class="card settings-card">
                    <div class="card-header">
                        <p class="eyebrow">"Settings"</p>
                    </div>
                    <div class="settings-section">
                        <div class="settings-row">
                            <div class="settings-label">
                                <span class="settings-title">"Shortcut"</span>
                                <span class="settings-hint">"Global hotkey to start and stop recording"</span>
                            </div>
                            <div class="settings-input-group">
                                <input
                                    id="shortcut-input"
                                    type="text"
                                    class="settings-input"
                                    value=move || shortcut.get()
                                    on:input=move |ev| set_shortcut.set(input_value(&ev))
                                    placeholder="e.g. Alt+Z"
                                />
                                <button class="ghost compact" on:click=save_shortcut_action>"Save"</button>
                            </div>
                        </div>
                        <p class="settings-status">{ move || shortcut_status.get() }</p>

                        <div class="settings-row">
                            <div class="settings-label">
                                <span class="settings-title">"Model Location"</span>
                                <code class="path-code">{ move || model_path.get() }</code>
                            </div>
                            <button class="ghost compact" on:click=change_path_action>
                                "Change"
                            </button>
                        </div>

                        <div class="settings-divider"></div>

                        <div class="settings-footer">
                            <button class="danger compact" on:click=reset_settings_action>
                                "Reset All Settings"
                            </button>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    }
}
