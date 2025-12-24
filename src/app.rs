use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{prelude::*, JsCast};
use crate::api::*;
use crate::components::recorder::RecorderSection;
use crate::components::settings::SettingsSection;
use crate::utils::apply_patch;

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
        let is_model_msg = current_status.is_empty()
            || current_status.starts_with("Preparing speech model")
            || current_status.starts_with("Downloading")
            || current_status.starts_with("Model download failed");
        if !is_model_msg {
            return;
        }

        spawn_local(async move {
            match fetch_download_progress().await {
                Ok(Some(progress)) => {
                    if let Some(error) = progress.error {
                        set_model_ready.set(false);
                        match fetch_model_path().await {
                            Ok(path) => set_model_error.set(Some(format!(
                                "Download failed: {}. Manual: {}",
                                error, path
                            ))),
                            Err(_) => set_model_error.set(Some(error)),
                        }
                        set_status.set("Model download failed.".to_string());
                    } else if progress.done {
                        set_model_ready.set(true);
                        set_model_error.set(None);
                        set_status.set("Tap to start recording.".to_string());
                    } else {
                        set_model_ready.set(false);
                        set_model_error.set(None);
                        let index = progress.file_index.min(progress.file_count);
                        let txt = if progress.total_bytes > 0 {
                            format!(
                                "Downloading ({:.1}/{:.1} MB, {}/{})",
                                progress.downloaded_bytes as f64 / 1e6,
                                progress.total_bytes as f64 / 1e6,
                                index,
                                progress.file_count
                            )
                        } else {
                            format!("Downloading file {}/{}...", index, progress.file_count)
                        };
                        set_status.set(txt);
                    }
                }
                Ok(None) => {
                    if let Ok(true) = check_model_ready_flag().await {
                        set_model_ready.set(true);
                        set_status.set("Tap to start recording.".to_string());
                    }
                }
                _ => {}
            }
        });
    }) as Box<dyn FnMut()>);

    if let Some(window) = leptos::web_sys::window() {
        let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            1_000,
        );
    }
    callback.forget();
}

#[component]
pub fn App() -> impl IntoView {
    let (transcription, set_transcription) = signal(String::new());
    let (transcribing, set_transcribing) = signal(false);
    let (is_recording, set_is_recording) = signal(false);
    let (status, set_status) = signal("Preparing speech model...".to_string());
    let (model_ready, set_model_ready) = signal(false);
    let (model_error, set_model_error) = signal::<Option<String>>(None);
    let (shortcut, set_shortcut) = signal(String::new());
    let (streaming_enabled, set_streaming_enabled) = signal(false);
    let (model_path, set_model_path) = signal(String::new());

    spawn_local(async move {
        let callback = Closure::wrap(Box::new(move |event: JsValue| {
            if let Ok(payload) = js_sys::Reflect::get(&event, &"payload".into()) {
                match serde_wasm_bindgen::from_value::<TranscriptionPatchDto>(payload) {
                    Ok(patch) => {
                        leptos::logging::log!(
                            "Applying patch: {:?} (current len: {})",
                            patch,
                            transcription.get_untracked().len()
                        );
                        set_transcription.update(|current| {
                            *current = apply_patch(current, &patch);
                        });
                    }
                    Err(e) => {
                        leptos::logging::error!("Failed to parse TranscriptionPatchDto: {:?}", e);
                    }
                }
            } else {
                leptos::logging::warn!("Event received but no payload found: {:?}", event);
            }
        }) as Box<dyn FnMut(JsValue)>);

        match listen("transcription_update", &callback).await {
            Ok(_) => leptos::logging::log!("Listening for transcription_update"),
            Err(e) => leptos::logging::error!("Failed to listen for transcription_update: {:?}", e),
        }
        callback.forget();
    });

    Effect::new(move |_| {
        start_model_progress_polling(
            is_recording,
            transcribing,
            status,
            set_status,
            set_model_ready,
            set_model_error,
        );
    });

    spawn_local(async move {
        if let Ok(true) = check_model_ready_flag().await {
            set_model_ready.set(true);
            set_status.set("Tap to start recording.".to_string());
        }

        let current = fetch_current_shortcut().await.ok().flatten();
        if let Some(s) = current {
            set_shortcut.set(s);
        } else if let Ok(Some(s)) = fetch_default_shortcut().await {
            set_shortcut.set(s);
        }

        if let Ok(path) = fetch_model_path().await {
            set_model_path.set(path);
        }
        if let Ok(enabled) = fetch_streaming_enabled().await {
            set_streaming_enabled.set(enabled);
        }
    });

    view! {
        <main class="shell">
            <header class="hero">
                <p class="eyebrow">"Local-first capture"</p>
                <h1>"Silent Keys"</h1>
            </header>

            <RecorderSection
                is_recording transcribing status set_status model_ready
                set_model_error model_error set_transcription streaming_enabled
                set_is_recording set_transcribing
            />

            <section class="grid">
                <div class="card">
                    <div class="card-header"><p class="eyebrow">"Transcript"</p></div>
                    <div class="transcription-body">
                        <p class="result-text">
                            {move || if transcription.get().is_empty() { "Your transcription will appear here.".to_string() } else { transcription.get() }}
                        </p>
                    </div>
                </div>

                <div class="card settings-card">
                    <div class="card-header"><p class="eyebrow">"Settings"</p></div>
                    <SettingsSection
                        model_path set_model_path streaming_enabled set_streaming_enabled
                        shortcut set_shortcut set_status
                    />
                </div>
            </section>
        </main>
    }
}
