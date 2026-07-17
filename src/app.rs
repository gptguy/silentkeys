use crate::api::*;
use crate::components::recorder::RecorderSection;
use crate::components::settings::SettingsSection;
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

fn download_status_text(progress: &ModelDownloadProgressDto) -> String {
    let index = progress.file_index.min(progress.file_count);
    if progress.total_bytes > 0 {
        format!(
            "Downloading ({:.1}/{:.1} MB, {}/{})",
            progress.downloaded_bytes as f64 / 1e6,
            progress.total_bytes as f64 / 1e6,
            index,
            progress.file_count
        )
    } else {
        format!("Downloading file {}/{}...", index, progress.file_count)
    }
}

#[derive(Clone, Copy)]
struct ModelView {
    set_status: WriteSignal<String>,
    set_model_ready: WriteSignal<bool>,
    set_model_error: WriteSignal<Option<String>>,
    set_language_options: WriteSignal<Vec<String>>,
}

impl ModelView {
    fn apply_state(self, state: EngineStateDto) {
        match state {
            EngineStateDto::Loaded => {
                self.set_model_ready.set(true);
                self.set_model_error.set(None);
                self.set_status.set("Tap to start recording.".to_string());
                spawn_local(async move {
                    if let Ok(languages) = fetch_asr_languages().await {
                        self.set_language_options.set(languages);
                    }
                });
            }
            EngineStateDto::Failed(error) => {
                self.set_model_ready.set(false);
                self.set_model_error.set(Some(error));
                self.set_status.set("Model download failed.".to_string());
            }
            EngineStateDto::Loading | EngineStateDto::Unloaded => {
                self.set_model_ready.set(false);
                self.set_model_error.set(None);
                self.set_status.set("Preparing speech model...".to_string());
            }
        }
    }

    fn apply_progress(self, progress: ModelDownloadProgressDto) {
        if !progress.done {
            self.set_status.set(download_status_text(&progress));
        }
    }
}

fn start_model_event_listeners(model_view: ModelView) {
    spawn_local(async move {
        let callback = Closure::wrap(Box::new(move |event: JsValue| {
            let Ok(payload) = js_sys::Reflect::get(&event, &"payload".into()) else {
                return;
            };
            match serde_wasm_bindgen::from_value::<EngineStateDto>(payload) {
                Ok(state) => model_view.apply_state(state),
                Err(error) => {
                    leptos::logging::error!("Failed to parse engine state: {:?}", error)
                }
            }
        }) as Box<dyn FnMut(JsValue)>);

        if let Err(error) = listen("engine_state", &callback).await {
            leptos::logging::error!("Failed to listen for engine state: {:?}", error);
        }
        callback.forget();

        if let Ok(state) = fetch_engine_state().await {
            model_view.apply_state(state);
        }
    });

    spawn_local(async move {
        let callback = Closure::wrap(Box::new(move |event: JsValue| {
            let Ok(payload) = js_sys::Reflect::get(&event, &"payload".into()) else {
                return;
            };
            match serde_wasm_bindgen::from_value::<ModelDownloadProgressDto>(payload) {
                Ok(progress) => model_view.apply_progress(progress),
                Err(error) => {
                    leptos::logging::error!("Failed to parse model progress: {:?}", error)
                }
            }
        }) as Box<dyn FnMut(JsValue)>);

        if let Err(error) = listen("model_download_progress", &callback).await {
            leptos::logging::error!("Failed to listen for model progress: {:?}", error);
        }
        callback.forget();
    });
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
    let (asr_language, set_asr_language) = signal("en-US".to_string());
    let (language_options, set_language_options) = signal(Vec::<String>::new());

    spawn_local(async move {
        let callback = Closure::wrap(Box::new(move |event: JsValue| {
            if let Ok(payload) = js_sys::Reflect::get(&event, &"payload".into()) {
                match serde_wasm_bindgen::from_value::<TranscriptionUpdateDto>(payload) {
                    Ok(update) => {
                        set_transcription.update(|current| match update {
                            TranscriptionUpdateDto::Append(text) => current.push_str(&text),
                            TranscriptionUpdateDto::Replace(text) => *current = text,
                        });
                    }
                    Err(e) => {
                        leptos::logging::error!("Failed to parse transcription update: {:?}", e);
                    }
                }
            } else {
                leptos::logging::warn!("Transcription event received without a payload");
            }
        }) as Box<dyn FnMut(JsValue)>);

        match listen("transcription_update", &callback).await {
            Ok(_) => leptos::logging::log!("Listening for transcription_update"),
            Err(e) => leptos::logging::error!("Failed to listen for transcription_update: {:?}", e),
        }
        callback.forget();
    });

    spawn_local(async move {
        let callback = Closure::wrap(Box::new(move |event: JsValue| {
            if let Some(message) = js_sys::Reflect::get(&event, &"payload".into())
                .ok()
                .and_then(|payload| payload.as_string())
            {
                set_status.set(message);
            }
        }) as Box<dyn FnMut(JsValue)>);

        if let Err(e) = listen("dictation_error", &callback).await {
            leptos::logging::error!("Failed to listen for dictation_error: {:?}", e);
        }
        callback.forget();
    });

    start_model_event_listeners(ModelView {
        set_status,
        set_model_ready,
        set_model_error,
        set_language_options,
    });

    spawn_local(async move {
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
        if let Ok(language) = fetch_asr_language().await {
            set_asr_language.set(language);
        }
    });

    view! {
        <main class="shell">
            <header class="hero">
                <p class="eyebrow">"Local-first capture"</p>
                <h1>"SilentKeys"</h1>
            </header>

            <RecorderSection
                is_recording transcribing status set_status model_ready
                set_model_error model_error set_transcription
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
                        shortcut set_shortcut asr_language set_asr_language language_options
                        is_recording transcribing set_status
                    />
                </div>
            </section>
        </main>
    }
}
