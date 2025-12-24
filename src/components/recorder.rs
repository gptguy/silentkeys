use crate::api::*;
use leptos::prelude::*;
use leptos::task::spawn_local;

#[component]
pub fn RecorderSection(
    is_recording: ReadSignal<bool>,
    set_is_recording: WriteSignal<bool>,
    transcribing: ReadSignal<bool>,
    set_transcribing: WriteSignal<bool>,
    status: ReadSignal<String>,
    set_status: WriteSignal<String>,
    model_ready: ReadSignal<bool>,
    set_model_error: WriteSignal<Option<String>>,
    model_error: ReadSignal<Option<String>>,
    set_transcription: WriteSignal<String>,
    streaming_enabled: ReadSignal<bool>,
) -> impl IntoView {
    let toggle_recording = move |_| {
        if !model_ready.get() {
            if model_error.get().is_some() {
                set_status.set("Retrying speech model download...".to_string());
                spawn_local(async move {
                    match retry_model_download_cmd().await {
                        Ok(_) => set_model_error.set(None),
                        Err(err) => {
                            set_model_error.set(Some(err.clone()));
                            set_status.set(format!("Could not download model: {}", err));
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
            set_transcription.set(String::new());
            spawn_local(async move {
                match start_recording_cmd().await {
                    Ok(_) => set_status.set("Recording... tap to stop.".to_string()),
                    Err(msg) => {
                        set_is_recording.set(false);
                        set_status.set(format!("Could not start recording: {}", msg));
                    }
                }
            });
            return;
        }

        spawn_local(async move {
            set_status.set("Stopping recording...".to_string());
            set_transcribing.set(true);
            let is_streaming = streaming_enabled.get_untracked();
            if !is_streaming {
                set_transcription.set(String::new());
            }

            match stop_recording_cmd().await {
                Ok(text) => {
                    set_is_recording.set(false);
                    if !is_streaming {
                        set_transcription.set(text);
                    }
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

    view! {
        <section class="card control-card">
            <div class="card-header">
                <div>
                    <p class="eyebrow">"Recorder"</p>
                    <h2>"One tap to capture"</h2>
                </div>
                <span class="pill"
                    class:live=move || is_recording.get()
                    class:glow=move || transcribing.get()
                    class:idle=move || !is_recording.get() && !transcribing.get()
                >
                    {move || if is_recording.get() { "Listening" } else if transcribing.get() { "Transcribing" } else { "Idle" }}
                </span>
            </div>
            <div class="control-row">
                <button
                    on:click=toggle_recording
                    disabled=move || transcribing.get() || (!model_ready.get() && model_error.get().is_none())
                >
                    {move || {
                        if !model_ready.get() {
                            if model_error.get().is_some() { "Retry model download" } else { "Preparing..." }
                        } else if transcribing.get() { "Working..." }
                        else if is_recording.get() { "Finish recording" }
                        else { "Start recording" }
                    }}
                </button>
                <div class="status-container">
                    <p class="inline-status">{ move || status.get() }</p>
                    {move || model_error.get().map(|err| view! {
                         <div class="error-details">
                            <p class="error-msg">{err}</p>
                            <code class="cmd-block">"hf download istupakov/parakeet-tdt-0.6b-v3-onnx"</code>
                         </div>
                    })}
                </div>
            </div>
        </section>
    }
}
