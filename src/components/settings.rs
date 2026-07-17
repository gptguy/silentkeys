use crate::api::*;
use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos::web_sys::{HtmlInputElement, HtmlSelectElement};
use wasm_bindgen::JsCast;

fn input_value(event: &leptos::ev::Event) -> String {
    event
        .target()
        .and_then(|target| target.dyn_into::<HtmlInputElement>().ok())
        .map(|input| input.value())
        .unwrap_or_default()
}

fn select_value(event: &leptos::ev::Event) -> String {
    event
        .target()
        .and_then(|target| target.dyn_into::<HtmlSelectElement>().ok())
        .map(|select| select.value())
        .unwrap_or_default()
}

fn language_label(language: &str) -> &str {
    match language {
        "en-US" => "English (US)",
        _ => language,
    }
}

async fn refresh_update_status(
    set_status: WriteSignal<String>,
    set_available: WriteSignal<Option<AppUpdateInfoDto>>,
) {
    match check_for_app_update_cmd().await {
        Ok(Some(update)) => {
            set_status.set(format!("Version {} is available.", update.version));
            set_available.set(Some(update));
        }
        Ok(None) => {
            set_available.set(None);
            set_status.set("SilentKeys is up to date.".to_string());
        }
        Err(error) => {
            set_available.set(None);
            set_status.set(format!("Update check failed: {error}"));
        }
    }
}

#[component]
pub fn SettingsSection(
    model_path: ReadSignal<String>,
    set_model_path: WriteSignal<String>,
    streaming_enabled: ReadSignal<bool>,
    set_streaming_enabled: WriteSignal<bool>,
    shortcut: ReadSignal<String>,
    set_shortcut: WriteSignal<String>,
    asr_language: ReadSignal<String>,
    set_asr_language: WriteSignal<String>,
    language_options: ReadSignal<Vec<String>>,
    is_recording: ReadSignal<bool>,
    transcribing: ReadSignal<bool>,
    set_status: WriteSignal<String>,
) -> impl IntoView {
    let (shortcut_status, set_shortcut_status) = signal(String::new());
    let (language_status, set_language_status) = signal(String::new());
    let (update_status, set_update_status) = signal("Checking for updates...".to_string());
    let (available_update, set_available_update) = signal::<Option<AppUpdateInfoDto>>(None);

    spawn_local(refresh_update_status(
        set_update_status,
        set_available_update,
    ));

    let change_path_action = move |_| {
        spawn_local(async move {
            match pick_model_folder_cmd().await {
                Ok(Some(path)) => match save_model_path(path.clone()).await {
                    Ok(_) => {
                        set_model_path.set(path);
                        set_status.set("Model path updated.".to_string());
                    }
                    Err(e) => set_status.set(format!("Failed to save path: {}", e)),
                },
                Ok(None) => {}
                Err(e) => set_status.set(format!("Failed to pick folder: {}", e)),
            }
        });
    };

    let save_shortcut_action = move |_| {
        let val = shortcut.get();
        set_shortcut_status.set("Saving shortcut...".to_string());
        spawn_local(async move {
            match save_shortcut(&val).await {
                Ok(saved) => set_shortcut_status.set(format!("Shortcut saved: {}", saved)),
                Err(err) => set_shortcut_status.set(format!("Failed to save shortcut: {}", err)),
            }
        });
    };

    let reset_settings_action = move |_| {
        spawn_local(async move {
            match reset_settings_cmd().await {
                Ok(_) => {
                    if let Ok(path) = fetch_model_path().await {
                        set_model_path.set(path);
                    }
                    if let Ok(Some(s)) = fetch_default_shortcut().await {
                        set_shortcut.set(s);
                    }
                    if let Ok(enabled) = fetch_streaming_enabled().await {
                        set_streaming_enabled.set(enabled);
                    }
                    if let Ok(language) = fetch_asr_language().await {
                        set_asr_language.set(language);
                    }
                    set_status.set("Settings reset.".to_string());
                }
                Err(e) => set_status.set(format!("Reset failed: {}", e)),
            }
        });
    };

    let change_language_action = move |event: leptos::ev::Event| {
        let language = select_value(&event);
        let previous = asr_language.get_untracked();
        if language == previous {
            return;
        }
        set_asr_language.set(language.clone());
        set_language_status.set("Applying...".to_string());
        spawn_local(async move {
            match save_asr_language(language).await {
                Ok(()) => set_language_status.set("Applied".to_string()),
                Err(error) => {
                    set_asr_language.set(previous);
                    set_language_status.set(error);
                }
            }
        });
    };

    let check_update_action = move |_| {
        set_update_status.set("Checking for updates...".to_string());
        spawn_local(refresh_update_status(
            set_update_status,
            set_available_update,
        ));
    };

    let install_update_action = move |_| {
        set_update_status.set("Installing update...".to_string());
        spawn_local(async move {
            match install_app_update_cmd().await {
                Ok(true) => set_update_status.set("Update installed. Restarting...".to_string()),
                Ok(false) => {
                    set_available_update.set(None);
                    set_update_status.set("SilentKeys is up to date.".to_string());
                }
                Err(err) => set_update_status.set(format!("Update install failed: {}", err)),
            }
        });
    };

    let update_details = move |update: &AppUpdateInfoDto| {
        let mut details = format!(
            "Current: {} - Latest: {}",
            update.current_version, update.version
        );
        if let Some(date) = update.date.as_deref() {
            details.push_str(&format!(" - Released: {}", date));
        }
        if let Some(body) = update.body.as_deref().filter(|body| !body.is_empty()) {
            details.push_str(&format!(" - {}", body));
        }
        details
    };

    view! {
        <div class="settings-section">
            <div class="settings-row">
                <div class="settings-label">
                    <span class="settings-title">"Speech Language"</span>
                    <span class="settings-hint">
                        {move || if language_status.get().is_empty() {
                            "English (US) avoids language detection.".to_string()
                        } else {
                            language_status.get()
                        }}
                    </span>
                </div>
                <select
                    class="settings-input settings-select"
                    prop:value=move || asr_language.get()
                    disabled=move || language_options.get().is_empty()
                        || is_recording.get()
                        || transcribing.get()
                    on:change=change_language_action
                >
                    <option value="system">"System language"</option>
                    <option value="auto">"Automatic detection"</option>
                    <For
                        each=move || language_options.get()
                        key=|language| language.clone()
                        children=move |language| {
                            let label = language_label(&language).to_string();
                            view! { <option value=language>{label}</option> }
                        }
                    />
                </select>
            </div>
            <div class="settings-row">
                <div class="settings-label">
                    <span class="settings-title">"Streaming Mode"</span>
                    <span class="settings-hint">"Show transcription in real-time"</span>
                </div>
                <button
                    class="toggle"
                    class:active=move || streaming_enabled.get()
                    on:click=move |_| {
                        let new_val = !streaming_enabled.get();
                        set_streaming_enabled.set(new_val);
                        spawn_local(async move { let _ = save_streaming_enabled(new_val).await; });
                    }
                >
                    <div class="toggle-track"><div class="toggle-thumb"></div></div>
                </button>
            </div>
            <div class="settings-row">
                <div class="settings-label">
                    <span class="settings-title">"Shortcut"</span>
                    <p class="settings-status">{ move || shortcut_status.get() }</p>
                </div>
                <div class="settings-input-group">
                    <input
                        type="text"
                        class="settings-input"
                        value=move || shortcut.get()
                        disabled=move || is_recording.get() || transcribing.get()
                        on:input=move |event| set_shortcut.set(input_value(&event))
                    />
                    <button
                        class="ghost compact"
                        disabled=move || is_recording.get() || transcribing.get()
                        on:click=save_shortcut_action
                    >
                        "Save"
                    </button>
                </div>
            </div>
            <div class="settings-row">
                <div class="settings-label">
                    <span class="settings-title">"Model Location"</span>
                    <code class="path-code">{ move || model_path.get() }</code>
                </div>
                <button class="ghost compact" on:click=change_path_action>"Change"</button>
            </div>
            <div class="settings-row">
                <div class="settings-label">
                    <span class="settings-title">"Updates"</span>
                    <span class="settings-hint">{ move || update_status.get() }</span>
                    {move || available_update.get().map(|update| view! {
                        <p class="settings-status">
                            {update_details(&update)}
                        </p>
                    })}
                </div>
                <div class="settings-input-group">
                    <button class="ghost compact" on:click=check_update_action>"Check"</button>
                    <button
                        class="ghost compact"
                        disabled=move || available_update.get().is_none()
                        on:click=install_update_action
                    >
                        "Install"
                    </button>
                </div>
            </div>
            <div class="settings-divider"></div>
            <div class="settings-footer">
                <button class="danger compact" on:click=reset_settings_action>"Reset All Settings"</button>
            </div>
        </div>
    }
}
