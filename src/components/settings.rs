use crate::api::*;
use leptos::prelude::*;
use leptos::task::spawn_local;

#[component]
pub fn SettingsSection(
    model_path: ReadSignal<String>,
    set_model_path: WriteSignal<String>,
    streaming_enabled: ReadSignal<bool>,
    set_streaming_enabled: WriteSignal<bool>,
    shortcut: ReadSignal<String>,
    set_shortcut: WriteSignal<String>,
    set_status: WriteSignal<String>,
) -> impl IntoView {
    let (shortcut_status, set_shortcut_status) = signal(String::new());

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
                    set_status.set("Settings reset.".to_string());
                }
                Err(e) => set_status.set(format!("Reset failed: {}", e)),
            }
        });
    };

    view! {
        <div class="settings-section">
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
                        on:input=move |ev| set_shortcut.set(crate::utils::input_value(&ev))
                    />
                    <button class="ghost compact" on:click=save_shortcut_action>"Save"</button>
                </div>
            </div>
            <div class="settings-row">
                <div class="settings-label">
                    <span class="settings-title">"Model Location"</span>
                    <code class="path-code">{ move || model_path.get() }</code>
                </div>
                <button class="ghost compact" on:click=change_path_action>"Change"</button>
            </div>
            <div class="settings-divider"></div>
            <div class="settings-footer">
                <button class="danger compact" on:click=reset_settings_action>"Reset All Settings"</button>
            </div>
        </div>
    }
}
