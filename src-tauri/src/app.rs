use std::time::Instant;

use tauri::{App, AppHandle, Builder, Emitter, Manager, RunEvent, WindowEvent};

use crate::commands;
#[cfg(desktop)]
use crate::desktop;
use crate::engine::SpeechEngine;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let context = tauri::generate_context!();

    let app = Builder::default()
        .plugin(tauri_plugin_single_instance::init(on_second_instance))
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir {
                        file_name: None,
                    }),
                ])
                .rotation_strategy(tauri_plugin_log::RotationStrategy::KeepAll)
                .max_file_size(2_000_000)
                .timezone_strategy(tauri_plugin_log::TimezoneStrategy::UseLocal)
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            app.manage(SpeechEngine::new(handle));
            setup(app)
        })
        .on_window_event(handle_window_event)
        .invoke_handler(tauri::generate_handler![
            commands::is_model_ready,
            commands::model_download_progress,
            commands::retry_model_download,
            commands::get_model_path,
            commands::set_model_path,
            commands::pick_model_folder,
            commands::start_recording,
            commands::stop_recording,
            commands::update_record_shortcut,
            commands::get_record_shortcut,
            commands::default_record_shortcut,
            commands::get_use_streaming,
            commands::set_use_streaming,
            commands::reset_settings
        ])
        .build(context)
        .expect("error while running tauri application");

    app.run(handle_run_event);
}

fn handle_window_event(window: &tauri::Window, event: &WindowEvent) {
    if let WindowEvent::CloseRequested { api, .. } = event {
        if let Err(err) = window.hide() {
            log::warn!("Failed to hide window on close request: {err}");
        }
        api.prevent_close();
    }
}

#[cfg(target_os = "macos")]
fn handle_run_event(app_handle: &AppHandle, event: RunEvent) {
    if let RunEvent::Reopen {
        has_visible_windows,
        ..
    } = event
    {
        if !has_visible_windows {
            if let Some(window) = app_handle.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn handle_run_event(_app_handle: &AppHandle, _event: RunEvent) {}

fn on_second_instance(app: &AppHandle, argv: Vec<String>, cwd: String) {
    log::info!("{}, {argv:?}, {cwd}", app.package_info().name);
    if let Err(err) = app.emit("single-instance", ()) {
        log::error!("Failed to emit single-instance event: {err}");
    }
}

fn setup(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(desktop)]
    {
        desktop::setup_desktop(app)?;
        prewarm_model(app.handle().clone());
    }

    Ok(())
}

#[cfg(desktop)]
fn prewarm_model(app_handle: AppHandle) {
    std::thread::spawn(move || {
        let start = Instant::now();
        let state = app_handle.state::<SpeechEngine>();
        let result = state.ensure_model_loaded();
        let elapsed = start.elapsed();

        match result {
            Ok(()) => {
                log::info!("ASR model pre-warmed successfully in {:?}", elapsed);
            }
            Err(err) => {
                log::error!("Failed to pre-warm ASR model after {:?}: {}", elapsed, err);
            }
        }
    });
}
