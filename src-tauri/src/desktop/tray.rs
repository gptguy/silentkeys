use std::path::PathBuf;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

const MENU_ITEM_QUIT: &str = "quit";
const MENU_ITEM_VIEW_LOGS: &str = "view_logs";

pub(super) fn init_tray(app: &AppHandle) -> tauri::Result<()> {
    let quit = MenuItem::with_id(app, MENU_ITEM_QUIT, "Quit", true, None::<&str>)?;
    let view_logs = MenuItem::with_id(
        app,
        MENU_ITEM_VIEW_LOGS,
        "View Log File",
        true,
        None::<&str>,
    )?;
    let menu = Menu::with_items(app, &[&view_logs, &quit])?;

    let mut tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, event| match event.id.as_ref() {
            MENU_ITEM_QUIT => {
                log::info!("Quit menu item clicked");
                app.exit(0);
            }
            MENU_ITEM_VIEW_LOGS => {
                log::info!("View Log File menu item clicked");
                if let Err(e) = open_log_file(app) {
                    log::error!("Failed to open log file: {}", e);
                    let _ = app
                        .dialog()
                        .message(format!("Failed to open log file: {}", e))
                        .kind(MessageDialogKind::Error)
                        .title("Warning")
                        .blocking_show();
                }
            }
            _ => log::debug!("Unhandled menu item: {:?}", event.id),
        });

    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    } else {
        log::warn!("Default window icon missing; using system tray default.");
    }

    tray.build(app)?;
    Ok(())
}

fn open_log_file(app: &AppHandle) -> Result<(), String> {
    let log_dir = app.path().app_log_dir().map_err(|e| e.to_string())?;
    if !log_dir.exists() {
        return Err("Log directory does not exist".to_string());
    }

    let log_file = newest_log_file(&log_dir).unwrap_or(log_dir);

    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";

    std::process::Command::new(cmd)
        .arg(log_file)
        .spawn()
        .map_err(|e| e.to_string())?;

    Ok(())
}

fn newest_log_file(log_dir: &PathBuf) -> Option<PathBuf> {
    let mut logs: Vec<_> = std::fs::read_dir(log_dir)
        .ok()?
        .flatten()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("log"))
                .unwrap_or(false)
        })
        .collect();

    logs.sort_by_key(|entry| {
        entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    logs.last().map(|entry| entry.path())
}
