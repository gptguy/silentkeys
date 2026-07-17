mod shortcuts;
mod tray;
mod typing;

pub use shortcuts::{
    default_record_shortcut, default_shortcut, get_record_shortcut, parse_shortcut_str,
    update_record_shortcut,
};
#[doc(hidden)]
pub use typing::{append_for_tests, deliver_for_tests, plan_final_delivery, FinalDelivery};

#[cfg(desktop)]
pub fn setup_desktop(app: &mut tauri::App) -> tauri::Result<()> {
    let handle = app.handle();
    tray::init_tray(handle)?;
    shortcuts::init_shortcuts(handle)?;
    Ok(())
}
