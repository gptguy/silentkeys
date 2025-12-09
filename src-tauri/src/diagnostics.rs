use cpal::traits::{DeviceTrait, HostTrait};
use ort::session::Session;

pub fn run_startup_checks() {
    log::info!("=== Startup Diagnostics ===");
    check_audio_devices();
    check_ort_environment();
    log::info!("===========================");
}

fn check_audio_devices() {
    let host = cpal::default_host();
    log::info!("Audio Host: {:?}", host.id());

    match host.default_input_device() {
        Some(device) => {
            log::info!(
                "Default Input Device: {}",
                device.name().unwrap_or_default()
            );
            if let Ok(config) = device.default_input_config() {
                log::info!("  Default Config: {:?}", config);
            }
        }
        None => log::error!("No default input device found!"),
    }

    log::info!("Available Input Devices:");
    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            log::info!("  {}. {}", i + 1, device.name().unwrap_or_default());
        }
    } else {
        log::error!("Failed to list input devices.");
    }
}

fn check_ort_environment() {
    log::info!("Checking ONNX Runtime environment...");
    match Session::builder() {
        Ok(_) => {
            log::info!("ORT Session Builder initialized successfully (Shared libs likely present).")
        }
        Err(e) => log::error!(
            "ORT Initialization failed: {}. Missing shared libraries (onnxruntime.dll/so)?",
            e
        ),
    }
}
