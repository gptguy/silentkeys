use silent_keys_lib::errors::UserFacing;
use silent_keys_lib::updater::{begin_exclusive_update_for_tests, AppUpdateError};

#[test]
fn updater_gate_is_exclusive_and_releases_on_drop() {
    let update = begin_exclusive_update_for_tests().expect("first update should reserve the gate");
    let conflict = match begin_exclusive_update_for_tests() {
        Ok(_) => panic!("second update should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(conflict, AppUpdateError::InstallInProgress));
    assert_eq!(
        conflict.user_message(),
        "An update installation is already running."
    );

    drop(update);
    assert!(begin_exclusive_update_for_tests().is_ok());

    assert_eq!(
        AppUpdateError::RecordingInProgress.user_message(),
        "Finish recording before installing an update."
    );
    assert_eq!(
        AppUpdateError::SettingsInProgress.user_message(),
        "Speech settings are being changed. Please try the update again."
    );
}
