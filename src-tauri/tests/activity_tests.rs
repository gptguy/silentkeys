use silent_keys_lib::activity::{try_begin, ActivityError, AppActivity};

#[test]
fn activity_guards_are_exclusive_and_release_on_drop() {
    let recording = try_begin(AppActivity::Recording).expect("recording should reserve activity");
    assert!(matches!(
        try_begin(AppActivity::Updating),
        Err(ActivityError::Busy(AppActivity::Recording))
    ));

    drop(recording);
    let updating = try_begin(AppActivity::Updating).expect("dropped guard should release activity");
    assert!(matches!(
        try_begin(AppActivity::Configuring),
        Err(ActivityError::Busy(AppActivity::Updating))
    ));

    drop(updating);
    assert!(try_begin(AppActivity::Configuring).is_ok());
}
