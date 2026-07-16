use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppActivity {
    Recording,
    Updating,
    Configuring,
}

#[derive(thiserror::Error, Debug)]
pub enum ActivityError {
    #[error("app is already busy with {0:?}")]
    Busy(AppActivity),
    #[error("app activity lock failed")]
    LockFailed,
}

pub struct ActivityGuard {
    activity: AppActivity,
}

static ACTIVE_ACTIVITY: OnceLock<Mutex<Option<AppActivity>>> = OnceLock::new();

fn state() -> &'static Mutex<Option<AppActivity>> {
    ACTIVE_ACTIVITY.get_or_init(|| Mutex::new(None))
}

pub fn try_begin(activity: AppActivity) -> Result<ActivityGuard, ActivityError> {
    let mut active = state().lock().map_err(|_| ActivityError::LockFailed)?;
    if let Some(current) = *active {
        return Err(ActivityError::Busy(current));
    }

    *active = Some(activity);
    Ok(ActivityGuard { activity })
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = state().lock() {
            if *active == Some(self.activity) {
                *active = None;
            }
        }
    }
}
