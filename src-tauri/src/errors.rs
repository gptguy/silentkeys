/// Maps internal failures to short, actionable text that is safe to show in the UI.
pub trait UserFacing {
    fn user_message(&self) -> &'static str;
}
