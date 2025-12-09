use silent_keys_lib::desktop::*;

#[test]
fn default_shortcut_is_valid() {
    let shortcut = default_shortcut();
    let shortcut_str = shortcut.into_string();
    assert!(shortcut_str.contains("alt") || shortcut_str.contains("Alt"));
    assert!(shortcut_str.contains("KeyZ"));
}

#[test]
fn parse_valid_shortcut() {
    let result = parse_shortcut_str("Alt+KeyZ");
    assert!(result.is_ok());
}

#[test]
fn parse_shortcut_with_modifier() {
    let result = parse_shortcut_str("Control+Shift+KeyS");
    assert!(result.is_ok());
}

#[test]
fn parse_invalid_shortcut() {
    let result = parse_shortcut_str("InvalidShortcut");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid"));
}

#[test]
fn parse_empty_shortcut() {
    let result = parse_shortcut_str("");
    assert!(result.is_err());
}

#[test]
fn default_record_shortcut_returns_string() {
    let shortcut = default_record_shortcut();
    assert!(!shortcut.is_empty());
    assert!(shortcut.contains("Alt") || shortcut.contains("Key"));
}
