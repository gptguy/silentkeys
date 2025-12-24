use crate::api::TranscriptionPatchDto;
use leptos::web_sys::HtmlInputElement;
use wasm_bindgen::JsCast;

pub fn input_value(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
        .map(|input| input.value())
        .unwrap_or_default()
}

pub fn slice_chars(text: &str, start: usize, end: usize) -> String {
    let total = text.chars().count();
    if start >= total || start >= end {
        return String::new();
    }
    let clamped_end = end.min(total);
    text.chars().skip(start).take(clamped_end - start).collect()
}

pub fn apply_patch(current: &str, patch: &TranscriptionPatchDto) -> String {
    let start = patch.start.min(patch.end) as usize;
    let end = patch.end.max(patch.start) as usize;
    let prefix = slice_chars(current, 0, start);
    let suffix = slice_chars(current, end, usize::MAX);
    format!("{prefix}{}{suffix}", patch.text)
}
