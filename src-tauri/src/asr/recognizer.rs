use std::{collections::HashSet, fmt, path::Path, time::Instant};

use parakeet_rs::{Nemotron, NemotronMode};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::errors::UserFacing;
use crate::settings::DEFAULT_ASR_LANGUAGE;

pub(crate) const STREAM_CHUNK_SAMPLES: usize = 8_960;
const STREAM_FLUSH_CHUNKS: usize = 3;
pub const AUTOMATIC_LANGUAGE: &str = "auto";
pub const SYSTEM_LANGUAGE: &str = "system";

#[derive(Deserialize)]
struct ModelConfig {
    prompt_dictionary: PromptDictionary,
}

struct PromptDictionary(Vec<(String, u16)>);

impl<'de> Deserialize<'de> for PromptDictionary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PromptVisitor;

        impl<'de> Visitor<'de> for PromptVisitor {
            type Value = PromptDictionary;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a language-to-prompt map")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(PromptDictionary(entries))
            }
        }

        deserializer.deserialize_map(PromptVisitor)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("Model download failed: {0}")]
    Download(String),
    #[error("Model integrity check failed: {0}")]
    Integrity(String),
    #[error("parse model configuration {path}: {source}")]
    Config {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("Unsupported speech language: {0}")]
    UnsupportedLanguage(String),
    #[error("{context}: {source}")]
    Nemotron {
        context: &'static str,
        #[source]
        source: parakeet_rs::Error,
    },
}

impl AsrError {
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    fn nemotron(context: &'static str, source: parakeet_rs::Error) -> Self {
        Self::Nemotron { context, source }
    }
}

impl UserFacing for AsrError {
    fn user_message(&self) -> &'static str {
        match self {
            Self::Download(_) => {
                "Could not download the speech model. Check your internet connection and try again."
            }
            Self::Integrity(_) | Self::Config { .. } => {
                "The speech model files are invalid. Download the model again."
            }
            Self::UnsupportedLanguage(_) => {
                "That language is not supported by the installed speech model."
            }
            Self::Nemotron { .. } => {
                "The speech engine failed to run. Try restarting the app or downloading the model again."
            }
            Self::Io { .. } => {
                "The app could not read or write its local files. Check disk space and permissions."
            }
        }
    }
}

pub struct AsrModel {
    model: Box<Nemotron>,
    languages: Vec<String>,
    accepted_languages: Vec<String>,
}

impl AsrModel {
    pub fn new(model_dir: impl AsRef<Path>, language_preference: &str) -> Result<Self, AsrError> {
        let start = Instant::now();
        let model_dir = model_dir.as_ref();
        let catalog = load_languages(model_dir)?;
        let mut model = Nemotron::from_pretrained(model_dir, None)
            .map_err(|error| AsrError::nemotron("load Nemotron model", error))?;
        if model.mode() == NemotronMode::Multilingual {
            let language = match apply_language(&mut model, &catalog.accepted, language_preference)
            {
                Ok(language) => language,
                Err(AsrError::UnsupportedLanguage(language)) => {
                    log::warn!(
                        "Stored speech language {language:?} is unsupported; using {DEFAULT_ASR_LANGUAGE}"
                    );
                    apply_language(&mut model, &catalog.accepted, DEFAULT_ASR_LANGUAGE)?
                }
                Err(error) => return Err(error),
            };
            log::info!("Nemotron ASR language hint: {language}");
        }
        log::info!("Nemotron ASR model initialized in {:?}", start.elapsed());
        Ok(Self {
            model: Box::new(model),
            languages: catalog.options,
            accepted_languages: catalog.accepted,
        })
    }

    pub fn languages(&self) -> &[String] {
        &self.languages
    }

    pub fn supports_language(&self, language: &str) -> bool {
        language == SYSTEM_LANGUAGE
            || language == AUTOMATIC_LANGUAGE
            || self.accepted_languages.iter().any(|item| item == language)
    }

    pub fn set_language(&mut self, language: &str) -> Result<String, AsrError> {
        let selected = apply_language(&mut self.model, &self.accepted_languages, language)?;
        self.model.reset();
        Ok(selected)
    }

    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        self.model.reset();
        self.model
            .transcribe_audio(samples)
            .map_err(|error| AsrError::nemotron("run offline transcription", error))
    }

    pub fn reset_state(&mut self) {
        self.model.reset();
    }

    pub(crate) fn advance_streaming(&mut self, samples: &[f32]) -> Result<String, AsrError> {
        let mut text = String::new();
        for chunk in samples.chunks(STREAM_CHUNK_SAMPLES) {
            text.push_str(
                &self
                    .model
                    .transcribe_chunk(chunk)
                    .map_err(|error| AsrError::nemotron("run streaming transcription", error))?,
            );
        }
        Ok(text)
    }

    pub(crate) fn finish_streaming(&mut self) -> Result<String, AsrError> {
        let silence = [0.0; STREAM_CHUNK_SAMPLES];
        let mut text = String::new();
        for _ in 0..STREAM_FLUSH_CHUNKS {
            text.push_str(
                &self
                    .model
                    .transcribe_chunk(&silence)
                    .map_err(|error| AsrError::nemotron("flush streaming transcription", error))?,
            );
        }
        Ok(text)
    }
}

fn load_languages(model_dir: &Path) -> Result<LanguageCatalog, AsrError> {
    let path = model_dir.join("config.json");
    let config = std::fs::read(&path)
        .map_err(|error| AsrError::io(format!("read model config {}", path.display()), error))?;
    let config: ModelConfig =
        serde_json::from_slice(&config).map_err(|source| AsrError::Config {
            path: path.display().to_string(),
            source,
        })?;
    Ok(language_catalog(config.prompt_dictionary.0))
}

fn apply_language(
    model: &mut Nemotron,
    languages: &[String],
    preference: &str,
) -> Result<String, AsrError> {
    if preference == SYSTEM_LANGUAGE {
        let mut last_error = None;
        for language in language_candidates_for_tests(&sys_locale::get_locale().unwrap_or_default())
        {
            match model.set_target_lang(&language) {
                Ok(()) => return Ok(language),
                Err(error) => last_error = Some(error),
            }
        }
        return Err(AsrError::nemotron(
            "configure system speech language",
            last_error.ok_or_else(|| AsrError::UnsupportedLanguage(preference.to_string()))?,
        ));
    }

    if preference != AUTOMATIC_LANGUAGE && !languages.iter().any(|item| item == preference) {
        return Err(AsrError::UnsupportedLanguage(preference.to_string()));
    }
    model
        .set_target_lang(preference)
        .map_err(|error| AsrError::nemotron("configure speech language", error))?;
    Ok(preference.to_string())
}

#[doc(hidden)]
pub fn language_candidates_for_tests(locale: &str) -> Vec<String> {
    let exact = locale
        .split(['.', '@'])
        .next()
        .unwrap_or_default()
        .replace('_', "-");
    let base = exact.split('-').next().unwrap_or_default();
    let mut candidates = Vec::with_capacity(3);
    if !exact.is_empty() {
        candidates.push(exact.clone());
    }
    if !base.is_empty() && base != exact {
        candidates.push(base.to_string());
    }
    candidates.push("auto".to_string());
    candidates
}

struct LanguageCatalog {
    accepted: Vec<String>,
    options: Vec<String>,
}

fn language_catalog(entries: Vec<(String, u16)>) -> LanguageCatalog {
    let mut prompts = HashSet::new();
    let mut accepted = Vec::with_capacity(entries.len());
    let mut options = Vec::new();
    for (language, prompt) in entries {
        if language == AUTOMATIC_LANGUAGE {
            continue;
        }
        if prompts.insert(prompt) {
            options.push(language.clone());
        }
        accepted.push(language);
    }
    accepted.sort();
    options.sort();
    LanguageCatalog { accepted, options }
}

#[doc(hidden)]
pub fn language_options_for_tests(entries: &[(&str, u16)]) -> Vec<String> {
    language_catalog(
        entries
            .iter()
            .map(|(language, prompt)| ((*language).to_string(), *prompt))
            .collect(),
    )
    .options
}
