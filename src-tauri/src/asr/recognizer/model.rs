use super::config::{AsrError, Transcript};
use crate::asr::FRAME_DURATION_SEC;
use num_cpus::get_physical;
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use regex::Regex;
use std::{fs, path::Path, sync::LazyLock, time::Instant};

const THREAD_ENV: &str = "ORT_THREADS";

fn resolve_thread_count() -> usize {
    std::env::var(THREAD_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(get_physical)
}

pub struct AsrModel {
    pub(super) encoder: Session,
    pub(super) decoder_joint: Session,
    pub(super) preprocessor: Session,
    pub vocab: Vec<String>,
    pub blank_idx: i32,
    pub vocab_size: usize,
    pub(super) cached_workspace: Option<crate::asr::decoder::DecoderWorkspace>,
    pub(super) cached_last_token: i32,
}

impl Drop for AsrModel {
    fn drop(&mut self) {
        log::debug!("Dropping ASR model");
    }
}

impl AsrModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, quantized: bool) -> Result<Self, AsrError> {
        let start = Instant::now();
        let threads = resolve_thread_count();
        let encoder = Self::init_session(&model_dir, "encoder-model", threads, quantized)?;
        let decoder_joint =
            Self::init_session(&model_dir, "decoder_joint-model", threads, quantized)?;
        let preprocessor = Self::init_session(&model_dir, "nemo128", threads, false)?;
        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;
        let vocab_size = vocab.len();

        log::info!("ASR model initialized in {:?}", start.elapsed());
        Ok(Self {
            encoder,
            decoder_joint,
            preprocessor,
            vocab,
            blank_idx,
            vocab_size,
            cached_workspace: None,
            cached_last_token: blank_idx,
        })
    }

    pub(crate) fn decoder_session(&self) -> &Session {
        &self.decoder_joint
    }
    pub(crate) fn decoder_session_mut(&mut self) -> &mut Session {
        &mut self.decoder_joint
    }

    fn init_session<P: AsRef<Path>>(
        dir: P,
        name: &str,
        threads: usize,
        try_q: bool,
    ) -> Result<Session, AsrError> {
        let mut file = format!("{name}.onnx");
        if try_q {
            let q = format!("{name}.int8.onnx");
            if dir.as_ref().join(&q).exists() {
                file = q;
            }
        }
        let opt = if cfg!(target_os = "windows") {
            GraphOptimizationLevel::Level1
        } else {
            GraphOptimizationLevel::Level3
        };
        Ok(Session::builder()?
            .with_optimization_level(opt)?
            .with_execution_providers(vec![CPUExecutionProvider::default().build()])?
            .with_parallel_execution(true)?
            .with_intra_threads(threads)?
            .with_inter_threads(threads)?
            .commit_from_file(dir.as_ref().join(file))?)
    }

    fn load_vocab<P: AsRef<Path>>(dir: P) -> Result<(Vec<String>, i32), AsrError> {
        let content = fs::read_to_string(dir.as_ref().join("vocab.txt"))?;
        let mut blank_idx = None;
        let entries: Vec<_> = content
            .lines()
            .filter_map(|l| {
                let mut p = l.split_whitespace();
                let t = p.next()?.replace('\u{2581}', " ");
                let i: usize = p.next()?.parse().ok()?;
                if t == "<blk>" {
                    blank_idx = Some(i as i32);
                }
                Some((t, i))
            })
            .collect();

        let mut vocab = vec![String::new(); entries.iter().map(|(_, i)| *i).max().unwrap_or(0) + 1];
        for (t, i) in entries {
            vocab[i] = t;
        }
        blank_idx.map(|idx| (vocab, idx)).ok_or_else(|| {
            AsrError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing <blk>",
            ))
        })
    }

    pub(crate) fn decode_tokens(&self, ids: Vec<i32>, timestamps: Vec<usize>) -> Transcript {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.vocab.get(id as usize).cloned())
            .collect();
        let text = match &*DECODE_SPACE_RE {
            Ok(re) => re
                .replace_all(
                    &tokens.join(""),
                    |c: &regex::Captures| if c.get(1).is_some() { " " } else { "" },
                )
                .to_string(),
            Err(_) => tokens.join(""),
        };
        Transcript {
            text,
            timestamps: timestamps
                .iter()
                .map(|&t| FRAME_DURATION_SEC * t as f32)
                .collect(),
            tokens,
        }
    }
}

static DECODE_SPACE_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));
