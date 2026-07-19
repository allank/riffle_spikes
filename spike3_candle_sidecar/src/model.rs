//! BGE-small-en-v1.5 inference via `candle`, reusing
//! `candle_transformers::models::bert::BertModel` (a standard BERT
//! implementation — BGE-small's architecture) rather than a hand-rolled
//! forward pass, following the same architecture and CLS-token +
//! L2-normalize pooling as the tract sidecar (`spike3_rust_sidecar`)
//! and onnx_test's onnxpath.go.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

const HIDDEN_SIZE: usize = 384;
const MAX_SEQ_LEN: usize = 512;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn load(model_path: &str, tokenizer_path: &str, device: Device) -> Result<Self, String> {
        let config_path = sibling_path(model_path, "config.json")?;
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("reading config.json at {config_path}: {e}"))?;
        let config: Config =
            serde_json::from_str(&config_json).map_err(|e| format!("parsing config.json: {e}"))?;

        // Safety: inherited from memmap2::MmapOptions, per VarBuilder's
        // own doc comment — model_path is trusted local input, same
        // posture as every other adapter-touching code in this repo.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                .map_err(|e| format!("loading model weights: {e}"))?
        };
        let model = BertModel::load(vb, &config).map_err(|e| format!("building model: {e}"))?;

        let mut tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("loading tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| format!("configuring tokenizer truncation: {e}"))?;
        // No padding: each chunk is encoded and run through the model
        // individually, matching onnxpath.go's and the tract sidecar's
        // per-sequence approach.
        tokenizer.with_padding(None::<PaddingParams>);

        Ok(Embedder {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed_batch(&self, chunks: &[String]) -> Result<Vec<Vec<f32>>, String> {
        chunks.iter().map(|chunk| self.embed_one(chunk)).collect()
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenizing: {e}"))?;

        let input_ids = ids_tensor(encoding.get_ids(), &self.device)?;
        let token_type_ids = ids_tensor(encoding.get_type_ids(), &self.device)?;
        let attention_mask = ids_tensor(encoding.get_attention_mask(), &self.device)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("running inference: {e}"))?;

        cls_and_normalize(&hidden)
    }
}

fn ids_tensor(ids: &[u32], device: &Device) -> Result<Tensor, String> {
    Tensor::new(ids, device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| format!("building input tensor: {e}"))
}

/// Takes the [CLS] token's (position 0) hidden state from a (1, seq_len,
/// hidden_size) output tensor and L2-normalizes it — BGE's documented
/// pooling method, not the BERT pooler head. Same math as the tract
/// sidecar's cls_and_normalize, ported not redesigned.
fn cls_and_normalize(hidden: &Tensor) -> Result<Vec<f32>, String> {
    let cls = hidden
        .i((0, 0))
        .map_err(|e| format!("indexing [CLS] token: {e}"))?
        .to_vec1::<f32>()
        .map_err(|e| format!("reading [CLS] token values: {e}"))?;

    if cls.len() != HIDDEN_SIZE {
        return Err(format!(
            "[CLS] vector has {} elements, want {HIDDEN_SIZE}",
            cls.len()
        ));
    }

    let norm: f64 = cls
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Ok(cls);
    }
    Ok(cls
        .into_iter()
        .map(|v| ((v as f64) / norm) as f32)
        .collect())
}

fn sibling_path(model_path: &str, filename: &str) -> Result<String, String> {
    let dir = std::path::Path::new(model_path)
        .parent()
        .ok_or_else(|| format!("{model_path} has no parent directory"))?;
    dir.join(filename)
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{model_path}'s directory is not valid UTF-8"))
}
