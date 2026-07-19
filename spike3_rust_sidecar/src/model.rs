//! BGE-small-en-v1.5 inference via `tract`, loading the same ONNX file
//! and following the same architecture as onnx_test's onnxpath.go: the
//! 3 ONNX inputs (input_ids, attention_mask, token_type_ids), the
//! last_hidden_state output, CLS-token + L2-normalize pooling.

use std::sync::Arc;

use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use tract_onnx::prelude::*;

const HIDDEN_SIZE: usize = 384;
const MAX_SEQ_LEN: usize = 512;

pub struct Embedder {
    plan: Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>,
    tokenizer: Tokenizer,
}

impl Embedder {
    pub fn load(model_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| format!("loading model: {e}"))?
            .into_optimized()
            .map_err(|e| format!("optimizing model: {e}"))?
            .into_runnable()
            .map_err(|e| format!("building runnable plan: {e}"))?;

        let mut tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("loading tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| format!("configuring tokenizer truncation: {e}"))?;
        // No padding: each chunk is encoded and run through the model
        // individually, matching onnxpath.go's per-sequence approach —
        // neither onnx_test adapter batches multiple sequences into one
        // padded tensor.
        tokenizer.with_padding(None::<PaddingParams>);

        Ok(Embedder {
            plan: model,
            tokenizer,
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

        let ids = encoding.get_ids();
        let seq_len = ids.len();

        let input_ids = ids_tensor(ids)?;
        let attention_mask = ids_tensor(encoding.get_attention_mask())?;
        let token_type_ids = ids_tensor(encoding.get_type_ids())?;

        let outputs = self
            .plan
            .run(tvec!(
                input_ids.into(),
                attention_mask.into(),
                token_type_ids.into()
            ))
            .map_err(|e| format!("running inference: {e}"))?;

        let hidden_view = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| format!("reading model output: {e}"))?;
        let hidden = hidden_view
            .as_slice()
            .ok_or("model output was not contiguous")?;

        cls_and_normalize(hidden, seq_len)
    }
}

fn ids_tensor(ids: &[u32]) -> Result<Tensor, String> {
    let values: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
    let seq_len = values.len();
    tract_ndarray::Array2::from_shape_vec((1, seq_len), values)
        .map(Tensor::from)
        .map_err(|e| format!("building input tensor: {e}"))
}

/// Takes the [CLS] token's (position 0) hidden state and L2-normalizes
/// it — BGE's documented pooling method, not the BERT pooler head. A
/// direct port of onnxpath.go's clsAndNormalize.
fn cls_and_normalize(hidden: &[f32], seq_len: usize) -> Result<Vec<f32>, String> {
    if hidden.len() < HIDDEN_SIZE {
        return Err(format!(
            "model output has {} elements, want at least {HIDDEN_SIZE} (seq_len={seq_len})",
            hidden.len()
        ));
    }

    let mut cls: Vec<f32> = hidden[..HIDDEN_SIZE].to_vec();
    let norm: f64 = cls
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for v in cls.iter_mut() {
            *v = ((*v as f64) / norm) as f32;
        }
    }
    Ok(cls)
}
