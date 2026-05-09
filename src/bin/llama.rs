use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use model::{Llama, LlamaConfig};
use std::io::Write;
use tokenizers::Tokenizer;

const MODEL_DIR: &str = "/ssd1/wy/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6";
const TOKENIZER_PATH: &str = "/ssd1/wy/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";
const CONFIG_PATH: &str = "/ssd1/wy/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json";
const WEIGHTS_PATH: &str = "/ssd1/wy/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";

const PROMPT: &str = "你好，请用一句话介绍 TinyLlama。";
const SAMPLE_LEN: usize = 128;

fn main() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => Device::Cpu,
    };
    let dtype = DType::F16;

    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(CONFIG_PATH)?)?;
    let config = config.into_config(false);
    let mut cache = model::Cache::new(true, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[WEIGHTS_PATH], dtype, &device)? };
    let llama = Llama::load(vb, &config)?;

    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).map_err(E::msg)?;
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id("</s>")
            .map(model::LlamaEosToks::Single)
    });

    let mut tokens = tokenizer
        .encode(PROMPT, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut token_stream = hello_candle::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("model dir: {MODEL_DIR}");
    println!("starting inference...");
    print!("{PROMPT}");
    std::io::stdout().flush()?;

    let mut logits_processor = LogitsProcessor::from_sampling(
        299792458,
        Sampling::TopKThenTopP {
            k: 50,
            p: 0.9,
            temperature: 0.8,
        },
    );

    let mut index_pos = 0usize;
    let mut generated = 0usize;

    for index in 0..SAMPLE_LEN {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };

        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => break,
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => {}
        }

        if let Some(text) = token_stream.next_token(next_token)? {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }

    if let Some(rest) = token_stream.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!("\n\n{generated} tokens generated");
    Ok(())
}