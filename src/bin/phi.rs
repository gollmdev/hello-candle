#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::quantized::gguf_file;
use candle::quantized::tokenizer::TokenizerFromGguf;
use candle::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;
use std::io::Write;
use tokenizers::Tokenizer;

use hello_candle::token_output_stream::TokenOutputStream;

const MODEL_PATH: &str =
	"/ssd1/wy/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/a64113399c2f6b8ad3e11c394733a2ddadaa7f33/Phi-3-mini-4k-instruct-q4.gguf";
const DEFAULT_PROMPT: &str = "Write a function to count prime numbers up to N. ";
const SAMPLE_LEN: usize = 1000;

fn format_size(size_in_bytes: usize) -> String {
	if size_in_bytes < 1_000 {
		format!("{size_in_bytes}B")
	} else if size_in_bytes < 1_000_000 {
		format!("{:.2}KB", size_in_bytes as f64 / 1e3)
	} else if size_in_bytes < 1_000_000_000 {
		format!("{:.2}MB", size_in_bytes as f64 / 1e6)
	} else {
		format!("{:.2}GB", size_in_bytes as f64 / 1e9)
	}
}

fn main() -> anyhow::Result<()> {
	println!(
		"avx: {}, neon: {}, simd128: {}, f16c: {}",
		candle::utils::with_avx(),
		candle::utils::with_neon(),
		candle::utils::with_simd128(),
		candle::utils::with_f16c()
	);

	let model_path = std::path::PathBuf::from(MODEL_PATH);
	let mut file = std::fs::File::open(&model_path)?;
	let start = std::time::Instant::now();
	let device = hello_candle::device(false)?;

	let model = {
		let mut ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
		if let Some(gguf_file::Value::String(model_kind)) = ct.metadata.get("tokenizer.ggml.model") {
			if model_kind.eq_ignore_ascii_case("llama") {
				ct.metadata.insert(
					"tokenizer.ggml.model".to_string(),
					gguf_file::Value::String("gpt2".to_string()),
				);
			}
		}
		let tokenizer: Tokenizer = Tokenizer::from_gguf(&ct)?;
		let mut total_size_in_bytes = 0;
		for (_, tensor) in ct.tensor_infos.iter() {
			let elem_count = tensor.shape.elem_count();
			total_size_in_bytes +=
				elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
		}
		println!(
			"loaded {:?} tensors ({}) in {:.2}s",
			ct.tensor_infos.len(),
			format_size(total_size_in_bytes),
			start.elapsed().as_secs_f32(),
		);
		let model = Phi3::from_gguf(false, ct, &mut file, &device)?;
		(model, tokenizer)
	};
	println!("model built");

	let (mut model, tokenizer) = model;
	let mut tos = TokenOutputStream::new(tokenizer);

	let prompt_str = DEFAULT_PROMPT.to_string();
	print!("{}", &prompt_str);
	let tokens = tos
		.tokenizer()
		.encode(prompt_str, true)
		.map_err(anyhow::Error::msg)?;
	let tokens = tokens.get_ids();
	let to_sample = SAMPLE_LEN.saturating_sub(1);
	let mut all_tokens = vec![];
	let mut logits_processor = LogitsProcessor::from_sampling(
		299792458,
		Sampling::All { temperature: 0.8 },
	);

	let start_prompt_processing = std::time::Instant::now();
	let mut next_token = {
		let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
		let logits = model.forward(&input, 0)?;
		let logits = logits.squeeze(0)?;
		logits_processor.sample(&logits)?
	};
	let prompt_dt = start_prompt_processing.elapsed();
	all_tokens.push(next_token);
	if let Some(t) = tos.next_token(next_token)? {
		print!("{t}");
		std::io::stdout().flush()?;
	}
	let eos_token = *tos
		.tokenizer()
		.get_vocab(true)
		.get("<|endoftext|>")
		.unwrap();
	let start_post_prompt = std::time::Instant::now();
	let mut sampled = 0;
	for index in 0..to_sample {
		let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
		let logits = model.forward(&input, tokens.len() + index)?;
		let logits = logits.squeeze(0)?;
		next_token = logits_processor.sample(&logits)?;
		all_tokens.push(next_token);
		if let Some(t) = tos.next_token(next_token)? {
			print!("{t}");
			std::io::stdout().flush()?;
		}
		sampled += 1;
		if next_token == eos_token {
			break;
		}
	}
	if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
		print!("{rest}");
	}
	std::io::stdout().flush()?;
	let dt = start_post_prompt.elapsed();
	println!(
		"\n\n{:4} prompt tokens processed: {:.2} token/s",
		tokens.len(),
		tokens.len() as f64 / prompt_dt.as_secs_f64(),
	);
	println!(
		"{sampled:4} tokens generated: {:.2} token/s",
		sampled as f64 / dt.as_secs_f64(),
	);
	Ok(())
}
