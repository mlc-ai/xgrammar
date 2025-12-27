#[cfg(feature = "hf")]
use hf_hub::{Repo, api::sync::ApiBuilder};
use xgrammar::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor, Grammar,
    GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType,
    allocate_token_bitmask, get_bitmask_shape,
};

/// Download tokenizer.json from HuggingFace model hub
#[cfg(feature = "hf")]
#[allow(dead_code)]
pub fn download_tokenizer_json(
    model_id: &str
) -> Result<std::path::PathBuf, String> {
    // Pass HF token explicitly from env to ensure access to gated models in CI/WSL
    let token = std::env::var("HF_TOKEN").ok();
    let api = ApiBuilder::new()
        .with_token(token)
        .build()
        .map_err(|e| e.to_string())?;
    let repo = api.repo(Repo::model(model_id.to_string()));
    repo.get("tokenizer.json").map_err(|e| e.to_string())
}

/// Create TokenizerInfo from HuggingFace model
#[cfg(feature = "hf")]
#[allow(dead_code)]
pub fn make_hf_tokenizer_info(model_id: &str) -> TokenizerInfo {
    let path =
        download_tokenizer_json(model_id).expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    TokenizerInfo::from_huggingface(&tk, None, None).unwrap()
}

/// Create a GrammarMatcher from a Grammar with minimal tokenizer info
pub fn matcher_from_grammar(gram: &Grammar) -> GrammarMatcher {
    let empty_vocab: Vec<&str> = vec![];
    let stop_ids: Option<Box<[i32]>> = None;
    let tokenizer_info =
        TokenizerInfo::new(&empty_vocab, VocabType::RAW, &stop_ids, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let cg = compiler.compile_grammar(gram).unwrap();
    GrammarMatcher::new(&cg, None, true, -1).unwrap()
}

/// Create a GrammarMatcher from a Grammar with a specific TokenizerInfo
#[allow(dead_code)]
pub fn matcher_from_grammar_with_tokenizer(
    gram: &Grammar,
    tokenizer_info: &TokenizerInfo,
) -> GrammarMatcher {
    let mut compiler = GrammarCompiler::new(tokenizer_info, 1, false, -1).unwrap();
    let cg = compiler.compile_grammar(gram).unwrap();
    GrammarMatcher::new(&cg, None, true, -1).unwrap()
}

/// Create a GrammarMatcher with rollback support
#[allow(dead_code)]
pub fn matcher_from_grammar_with_tokenizer_and_rollback(
    gram: &Grammar,
    tokenizer_info: &TokenizerInfo,
    max_rollback_tokens: i32,
) -> GrammarMatcher {
    let mut compiler = GrammarCompiler::new(tokenizer_info, 1, false, -1).unwrap();
    let cg = compiler.compile_grammar(gram).unwrap();
    GrammarMatcher::new(&cg, None, false, max_rollback_tokens).unwrap()
}

/// Check if a grammar accepts a string
#[allow(dead_code)]
pub fn is_grammar_accept_string(
    grammar: &Grammar,
    input: &str,
) -> bool {
    let mut matcher = matcher_from_grammar(grammar);
    let accepted = matcher.accept_string(input, false);
    if !accepted {
        return false;
    }
    matcher.is_terminated()
}

/// Helper to create a DLTensor from a bitmask slice
#[allow(dead_code)]
pub fn create_bitmask_dltensor(
    bitmask_data: &mut [i32],
    batch_size: usize,
    vocab_size: usize,
) -> (DLTensor, Vec<i64>, Vec<i64>) {
    let (_, bitmask_size) = get_bitmask_shape(batch_size, vocab_size);
    let mut shape = vec![batch_size as i64, bitmask_size as i64];
    let mut strides = vec![bitmask_size as i64, 1];

    let tensor = DLTensor {
        data: bitmask_data.as_mut_ptr() as *mut std::ffi::c_void,
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLInt as u8,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };

    (tensor, shape, strides)
}

/// Get bitmask and return it (for comparison)
#[allow(dead_code)]
pub fn get_next_token_bitmask_helper(
    matcher: &mut GrammarMatcher,
    vocab_size: usize,
) -> Box<[i32]> {
    let mut bitmask_data = allocate_token_bitmask(1, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, 1, vocab_size);
    matcher.fill_next_token_bitmask(&mut tensor, 0, false);
    bitmask_data
}

/// Check if a token is accepted in the bitmask
#[allow(dead_code)]
pub fn is_token_accepted_helper(
    token_id: i32,
    bitmask: &[i32],
) -> bool {
    let word_idx = (token_id / 32) as usize;
    let bit_idx = token_id % 32;
    if word_idx >= bitmask.len() {
        return false;
    }
    (bitmask[word_idx] & (1 << bit_idx)) != 0
}

/// Get list of accepted tokens from bitmask
#[allow(dead_code)]
pub fn get_accepted_tokens_helper(
    bitmask: &[i32],
    vocab_size: usize,
) -> Box<[usize]> {
    let mut accepted = Vec::new();
    for i in 0..vocab_size {
        if is_token_accepted_helper(i as i32, bitmask) {
            accepted.push(i);
        }
    }
    accepted.into_boxed_slice()
}
