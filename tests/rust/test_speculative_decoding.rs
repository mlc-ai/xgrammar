mod test_utils;

use serial_test::serial;
use test_utils::*;

use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType, allocate_token_bitmask, testing};

fn create_i32_1d_dltensor(data: &mut [i32]) -> (xgrammar::DLTensor, Vec<i64>, Vec<i64>) {
    let mut shape = vec![data.len() as i64];
    let mut strides = vec![1i64];
    let tensor = xgrammar::DLTensor {
        data: data.as_mut_ptr() as *mut std::ffi::c_void,
        device: xgrammar::DLDevice {
            device_type: xgrammar::DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: xgrammar::DLDataType {
            code: xgrammar::DLDataTypeCode::kDLInt as u8,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };
    (tensor, shape, strides)
}

#[test]
#[serial]
fn test_traverse_draft_tree_linear() {
    let grammar = Grammar::builtin_json_grammar();
    let vocab = ["a", "b", "c", "{", "}", "\"", ":", ",", " ", "true", "false", "null"];
    let tok = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();

    let num_nodes = 3usize;
    let mut retrieve_next_token: Vec<i64> = vec![1, 2, -1];
    let mut retrieve_next_sibling: Vec<i64> = vec![-1, -1, -1];
    let mut draft_tokens: Vec<i64> = vec![3, 6, 4]; // {, :, }

    let (rt, _rt_shape, _rt_strides) = create_i64_1d_dltensor(&mut retrieve_next_token);
    let (rs, _rs_shape, _rs_strides) = create_i64_1d_dltensor(&mut retrieve_next_sibling);
    let (dt, _dt_shape, _dt_strides) = create_i64_1d_dltensor(&mut draft_tokens);

    let vocab_size = vocab.len();
    let mut bitmask_data = allocate_token_bitmask(num_nodes, vocab_size);
    let (mut bitmask_tensor, _bshape, _bstrides) =
        create_bitmask_dltensor(&mut bitmask_data, num_nodes, vocab_size);

    testing::traverse_draft_tree(&rt, &rs, &dt, &mut matcher, &mut bitmask_tensor).unwrap();

    // At the start of JSON parsing, not all tokens should be allowed (e.g. "a" is rejected).
    let rejected = testing::get_masked_tokens_from_bitmask(&bitmask_tensor, vocab_size as i32, 0);
    assert!(!rejected.is_empty());
}

#[test]
#[serial]
fn test_traverse_draft_tree_with_siblings() {
    let grammar = Grammar::builtin_json_grammar();
    let vocab = ["a", "b", "c", "{", "}", "\"", ":", ",", " ", "true", "false", "null"];
    let tok = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();

    // Tree:
    //   0
    //  / \
    // 1   2
    let num_nodes = 3usize;
    let mut retrieve_next_token: Vec<i64> = vec![1, -1, -1];
    let mut retrieve_next_sibling: Vec<i64> = vec![-1, 2, -1];
    let mut draft_tokens: Vec<i64> = vec![3, 5, 4]; // {, ", }

    let (rt, _rt_shape, _rt_strides) = create_i64_1d_dltensor(&mut retrieve_next_token);
    let (rs, _rs_shape, _rs_strides) = create_i64_1d_dltensor(&mut retrieve_next_sibling);
    let (dt, _dt_shape, _dt_strides) = create_i64_1d_dltensor(&mut draft_tokens);

    let vocab_size = vocab.len();
    let mut bitmask_data = allocate_token_bitmask(num_nodes, vocab_size);
    let (mut bitmask_tensor, _bshape, _bstrides) =
        create_bitmask_dltensor(&mut bitmask_data, num_nodes, vocab_size);

    testing::traverse_draft_tree(&rt, &rs, &dt, &mut matcher, &mut bitmask_tensor).unwrap();

    let rejected = testing::get_masked_tokens_from_bitmask(&bitmask_tensor, vocab_size as i32, 0);
    assert!(!rejected.is_empty());
}

#[test]
#[serial]
fn test_traverse_draft_tree_shape_assertion() {
    let grammar = Grammar::builtin_json_grammar();
    let vocab = ["a", "b", "c", "{", "}", "\"", ":", ",", " ", "true", "false", "null"];
    let tok = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();

    let mut retrieve_next_token: Vec<i64> = vec![1, 2, -1];
    let mut retrieve_next_sibling_wrong_shape: Vec<i64> = vec![-1, -1];
    let mut retrieve_next_sibling_wrong_dtype: Vec<i32> = vec![-1, -1, -1];
    let mut draft_tokens_wrong_dtype: Vec<i32> = vec![3, 6, 4];

    let (rt, _rt_shape, _rt_strides) = create_i64_1d_dltensor(&mut retrieve_next_token);
    let (rs_wrong_shape, _rs_shape, _rs_strides) =
        create_i64_1d_dltensor(&mut retrieve_next_sibling_wrong_shape);
    let (rs_wrong_dtype, _rs_shape2, _rs_strides2) =
        create_i32_1d_dltensor(&mut retrieve_next_sibling_wrong_dtype);
    let (dt_wrong_dtype, _dt_shape, _dt_strides) =
        create_i32_1d_dltensor(&mut draft_tokens_wrong_dtype);

    let vocab_size = vocab.len();
    let mut bitmask_data = allocate_token_bitmask(3, vocab_size);
    let (mut bitmask_tensor, _bshape, _bstrides) =
        create_bitmask_dltensor(&mut bitmask_data, 3, vocab_size);

    assert!(testing::traverse_draft_tree(&rt, &rs_wrong_shape, &dt_wrong_dtype, &mut matcher, &mut bitmask_tensor).is_err());
    assert!(testing::traverse_draft_tree(&rt, &rs_wrong_dtype, &dt_wrong_dtype, &mut matcher, &mut bitmask_tensor).is_err());
}


