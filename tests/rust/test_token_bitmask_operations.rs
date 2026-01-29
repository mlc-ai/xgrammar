#![allow(clippy::needless_range_loop)]

mod test_utils;

use serial_test::serial;
use test_utils::*;

use xgrammar::{
    allocate_token_bitmask, apply_token_bitmask_inplace_cpu, get_bitmask_shape,
    reset_token_bitmask, testing,
};

fn pack_bool_masks_to_bitmask_data(
    bool_masks: &[Vec<bool>],
    vocab_size: usize,
) -> Box<[i32]> {
    let batch = bool_masks.len();
    let (_, bitmask_size) = get_bitmask_shape(batch, vocab_size);
    let mut out = vec![0i32; batch * bitmask_size];
    for (row, mask) in bool_masks.iter().enumerate() {
        assert_eq!(mask.len(), vocab_size);
        for (tok, &allowed) in mask.iter().enumerate() {
            if allowed {
                let word = tok / 32;
                let bit = tok % 32;
                out[row * bitmask_size + word] |= 1i32 << bit;
            }
        }
    }
    out.into_boxed_slice()
}

#[test]
#[serial]
fn test_allocate_reset_token_bitmask() {
    let batch_size = 10usize;
    let vocab_size = 128_005usize;
    let (_, bitmask_size) = get_bitmask_shape(batch_size, vocab_size);
    let mut bitmask = allocate_token_bitmask(batch_size, vocab_size);
    assert_eq!(bitmask.len(), batch_size * bitmask_size);
    assert!(bitmask.iter().all(|&x| x == -1i32));

    bitmask.fill(0);
    reset_token_bitmask(&mut bitmask);
    assert!(bitmask.iter().all(|&x| x == -1i32));
}

#[test]
#[serial]
fn test_get_masked_tokens_from_bitmask() {
    let token_mask_sizes = [1024usize, 32_000usize, 32_001usize, 32_011usize];
    for &vocab_size in &token_mask_sizes {
        let mask0: Vec<bool> = (0..vocab_size).map(|i| i % 3 == 0).collect();
        let mask1: Vec<bool> = (0..vocab_size).map(|i| i % 5 == 0).collect();
        let mut bitmask_data = pack_bool_masks_to_bitmask_data(&[mask0.clone(), mask1.clone()], vocab_size);
        let (tensor, _shape, _strides) = create_bitmask_dltensor(&mut bitmask_data, 2, vocab_size);

        for (index, mask) in [mask0, mask1].into_iter().enumerate() {
            let expected: Vec<i32> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &allowed)| if allowed { None } else { Some(i as i32) })
                .collect();
            let got = testing::get_masked_tokens_from_bitmask(&tensor, vocab_size as i32, index as i32);
            assert_eq!(&*got, &*expected, "vocab_size={vocab_size}, index={index}");
        }
    }
}

#[test]
#[serial]
fn test_is_single_token_bitmask() {
    let batch = 2usize;
    let vocab_size = 1024usize;
    let token_id = 100usize;

    let mask0 = vec![false; vocab_size];
    let mut mask1 = vec![false; vocab_size];
    let mut bitmask_data = pack_bool_masks_to_bitmask_data(&[mask0.clone(), mask1.clone()], vocab_size);
    let (tensor, _shape, _strides) = create_bitmask_dltensor(&mut bitmask_data, batch, vocab_size);
    assert_eq!(
        testing::is_single_token_bitmask(&tensor, vocab_size as i32, 1),
        (false, -1)
    );

    mask1[token_id] = true;
    let mut bitmask_data = pack_bool_masks_to_bitmask_data(&[mask0.clone(), mask1.clone()], vocab_size);
    let (tensor, _shape, _strides) = create_bitmask_dltensor(&mut bitmask_data, batch, vocab_size);
    assert_eq!(
        testing::is_single_token_bitmask(&tensor, vocab_size as i32, 1),
        (true, token_id as i32)
    );

    mask1[token_id + 1] = true;
    let mut bitmask_data = pack_bool_masks_to_bitmask_data(&[mask0, mask1], vocab_size);
    let (tensor, _shape, _strides) = create_bitmask_dltensor(&mut bitmask_data, batch, vocab_size);
    assert_eq!(
        testing::is_single_token_bitmask(&tensor, vocab_size as i32, 1),
        (false, -1)
    );
}

#[test]
#[serial]
fn test_apply_token_bitmask_inplace_cpu_basic() {
    // Keep logits at odd positions.
    let vocab_size = 10usize;
    let bool_mask: Vec<bool> = (0..vocab_size).map(|i| i % 2 == 1).collect();
    let mut bitmask_data = pack_bool_masks_to_bitmask_data(std::slice::from_ref(&bool_mask), vocab_size);
    let (bitmask_tensor, _bshape, _bstrides) = create_bitmask_dltensor(&mut bitmask_data, 1, vocab_size);

    let mut logits: Vec<f32> = (1..=vocab_size).map(|x| x as f32).collect();
    let (mut logits_tensor, _lshape, _lstrides) = create_f32_1d_dltensor(&mut logits);

    // Note: the C++ API accepts 1D logits + 2D bitmask.
    apply_token_bitmask_inplace_cpu(&mut logits_tensor, &bitmask_tensor, Some(vocab_size as i32), None)
        .unwrap();

    for i in 0..vocab_size {
        let expected = if bool_mask[i] { (i + 1) as f32 } else { f32::NEG_INFINITY };
        assert_eq!(logits[i], expected, "i={i}");
    }
}

#[test]
#[serial]
fn test_apply_token_bitmask_inplace_cpu_shape_stride_mismatch() {
    let col = 100usize;
    let batch = 2usize;
    let vocab_size = col;

    // Row 0 keeps even indices, row 1 keeps odd indices.
    let bool_masks: Vec<Vec<bool>> = (0..batch)
        .map(|row| (0..col).map(|i| (i % 2 == 0) == (row == 0)).collect())
        .collect();
    let mut bitmask_data = pack_bool_masks_to_bitmask_data(&bool_masks, vocab_size);
    let (bitmask_tensor, _bshape, _bstrides) = create_bitmask_dltensor(&mut bitmask_data, batch, vocab_size);

    // Create a (2, col+1) buffer and view it as a (2, col) with stride0 = col+1.
    let stride0 = (col + 1) as i64;
    let mut master: Vec<f32> = Vec::with_capacity(batch * (col + 1));
    for row in 0..batch {
        for i in 0..(col + 1) {
            master.push(i as f32 + if row == 0 { 0.1 } else { 0.2 });
        }
    }
    let original = master.clone();
    let (mut logits_tensor, _lshape, _lstrides) =
        create_f32_2d_dltensor(&mut master, batch, col, stride0, 1);

    apply_token_bitmask_inplace_cpu(&mut logits_tensor, &bitmask_tensor, Some(vocab_size as i32), None)
        .unwrap();

    for row in 0..batch {
        for i in 0..col {
            let idx = row * (col + 1) + i;
            let expected = if bool_masks[row][i] {
                original[idx]
            } else {
                f32::NEG_INFINITY
            };
            assert_eq!(master[idx], expected, "row={row}, i={i}");
        }
        // padding element untouched
        let pad_idx = row * (col + 1) + col;
        assert_eq!(master[pad_idx], original[pad_idx], "row={row} padding mutated");
    }
}

#[test]
#[serial]
fn test_apply_token_bitmask_inplace_cpu_indices() {
    let batch = 3usize;
    let vocab_size = 64usize;

    let mut bool_masks: Vec<Vec<bool>> = Vec::new();
    for row in 0..batch {
        // Different pattern per row.
        bool_masks.push((0..vocab_size).map(|i| (i + row) % 3 == 0).collect());
    }
    let mut bitmask_data = pack_bool_masks_to_bitmask_data(&bool_masks, vocab_size);
    let (bitmask_tensor, _bshape, _bstrides) = create_bitmask_dltensor(&mut bitmask_data, batch, vocab_size);

    let mut logits: Vec<f32> = (0..(batch * vocab_size)).map(|i| i as f32).collect();
    let original = logits.clone();
    let (mut logits_tensor, _lshape, _lstrides) =
        create_f32_2d_dltensor(&mut logits, batch, vocab_size, vocab_size as i64, 1);

    // Only apply to rows 0 and 2.
    let indices = [0i32, 2i32];
    apply_token_bitmask_inplace_cpu(
        &mut logits_tensor,
        &bitmask_tensor,
        Some(vocab_size as i32),
        Some(&indices),
    )
    .unwrap();

    for row in 0..batch {
        for i in 0..vocab_size {
            let idx = row * vocab_size + i;
            let expected = if row == 1 || bool_masks[row][i] {
                original[idx]
            } else {
                f32::NEG_INFINITY
            };
            assert_eq!(logits[idx], expected, "row={row}, i={i}");
        }
    }
}

#[test]
#[serial]
fn test_apply_token_bitmask_inplace_cpu_vocab_size() {
    let cases = [
        ((2usize, 130usize), (2usize, 4usize), None),
        ((2usize, 120usize), (2usize, 4usize), None),
        ((2usize, 130usize), (2usize, 4usize), Some(120i32)),
    ];

    for (logits_shape, bitmask_shape, vocab_size_override) in cases {
        let (batch, logits_vocab) = logits_shape;
        let (bitmask_batch, bitmask_size) = bitmask_shape;
        let bitmask_vocab_size = bitmask_size * 32;

        let mut bitmask_data = vec![0i32; bitmask_batch * bitmask_size];
        let (bitmask_tensor, _bshape, _bstrides) =
            create_bitmask_dltensor(&mut bitmask_data, bitmask_batch, bitmask_vocab_size);

        let mut logits = vec![1.0f32; batch * logits_vocab];
        let (mut logits_tensor, _lshape, _lstrides) =
            create_f32_2d_dltensor(&mut logits, batch, logits_vocab, logits_vocab as i64, 1);

        let vocab_size = vocab_size_override.unwrap_or_else(|| {
            std::cmp::min(logits_vocab, bitmask_vocab_size) as i32
        });
        apply_token_bitmask_inplace_cpu(
            &mut logits_tensor,
            &bitmask_tensor,
            Some(vocab_size),
            None,
        )
        .unwrap();

        for row in 0..batch {
            for col in 0..logits_vocab {
                let idx = row * logits_vocab + col;
                let expected = if (col as i32) < vocab_size {
                    f32::NEG_INFINITY
                } else {
                    1.0
                };
                assert_eq!(
                    logits[idx], expected,
                    "row={row}, col={col}, vocab_size={vocab_size}"
                );
            }
        }
    }
}

#[test]
#[serial]
fn test_apply_token_bitmask_inplace_cpu_indices_mismatch() {
    let cases = [
        (3usize, 3usize, 128usize, vec![0i32, 1i32]),
        (2usize, 3usize, 128usize, vec![0i32]),
        (3usize, 2usize, 130usize, vec![0i32]),
    ];

    for (logits_batch, bitmask_batch, vocab_size, indices) in cases {
        let bitmask_vocab_size = vocab_size.div_ceil(32) * 32;
        let (_, bitmask_size) = get_bitmask_shape(bitmask_batch, bitmask_vocab_size);
        let mut bitmask_data = vec![0i32; bitmask_batch * bitmask_size];
        let (bitmask_tensor, _bshape, _bstrides) =
            create_bitmask_dltensor(&mut bitmask_data, bitmask_batch, bitmask_vocab_size);

        let mut logits = vec![1.0f32; logits_batch * vocab_size];
        let (mut logits_tensor, _lshape, _lstrides) =
            create_f32_2d_dltensor(&mut logits, logits_batch, vocab_size, vocab_size as i64, 1);

        let original = logits.clone();
        apply_token_bitmask_inplace_cpu(
            &mut logits_tensor,
            &bitmask_tensor,
            Some(vocab_size as i32),
            Some(&indices),
        )
        .unwrap();

        for row in 0..logits_batch {
            for col in 0..vocab_size {
                let idx = row * vocab_size + col;
                let expected = if indices.contains(&(row as i32)) {
                    f32::NEG_INFINITY
                } else {
                    original[idx]
                };
                assert_eq!(
                    logits[idx], expected,
                    "row={row}, col={col}, vocab_size={vocab_size}"
                );
            }
        }
    }
}


