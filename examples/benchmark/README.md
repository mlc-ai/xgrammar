
## Run Benchmark

### Benchmark Grammar Compile and Mask Generation

#### Dependencies
```
outlines                          0.1.3
outlines_core                     0.1.14
lm-format-enforcer                0.10.6
```

#### Run
```bash
python3 bench_grammar_compile_mask_gen.py [-h] [--backend {xgrammar,outlines,lmformatenforcer}]
                                          [--num_iters NUM_ITERS] [--num_warmup NUM_WARMUP]
```


### Benchmark Apply Token Bitmask Inplace Kernels

#### Run
```bash
python3 bench_apply_token_bitmask_inplace.py [-h] [--impl {cuda,triton}]
                                             [--batch_size BATCH_SIZE] [--vocab_size VOCAB_SIZE]
                                             [--masked_cnt MASKED_CNT] [--stride STRIDE]
                                             [--logits_dtype {float32,float16,bfloat16}]
                                             [--warmup WARMUP] [--rep REP]
```

#### Results

| GPU            | Batch size | Vocab size | Masked cnt | Triton (μs)  | CUDA (μs) | Speedup |
|:--------------:|-----------:|-----------:|-----------:|-------------:|----------:|--------:|
| H100 80GB HBM3 |          1 |       128k |         1k |         5.95 |      6.57 |   0.91x |
|                |          1 |       128k |        64k |         6.38 |      6.46 |   0.99x |
|                |          1 |       128k |       127k |         6.69 |      6.48 |   1.03x |
|                |          8 |       128k |         1k |         6.77 |      6.94 |   0.98x |
|                |          8 |       128k |        64k |         8.05 |      9.19 |   0.88x |
|                |          8 |       128k |       127k |         8.49 |      8.08 |   1.05x |
|                |         64 |       128k |         1k |        14.97 |     13.82 |   1.08x |
|                |         64 |       128k |        64k |        43.13 |     30.98 |   1.39x |
|                |         64 |       128k |       127k |        33.85 |     21.43 |   1.58x |
|                |        512 |       128k |         1k |        82.65 |     61.13 |   1.35x |
|                |        512 |       128k |        64k |       293.51 |    194.06 |   1.51x |
|                |        512 |       128k |       127k |       240.11 |    119.77 |   2.00x |
|                |       4096 |       128k |         1k |       566.17 |    417.33 |   1.36x |
|                |       4096 |       128k |        64k |      2198.59 |   1491.79 |   1.47x |
|                |       4096 |       128k |       127k |      1812.39 |    897.17 |   2.02x |
| A100 SXM4 80GB |          1 |       128k |         1k |         8.32 |      7.97 |   1.04x |
|                |          1 |       128k |        64k |         9.26 |      8.24 |   1.12x |
|                |          1 |       128k |       127k |         8.81 |      8.71 |   1.01x |
|                |          8 |       128k |         1k |         9.56 |     10.31 |   0.93x |
|                |          8 |       128k |        64k |        12.72 |     13.22 |   0.96x |
|                |          8 |       128k |       127k |        13.45 |     11.27 |   1.19x |
|                |         64 |       128k |         1k |        22.95 |     25.57 |   0.90x |
|                |         64 |       128k |        64k |        58.52 |     56.47 |   1.04x |
|                |         64 |       128k |       127k |        44.83 |     39.29 |   1.14x |
|                |        512 |       128k |         1k |       132.92 |    108.60 |   1.22x |
|                |        512 |       128k |        64k |       362.08 |    349.54 |   1.04x |
|                |        512 |       128k |       127k |       306.75 |    233.20 |   1.32x |
|                |       4096 |       128k |         1k |       955.99 |    777.94 |   1.23x |
|                |       4096 |       128k |        64k |      2756.63 |   2707.57 |   1.02x |
|                |       4096 |       128k |       127k |      2472.82 |   1782.41 |   1.39x |
