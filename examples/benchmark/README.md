
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
python3 bench_apply_token_bitmask_inplace.py [-h] [--kernel {cuda,triton}] [--batch_size BATCH_SIZE] [--vocab_size VOCAB_SIZE] [--num_warmup NUM_WARMUP] [--num_iters NUM_ITERS]
```

#### Results

| GPU            | Batch size | Vocab size | Triton kernel (μs)  | CUDA kernel (μs) | Speedup ratio  |
|----------------|------------|------------|---------------------|------------------|----------------|
| H100 80GB HBM3 | 16         | 128K       | 75.4                | 6.5              | 11.65x         |
|                | 128        | 128K       | 79.0                | 52.4             | 1.51x          |
|                | 2048       | 128K       | 1048.5              | 714.5            | 1.47x          |
| A100 SXM4 80GB | 16         | 128K       | 137.1               | 9.6              | 14.30x         |
|                | 128        | 128K       | 140.3               | 88.0             | 1.59x          |
|                | 2048       | 128K       | 1439.9              | 1293.2           | 1.11x          |
