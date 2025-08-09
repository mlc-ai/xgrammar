
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
python3 examples/benchmark/bench_apply_token_bitmask_inplace.py
```

#### Results
H100
|   Batch |   Vocab |   Masked cnt |   Torch Compile |         Triton  |
|    size |    size |              |     Baseline us |    us (speedup) |
|--------:|--------:|-------------:|----------------:|----------------:|
|       1 |  128000 |            1 |            6.04 |    5.52 (1.09x) |
|       1 |  128000 |        64000 |            5.96 |    6.16 (0.97x) |
|       1 |  128000 |       127000 |            6.01 |    6.27 (0.96x) |
|       8 |  128000 |            1 |           10.90 |    6.04 (1.81x) |
|       8 |  128000 |        64000 |           10.90 |    7.76 (1.40x) |
|       8 |  128000 |       127000 |           10.91 |    8.02 (1.36x) |
|      64 |  128000 |            1 |           48.72 |   13.36 (3.65x) |
|      64 |  128000 |        64000 |           48.74 |   46.35 (1.05x) |
|      64 |  128000 |       127000 |           48.74 |   33.26 (1.47x) |
|     512 |  128000 |            1 |          350.11 |   67.43 (5.19x) |
|     512 |  128000 |        64000 |          347.57 |  330.76 (1.05x) |
|     512 |  128000 |       127000 |          345.73 |  250.06 (1.38x) |
|    4096 |  128000 |            1 |         2903.81 |  494.67 (5.87x) |
|    4096 |  128000 |        64000 |         2855.70 | 2516.79 (1.13x) |
|    4096 |  128000 |       127000 |         2720.98 | 1936.44 (1.41x) |
