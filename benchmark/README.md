
## Run Benchmark

### Benchmark Grammar Compile and Mask Generation

#### Dependencies
```
outlines                          0.1.3
outlines_core                     0.1.14
lm-format-enforcer                0.10.6
tqdm                              4.66.5
```

#### Run
```bash
python3 bench_grammar_compile_mask_gen.py [-h] [--backend {xgrammar,outlines,lmformatenforcer}]
                                          [--num_iters NUM_ITERS] [--num_warmup NUM_WARMUP]
```

#### Results

Hardware:

```
CPU: AMD Ryzen 9 7950X 16-Core Processor
GPU: NVIDIA GeForce RTX 4090
```

Results:

```
Backend: xgrammar
Fail count: 0 / 99
Grammar preprocessing time (ms): 61.9149
Mask generation time (us/token): 35.7277
Backend: outlines
Fail count (per iter): 7 / 99
Grammar preprocessing time (ms): 1333.1387
Mask generation time (us/token): 125.2214
Backend: lmformatenforcer
Fail count: 6 / 99
Grammar preprocessing time (ms): 2.7900
Mask generation time (us/token): 6147.1414
```
