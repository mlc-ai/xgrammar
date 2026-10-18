# Unity Builds

`XGRAMMAR_ENABLE_UNITY_BUILD` is an optional CMake setting, disabled by default.
It combines the core library's C++ source files into batches of up to four files
and compiles each batch as one translation unit. Test sources and Python binding
sources keep their usual compilation settings.

## When to Try It

Compiling a batch can avoid parsing the same headers separately for each source
file. This can shorten clean builds or rebuilds after a widely included header
changes. The result depends on the compiler, build flags, available memory and
number of parallel jobs.

There are tradeoffs:

- A compiler process handles more code at once and can use more memory.
- Fewer independent compilation tasks remain, which can limit parallelism on
  machines with many cores.
- Editing one source file recompiles its whole batch, including unchanged
  neighbors. Check the files you edit often, as well as clean builds.

Combining files also puts their private names in the same translation unit.
If a compiler or platform configuration encounters a name collision, disable
the option and include that configuration when reporting the build failure.

## Enable or Disable It

Add `-DXGRAMMAR_ENABLE_UNITY_BUILD=ON` to the CMake configuration command, or set
`XGRAMMAR_ENABLE_UNITY_BUILD` in `config.cmake`. Reconfigure with `OFF` to restore
separate compilation, changing the setting in the same place. A value set in
`config.cmake` overrides the cached command-line value. The batch size is four;
this option changes only the `xgrammar` target.

```bash
cmake -S . -B build -G Ninja -DXGRAMMAR_ENABLE_UNITY_BUILD=ON
cmake --build build --parallel 4

cmake -S . -B build -DXGRAMMAR_ENABLE_UNITY_BUILD=OFF
cmake --build build --parallel 4
```

## Compare on Your Machine

For a native C++ comparison, set these values in `config.cmake`:

```cmake
set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)
set(XGRAMMAR_BUILD_CXX_TESTS ON)
```

Use separate build directories with the same compiler and flags. Disable
compiler caches for the clean-build comparison:

```bash
cmake -S . -B build-regular -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DXGRAMMAR_ENABLE_UNITY_BUILD=OFF
cmake -S . -B build-unity -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DXGRAMMAR_ENABLE_UNITY_BUILD=ON

cmake --build build-regular --parallel 4
cmake --build build-unity --parallel 4
ctest --test-dir build-regular --output-on-failure
ctest --test-dir build-unity --output-on-failure
```

Warm both builds once, then alternate at least three clean builds of each
configuration. Run `cmake --build <directory> --target clean` before each timed
clean build. Keep the job count fixed within each comparison; repeat with the
job counts you use, such as 1, 2, 4 and 8. More jobs than available CPUs measures
oversubscription rather than scaling to a larger machine.

Include the complete build and linking in elapsed time. Keep cleaning and tests
outside the timer, and compare the medians and variation across repetitions.
Also measure rebuilds after editing one implementation file and a shared header.
Record the machine's CPU count, compiler version and memory usage alongside the
timings. A faster clean build alone may not justify slower everyday edits.
