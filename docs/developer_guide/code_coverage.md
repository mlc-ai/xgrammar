# Code Coverage

The script [`run_coverage.sh`](https://github.com/mlc-ai/xgrammar/blob/main/scripts/run_coverage.sh) offers a way to test the code coverage
of the XGrammar library.

To run the coverage test, please follow these steps:

1. In `config.cmake`, set the variable `XGRAMMAR_ENABLE_COVERAGE` to `ON`.
2. Compile the XGrammar library with the configured settings.
3. Run the script `run_coverage.sh` in the root directory of the XGrammar library.

After running the script, you will find the coverage report in the
`coverage_report` directory.

Please note that code coverage tools are merely aids to help identify which parts of the code have not been tested. However, pursuing 100% code coverage is not advisable. It can actually have [negative consequences](https://neatstack.substack.com/p/stop-using-code-coverage-as-a-quality).
