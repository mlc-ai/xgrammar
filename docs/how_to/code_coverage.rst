.. _how-to-test-code-coverage:

Code Coverage Test
==================

The script ``run_coverage.sh`` offers a way to test the code coverage
of the XGrammar library.

To run the coverage test, please follow these steps:

#. In ``config.cmake``, set the variable ``XGRAMMAR_ENABLE_COVERAGE`` to ``ON``.
#. Compile the XGrammar library with the configured settings.
#. Run the script ``run_coverage.sh`` in the root directory of the XGrammar library.

After running the script, you will find the coverage report in the
``coverage_report`` directory.

You can modify the script to change the test cases or the output directory.
