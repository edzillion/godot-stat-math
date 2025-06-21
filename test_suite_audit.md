# Godot Stat Math Test Suite Audit

This document outlines the findings of a comprehensive audit of the test suites in `addons/godot-stat-math/tests/core/` against the standards defined in `godot_stat_math_testing_standards.md`. The goal is to identify all areas of non-compliance and create a clear action plan for remediation.

## High-Level Summary

- **Overall Compliance**: The test suites show a strong foundation in data-driven principles, but compliance is inconsistent. While many tests use data tables and proper tolerances, there are significant deviations from the standards regarding data sources, test structure, and the handling of hardcoded values.
- **Key Areas for Improvement**:
    - **Data-Driven Purity**: Eliminate all hardcoded test data from function parameters (`test_parameters`) and test bodies, moving it to the central `/tables/` files.
    - **Test Structure Consistency**: Uniformly apply the `SCIPY VALIDATION`, `MATHEMATICAL PROPERTY`, and `PARAMETER VALIDATION` sections across all test files.
    - **Elimination of "Magic Numbers"**: Replace all non-obvious numerical literals with well-documented values from data tables or named constants.
    - **Comprehensive Error Testing**: Ensure all parameter validation tests check for both the `push_error` message and the correct sentinel return value (e.g., `NAN`).
    - **`distributions_test.gd` Overhaul**: This file requires a complete refactoring to replace brittle, non-deterministic tests with robust, data-driven validation.

---

## Audit Checklist by File

### `addons/godot-stat-math/tests/core/basic_stats_test.gd`

- [ ] **Improve Weak Assertion**: `test_median_absolute_deviation_bimodal_data` asserts `is_greater_equal(0.0)`. This is a weak test. The expected Median Absolute Deviation (MAD) should be calculated, added to the data table, and asserted directly.
- [ ] **Remove Data Conversion Helper**: Tests repeatedly use `StatMath.HelperFunctions.convert_to_float_array`. The test data in `basic_stats_test_data.gd` should be stored as `Array[float]` directly to simplify the tests and adhere to the "Direct table data lookups without transformation" standard.
- [ ] **Add Structure Headers**: The file lacks the standard `# =============================================================================` section headers for `SCIPY VALIDATION`, `MATHEMATICAL PROPERTY`, and `PARAMETER VALIDATION`. These should be added to organize the tests.
- [ ] **Check error return values**: The error tests for `mean`, `median`, `variance` etc. check for `push_error` but do not check that the function returns `NAN` as specified in the standards.

### `addons/godot-stat-math/tests/core/cdf_functions_test.gd`

- [ ] **Eliminate Hardcoded "Magic Numbers"**: Tests like `test_normal_cdf_median` and `test_exponential_cdf_median` assert that the result is `is_between(0.49, 0.51)`. They should assert equality with `0.5` using an appropriate tolerance. The value `0.693147` (`ln(2)`) should be documented as a mathematical relationship or moved to a data table if higher precision is needed.
- [ ] **Standardize File Structure**: The section headers use "PHASE 3" prefixes (e.g., `PHASE 3: ENHANCED CDF TESTING`). These should be removed and standardized to the required `SCIPY VALIDATION TESTS`, `MATHEMATICAL PROPERTY TESTS`, and `PARAMETER VALIDATION TESTS` sections.
- [ ] **Check error return values**: The parameter validation tests do not check that the functions return `NAN` on error.

### `addons/godot-stat-math/tests/core/cdf_pdf_integration_test.gd`

- [ ] **Refactor Hardcoded `test_cross_function_probability_consistency` data**: The `distributions` array is hardcoded test data that should be moved to the data table.
- [ ] **Review Tolerances**: The test `test_end_to_end_normal_distribution_workflow` uses `StatMath.NEGATIVE_BINOMIAL_TOLERANCE`, `StatMath.HIGH_DISTRIBUTION_TOLERANCE` and `StatMath.DEFAULT_TOLERANCE_FACTOR`. These seem arbitrary and should be reviewed for appropriateness, and the test itself might be too complex and non-deterministic. End-to-end workflow tests should be deterministic.
- [ ] **Check error return values**: The `match` statements in `test_cdf_monotonicity_all_distributions` and `test_cross_function_probability_consistency` push an error for unknown distributions but do not test that the `current_cdf` is `NAN`. This should be tested explicitly.

### `addons/godot-stat-math/tests/core/distributions_test.gd`

- [ ] **MAJOR OVERHAUL REQUIRED**: This entire file deviates significantly from the testing standards.
- [ ] **Eliminate Statistical Property Tests**: Tests like `test_randi_binomial_statistical_properties` are non-deterministic and rely on the Law of Large Numbers. Per the standards, tests must be deterministic. These should be replaced with tests that validate against pre-computed values or test specific properties of the random number generation algorithm with a fixed seed.
- [ ] **Eliminate `pass`ed Tests**: `test_randi_binomial_typical_case` is marked as deprecated with a `pass`. It should be removed.
- [ ] **Remove `print` Statements**: `test_randi_geometric_p_very_small_expect_large_or_inf` contains a `print` statement that should be removed.
- [ ] **Replace Vague Assertions**: Many tests use vague assertions (e.g., `assert_bool(result >= 1).is_true()`). Tests should have precise, data-driven expected outcomes.
- [ ] **Refactor All Hardcoded Values**: Nearly every test in this file uses hardcoded values and parameters. All of this data should be moved to a new `distributions_test_data.gd` file. For example:
    - **`test_randi_binomial_statistical_properties`**: The parameters `n_trials = 20`, `p = 0.4`, and `sample_size = 2000` are hardcoded for a non-deterministic statistical test. This test should be replaced with a deterministic one, using a fixed seed, with its parameters and exact expected outcome defined in the data table.
    - **`test_randi_pseudo_typical_case`**: The parameter `c_param = 0.3` is a "magic number" used for a weak assertion (`result >= 1 and result <= 3`). A proper data-driven test would have an entry in the data table specifying the input `0.3`, a fixed seed, and the *exact* expected integer result.
    - **`test_randi_seige_typical_case`**: The parameters `w=0.5, c_0=0.1, c_win=0.2, c_lose=-0.05` are all hardcoded values defining a "typical" scenario. This entire scenario should be defined in a data table, including a seed and the exact, pre-calculated number of trials expected as a result.
    - **`test_randi_poisson_typical_case`**: The `lambda = 3.0` parameter is a hardcoded value used in a weak test that only asserts the result is non-negative. The data table should contain entries that, for a given seed and lambda, specify the exact integer that the function is expected to produce.
    - **`test_randi_seige_initial_capture_guaranteed`**: The parameters `0.5, 1.0, 0.1, -0.1` are hardcoded. While it's testing a boundary condition (`c_0 = 1.0`), the other parameters are arbitrary. A data table should define this specific boundary case to make the test intentional and deterministic.
- [ ] **Check error return values**: All parameter validation tests are missing checks for the sentinel return value.

### `addons/godot-stat-math/tests/core/error_functions_test.gd`

- [ ] **Add Structure Headers**: The file lacks the standard section headers. Tests like `test_error_function_inverse_round_trip` are acceptable mathematical property tests and should be organized under the `MATHEMATICAL PROPERTY TESTS` header.
- [ ] **Check error return values**: `test_error_function_inverse_invalid_gt_one` and other validation tests are missing assertions for the `NAN` return value.
- [ ] **Consolidate Scipy Tests**: The tests `test_error_function_positive` and `test_complementary_error_function_positive` fetch single cases from the test data. These should be converted to parametrized loops that iterate over all cases in the data table for better coverage.

### `addons/godot-stat-math/tests/core/helper_functions_test.gd`

- [ ] **Standardize File Structure**: The file is missing the standard section headers. These should be added to group tests into `SCIPY VALIDATION`, `MATHEMATICAL PROPERTY`, and `PARAMETER VALIDATION`.
- [ ] **Review `test_lower_incomplete_gamma_regularized_convergence_warning`**: This test is empty. It should be implemented or removed.
- [ ] **Check error return values**: Most parameter validation tests are missing assertions for the sentinel return value (`NAN` or `-INF`).

### `addons/godot-stat-math/tests/core/pmf_pdf_functions_test.gd`

- [ ] **Eliminate `test_parameters`**: Multiple tests (`test_normal_pdf_parametrized`, `test_exponential_pdf_parametrized`, `test_uniform_pdf_parametrized`, etc.) use hardcoded `test_parameters` in the function signature. This is a major violation of the "Single Source of Truth" principle. All this data must be moved to `pmf_pdf_test_data.gd` and the tests refactored to loop over the data table.
- [ ] **Consolidate Scipy Tests**: Tests like `test_gamma_pdf_basic` test a single hardcoded case. This should be expanded into a comprehensive, data-driven test that pulls all gamma PDF cases from the data table.
- [ ] **Standardize File Structure**: The file has some section headers but they should be checked for consistency with the official standard (`SCIPY VALIDATION TESTS`, `MATHEMATICAL PROPERTY TESTS`, `PARAMETER VALIDATION TESTS`).

### `addons/godot-stat-math/tests/core/_cdf_ppf_round_trip_consistency.gd`

- [ ] **Refactor Hardcoded Test Data**: The `test_*_cdf_ppf_round_trip_consistency` tests all use hardcoded arrays of test cases. This data should be moved to `ppf_test_data.gd`.
- [ ] **Eliminate Hardcoded "Magic Numbers"**: `test_exponential_median_special_value` asserts against a hardcoded value for `ln(2)`. This should be documented as a mathematical constant or the test should be data-driven if higher precision is required. `assert_float(result).is_equal_approx(log(2.0), StatMath.SPECIAL_VALUES_TOLERANCE)` would be better.
- [ ] **Standardize File Structure**: The tests are well-organized, but they should be placed under the standard `SCIPY VALIDATION`, `MATHEMATICAL PROPERTY`, and `PARAMETER VALIDATION` headers.
- [ ] **Check error return values**: Parameter validation tests are missing checks for the sentinel return values.

### `addons/godot-stat-math/tests/core/sampling_gen_test.gd`
*Note:* addons/godot-stat-math/tables/sobol_data.gd contains attested sobol direction numbers. 

- [ ] **Review `test_coordinated_shuffle_deterministic`**: This test asserts that two shuffles are different. This can be flaky. A better test would be to have pre-calculated shuffles for given indices in the test data file and assert equality.
- [ ] **Consider Data-Driven Approach**: While harder for sampling, key properties of the sequences (e.g., the first 5 points of a Sobol sequence in 2D) could be stored in a data file and tested against directly to ensure the underlying algorithms are correct and deterministic.
- [ ] **Standardize File Structure**: The file lacks the standard section headers.
- [ ] **Check error return values**: `test_generate_samples_unified_interface_edge_cases` checks for the `push_error` but doesn't verify the return value (which should probably be an empty array or null). 