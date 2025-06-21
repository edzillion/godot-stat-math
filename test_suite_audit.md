# Godot Stat Math - Test Suite Audit

This document outlines the necessary tasks to align all test suites in `addons/godot-stat-math/tests/core` with the standards defined in `godot_stat_math_testing_standards.md`. The audit focuses on converting tests to a data-driven methodology, eliminating hardcoded values, improving structure, and removing deprecated patterns.

## General / Cross-Suite Tasks

- [ ] **Standardize File Structure**: Ensure every test suite has the standard file structure with `# =============================================================================` headers for `SCIPY VALIDATION TESTS`, `MATHEMATICAL PROPERTY TESTS`, and `PARAMETER VALIDATION TESTS`.
- [ ] **Review Tolerances**: Audit the use of tolerance values across all tests to ensure the most appropriate constant from `StatMath` is used for each comparison, as per the guidelines.

---

## `basic_stats_test.gd`

- [ ] **Data-Driven `test_mean_...`**: The tests for `mean`, `median`, `mode`, `variance`, and `stdev` use hardcoded arrays and expected values. These should be converted to use a data-driven approach with `basic_stats_test_data.gd`, sourcing examples from a standard statistical textbook or `numpy`.
- [ ] **Refactor `test_trimmed_mean`**: This test is complex and has hardcoded values. It should be broken down and converted to a data-driven pattern. The source of the expected value (`7.77...`) is unclear and must be documented.
- [ ] **Data-Driven `test_covariance`**: Convert to a data-driven test. The expected value `1.66667` is a "magic number" and needs to be sourced from a validated calculation.
- [ ] **Data-Driven `test_correlation`**: Convert to a data-driven test. The expected value `0.932504` is a "magic number".
- [ ] **Data-Driven `test_skewness` and `test_kurtosis`**: Convert these to data-driven tests. Expected values are hardcoded.
- [ ] **Parameter Validation for `mode`**: The test for an empty array in `test_mode_empty_array_returns_empty_array` correctly checks for an empty array return, but it should also test for the logged error message to ensure complete validation.
- [ ] **Refactor `test_percentile` and `test_quantile`**: These have many hardcoded values. They should be consolidated and converted to a single parametrized, data-driven test.
- [ ] **`test_summary_stats`**: This test relies on the output of many other functions. While useful as an integration test, its assertions depend on hardcoded values. It should be updated to pull expected values from the test data table after the core stat functions are made data-driven.

---

## `cdf_functions_test.gd`

- [ ] **Convert `test_normal_cdf_...`**: The tests `test_normal_cdf_positive_z_score`, `test_normal_cdf_negative_z_score`, and `test_normal_cdf_zero_z_score` contain hardcoded precision values (`0.975`, `0.025`). These should be removed and the tests refactored into a single, parametrized, data-driven test sourcing data from `cdf_test_data.gd`.
- [ ] **Consolidate `test_student_t_cdf_...`**: The multiple `student_t_cdf` tests should be merged into a single parametrized, data-driven test.
- [ ] **Consolidate `test_chi_squared_cdf_...`**: Merge the multiple `chi_squared_cdf` tests into a single parametrized, data-driven test.
- [ ] **Consolidate `test_f_cdf_...`**: Merge the multiple `f_cdf` tests into a single parametrized, data-driven test.
- [ ] **Consolidate `test_gamma_cdf...`**: Merge the multiple `gamma_cdf` tests into a single parametrized, data-driven test.
- [ ] **Review Mathematical Property Tests**: The tests under "MATHEMATICAL PROPERTY TESTS" (e.g., `test_uniform_cdf_below_range`, `test_exponential_cdf_at_zero`) use acceptable hardcoded constants (`0.0`, `1.0`). However, they should be reviewed to ensure they are clearly separated and documented as mathematical properties, not requiring scipy data.

---

## `cdf_pdf_integration_test.gd`

- [ ] **Standardize Structure**: The file is missing the standard header comments for test sections.
- [ ] **Document Test Logic**: The tests `test_pdf_cdf_consistency_...` perform numerical integration. The logic and the choice of tolerance (`NUMERICAL_INTEGRATION_TOLERANCE`) should be clearly documented within the test functions.
- [ ] **Data-Driven Approach**: The parameters for these tests (`x_values`, `mu`, `sigma`, etc.) are hardcoded. Consider moving these scenarios to a test data table to make the tests cleaner and more maintainable, even if the "expected" value is calculated dynamically.

---

## `distributions_test.gd`

- [ ] **Eliminate `_string_to_enum`**: The helper function `_string_to_enum` is a deprecated anti-pattern and must be removed. All tests should use the `StatMath.SupportedDistributions` enum directly.
- [ ] **Eliminate `_get_cdf_value` and `_get_ppf_value`**: These generic helper wrappers are anti-patterns. Tests must call `StatMath.cdf()` and `StatMath.ppf()` directly. Refactor all calling tests.
- [ ] **Refactor `test_distributions_cdf_wrapper` and `test_distributions_ppf_wrapper`**: These large, complex tests rely on the anti-pattern helpers. They must be broken down into individual, data-driven tests for each distribution, calling the `StatMath` API directly.
- [ ] **Refactor `test_valid_distributions`**: This test uses a loop and string names. It should be refactored to directly test the desired functionality without relying on string-to-enum conversion.
- [ ] **Data-Driven `test_distribution_moments_...`**: The moment tests (`_mean`, `_variance`, etc.) use hardcoded expected values. These should be converted to data-driven tests using `distributions_test_data.gd`.
- [ ] **Parameter Validation**: Add tests for invalid distribution parameters passed to `StatMath.distribution()`.

---

## `error_functions_test.gd`

- [ ] **Review Hardcoded Values**: The test `test_erf_precision_values` contains hardcoded precision values (`0.84270079`, etc.). The standards doc uses this as a specific example of a value that **must** be converted to the data-driven pattern. This test needs to be converted.
- [ ] **Review `test_gamma_known_values`**: The gamma function tests contain a mix of integer factorials (acceptable) and floating point results (`gamma(1.5) -> 0.886226925...`). The floating point cases should be moved to a data-driven test.
- [ ] **Standardize Structure**: The file is missing the standard header comments for test sections.

---

## `helper_functions_test.gd`

- [ ] **Data-Driven `test_beta_function_known_values`**: This test contains hardcoded floating-point results (`0.0095238...`). These must be converted to a data-driven test.
- [ ] **Data-Driven `test_incomplete_beta_known_values`**: This test contains hardcoded precision values and must be converted to a data-driven test.
- [ ] **Review `test_factorial_known_values`**: This test uses acceptable hardcoded integer factorial results. Confirm no floating-point "magic numbers" are present.
- [ ] **Parameter Validation for `beta` function**: Add a test to validate that the `beta` function correctly handles `a <= 0` or `b <= 0` and logs the appropriate error.
- [ ] **Standardize Structure**: The file is missing the standard header comments for test sections for some parts of the file.

---

## `pmf_pdf_functions_test.gd`

- [ ] **Consolidate and Convert `..._pmf_...` tests**: All PMF tests (`test_poisson_pmf`, `test_binomial_pmf`, `test_hypergeometric_pmf`) use hardcoded values and should be consolidated into single, parametrized, data-driven tests for each type.
- [ ] **Consolidate and Convert `..._pdf_...` tests**: All PDF tests (`test_normal_pdf`, `test_student_t_pdf`, `test_gamma_pdf`, `test_f_pdf`, `test_chi_squared_pdf`) use hardcoded values and should be consolidated into single, parametrized, data-driven tests.
- [ ] **Review Mathematical Property Tests**: Ensure tests like `test_uniform_pdf_inside_range` are correctly categorized and documented as mathematical properties.

---

## `ppf_functions_test.gd`

- [ ] **Consolidate and Convert All Tests**: Every test in this suite (`test_normal_ppf`, `test_student_t_ppf`, `test_chi_squared_ppf`, `test_f_ppf`) uses hardcoded precision values for probabilities and expected outcomes. All of them must be refactored into single, parametrized, data-driven tests for each function type, using `ppf_test_data.gd`.
- [ ] **Add `CDF_PPF_CONSISTENCY` tests**: Add round-trip consistency tests that verify `ppf(cdf(x)) == x` within the `StatMath.CDF_PPF_CONSISTENCY_TOLERANCE`. These should also be data-driven.
- [ ] **Validate Tolerance Usage**: Double-check that all PPF tests use `StatMath.INVERSE_FUNCTION_TOLERANCE` as specified in the standards.

---

## `sampling_gen_test.gd`

- [ ] **Refactor Statistical Property Tests**: Tests like `test_sample_from_array_mean_and_std_dev` and `test_sample_normal_mean_and_std_dev` perform statistical tests on generated samples. The logic is complex, and the tolerances are loose. These tests need to be reviewed.
    - [ ] **Document Methodology**: The statistical methodology (e.g., number of samples, choice of tolerance) needs to be clearly documented in the test.
    - [ ] **Seed RNG**: Ensure the random number generator is seeded for reproducibility. The tests seem to do this already, but it's critical.
    - [ ] **Review Tolerances**: The tolerances are very loose (e.g., `0.1` or `0.2`). While this is expected for sampling tests, the rationale for the specific value should be documented.
- [ ] **Validate `test_shuffle_array`**: This test checks that the shuffled array is not equal to the original. This could fail by chance. A better test might check that the elements are the same but the order is likely different, or run it multiple times.
- [ ] **Error/Parameter Validation**: Add tests for the sampling generators to ensure they handle invalid parameters correctly (e.g., sample size > population size without replacement).
- [ ] **Eliminate `_build_test_parameters`**: This function is a deprecated `Manual data transformation function` anti-pattern. The tests should be refactored to not use it.
- [ ] **Remove `test_sample_from_cdf_pmf`**: This test appears to be disabled (`#@disabled`). It should either be fixed and enabled or removed entirely. If fixed, it must be made data-driven.

--- 