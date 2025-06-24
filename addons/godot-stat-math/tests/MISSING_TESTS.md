# Missing Tests Documentation

This document tracks functionality that lacks proper test coverage and needs test implementation. It is organized by module, then by function, detailing missing tests for each category based on a full codebase review.

---

## I. Helper Functions (`helper_functions.gd`)

**Note on `gamma` and `log_gamma`**: These functions are duplicated in `error_functions.gd`. The tests below should target the `helper_functions.gd` implementation, and the versions in `error_functions.gd` should be deprecated and removed to resolve the code smell.

### `gamma_function()`
- **Scipy Validation**:
    - [ ] `test_gamma_function_scipy_validation()`: Test against `scipy.special.gamma` with various positive float values (e.g., 0.5, 1.5, 2.7).
- **Mathematical Property**:
    - [ ] `test_gamma_function_factorial_relationship()`: Test `Γ(n) = (n-1)!` for positive integers.
    - [ ] `test_gamma_function_reflection_formula()`: Test `Γ(z)Γ(1-z) = π/sin(πz)` for non-integer z.
- **Parameter Validation**:
    - [ ] `test_gamma_function_poles_at_non_positive_integers()`: Test that `gamma_function(0)`, `gamma_function(-1)`, etc., return `INF`.

### `log_gamma()`
- **Scipy Validation**:
    - [ ] `test_log_gamma_scipy_validation()`: Test against `scipy.special.gammaln` with various positive float values.
- **Mathematical Property**:
    - [ ] `test_log_gamma_recursion_property()`: Test `log_gamma(z+1) = log_gamma(z) + log(z)`.
- **Parameter Validation**:
    - [ ] `test_log_gamma_invalid_input_non_positive()`: Test that `z <= 0` pushes an error and returns `NAN`.

### `log_binomial_coef()`
- **Mathematical Property**:
    - [ ] `test_log_binomial_coef_symmetry()`: Test `log_binomial_coef(n, k) == log_binomial_coef(n, n-k)`.

### `log_beta_function_direct()`
- **Mathematical Property**:
    - [ ] `test_log_beta_function_direct_symmetry()`: Test `log_beta(a, b) == log_beta(b, a)`.

### `get_cdf_value()`
- **Behavioral & Error Tests**:
    - [ ] `test_get_cdf_value_with_valid_enum()`: Test with `StatMath.SupportedDistributions` enum.
    - [ ] `test_get_cdf_value_with_valid_string()`: Test with string distribution names.
    - [ ] `test_get_cdf_value_invalid_distribution_type()`: Test error for invalid type: "Invalid distribution type...".
    - [ ] `test_get_cdf_value_unimplemented_distribution()`: Test error for unimplemented CDF: "CDF function not implemented...".

### `get_ppf_value()`
- **Behavioral & Error Tests**:
    - [ ] `test_get_ppf_value_with_valid_enum()`: Test with `StatMath.SupportedDistributions` enum.
    - [ ] `test_get_ppf_value_with_valid_string()`: Test with string distribution names.
    - [ ] `test_get_ppf_value_invalid_distribution_type()`: Test error for invalid type: "Invalid distribution type...".
    - [ ] `test_get_ppf_value_unimplemented_distribution()`: Test error for unimplemented PPF: "PPF function not implemented...".

### `validate_indices()`
- **Behavioral & Error Tests**:
    - [ ] `test_validate_indices_valid_samples()`: Test with a valid index array.
    - [ ] `test_validate_indices_negative_index()`: Test error: "Sample index must be non-negative".
    - [ ] `test_validate_indices_index_too_large()`: Test error: "Sample index must be less than population size".

### `validate_unique_indices()`
- **Behavioral & Error Tests**:
    - [ ] `test_validate_unique_indices_valid_unique_samples()`: Test with a valid unique index array.
    - [ ] `test_validate_unique_indices_duplicate_found()`: Test error: "Sample indices must be unique".
    - [ ] `test_validate_unique_indices_size_mismatch()`: Test error: "Number of unique indices must equal sample size".

### `convert_to_float_array()`
- **Property-based Tests**:
    - [ ] `test_convert_to_float_array_valid_conversion()`: Test `Array` to `Array[float]` conversion.
    - [ ] `test_convert_to_float_array_preserves_order()`: Test that element order is maintained.

---

## II. Basic Stats (`basic_stats.gd`)

### `percentile()`
- **Scipy Validation**:
    - [ ] `test_percentile_scipy_validation()`: Test against `scipy.stats.scoreatpercentile` for various percentiles (e.g., 25th, 75th) on different datasets.
- **Mathematical Property**:
    - [ ] `test_percentile_boundary_properties()`: Test `percentile(data, 0) == minimum(data)` and `percentile(data, 100) == maximum(data)`.
    - [ ] `test_percentile_median_property()`: Test `percentile(data, 50) == median(data)`.
- **Parameter Validation**:
    - [ ] `test_percentile_empty_array()`: Test that an empty `data` array pushes an error and returns `NAN`.
    - [ ] `test_percentile_value_out_of_range()`: Test that a `percentile_value` outside `[0, 100]` pushes an error and returns `NAN`.

### `summary_statistics()`
- **Behavioral & Integration Test**:
    - [ ] `test_summary_statistics_consistency()`: Test that the values in the returned dictionary match the results from calling the individual functions (`mean`, `median`, `variance`, etc.) on the same dataset.
- **Parameter Validation**:
    - [ ] `test_summary_statistics_empty_array()`: Test that an empty `data` array pushes an error and returns an empty `Dictionary`.

---

## III. CDF Functions (`cdf_functions.gd`)

This module has excellent Scipy and Mathematical Property test coverage. The only gaps are in parameter validation for some of the newer functions.

- **Parameter Validation**:
    - [ ] **`logistic_cdf`**: Test invalid `scale` (<= 0).
    - [ ] **`lognormal_cdf`**: Test invalid `sigma` (<= 0).
    - [ ] **`cauchy_cdf`**: Test invalid `scale` (<= 0).
    - [ ] **`hypergeometric_cdf`**: Test invalid parameters (e.g., `N < 0`, `K < 0`, `n > N`).
    - [ ] **`gumbel_cdf`**: Test invalid `scale` (<= 0).

---

## IV. Distributions (`distributions.gd`)

The testing for this module has two major issues: a complete lack of Scipy validation and a misnamed test file.

-   **Missing Scipy Validation / Statistical Validation**:
    -   **Issue**: There are no tests to verify that the random number generators produce distributions with the correct statistical properties (e.g., mean, variance) over a large sample, which would be the equivalent of Scipy validation for this module.
    -   **Recommendation**: A new test file, `tests/core/distributions/statistical_validation_test.gd`, should be created.
    -   **Specific Gaps**: For every distribution function (e.g., `randf_normal`, `randi_binomial`, etc.):
        -   [ ] Generate a large sample (e.g., 10,000 variates).
        -   [ ] Use `BasicStats` functions to calculate the sample's `mean` and `variance`.
        -   [ ] Assert that the calculated `mean` and `variance` are approximately equal to the theoretical `mean` and `variance` defined by the function's input parameters.

-   **Redundant and Misnamed Test File**:
    -   **File**: `tests/core/distributions/scipy_validation_test.gd`
    -   **Issue**: This file is named `scipy_validation_test.gd` but contains no validation against Scipy. It performs basic, deterministic checks that are largely redundant with `mathematical_property_test.gd`.
    -   **Recommendation**: The file should be removed. Any unique, valuable checks it performs should be migrated to `mathematical_property_test.gd`.

---

## V. Error Functions (`error_functions.gd`)

This module has significant gaps in both Scipy validation and mathematical properties for many of its functions. The duplicated `gamma` and `log_gamma` functions should also be addressed.

- **`erf`**:
    - [ ] **Mathematical Property**: Test odd function property `erf(-x) == -erf(x)`.
    - [ ] **Parameter Validation**: Although it takes any float, add tests for `INF` and `NAN` inputs.
- **`erfc`**:
    - [ ] **Mathematical Property**: Test property `erfc(x) + erf(x) == 1`.
    - [ ] **Parameter Validation**: Add tests for `INF` and `NAN` inputs.
- **`erf_inv`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.erfinv`.
- **`erfc_inv`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.erfcinv`.
- **`gamma`**:
    - [ ] **Mathematical Property**: Test reflection formula `Γ(z)Γ(1-z) = π/sin(πz)`.
- **`log_gamma`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.gammaln`.
- **`incomplete_gamma`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.gammainc`.
    - [ ] **Mathematical Property**: Test that `incomplete_gamma(a, inf) == gamma(a)`.
    - [ ] **Parameter Validation**: Test `a <= 0` and `x < 0`.
- **`regularized_gamma_q`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.gammaincc`.
    - [ ] **Mathematical Property**: Test that `regularized_gamma_p + regularized_gamma_q == 1`.
    - [ ] **Parameter Validation**: Test `a <= 0` and `x < 0`.
- **`beta_function`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.beta`.
    - [ ] **Mathematical Property**: Test symmetry `B(a, b) == B(b, a)`.
    - [ ] **Parameter Validation**: Test `a <= 0` or `b <= 0`.
- **`incomplete_beta`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.betainc`.
    - [ ] **Mathematical Property**: Test `incomplete_beta(x, a, b) + incomplete_beta(1-x, b, a) == beta_function(a, b)`.
    - [ ] **Parameter Validation**: Test `a <= 0`, `b <= 0`, or `x` not in `[0, 1]`.
- **`regularized_beta`**:
    - [ ] **Scipy Validation**: Test against `scipy.special.betainc`. (Note: Scipy's `betainc` is the regularized version).
    - [ ] **Mathematical Property**: Test boundary conditions `regularized_beta(0, a, b) == 0` and `regularized_beta(1, a, b) == 1`.
    - [ ] **Parameter Validation**: Test `a <= 0`, `b <= 0`, or `x` not in `[0, 1]`.

---

## VI. PMF/PDF Functions (`pmf_pdf_functions.gd`)

This module has good coverage overall, but several functions are completely untested.

- **`geometric_pmf`**:
    - [ ] **Scipy Validation**: Test against `scipy.stats.geom.pmf`.
    - [ ] **Mathematical Property**: Test that the sum over the support equals 1.
    - [ ] **Parameter Validation**: Test invalid `p_prob` (not in `(0, 1]`).
- **`hypergeometric_pmf`**:
    - [ ] **Scipy Validation**: Test against `scipy.stats.hypergeom.pmf`.
    - [ ] **Mathematical Property**: Test symmetry property.
    - [ ] **Parameter Validation**: Test invalid parameters (e.g., `k > n`, `n > K`, `K > N`).
- **`cauchy_pdf`**:
    - [ ] **Scipy Validation**: Test against `scipy.stats.cauchy.pdf`.
    - [ ] **Mathematical Property**: Test that the mode is at `x0`.
    - [ ] **Parameter Validation**: Test `scale <= 0`.
- **`f_pdf`**:
    - [ ] **Scipy Validation**: Add data-driven tests against `scipy.stats.f.pdf`.
    - [ ] **Mathematical Property**: Test the relationship between the F-distribution and the Beta distribution.
- **`pareto_pdf`**:
    - [ ] **Scipy Validation**: Test against `scipy.stats.pareto.pdf`.
    - [ ] **Mathematical Property**: Test that the mode is at `scale`.
    - [ ] **Parameter Validation**: Test `scale <= 0` or `shape <= 0`.

---

## VII. PPF Functions (`ppf_functions.gd`)

The test coverage for the `ppf_functions.gd` module is **critically insufficient**. Many essential PPF functions are not implemented at all, and those that exist have testing gaps.

### 1. Missing Function Implementations
- A large number of distributions implemented in `CdfFunctions` are missing a corresponding `ppf` function. The following need to be implemented and tested:
    - [ ] `beta_ppf`
    - [ ] `gamma_ppf`
    - [ ] `chi_square_ppf`
    - [ ] `f_ppf`
    - [ ] `t_ppf`
    - [ ] `binomial_ppf`
    - [ ] `poisson_ppf`
    - [ ] `geometric_ppf`
    - [ ] `negative_binomial_ppf`

### 2. Gaps in Existing Functions

- **`uniform_ppf`**:
    - [ ] **Parameter Validation**: Test invalid probability `p` and `a > b`.
- **`pareto_ppf`**:
    - [ ] **Mathematical Property**: Test CDF-PPF round-trip consistency. Test boundary conditions.
    - [ ] **Parameter Validation**: Test invalid `p` and non-positive `scale` or `shape`.
- **`weibull_ppf`**:
    - [ ] **Parameter Validation**: Test invalid `p` and non-positive `scale` or `shape`.

---

## VIII. Sampling Gen (`sampling_gen.gd`)

This module suffers from a complete lack of Scipy validation and a misnamed test file.

-   **Missing Scipy Validation**:
    -   **Issue**: No functions in this module are validated against a known standard library like Scipy's `qmc` (Quasi-Monte Carlo) module.
    -   **Recommendation**: A new `scipy_validation_test.gd` should be created (after renaming/refactoring the existing one).
    -   **Specific Gaps**:
        -   [ ] **`generate_samples` (Sobol)**: Validate output against `scipy.stats.qmc.Sobol`.
        -   [ ] **`generate_samples` (Halton)**: Validate output against `scipy.stats.qmc.Halton`.
        -   [ ] **`generate_samples` (Latin Hypercube)**: Validate against `scipy.stats.qmc.LatinHypercube`.

-   **Redundant and Misnamed Test File**:
    -   **File**: `tests/core/sampling_gen/scipy_validation_test.gd`
    -   **Issue**: This file is named for Scipy validation but contains property and determinism tests, which are already covered in the mathematical properties test file.
    -   **Recommendation**: Rename this file to `property_test_suite.gd` or similar and merge the few unique tests from `mathematical_property_test.gd` into it.

- **`coordinated_batch_shuffles`**:
    - [ ] **Parameter Validation**: Test invalid `deck_size` or `n_shuffles` (e.g., negative).