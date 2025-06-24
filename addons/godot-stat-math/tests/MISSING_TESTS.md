# Missing Tests Documentation

This document tracks functionality that lacks proper test coverage and needs test implementation. It is organized by module, then by function, detailing missing tests for each category based on a full codebase review.

---

## I. Helper Functions (`helper_functions.gd`)

**Note on `gamma` and `log_gamma`**: These functions are duplicated in `error_functions.gd`. The tests below should target the `helper_functions.gd` implementation, and the versions in `error_functions.gd` should be deprecated and removed to resolve the code smell.

### `gamma_function()`
- **Scipy Validation**:
    - [*] `test_gamma_function_scipy_validation()`: Test against `scipy.special.gamma` with various positive float values (e.g., 0.5, 1.5, 2.7).
- **Mathematical Property**:
    - [*] `test_gamma_function_factorial_relationship()`: Test `Γ(n) = (n-1)!` for positive integers.
    - [*] `test_gamma_function_reflection_formula()`: Test `Γ(z)Γ(1-z) = π/sin(πz)` for non-integer z.
- **Parameter Validation**:
    - [*] `test_gamma_function_poles_at_non_positive_integers()`: Test that `gamma_function(0)`, `gamma_function(-1)`, etc., return `INF`.

### `log_gamma()`
- **Scipy Validation**:
    - [*] `test_log_gamma_scipy_validation()`: Test against `scipy.special.gammaln` with various positive float values.
- **Mathematical Property**:
    - [*] `test_log_gamma_recursion_property()`: Test `log_gamma(z+1) = log_gamma(z) + log(z)`.
- **Parameter Validation**:
    - [*] `test_log_gamma_invalid_input_non_positive()`: Test that `z <= 0` pushes an error and returns `NAN`.

### `log_binomial_coef()`
- **Mathematical Property**:
    - [*] `test_log_binomial_coef_symmetry()`: Test `log_binomial_coef(n, k) == log_binomial_coef(n, n-k)`.

### `log_beta_function_direct()`
- **Mathematical Property**:
    - [*] `test_log_beta_function_direct_symmetry()`: Test `log_beta(a, b) == log_beta(b, a)`.

### `get_cdf_value()`
- **Behavioral & Error Tests**:
    - [*] `test_get_cdf_value_with_valid_enum()`: Test with `StatMath.SupportedDistributions` enum.
    - [*] `test_get_cdf_value_with_valid_string()`: Test with string distribution names.
    - [*] `test_get_cdf_value_invalid_distribution_type()`: Test error for invalid type: "Invalid distribution type...".
    - [*] `test_get_cdf_value_unimplemented_distribution()`: Test error for unimplemented CDF: "CDF function not implemented...".

### `get_ppf_value()`
- **Behavioral & Error Tests**:
    - [*] `test_get_ppf_value_with_valid_enum()`: Test with `StatMath.SupportedDistributions` enum.
    - [*] `test_get_ppf_value_with_valid_string()`: Test with string distribution names.
    - [*] `test_get_ppf_value_invalid_distribution_type()`: Test error for invalid type: "Invalid distribution type...".
    - [*] `test_get_ppf_value_unimplemented_distribution()`: Test error for unimplemented PPF: "PPF function not implemented...".

### `validate_indices()`
- **Behavioral & Error Tests**:
    - [*] `test_validate_indices_valid_samples()`: Test with a valid index array.
    - [*] `test_validate_indices_negative_index()`: Test error: "Sample index must be non-negative".
    - [*] `test_validate_indices_index_too_large()`: Test error: "Sample index must be less than population size".

### `validate_unique_indices()`
- **Behavioral & Error Tests**:
    - [*] `test_validate_unique_indices_valid_unique_samples()`: Test with a valid unique index array.
    - [*] `test_validate_unique_indices_duplicate_found()`: Test error: "Sample indices must be unique".
    - [*] `test_validate_unique_indices_size_mismatch()`: Test error: "Number of unique indices must equal sample size".

### `convert_to_float_array()`
- **Property-based Tests**:
    - [*] `test_convert_to_float_array_valid_conversion()`: Test `Array` to `Array[float]` conversion.
    - [*] `test_convert_to_float_array_preserves_order()`: Test that element order is maintained.

---

## II. Basic Stats (`basic_stats.gd`)

### `percentile()`
- **Scipy Validation**:
    - [*] `test_percentile_scipy_validation()`: Test against `scipy.stats.scoreatpercentile` for various percentiles (e.g., 25th, 75th) on different datasets.
- **Mathematical Property**:
    - [*] `test_percentile_boundary_properties()`: Test `percentile(data, 0) == minimum(data)` and `percentile(data, 100) == maximum(data)`.
    - [*] `test_percentile_median_property()`: Test `percentile(data, 50) == median(data)`.
- **Parameter Validation**:
    - [*] `test_percentile_empty_array()`: Test that an empty `data` array pushes an error and returns `NAN`.
    - [*] `test_percentile_value_out_of_range()`: Test that a `percentile_value` outside `[0, 100]` pushes an error and returns `NAN`.

### `summary_statistics()`
- **Behavioral & Integration Test**:
    - [*] `test_summary_statistics_consistency()`: Test that the values in the returned dictionary match the results from calling the individual functions (`mean`, `median`, `variance`, etc.) on the same dataset.
- **Parameter Validation**:
    - [*] `test_summary_statistics_empty_array()`: Test that an empty `data` array pushes an error and returns an empty `Dictionary`.

---

## III. CDF Functions (`cdf_functions.gd`)

This module has excellent Scipy and Mathematical Property test coverage. Some functions need implementation.

### Functions Needing Implementation (Easy Wins - Distributions Exist):
- **`lognormal_cdf`** - *(**IMPLEMENTED**)*:
    - [*] **Implementation**: Add CDF function to complement existing distribution
    - [*] **Scipy Validation**: Test against `scipy.stats.lognorm.cdf`.
    - [*] **Parameter Validation**: Test invalid `sigma` (<= 0).
- **`cauchy_cdf`** - *(**IMPLEMENTED**)*:
    - [*] **Implementation**: Add CDF function to complement existing distribution  
    - [*] **Scipy Validation**: Test against `scipy.stats.cauchy.cdf`.
    - [*] **Parameter Validation**: Test invalid `scale` (<= 0).


### Existing Functions - Parameter Validation Only:
- **Parameter Validation**:
    - [*] **`uniform_cdf`**: Test invalid parameters (`a > b`).

---

## IV. Distributions (`distributions.gd`)

***RESOLVED**: Statistical validation has been implemented using a lean statistical validation approach.

-   ***Statistical Validation Implemented**:
    -   **File**: `tests/core/distributions/statistical_validation_test.gd` - **CREATED**
    -   **Approach**: Uses "lean statistical validation" with small sample sizes (200-1000) for fast execution while catching algorithmic flaws
    -   **Coverage**: Tests implemented for:
        -   [*] **Discrete distributions**: Bernoulli, Binomial, Geometric, Poisson (chi-squared tests and mean validation)
        -   [*] **Continuous distributions**: Normal, Exponential, Uniform, Beta (mean/variance validation)
    -   **Performance**: All tests execute in ~300ms, suitable for CI/CD pipelines
    -   **Method**: Uses fixed seeds for deterministic behavior and statistical tolerances based on standard errors

-   **File Renaming Required**:
    -   **File**: `tests/core/distributions/scipy_validation_test.gd` - **NEEDS RENAMING**
    -   **Issue**: This file is misnamed - it contains boundary condition tests, not scipy validation
    -   **Recommendation**: Rename to `boundary_condition_test.gd` to accurately reflect its contents

**Remaining Distribution Functions to Add**:

**Easy Additions** (standard mean/variance validation):
-   [*] **`randf_gamma`**: Test E[X] = α×θ, Var(X) = α×θ² 
-   [*] **`randf_erlang`**: Test E[X] = k/λ, Var(X) = k/λ²
-   [*] **`randi_uniform`**: Test discrete uniformity using chi-squared goodness-of-fit

**Special Cases** (require different validation approaches):
-   [*] **`randf_cauchy`**: Cannot test mean/variance (undefined due to heavy tails)
    -   [*] Test location parameter: median ≈ location
    -   [*] Test symmetry: count(x < location) ≈ count(x > location)
    -   [*] Test scale parameter effects on spread

**Custom Distributions** (behavioral validation):
-   [*] **`randi_pseudo`**: Test that success probability increases correctly by c_param each trial
-   [*] **`randi_seige`**: Test capture mechanics work as expected with win/loss probability changes

**Additional Continuous Distributions**:
-   [*] **`randf_triangular`**: Test mean = (a+b+c)/3, range validation
-   [*] **`randf_pareto`**: Test mean = αβ/(α-1) for α > 1, support validation  
-   [*] **`randf_weibull`**: Test mean = scale×Γ(1+1/shape), variance calculations
-   [*] **`randf_lognormal`**: Test mean = exp(μ + σ²/2), variance validation

---

## V. Error Functions (`error_functions.gd`)

This module has significant gaps in both Scipy validation and mathematical properties for many of its functions. The duplicated `gamma` and `log_gamma` functions should also be addressed.

- **`erf`**:
    - [*] **Mathematical Property**: Test odd function property `erf(-x) == -erf(x)`.
    - [*] **Parameter Validation**: Although it takes any float, add tests for `INF` and `NAN` inputs.
- **`erfc`**:
    - [*] **Mathematical Property**: Test property `erfc(x) + erf(x) == 1`.
    - [*] **Parameter Validation**: Add tests for `INF` and `NAN` inputs.
- **`erf_inv`**:
    - [*] **Scipy Validation**: Test against `scipy.special.erfinv`.
- **`erfc_inv`**:
    - [*] **Scipy Validation**: Test against `scipy.special.erfcinv`.
- **`gamma`**:
    - [*] **Mathematical Property**: Test reflection formula `Γ(z)Γ(1-z) = π/sin(πz)`.
- **`log_gamma`**:
    - [*] **Scipy Validation**: Test against `scipy.special.gammaln`.
**NOTE**: The following functions mentioned in the original missing tests list are implemented in other modules:
- `beta_function` → Implemented in `HelperFunctions.beta_function()` - **fully tested**
- `incomplete_beta` → Implemented in `HelperFunctions.incomplete_beta()` (regularized version) - **fully tested**  
- `lower_incomplete_gamma_regularized` → Implemented in `HelperFunctions.lower_incomplete_gamma_regularized()` - **fully tested**

**Functions not implemented anywhere**:
- `incomplete_gamma` (raw, non-regularized) - Not implemented
- `regularized_gamma_q` (upper incomplete gamma) - Not implemented, but could be computed as `1.0 - lower_incomplete_gamma_regularized(a, z)`

These belong in their respective module sections if they need to be implemented.

---

## VI. PMF/PDF Functions (`pmf_pdf_functions.gd`)

This module has good coverage overall, but several functions need implementation.

### Functions Needing Implementation (Easy Wins - Distributions Exist):
- **`geometric_pmf`** - *(**IMPLEMENTED**)*:
    - [*] **Implementation**: Add PMF function to complement existing distribution
    - [*] **Scipy Validation**: Test against `scipy.stats.geom.pmf`.
    - [*] **Mathematical Property**: Test that the sum over the support equals 1.
    - [*] **Parameter Validation**: Test invalid `p_prob` (not in `(0, 1]`).
- **`cauchy_pdf`** - *(**IMPLEMENTED**)*:
    - [*] **Implementation**: Add PDF function to complement existing distribution
    - [*] **Scipy Validation**: Test against `scipy.stats.cauchy.pdf`.
    - [*] **Mathematical Property**: Test that the mode is at `x0`.
    - [*] **Parameter Validation**: Test `scale <= 0`.
- **`pareto_pdf`** - *(**IMPLEMENTED**)*:
    - [*] **Implementation**: Add PDF function to complement existing distribution
    - [*] **Scipy Validation**: Test against `scipy.stats.pareto.pdf`.
    - [*] **Mathematical Property**: Test that the mode is at `scale`.
    - [*] **Parameter Validation**: Test `scale <= 0` or `shape <= 0`.


### Existing Functions Needing Tests:
- **`f_pdf`**:
    - [*] **Scipy Validation**: Add data-driven tests against `scipy.stats.f.pdf`.
    - [*] **Mathematical Property**: Test the relationship between the F-distribution and the Beta distribution.

---

## VII. PPF Functions (`ppf_functions.gd`)

The test coverage for the `ppf_functions.gd` module is **critically insufficient**. Many essential PPF functions are not implemented at all, and those that exist have testing gaps.

### 1. *RESOLVED: PPF Functions Are Actually Implemented
- **Issue**: These functions were incorrectly listed as missing - they all exist in `ppf_functions.gd`
- **Status**: All major PPF functions are implemented and need testing coverage:
    - [*] `beta_ppf` - EXISTS (line 131)
    - [*] `gamma_ppf` - EXISTS (line 249)
    - [*] `chi_square_ppf` - EXISTS (line 339)
    - [*] `f_ppf` - EXISTS (line 360)
    - [*] `t_ppf` - EXISTS (line 435)
    - [*] `binomial_ppf` - EXISTS (line 506)
    - [*] `poisson_ppf` - EXISTS (line 551)
    - [*] `geometric_ppf` - EXISTS (line 608)
    - [*] `negative_binomial_ppf` - EXISTS (line 651)

### 2. Gaps in Existing Functions

- **`uniform_ppf`**:
    - [*] **Parameter Validation**: Test invalid probability `p` and `a > b`.
- **`pareto_ppf`**:
    - [*] **Mathematical Property**: Test CDF-PPF round-trip consistency. Test boundary conditions.
    - [*] **Parameter Validation**: Test invalid `p` and non-positive `scale` or `shape`.
- **`weibull_ppf`**:
    - [*] **Parameter Validation**: Test invalid `p` and non-positive `scale` or `shape`.

---

## VIII. Sampling Gen (`sampling_gen.gd`)

This module suffers from a complete lack of Scipy validation and a misnamed test file.

-   **Missing Scipy Validation**:
    -   **Issue**: No functions in this module are validated against a known standard library like Scipy's `qmc` (Quasi-Monte Carlo) module.
    -   **Recommendation**: A new `scipy_validation_test.gd` should be created (after renaming/refactoring the existing one).
    -   **Specific Gaps**:
        -   [*] **`generate_samples` (Sobol)**: Validate output against `scipy.stats.qmc.Sobol`.
        -   [*] **`generate_samples` (Halton)**: Validate output against `scipy.stats.qmc.Halton`.
        -   [*] **`generate_samples` (Latin Hypercube)**: Validate against `scipy.stats.qmc.LatinHypercube`.

-   **Tests in wrong file**: *(RESOLVED)*
    -   **File**: `tests/core/sampling_gen/scipy_validation_test.gd` - **FIXED**
    -   **Issue**: This file was misnamed - it contained property and determinism tests, not scipy validation
    -   **Resolution**: Renamed the old file to `property_tests_backup.gd` and created proper scipy validation tests

- **`coordinated_batch_shuffles`**:
    - [*] **Parameter Validation**: Test invalid `deck_size` or `n_shuffles` (e.g., negative).