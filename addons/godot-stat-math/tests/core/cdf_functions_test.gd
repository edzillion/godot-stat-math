# res://addons/godot-stat-math/tests/core/cdf_functions_test.gd
class_name CdfFunctionsTest extends GdUnitTestSuite


# simple_data eliminated - using scipy-generated CDF_TEST_DATA instead
const CDF_TEST_DATA = preload("res://addons/godot-stat-math/tables/cdf_test_data.gd")

# Manual data transformation function ELIMINATED - data structure optimized for direct usage

# =============================================================================
# PHASE 3: ENHANCED CDF TESTING - SCIPY VALIDATED DATA
# =============================================================================

## Tests normal CDF function with comprehensive test data
func test_normal_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["normal_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Tests exponential CDF function with comprehensive test data
func test_exponential_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["exponential_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.exponential_cdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Tests gamma CDF function with comprehensive test data
func test_gamma_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["gamma_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.gamma_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Tests beta CDF function with comprehensive test data
func test_beta_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["beta_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.beta_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Tests Chi-Square CDF function with comprehensive test data
func test_chi_square_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["chi_square_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.chi_square_cdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Tests t-distribution CDF function with comprehensive test data
func test_t_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["t_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.t_cdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.NUMERICAL_TOLERANCE)

## Tests F-distribution CDF function with comprehensive test data
func test_f_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["f_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.f_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.NUMERICAL_TOLERANCE)

## Tests Weibull CDF function with comprehensive test data
func test_weibull_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["weibull_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.weibull_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

# =============================================================================
# PHASE 3: MONOTONICITY TESTING FOR ALL DISTRIBUTIONS  
# =============================================================================

## Tests normal CDF monotonicity 
func test_normal_cdf_monotonicity() -> void:
	var test_points: Array[float] = [-3.0, -1.0, 0.0, 1.0, 3.0]
	var prev_cdf: float = -1.0
	
	for x in test_points:
		var current_cdf: float = StatMath.CdfFunctions.normal_cdf(x, 0.0, 1.0)
		assert_float(current_cdf).is_greater_equal(prev_cdf)
		assert_float(current_cdf).is_between(0.0, 1.0)
		prev_cdf = current_cdf

## Tests exponential CDF monotonicity
func test_exponential_cdf_monotonicity() -> void:
	var test_points: Array[float] = [0.0, 0.5, 1.0, 2.0, 5.0]
	var prev_cdf: float = -1.0
	
	for x in test_points:
		var current_cdf: float = StatMath.CdfFunctions.exponential_cdf(x, 1.0)
		assert_float(current_cdf).is_greater_equal(prev_cdf)
		assert_float(current_cdf).is_between(0.0, 1.0)
		prev_cdf = current_cdf

## Tests gamma CDF monotonicity
func test_gamma_cdf_monotonicity() -> void:
	var test_points: Array[float] = [0.0, 0.5, 1.0, 2.0, 4.0]
	var prev_cdf: float = -1.0
	
	for x in test_points:
		var current_cdf: float = StatMath.CdfFunctions.gamma_cdf(x, 2.0, 1.0)
		assert_float(current_cdf).is_greater_equal(prev_cdf)
		assert_float(current_cdf).is_between(0.0, 1.0)
		prev_cdf = current_cdf

# =============================================================================
# PHASE 3: BOUNDARY VALUE TESTING
# =============================================================================

## Tests CDF boundary behavior for normal distribution
func test_normal_cdf_boundary_values() -> void:
	# Extreme negative values should approach 0
	var result_neg: float = StatMath.CdfFunctions.normal_cdf(-1000.0, 0.0, 1.0)
	assert_float(result_neg).is_equal_approx(0.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Extreme positive values should approach 1  
	var result_pos: float = StatMath.CdfFunctions.normal_cdf(1000.0, 0.0, 1.0)
	assert_float(result_pos).is_equal_approx(1.0, StatMath.BOUNDARY_TOLERANCE)

## Tests CDF boundary behavior for exponential distribution
func test_exponential_cdf_boundary_values() -> void:
	# x = 0 should give CDF = 0
	var result_zero: float = StatMath.CdfFunctions.exponential_cdf(0.0, 1.0)
	assert_float(result_zero).is_equal_approx(0.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Negative x should give CDF = 0
	var result_neg: float = StatMath.CdfFunctions.exponential_cdf(-1.0, 1.0)
	assert_float(result_neg).is_equal_approx(0.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Large x should approach 1
	var result_large: float = StatMath.CdfFunctions.exponential_cdf(100.0, 1.0)
	assert_float(result_large).is_equal_approx(1.0, StatMath.BOUNDARY_TOLERANCE)

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

## Tests CDF values at median for normal distribution
func test_normal_cdf_median() -> void:
	# Standard normal median should be at x=0, giving CDF=0.5
	var result: float = StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 1.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

## Tests CDF values at median for exponential distribution  
func test_exponential_cdf_median() -> void:
	# Exponential median at ln(2) should give CDF = 0.5 exactly
	var ln_2: float = log(2.0)  # Mathematical constant: natural log of 2
	var result: float = StatMath.CdfFunctions.exponential_cdf(ln_2, 1.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

## Tests CDF values at median for uniform distribution
func test_uniform_cdf_median() -> void:
	# Uniform distribution midpoint should give CDF = 0.5 exactly
	var result: float = StatMath.CdfFunctions.uniform_cdf(4.0, 2.0, 6.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

## Individual parameter validation tests - ELIMINATED complex string-based match pattern

# Normal CDF validation tests
func test_normal_cdf_validation_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(1.0, 0.0, 0.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive for Normal CDF. Received: 0.0")

func test_normal_cdf_validation_sigma_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(1.0, 0.0, -1.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive for Normal CDF. Received: -1.0")

# Exponential CDF validation tests
func test_exponential_cdf_validation_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive for Exponential CDF. Received: 0.0")

func test_exponential_cdf_validation_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive for Exponential CDF. Received: -1.0")

# Gamma CDF validation tests
func test_gamma_cdf_validation_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 0.0, 1.0)
	await assert_error(test_call).is_push_error("Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=0.0, theta_scale=1.0")

func test_gamma_cdf_validation_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 1.0, 0.0)
	await assert_error(test_call).is_push_error("Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=1.0, theta_scale=0.0")

# Beta CDF validation tests
func test_beta_cdf_validation_alpha_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, 0.0, 1.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=0.0, beta_param=1.0")

func test_beta_cdf_validation_beta_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, 1.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=1.0, beta_param=-1.0")

# Chi-Square CDF validation tests
func test_chi_square_cdf_validation_df_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: 0.0")

func test_chi_square_cdf_validation_df_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, -1.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: -1.0")

# t-Distribution CDF validation tests
func test_t_cdf_validation_df_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: 0.0")

func test_t_cdf_validation_df_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, -5.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: -5.0")

# F-Distribution CDF validation tests
func test_f_cdf_validation_d1_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 0.0, 5.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=0.0, d2_df=5.0")

func test_f_cdf_validation_d2_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 5.0, -1.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=5.0, d2_df=-1.0")

# Weibull CDF validation tests
func test_weibull_cdf_validation_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive for Weibull CDF. Received: 0.0")

func test_weibull_cdf_validation_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 1.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive for Weibull CDF. Received: -1.0")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

# --- Uniform CDF ---
func test_uniform_cdf_basic_range() -> void:
	var a: float = 2.0
	var b: float = 5.0
	var x: float = 3.0
	var result: float = StatMath.CdfFunctions.uniform_cdf(x, a, b)
	assert_float(result).is_equal_approx((x - a) / (b - a), StatMath.FLOAT_TOLERANCE)

func test_uniform_cdf_x_below_a() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(1.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_uniform_cdf_x_above_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(6.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_uniform_cdf_a_equals_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(2.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_uniform_cdf_invalid_a_greater_than_b() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.uniform_cdf(2.0, 5.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter a must be less than or equal to b for Uniform CDF. Received a=5.0, b=2.0")

# --- Normal CDF ---
func test_normal_cdf_standard_normal() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(0.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_normal_cdf_mu_sigma() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(2.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_normal_cdf_known_value() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["normal_cdf"]
	var case: Dictionary = test_data[0]  # First case: normal_cdf(1.96, 0.0, 1.0)
	var result: float = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_normal_cdf_invalid_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 0.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive for Normal CDF. Received: 0.0")

# --- Exponential CDF ---
func test_exponential_cdf_typical() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["exponential_cdf"]
	var case: Dictionary = test_data[1]  # Second case: exponential_cdf(1.0, 1.0)
	var result: float = StatMath.CdfFunctions.exponential_cdf(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_exponential_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.exponential_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_exponential_cdf_invalid_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive for Exponential CDF. Received: 0.0")

# --- Beta CDF ---
func test_beta_cdf_symmetric() -> void:
	# For symmetric alpha=beta, CDF at 0.5 should be 0.5
	var result: float = StatMath.CdfFunctions.beta_cdf(0.5, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_beta_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_beta_cdf_x_one() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_beta_cdf_invalid_alpha_beta() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=-1.0, beta_param=2.0")

# --- Gamma CDF ---
func test_gamma_cdf_known_value() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["gamma_cdf"]
	var case: Dictionary = test_data[0]  # First case: gamma_cdf(2.0, 2.0, 1.0)
	var result: float = StatMath.CdfFunctions.gamma_cdf(case["params"][0], case["params"][1], case["params"][2])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_gamma_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.gamma_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_gamma_cdf_invalid_shape_scale() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=0.0, theta_scale=2.0")

# --- Chi-Square CDF ---
func test_chi_square_cdf_known_value() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["chi_square_cdf"]
	var case: Dictionary = test_data[0]  # First case: chi_square_cdf(3.841, 1.0)
	var result: float = StatMath.CdfFunctions.chi_square_cdf(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_chi_square_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.chi_square_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_chi_square_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: 0.0")

# --- F-Distribution CDF ---
func test_f_cdf_known_value() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["f_cdf"]
	var case: Dictionary = test_data[0]  # First case: f_cdf(1.5, 2.0, 2.0)
	var result: float = StatMath.CdfFunctions.f_cdf(case["params"][0], case["params"][1], case["params"][2])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_f_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.f_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_f_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=0.0, d2_df=2.0")

# --- Student's t-Distribution CDF ---
func test_t_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.t_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_t_cdf_known_value() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["t_cdf"]
	var case: Dictionary = test_data[0]  # First case: t_cdf(1.0, 10.0)
	var result: float = StatMath.CdfFunctions.t_cdf(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.NUMERICAL_TOLERANCE) # Slightly lower tolerance for t-dist approximation

func test_t_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: 0.0")

# --- Binomial CDF ---
func test_binomial_cdf_known_value() -> void:
	# Test standard case with p=0.5 - should equal 0.5 for symmetric distribution
	var result: float = StatMath.CdfFunctions.binomial_cdf(2, 5, 0.5)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_binomial_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(-1, 5, 0.5)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_binomial_cdf_k_ge_n() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(5, 5, 0.5)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_binomial_cdf_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, -1, 0.5)
	await assert_error(test_call).is_push_error("Number of trials (n_trials) must be non-negative. Received: -1")

func test_binomial_cdf_invalid_p() -> void:
	var test_call_low: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, 5, -0.1)
	await assert_error(test_call_low).is_push_error("Probability (p_prob) must be between 0.0 and 1.0. Received: -0.1")
	
	var test_call_high: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, 5, 1.1)
	await assert_error(test_call_high).is_push_error("Probability (p_prob) must be between 0.0 and 1.0. Received: 1.1")

# --- Poisson CDF ---
func test_poisson_cdf_known_value() -> void:
	# Test standard case where k equals lambda parameter
	var result: float = StatMath.CdfFunctions.poisson_cdf(2, 2.0)
	assert_float(result).is_greater(0.5) # Should be above 0.5 for k=lambda case
	assert_float(result).is_less(0.8) # Reasonable upper bound

func test_poisson_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.poisson_cdf(-1, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_poisson_cdf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.poisson_cdf(2, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be non-negative for Poisson CDF. Received: -1.0")

# --- Geometric CDF ---
func test_geometric_cdf_known_value() -> void:
	# Value from 1 - (1-p)^k = 1 - (0.5)^3 = 0.875
	var result: float = StatMath.CdfFunctions.geometric_cdf(3, 0.5)
	assert_float(result).is_equal_approx(0.875, StatMath.FLOAT_TOLERANCE)

func test_geometric_cdf_k_less_than_1() -> void:
	var result: float = StatMath.CdfFunctions.geometric_cdf(0, 0.5)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_geometric_cdf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.geometric_cdf(2, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")

# --- Negative Binomial CDF ---
func test_negative_binomial_cdf_known_value() -> void:
	# P(k<=5; r=3, p=0.5) = sum of PMF for k=3,4,5
	# PMF(k;r,p) = C(k-1,r-1) * p^r * (1-p)^(k-r)
	# P(3) = C(2,2)*0.5^3*0.5^0 = 0.125
	# P(4) = C(3,2)*0.5^3*0.5^1 = 3 * 0.125 * 0.5 = 0.1875
	# P(5) = C(4,2)*0.5^3*0.5^2 = 6 * 0.125 * 0.25 = 0.1875
	# Sum = 0.125 + 0.1875 + 0.1875 = 0.5
	var result: float = StatMath.CdfFunctions.negative_binomial_cdf(5, 3, 0.5)
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_negative_binomial_cdf_k_less_than_r() -> void:
	var result: float = StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.5)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_negative_binomial_cdf_invalid_r() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 0, 0.5)
	await assert_error(test_call).is_push_error("Number of successes (r_successes) must be positive. Received: 0")

func test_negative_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")

# --- Pareto CDF ---
func test_pareto_cdf_x_equals_scale() -> void:
	var result: float = StatMath.CdfFunctions.pareto_cdf(2.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_pareto_cdf_basic_calculation() -> void:
	# For x = 4, scale = 2, shape = 3: F(4) = 1 - (2/4)^3 = 1 - 0.125 = 0.875
	var result: float = StatMath.CdfFunctions.pareto_cdf(4.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.875, StatMath.FLOAT_TOLERANCE)

func test_pareto_cdf_x_below_scale() -> void:
	var result: float = StatMath.CdfFunctions.pareto_cdf(1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_pareto_cdf_large_x() -> void:
	# For very large x, CDF should approach 1
	var result: float = StatMath.CdfFunctions.pareto_cdf(1000.0, 2.0, 3.0)
	assert_float(result).is_greater(0.99)
	assert_float(result).is_less_equal(1.0)

func test_pareto_cdf_different_shapes() -> void:
	var x: float = 4.0
	var scale: float = 2.0
	
	# Higher shape parameter = faster decay, higher CDF for same x
	var cdf_shape_1: float = StatMath.CdfFunctions.pareto_cdf(x, scale, 1.0)
	var cdf_shape_3: float = StatMath.CdfFunctions.pareto_cdf(x, scale, 3.0)
	var cdf_shape_5: float = StatMath.CdfFunctions.pareto_cdf(x, scale, 5.0)
	
	assert_float(cdf_shape_1).is_less(cdf_shape_3)
	assert_float(cdf_shape_3).is_less(cdf_shape_5)

func test_pareto_cdf_monotonicity() -> void:
	# CDF should be monotonically increasing
	var scale: float = 2.0
	var shape: float = 3.0
	
	var x1: float = 2.5
	var x2: float = 3.0
	var x3: float = 4.0
	var x4: float = 6.0
	
	var cdf1: float = StatMath.CdfFunctions.pareto_cdf(x1, scale, shape)
	var cdf2: float = StatMath.CdfFunctions.pareto_cdf(x2, scale, shape)
	var cdf3: float = StatMath.CdfFunctions.pareto_cdf(x3, scale, shape)
	var cdf4: float = StatMath.CdfFunctions.pareto_cdf(x4, scale, shape)
	
	assert_float(cdf1).is_less_equal(cdf2)
	assert_float(cdf2).is_less_equal(cdf3)
	assert_float(cdf3).is_less_equal(cdf4)

func test_pareto_cdf_bounds() -> void:
	# CDF should always be between 0 and 1
	var test_cases: Array[Array] = [
		[2.0, 1.0, 0.5], [5.0, 3.0, 2.0], [10.0, 2.0, 4.0],
		[100.0, 10.0, 1.5], [1.5, 1.0, 10.0]
	]
	
	for case in test_cases:
		var x: float = case[0]
		var scale: float = case[1]
		var shape: float = case[2]
		
		var result: float = StatMath.CdfFunctions.pareto_cdf(x, scale, shape)
		
		assert_float(result).is_greater_equal(0.0)
		assert_float(result).is_less_equal(1.0)
		assert_bool(is_nan(result)).is_false()

func test_pareto_cdf_deterministic() -> void:
	# Same parameters should give same results
	var x: float = 5.0
	var scale: float = 2.0
	var shape: float = 3.0
	
	var result1: float = StatMath.CdfFunctions.pareto_cdf(x, scale, shape)
	var result2: float = StatMath.CdfFunctions.pareto_cdf(x, scale, shape)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

func test_pareto_cdf_invalid_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive for Pareto CDF. Received: 0.0")

func test_pareto_cdf_invalid_scale_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive for Pareto CDF. Received: -1.0")

func test_pareto_cdf_invalid_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive for Pareto CDF. Received: 0.0")

func test_pareto_cdf_invalid_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive for Pareto CDF. Received: -1.0")

# --- Game Development Use Cases for Pareto CDF ---

func test_pareto_cdf_wealth_distribution_probability() -> void:
	# Example: probability that a player's wealth is below a certain threshold
	var wealth_threshold: float = 500.0
	var min_wealth: float = 100.0 # scale parameter
	var inequality_factor: float = 2.0 # shape parameter (lower = more inequality)
	
	var prob_below_threshold: float = StatMath.CdfFunctions.pareto_cdf(wealth_threshold, min_wealth, inequality_factor)
	
	# Should be a valid probability
	assert_float(prob_below_threshold).is_between(0.0, 1.0)
	
	# With shape=2 and threshold=5*scale, should be significant probability
	assert_float(prob_below_threshold).is_greater(0.5)

func test_pareto_cdf_loot_rarity_distribution() -> void:
	# Example: probability of getting loot below certain value
	var loot_values: Array[float] = [10.0, 50.0, 100.0, 500.0]
	var min_loot_value: float = 10.0
	var rarity_shape: float = 3.0 # Higher shape = less extreme values
	
	var probabilities: Array[float] = []
	for loot_value in loot_values:
		var prob: float = StatMath.CdfFunctions.pareto_cdf(loot_value, min_loot_value, rarity_shape)
		probabilities.append(prob)
	
	# Probabilities should increase with loot value
	for i in range(probabilities.size() - 1):
		assert_float(probabilities[i]).is_less_equal(probabilities[i + 1])
	
	# At minimum value, probability should be 0
	assert_float(probabilities[0]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_pareto_cdf_damage_resistance_calculation() -> void:
	# Example: probability that damage dealt is below player's resistance
	var player_resistance: float = 75.0
	var min_damage: float = 25.0
	var damage_scaling: float = 2.5
	
	var prob_resist: float = StatMath.CdfFunctions.pareto_cdf(player_resistance, min_damage, damage_scaling)
	
	# Should be valid probability
	assert_float(prob_resist).is_between(0.0, 1.0)
	
	# Can use this probability for resist chance calculations
	assert_bool(prob_resist > 0.0).is_true()

func test_pareto_cdf_market_price_analysis() -> void:
	# Example: analyzing probability of items being below market price
	var market_price: float = 200.0
	var base_item_value: float = 50.0
	var market_volatility: float = 1.5 # Lower shape = more price volatility
	
	var prob_below_market: float = StatMath.CdfFunctions.pareto_cdf(market_price, base_item_value, market_volatility)
	
	# Should be valid probability for economic calculations
	assert_float(prob_below_market).is_between(0.0, 1.0)
	
	# With low shape parameter, most items should be near base value
	assert_float(prob_below_market).is_greater(0.3)

# --- Weibull CDF ---
func test_weibull_cdf_known_value() -> void:
	# Using table data instead of hardcoded values
	var test_data: Array = CDF_TEST_DATA.VALUES["weibull_cdf"]
	var case: Dictionary = test_data[0]  # First case: weibull_cdf(1.5, 1.0, 2.0)
	var result: float = StatMath.CdfFunctions.weibull_cdf(case["params"][0], case["params"][1], case["params"][2])
	assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

func test_weibull_cdf_basic_calculation() -> void:
	# For x = 2, scale = 2, shape = 2: F(2) = 1 - exp(-(2/2)^2) = 1 - exp(-1) ≈ 0.632
	var result: float = StatMath.CdfFunctions.weibull_cdf(2.0, 2.0, 2.0)
	var expected: float = 1.0 - exp(-1.0)
	assert_float(result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

func test_weibull_cdf_x_below_zero() -> void:
	var result: float = StatMath.CdfFunctions.weibull_cdf(-1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_weibull_cdf_exponential_case() -> void:
	# When shape = 1, Weibull becomes exponential: F(x) = 1 - exp(-x/λ)
	var x: float = 3.0
	var scale: float = 2.0 # This is lambda_scale
	var shape: float = 1.0
	
	var weibull_result: float = StatMath.CdfFunctions.weibull_cdf(x, scale, shape)
	# For exponential, lambda_param is the rate, which is 1.0 / scale
	var exponential_result: float = StatMath.CdfFunctions.exponential_cdf(x, 1.0 / scale)
	
	assert_float(weibull_result).is_equal_approx(exponential_result, StatMath.FLOAT_TOLERANCE)

func test_weibull_cdf_monotonicity() -> void:
	# CDF should be monotonically increasing
	var scale: float = 2.0
	var shape: float = 2.0
	
	var x1: float = 0.5
	var x2: float = 1.0
	var x3: float = 2.0
	var x4: float = 4.0
	
	var cdf1: float = StatMath.CdfFunctions.weibull_cdf(x1, scale, shape)
	var cdf2: float = StatMath.CdfFunctions.weibull_cdf(x2, scale, shape)
	var cdf3: float = StatMath.CdfFunctions.weibull_cdf(x3, scale, shape)
	var cdf4: float = StatMath.CdfFunctions.weibull_cdf(x4, scale, shape)
	
	assert_float(cdf1).is_less_equal(cdf2)
	assert_float(cdf2).is_less_equal(cdf3)
	assert_float(cdf3).is_less_equal(cdf4)

func test_weibull_cdf_bounds() -> void:
	# CDF should always be between 0 and 1
	var test_cases: Array[Array] = [
		[1.0, 1.0, 0.5], [2.0, 3.0, 2.0], [5.0, 2.0, 4.0],
		[10.0, 10.0, 1.5], [0.5, 1.0, 10.0]
	]
	
	for case in test_cases:
		var x: float = case[0]
		var shape: float = case[1]
		var scale: float = case[2]
		
		var result: float = StatMath.CdfFunctions.weibull_cdf(x, scale, shape)
		
		assert_float(result).is_greater_equal(0.0)
		assert_float(result).is_less_equal(1.0)
		assert_bool(is_nan(result)).is_false()

func test_weibull_cdf_invalid_shape() -> void:
	var test_call_zero: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 1.0, 0.0)
	await assert_error(test_call_zero).is_push_error("Shape parameter must be positive for Weibull CDF. Received: 0.0")

	var test_call_neg: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 1.0, -1.0)
	await assert_error(test_call_neg).is_push_error("Shape parameter must be positive for Weibull CDF. Received: -1.0")

func test_weibull_cdf_invalid_scale() -> void:
	var test_call_zero: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 0.0, 1.0)
	await assert_error(test_call_zero).is_push_error("Scale parameter must be positive for Weibull CDF. Received: 0.0")

	var test_call_neg: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, -1.0, 1.0)
	await assert_error(test_call_neg).is_push_error("Scale parameter must be positive for Weibull CDF. Received: -1.0")

# --- Game Development Use Cases for Weibull CDF ---

func test_weibull_cdf_equipment_failure_probability() -> void:
	# Example: probability that equipment fails before a certain time
	var time_threshold: float = 800.0  # Hours
	var characteristic_life: float = 1000.0  # Hours
	var wear_pattern: float = 2.5  # Increasing failure rate
	
	var failure_probability: float = StatMath.CdfFunctions.weibull_cdf(time_threshold, characteristic_life, wear_pattern)
	
	# Should be a valid probability
	assert_float(failure_probability).is_between(0.0, 1.0)
	
	# Should be reasonable for this scenario (some probability of early failure)
	assert_float(failure_probability).is_greater(0.1)
	assert_float(failure_probability).is_less(0.9)

func test_weibull_cdf_survival_analysis() -> void:
	# Example: probability of surviving less than a certain time
	var survival_times: Array[float] = [100.0, 300.0, 600.0, 1000.0]
	var base_survival: float = 500.0  # Base survival time
	var hazard_pattern: float = 1.8  # Increasing hazard
	
	var probabilities: Array[float] = []
	for time in survival_times:
		var prob: float = StatMath.CdfFunctions.weibull_cdf(time, base_survival, hazard_pattern)
		probabilities.append(prob)
	
	# Probabilities should increase with time
	for i in range(probabilities.size() - 1):
		assert_float(probabilities[i]).is_less_equal(probabilities[i + 1])
	
	# All should be valid probabilities
	for prob in probabilities:
		assert_float(prob).is_between(0.0, 1.0)

func test_weibull_cdf_wind_speed_distribution() -> void:
	# Example: probability of wind speed being below threshold (Rayleigh case)
	var wind_threshold: float = 20.0  # km/h
	var characteristic_wind: float = 15.0  # km/h
	var rayleigh_shape: float = 2.0  # Rayleigh distribution
	
	var prob_below_threshold: float = StatMath.CdfFunctions.weibull_cdf(wind_threshold, characteristic_wind, rayleigh_shape)
	
	# Should be valid probability
	assert_float(prob_below_threshold).is_between(0.0, 1.0)
	
	# Should be reasonable for wind speed analysis
	assert_float(prob_below_threshold).is_greater(0.3)

func test_weibull_cdf_component_reliability() -> void:
	# Example: reliability analysis for electronic components
	var operating_time: float = 4000.0  # Hours
	var design_life: float = 5000.0  # Hours
	var reliability_shape: float = 3.0  # Sharp wear-out
	
	var failure_probability: float = StatMath.CdfFunctions.weibull_cdf(operating_time, design_life, reliability_shape)
	
	# Should be valid probability
	assert_float(failure_probability).is_between(0.0, 1.0)
	
	# Before design life, failure probability should be relatively low
	assert_float(failure_probability).is_less(0.7)

func test_weibull_cdf_quest_completion_analysis() -> void:
	# Example: analyzing quest completion time distributions
	var completion_times: Array[float] = [30.0, 60.0, 120.0, 180.0]
	var typical_time: float = 90.0  # Minutes
	var difficulty_curve: float = 2.2  # Increasing difficulty
	
	var completion_probabilities: Array[float] = []
	for time in completion_times:
		var prob: float = StatMath.CdfFunctions.weibull_cdf(time, typical_time, difficulty_curve)
		completion_probabilities.append(prob)
	
	# Should maintain monotonicity
	for i in range(completion_probabilities.size() - 1):
		assert_float(completion_probabilities[i]).is_less_equal(completion_probabilities[i + 1])
	
	# All should be valid
	for prob in completion_probabilities:
		assert_float(prob).is_between(0.0, 1.0)

func test_weibull_cdf_resource_depletion_modeling() -> void:
	# Example: modeling resource node depletion probability
	var extraction_time: float = 1500.0  # Time units
	var resource_lifetime: float = 2000.0  # Expected lifetime
	var depletion_pattern: float = 1.2  # Slight acceleration
	
	var depletion_probability: float = StatMath.CdfFunctions.weibull_cdf(extraction_time, resource_lifetime, depletion_pattern)
	
	# Should be valid probability
	assert_float(depletion_probability).is_between(0.0, 1.0)
	
	# Should be reasonable for resource management
	assert_float(depletion_probability).is_greater(0.2)
	assert_float(depletion_probability).is_less(0.9)

func test_weibull_cdf_network_latency_analysis() -> void:
	# Example: network latency spike duration analysis
	var latency_threshold: float = 100.0  # Milliseconds
	var typical_spike_duration: float = 75.0  # Milliseconds
	var recovery_pattern: float = 2.8  # Sharp recovery
	
	var prob_short_spike: float = StatMath.CdfFunctions.weibull_cdf(latency_threshold, typical_spike_duration, recovery_pattern)
	
	# Should be valid probability
	assert_float(prob_short_spike).is_between(0.0, 1.0)
	
	# Most spikes should be relatively short
	assert_float(prob_short_spike).is_greater(0.4) 
