# res://addons/godot-stat-math/tests/core/cdf_functions/cdf_functions_parameter_validation_tests.gd
class_name CdfFunctionsParameterValidationTests extends GdUnitTestSuite


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

# --- Uniform CDF Parameter Validation ---
func test_uniform_cdf_invalid_a_greater_than_b() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.uniform_cdf(2.0, 4.0, 1.0)
	await assert_error(test_call).is_push_error("Parameter a must be less than or equal to b for Uniform CDF. Received a=4.0, b=1.0")
	
	# Test return value is NAN
	var result: float = StatMath.CdfFunctions.uniform_cdf(2.0, 4.0, 1.0)
	assert_bool(is_nan(result)).is_true()

# --- Normal CDF Parameter Validation ---
func test_normal_cdf_invalid_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 0.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive for Normal CDF. Received: 0.0")


# --- Exponential CDF Parameter Validation ---
func test_exponential_cdf_invalid_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive for Exponential CDF. Received: 0.0")


# --- Beta CDF Parameter Validation ---
func test_beta_cdf_invalid_alpha_beta() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=-1.0, beta_param=2.0")


# --- Gamma CDF Parameter Validation ---
func test_gamma_cdf_invalid_shape_scale() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=0.0, theta_scale=2.0")


# --- Chi-Square CDF Parameter Validation ---
func test_chi_square_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: 0.0")


# --- F-Distribution CDF Parameter Validation ---
func test_f_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=0.0, d2_df=2.0")


# --- Student's t-Distribution CDF Parameter Validation ---
func test_t_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: 0.0")


# --- Binomial CDF Parameter Validation ---
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


# --- Poisson CDF Parameter Validation ---
func test_poisson_cdf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.poisson_cdf(2, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be non-negative for Poisson CDF. Received: -1.0")


# --- Geometric CDF Parameter Validation ---
func test_geometric_cdf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.geometric_cdf(2, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")


# --- Negative Binomial CDF Parameter Validation ---
func test_negative_binomial_cdf_invalid_r() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 0, 0.5)
	await assert_error(test_call).is_push_error("Number of successes (r_successes) must be positive. Received: 0")


func test_negative_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")


# --- Pareto CDF Parameter Validation ---
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


# --- Weibull CDF Parameter Validation ---
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


# =============================================================================
# NEW FUNCTIONS PARAMETER VALIDATION TESTS
# =============================================================================

# --- Cauchy CDF Parameter Validation ---
func test_cauchy_cdf_invalid_scale() -> void:
	var test_call1: Callable = func():
		StatMath.CdfFunctions.cauchy_cdf(0.0, 0.0, -1.0)
	await assert_error(test_call1).is_push_error("Scale parameter must be positive for Cauchy CDF. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.CdfFunctions.cauchy_cdf(0.0, 0.0, 0.0)
	await assert_error(test_call2).is_push_error("Scale parameter must be positive for Cauchy CDF. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.CdfFunctions.cauchy_cdf(0.0, 0.0, -1.0)
	var result2: float = StatMath.CdfFunctions.cauchy_cdf(0.0, 0.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()


# --- Lognormal CDF Parameter Validation ---
func test_lognormal_cdf_invalid_sigma() -> void:
	var test_call1: Callable = func():
		StatMath.CdfFunctions.lognormal_cdf(1.0, 0.0, -1.0)
	await assert_error(test_call1).is_push_error("Standard deviation (sigma) must be positive for Lognormal CDF. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.CdfFunctions.lognormal_cdf(1.0, 0.0, 0.0)
	await assert_error(test_call2).is_push_error("Standard deviation (sigma) must be positive for Lognormal CDF. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.CdfFunctions.lognormal_cdf(1.0, 0.0, -1.0)
	var result2: float = StatMath.CdfFunctions.lognormal_cdf(1.0, 0.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true() 
