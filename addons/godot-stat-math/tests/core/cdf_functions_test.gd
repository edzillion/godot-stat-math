# addons/godot-stat-math/tests/core/cdf_functions_test.gd
class_name CdfFunctionsTest extends GdUnitTestSuite

# --- Uniform CDF ---
func test_uniform_cdf_basic_range() -> void:
	var a: float = 2.0
	var b: float = 5.0
	var x: float = 3.0
	var result: float = StatMath.CdfFunctions.uniform_cdf(x, a, b)
	assert_float(result).is_equal_approx((x - a) / (b - a), 1e-7)

func test_uniform_cdf_x_below_a() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(1.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_uniform_cdf_x_above_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(6.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_uniform_cdf_a_equals_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(2.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_uniform_cdf_invalid_a_greater_than_b() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.uniform_cdf(2.0, 5.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter a must be less than or equal to b for Uniform CDF.")

# --- Normal CDF ---
func test_normal_cdf_standard_normal() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(0.0)
	assert_float(result).is_equal_approx(0.5, 1e-6)

func test_normal_cdf_mu_sigma() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(2.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.5, 1e-6)

func test_normal_cdf_invalid_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Standard deviation (sigma) must be positive for Normal CDF.")

# --- Exponential CDF ---
func test_exponential_cdf_typical() -> void:
	var result: float = StatMath.CdfFunctions.exponential_cdf(1.0, 2.0)
	assert_float(result).is_greater_equal(0.0)
	assert_float(result).is_less_equal(1.0)

func test_exponential_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.exponential_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_exponential_cdf_invalid_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive for Exponential CDF.")

# --- Beta CDF ---
func test_beta_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_beta_cdf_x_one() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_cdf_invalid_alpha_beta() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape parameters (alpha, beta_param) must be positive for Beta CDF.")

# --- Gamma CDF ---
func test_gamma_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.gamma_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_gamma_cdf_invalid_shape_scale() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF.")

# --- Chi-Square CDF ---
func test_chi_square_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.chi_square_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_square_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Degrees of freedom (k_df) must be positive for Chi-Square CDF.")

# --- F-Distribution CDF ---
func test_f_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.f_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF.")

# --- Student's t-Distribution CDF ---
func test_t_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.t_cdf(0.0, 2.0)
	assert_float(result).is_greater_equal(0.0)
	assert_float(result).is_less_equal(1.0)

func test_t_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF.")

# --- Binomial CDF ---
func test_binomial_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(-1, 5, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_binomial_cdf_k_ge_n() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(5, 5, 0.5)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_binomial_cdf_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, -1, 0.5)
	await assert_error(test_call).is_runtime_error("Assertion failed: Number of trials (n_trials) must be non-negative.")

func test_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, 5, -0.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Probability (p_prob) must be between 0.0 and 1.0.")

# --- Poisson CDF ---
func test_poisson_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.poisson_cdf(-1, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_poisson_cdf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.poisson_cdf(2, -1.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be non-negative for Poisson CDF.")

# --- Geometric CDF ---
func test_geometric_cdf_k_less_than_1() -> void:
	var result: float = StatMath.CdfFunctions.geometric_cdf(0, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_geometric_cdf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.geometric_cdf(2, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Success probability (p_prob) must be in (0,1].")

# --- Negative Binomial CDF ---
func test_negative_binomial_cdf_k_less_than_r() -> void:
	var result: float = StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_negative_binomial_cdf_invalid_r() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 0, 0.5)
	await assert_error(test_call).is_runtime_error("Assertion failed: Number of successes (r_successes) must be positive.")

func test_negative_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Success probability (p_prob) must be in (0,1].") 
