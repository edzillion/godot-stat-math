# addons/godot-stat-math/tests/core/pmf_pdf_functions_test.gd
class_name PmfPdfFunctionsTest extends GdUnitTestSuite

# --- Binomial PMF ---
func test_binomial_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(2, 5, 0.5)
	assert_float(result).is_equal_approx(0.3125, 1e-7) # C(5,2) * 0.5^2 * 0.5^3 = 10 * 0.25 * 0.125 = 0.3125

func test_binomial_pmf_k_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(0, 5, 0.5)
	assert_float(result).is_equal_approx(0.03125, 1e-7)

func test_binomial_pmf_k_equals_n() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(5, 5, 0.5)
	assert_float(result).is_equal_approx(0.03125, 1e-7)

func test_binomial_pmf_k_greater_than_n() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(6, 5, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_binomial_pmf_p_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(0, 5, 0.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.binomial_pmf(1, 5, 0.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_binomial_pmf_p_one() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(5, 5, 1.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.binomial_pmf(4, 5, 1.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_binomial_pmf_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, -1, 0.5)
	await assert_error(test_call).is_push_error("Number of trials (n_trials) must be non-negative. Received: -1")

func test_binomial_pmf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, 5, -0.1)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be between 0.0 and 1.0. Received: -0.1")

# --- Poisson PMF ---
func test_poisson_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(2, 3.0)
	assert_float(result).is_equal_approx(0.2240418, 1e-7) # (3^2 * e^-3) / 2! = 9 * e^-3 / 2

func test_poisson_pmf_k_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(0, 3.0)
	assert_float(result).is_equal_approx(exp(-3.0), 1e-7)

func test_poisson_pmf_lambda_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(0, 0.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.poisson_pmf(1, 0.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_poisson_pmf_k_negative() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(-1, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_poisson_pmf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.poisson_pmf(2, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be non-negative. Received: -1.0")

# --- Negative Binomial PMF ---
func test_negative_binomial_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(5, 2, 0.5)
	assert_float(result).is_equal_approx(0.125, 1e-7) # C(4,1) * 0.5^2 * 0.5^3 = 4 * 0.25 * 0.125 = 0.125

func test_negative_binomial_pmf_k_equals_r() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 0.5)
	assert_float(result).is_equal_approx(0.25, 1e-7)

func test_negative_binomial_pmf_k_less_than_r() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(1, 2, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_negative_binomial_pmf_p_one() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 1.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(3, 2, 1.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_negative_binomial_pmf_invalid_r_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 0, 0.5)
	await assert_error(test_call).is_push_error("Number of required successes (r_successes) must be positive. Received: 0")

func test_negative_binomial_pmf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0") 


# =============================================================================
# PDF TESTS
# =============================================================================

# --- Normal PDF ---
func test_normal_pdf_standard_normal() -> void:
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, 1.0)
	assert_float(result).is_equal_approx(1.0 / sqrt(2.0 * PI), 1e-7) # At mean, standard normal

func test_normal_pdf_one_std_dev() -> void:
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(1.0, 0.0, 1.0)
	var expected: float = (1.0 / sqrt(2.0 * PI)) * exp(-0.5)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_normal_pdf_custom_parameters() -> void:
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(5.0, 3.0, 2.0)
	var expected: float = (1.0 / (2.0 * sqrt(2.0 * PI))) * exp(-1.0) # (5-3)^2 / (2*4) = 1
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_normal_pdf_invalid_sigma() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, -1.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive. Received: -1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_normal_pdf_invalid_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, 0.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive. Received: 0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# --- Exponential PDF ---
func test_exponential_pdf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(1.0, 2.0)
	assert_float(result).is_equal_approx(2.0 * exp(-2.0), 1e-7)

func test_exponential_pdf_at_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(0.0, 1.5)
	assert_float(result).is_equal_approx(1.5, 1e-7)

func test_exponential_pdf_negative_x() -> void:
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(-1.0, 1.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_exponential_pdf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.exponential_pdf(1.0, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive. Received: -1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(1.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_exponential_pdf_invalid_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.exponential_pdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive. Received: 0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(1.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# --- Uniform PDF ---
func test_uniform_pdf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.uniform_pdf(2.5, 1.0, 4.0)
	assert_float(result).is_equal_approx(1.0 / 3.0, 1e-7)

func test_uniform_pdf_at_boundary() -> void:
	var result1: float = StatMath.PmfPdfFunctions.uniform_pdf(1.0, 1.0, 4.0)
	var result2: float = StatMath.PmfPdfFunctions.uniform_pdf(4.0, 1.0, 4.0)
	assert_float(result1).is_equal_approx(1.0 / 3.0, 1e-7)
	assert_float(result2).is_equal_approx(1.0 / 3.0, 1e-7)

func test_uniform_pdf_outside_range() -> void:
	var result1: float = StatMath.PmfPdfFunctions.uniform_pdf(0.5, 1.0, 4.0)
	var result2: float = StatMath.PmfPdfFunctions.uniform_pdf(4.5, 1.0, 4.0)
	assert_float(result1).is_equal_approx(0.0, 1e-7)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_uniform_pdf_invalid_parameters() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.uniform_pdf(2.0, 4.0, 1.0)
	await assert_error(test_call).is_push_error("Parameter b must be greater than a. Received a=4.0, b=1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.uniform_pdf(2.0, 4.0, 1.0)
	assert_bool(is_nan(result)).is_true()

func test_uniform_pdf_invalid_parameters_equal() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.uniform_pdf(2.0, 3.0, 3.0)
	await assert_error(test_call).is_push_error("Parameter b must be greater than a. Received a=3.0, b=3.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.uniform_pdf(2.0, 3.0, 3.0)
	assert_bool(is_nan(result)).is_true()


# --- Gamma PDF ---
func test_gamma_pdf_basic() -> void:
	# For Gamma(2, 1), PDF at x=1 is e^(-1) = exp(-1)
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(exp(-1.0), 1e-7)

func test_gamma_pdf_at_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(0.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_gamma_pdf_negative_x() -> void:
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(-1.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_gamma_pdf_invalid_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, -1.0, 1.0)
	await assert_error(test_call).is_push_error("Shape parameter (k_shape) must be positive. Received: -1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, -1.0, 1.0)
	assert_bool(is_nan(result)).is_true()

func test_gamma_pdf_invalid_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, 0.0, 1.0)
	await assert_error(test_call).is_push_error("Shape parameter (k_shape) must be positive. Received: 0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 0.0, 1.0)
	assert_bool(is_nan(result)).is_true()

func test_gamma_pdf_invalid_scale_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Scale parameter (theta_scale) must be positive. Received: -1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_gamma_pdf_invalid_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Scale parameter (theta_scale) must be positive. Received: 0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# --- Beta PDF ---
func test_beta_pdf_basic() -> void:
	# Beta(2,2) at x=0.5 should be 6 * 0.5 * 0.5 = 1.5
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.5, 1e-7)

func test_beta_pdf_uniform_case() -> void:
	# Beta(1,1) is uniform on [0,1], so PDF = 1
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(0.3, 1.0, 1.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_pdf_at_boundaries() -> void:
	var result1: float = StatMath.PmfPdfFunctions.beta_pdf(0.0, 2.0, 2.0)
	var result2: float = StatMath.PmfPdfFunctions.beta_pdf(1.0, 2.0, 2.0)
	assert_float(result1).is_equal_approx(0.0, 1e-7)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_beta_pdf_outside_range() -> void:
	var result1: float = StatMath.PmfPdfFunctions.beta_pdf(-0.1, 2.0, 2.0)
	var result2: float = StatMath.PmfPdfFunctions.beta_pdf(1.1, 2.0, 2.0)
	assert_float(result1).is_equal_approx(0.0, 1e-7)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_beta_pdf_invalid_alpha_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=-1.0, beta_param=2.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, -1.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_pdf_invalid_alpha_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=0.0, beta_param=2.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_pdf_invalid_beta_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=2.0, beta_param=-1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_pdf_invalid_beta_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=2.0, beta_param=0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# --- Chi-squared PDF ---
func test_chi_squared_pdf_basic() -> void:
	# Chi-squared is Gamma(k/2, 2), so we can test against known values
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(2.0, 2.0)
	var expected: float = 0.5 * exp(-1.0) # Gamma(1, 2) at x=2
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_chi_squared_pdf_at_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_squared_pdf_negative_x() -> void:
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(-1.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_squared_pdf_invalid_df_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, -1.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive. Received: -1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_chi_squared_pdf_invalid_df_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive. Received: 0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# --- Student's t PDF ---
func test_t_pdf_basic() -> void:
	# For large df, t-distribution approaches standard normal
	var result: float = StatMath.PmfPdfFunctions.t_pdf(0.0, 1000.0)
	var expected_normal: float = 1.0 / sqrt(2.0 * PI)
	assert_float(result).is_equal_approx(expected_normal, 1e-2) # Less precise for large df

func test_t_pdf_df_one() -> void:
	# t(1) is Cauchy distribution, PDF at 0 is 1/π
	var result: float = StatMath.PmfPdfFunctions.t_pdf(0.0, 1.0)
	assert_float(result).is_equal_approx(1.0 / PI, 1e-7)

func test_t_pdf_symmetry() -> void:
	var result1: float = StatMath.PmfPdfFunctions.t_pdf(1.0, 5.0)
	var result2: float = StatMath.PmfPdfFunctions.t_pdf(-1.0, 5.0)
	assert_float(result1).is_equal_approx(result2, 1e-7)

func test_t_pdf_invalid_df_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.t_pdf(0.0, -1.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive. Received: -1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.t_pdf(0.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_t_pdf_invalid_df_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.t_pdf(0.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive. Received: 0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.t_pdf(0.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# --- F PDF ---
func test_f_pdf_basic() -> void:
	# Test that F PDF gives reasonable values
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 3.0)
	assert_float(result).is_greater(0.0)
	assert_float(result).is_less(10.0) # Sanity check

func test_f_pdf_at_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.f_pdf(0.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_pdf_negative_x() -> void:
	var result: float = StatMath.PmfPdfFunctions.f_pdf(-1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_pdf_invalid_df1_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, -1.0, 3.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=-1.0, d2_df=3.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, -1.0, 3.0)
	assert_bool(is_nan(result)).is_true()

func test_f_pdf_invalid_df1_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, 0.0, 3.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=0.0, d2_df=3.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 0.0, 3.0)
	assert_bool(is_nan(result)).is_true()

func test_f_pdf_invalid_df2_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=2.0, d2_df=-1.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, -1.0)
	assert_bool(is_nan(result)).is_true()

func test_f_pdf_invalid_df2_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=2.0, d2_df=0.0")
	
	# Test return value is NAN
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()


# =============================================================================
# PARAMETRIZED AND COMPREHENSIVE TESTS
# =============================================================================

func test_normal_pdf_parametrized(x: float, mu: float, sigma: float, expected: float, test_parameters := [
	[0.0, 0.0, 1.0, 1.0 / sqrt(2.0 * PI)],  # Standard normal at mean
	[1.0, 0.0, 1.0, (1.0 / sqrt(2.0 * PI)) * exp(-0.5)],  # One std dev from mean
	[2.0, 2.0, 1.0, 1.0 / sqrt(2.0 * PI)],  # Different mean, at mean
	[5.0, 3.0, 2.0, (1.0 / (2.0 * sqrt(2.0 * PI))) * exp(-1.0)],  # Custom parameters
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(x, mu, sigma)
	assert_float(result).is_equal_approx(expected, 1e-6)

func test_exponential_pdf_parametrized(x: float, lambda_param: float, expected: float, test_parameters := [
	[0.0, 1.0, 1.0],  # At x=0
	[1.0, 1.0, exp(-1.0)],  # At x=1, lambda=1
	[0.0, 2.0, 2.0],  # At x=0, lambda=2
	[2.0, 0.5, 0.5 * exp(-1.0)],  # At x=2, lambda=0.5
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, lambda_param)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_uniform_pdf_parametrized(x: float, a: float, b: float, expected: float, test_parameters := [
	[2.5, 1.0, 4.0, 1.0/3.0],  # Middle of range
	[1.0, 1.0, 4.0, 1.0/3.0],  # At lower boundary
	[4.0, 1.0, 4.0, 1.0/3.0],  # At upper boundary
	[0.5, 1.0, 4.0, 0.0],  # Outside range (below)
	[4.5, 1.0, 4.0, 0.0],  # Outside range (above)
	[0.0, -2.0, 2.0, 0.25],  # Symmetric around 0
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.uniform_pdf(x, a, b)
	assert_float(result).is_equal_approx(expected, 1e-7)

# Test PDF properties - all PDFs should be non-negative
func test_pdf_non_negative_property() -> void:
	var test_x_vals: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	
	# Normal PDF - always positive
	for x in test_x_vals:
		var normal_result: float = StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, 1.0)
		assert_float(normal_result).is_greater_equal(0.0)
	
	# Exponential PDF - positive for x >= 0
	for x in test_x_vals:
		var exp_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, 1.0)
		assert_float(exp_result).is_greater_equal(0.0)
	
	# Uniform PDF
	for x in test_x_vals:
		var uniform_result: float = StatMath.PmfPdfFunctions.uniform_pdf(x, -1.0, 3.0)
		assert_float(uniform_result).is_greater_equal(0.0)

# Test that Gamma PDF reduces to Exponential when shape=1
func test_gamma_pdf_exponential_special_case() -> void:
	var test_x_vals: Array[float] = [0.5, 1.0, 2.0, 3.0]
	var scale: float = 2.0  # theta
	var rate: float = 1.0 / scale  # lambda = 1/theta
	
	for x in test_x_vals:
		if x > 0:
			var gamma_result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, 1.0, scale)
			var exp_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, rate)
			assert_float(gamma_result).is_equal_approx(exp_result, 1e-6)

# Test Beta PDF boundary behavior
func test_beta_pdf_boundary_behavior() -> void:
	# Test values very close to boundaries
	var epsilon: float = 1e-10
	
	# Values just inside the range should give finite results
	var result_near_zero: float = StatMath.PmfPdfFunctions.beta_pdf(epsilon, 2.0, 2.0)
	var result_near_one: float = StatMath.PmfPdfFunctions.beta_pdf(1.0 - epsilon, 2.0, 2.0)
	
	assert_bool(is_finite(result_near_zero)).is_true()
	assert_bool(is_finite(result_near_one)).is_true()
	assert_float(result_near_zero).is_greater(0.0)
	assert_float(result_near_one).is_greater(0.0)

# Test Chi-squared relationship to Gamma
func test_chi_squared_gamma_relationship() -> void:
	var test_x_vals: Array[float] = [1.0, 2.0, 5.0]
	var df: float = 4.0
	
	for x in test_x_vals:
		if x > 0:
			var chi_sq_result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, df)
			var gamma_result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, df / 2.0, 2.0)
			assert_float(chi_sq_result).is_equal_approx(gamma_result, 1e-6)

# Test numerical stability for extreme values
func test_pdf_numerical_stability() -> void:
	# Test very large x values don't cause overflow/NaN
	var large_x: float = 100.0
	
	var normal_large: float = StatMath.PmfPdfFunctions.normal_pdf(large_x, 0.0, 1.0)
	var exp_large: float = StatMath.PmfPdfFunctions.exponential_pdf(large_x, 1.0)
	var gamma_large: float = StatMath.PmfPdfFunctions.gamma_pdf(large_x, 2.0, 1.0)
	
	# Should be very small but finite
	assert_bool(is_finite(normal_large)).is_true()
	assert_bool(is_finite(exp_large)).is_true()
	assert_bool(is_finite(gamma_large)).is_true()
	assert_float(normal_large).is_greater_equal(0.0)
	assert_float(exp_large).is_greater_equal(0.0)
	assert_float(gamma_large).is_greater_equal(0.0)
