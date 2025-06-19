# addons/godot-stat-math/tests/core/pmf_pdf_functions_test.gd
class_name PmfPdfFunctionsTest extends GdUnitTestSuite

# --- Binomial PMF ---
func test_binomial_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(2, 5, 0.5)
	assert_float(result).is_equal_approx(0.3125, 1e-7) # C(5,2) * 0.5^2 * 0.5^3 = 10 * 0.25 * 0.125 = 0.3125

func test_binomial_pmf_edge_cases(k: int, n: int, p: float, expected: float, test_parameters := [
	[0, 5, 0.5, 0.03125],  # k = 0
	[5, 5, 0.5, 0.03125],  # k = n
	[6, 5, 0.5, 0.0],      # k > n
	[0, 5, 0.0, 1.0],      # p = 0
	[5, 5, 1.0, 1.0],      # p = 1
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(k, n, p)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_binomial_pmf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, -1, 0.5)
	await assert_error(test_call1).is_push_error("Number of trials (n_trials) must be non-negative. Received: -1")

	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, 5, -0.1)
	await assert_error(test_call2).is_push_error("Success probability (p_prob) must be between 0.0 and 1.0. Received: -0.1")

# --- Poisson PMF ---
func test_poisson_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(2, 3.0)
	assert_float(result).is_equal_approx(0.2240418, 1e-7) # (3^2 * e^-3) / 2! = 9 * e^-3 / 2

func test_poisson_pmf_edge_cases(k: int, lambda_param: float, expected: float, test_parameters := [
	[0, 3.0, exp(-3.0)],  # k = 0
	[0, 0.0, 1.0],        # lambda = 0
	[-1, 3.0, 0.0],       # k < 0
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(k, lambda_param)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_poisson_pmf_invalid_parameters() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.poisson_pmf(2, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be non-negative. Received: -1.0")

# --- Negative Binomial PMF ---
func test_negative_binomial_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(5, 2, 0.5)
	assert_float(result).is_equal_approx(0.125, 1e-7) # C(4,1) * 0.5^2 * 0.5^3 = 4 * 0.25 * 0.125 = 0.125

func test_negative_binomial_pmf_edge_cases(k: int, r: int, p: float, expected: float, test_parameters := [
	[2, 2, 0.5, 0.25],  # k = r
	[1, 2, 0.5, 0.0],   # k < r
	[2, 2, 1.0, 1.0],   # p = 1
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(k, r, p)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_negative_binomial_pmf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 0, 0.5)
	await assert_error(test_call1).is_push_error("Number of required successes (r_successes) must be positive. Received: 0")

	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 0.0)
	await assert_error(test_call2).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0") 


# =============================================================================
# PDF TESTS
# =============================================================================

# --- Normal PDF ---
func test_normal_pdf_parametrized(x: float, mu: float, sigma: float, expected: float, test_parameters := [
	[0.0, 0.0, 1.0, 1.0 / sqrt(2.0 * PI)],  # Standard normal at mean
	[1.0, 0.0, 1.0, (1.0 / sqrt(2.0 * PI)) * exp(-0.5)],  # One std dev from mean
	[2.0, 2.0, 1.0, 1.0 / sqrt(2.0 * PI)],  # Different mean, at mean
	[5.0, 3.0, 2.0, (1.0 / (2.0 * sqrt(2.0 * PI))) * exp(-1.0)],  # Custom parameters
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(x, mu, sigma)
	assert_float(result).is_equal_approx(expected, 1e-6)

func test_normal_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, -1.0)
	await assert_error(test_call1).is_push_error("Standard deviation (sigma) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, 0.0)
	await assert_error(test_call2).is_push_error("Standard deviation (sigma) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, -1.0)
	var result2: float = StatMath.PmfPdfFunctions.normal_pdf(0.0, 0.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()

# --- Exponential PDF ---
func test_exponential_pdf_parametrized(x: float, lambda_param: float, expected: float, test_parameters := [
	[0.0, 1.0, 1.0],  # At x=0
	[1.0, 1.0, exp(-1.0)],  # At x=1, lambda=1
	[0.0, 2.0, 2.0],  # At x=0, lambda=2
	[2.0, 0.5, 0.5 * exp(-1.0)],  # At x=2, lambda=0.5
	[-1.0, 1.0, 0.0],  # Negative x should return 0
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, lambda_param)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_exponential_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.exponential_pdf(1.0, -1.0)
	await assert_error(test_call1).is_push_error("Rate parameter (lambda_param) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.exponential_pdf(1.0, 0.0)
	await assert_error(test_call2).is_push_error("Rate parameter (lambda_param) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.exponential_pdf(1.0, -1.0)
	var result2: float = StatMath.PmfPdfFunctions.exponential_pdf(1.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()

# --- Uniform PDF ---
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

func test_uniform_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.uniform_pdf(2.0, 4.0, 1.0)
	await assert_error(test_call1).is_push_error("Parameter b must be greater than a. Received a=4.0, b=1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.uniform_pdf(2.0, 3.0, 3.0)
	await assert_error(test_call2).is_push_error("Parameter b must be greater than a. Received a=3.0, b=3.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.uniform_pdf(2.0, 4.0, 1.0)
	var result2: float = StatMath.PmfPdfFunctions.uniform_pdf(2.0, 3.0, 3.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()

# --- Gamma PDF ---
func test_gamma_pdf_basic() -> void:
	# For Gamma(2, 1), PDF at x=1 is e^(-1) = exp(-1)
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(exp(-1.0), 1e-7)

func test_gamma_pdf_edge_cases(x: float, k_shape: float, theta_scale: float, expected: float, test_parameters := [
	[0.0, 2.0, 1.0, 0.0],   # At x=0
	[-1.0, 2.0, 1.0, 0.0],  # Negative x
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, k_shape, theta_scale)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_gamma_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, -1.0, 1.0)
	await assert_error(test_call1).is_push_error("Shape parameter (k_shape) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, 0.0, 1.0)
	await assert_error(test_call2).is_push_error("Shape parameter (k_shape) must be positive. Received: 0.0")
	
	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, -1.0)
	await assert_error(test_call3).is_push_error("Scale parameter (theta_scale) must be positive. Received: -1.0")
	
	var test_call4: Callable = func():
		StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 0.0)
	await assert_error(test_call4).is_push_error("Scale parameter (theta_scale) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, -1.0, 1.0)
	var result2: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 0.0, 1.0)
	var result3: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, -1.0)
	var result4: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()
	assert_bool(is_nan(result3)).is_true()
	assert_bool(is_nan(result4)).is_true()

# --- Beta PDF ---
func test_beta_pdf_scipy_validated() -> void:
	var test_data: Array = PmfPdfTestData.VALUES["beta_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.beta_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], 1e-7)

func test_beta_pdf_uniform_special_case(x: float, test_parameters := [
	[0.2], [0.4], [0.6], [0.8],
]) -> void:
	# Beta(1,1) is uniform on [0,1]
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(x, 1.0, 1.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_pdf_outside_range(x: float, alpha: float, beta_param: float, expected: float, test_parameters := [
	[-0.1, 2.0, 2.0, 0.0],  # Below range
	[1.1, 2.0, 2.0, 0.0],   # Above range
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta_param)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_beta_pdf_symmetry(x: float, alpha: float, beta_param: float, test_parameters := [
	[0.3, 2.0, 2.0],  # Basic symmetry test
	[0.2, 3.0, 3.0],  # Different parameters
	[0.4, 1.5, 1.5],  # Non-integer parameters
]) -> void:
	# For alpha = beta, PDF should be symmetric around x = 0.5
	var x_mirror: float = 1.0 - x
	var pdf1: float = StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta_param)
	var pdf2: float = StatMath.PmfPdfFunctions.beta_pdf(x_mirror, alpha, beta_param)
	assert_float(pdf1).is_equal_approx(pdf2, 1e-7)

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

func test_beta_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, -1.0, 2.0)
	await assert_error(test_call1).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=-1.0, beta_param=2.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, 0.0, 2.0)
	await assert_error(test_call2).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=0.0, beta_param=2.0")
	
	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, -1.0)
	await assert_error(test_call3).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=2.0, beta_param=-1.0")
	
	var test_call4: Callable = func():
		StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, 0.0)
	await assert_error(test_call4).is_push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=2.0, beta_param=0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, -1.0, 2.0)
	var result2: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 0.0, 2.0)
	var result3: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, -1.0)
	var result4: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 2.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()
	assert_bool(is_nan(result3)).is_true()
	assert_bool(is_nan(result4)).is_true()

# --- Chi-squared PDF ---
func test_chi_squared_pdf_scipy_validated() -> void:
	var test_data: Array = PmfPdfTestData.VALUES["chi_squared_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], 1e-7)

func test_chi_squared_pdf_edge_cases(x: float, k_df: float, expected: float, test_parameters := [
	[0.0, 2.0, 0.0],   # At x=0
	[-1.0, 2.0, 0.0],  # Negative x
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, k_df)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_chi_squared_pdf_exponential_relationship(x: float, test_parameters := [
	[1.0], [1.5], [2.0], [3.0], [5.0],
]) -> void:
	# Chi-squared with df=2 is equivalent to exponential with rate=1/2
	var chi_squared_result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, 2.0)
	var exponential_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, 0.5)
	assert_float(chi_squared_result).is_equal_approx(exponential_result, 1e-7)

func test_chi_squared_pdf_gamma_relationship(x: float, test_parameters := [
	[1.0],
	[2.0],
	[5.0],
]) -> void:
	# Chi-squared is Gamma(k/2, 2)
	var df: float = 4.0
	
	if x > 0:
		var chi_sq_result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, df)
		var gamma_result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, df / 2.0, 2.0)
		assert_float(chi_sq_result).is_equal_approx(gamma_result, 1e-6)

func test_chi_squared_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, -1.0)
	await assert_error(test_call1).is_push_error("Degrees of freedom (k_df) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, 0.0)
	await assert_error(test_call2).is_push_error("Degrees of freedom (k_df) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, -1.0)
	var result2: float = StatMath.PmfPdfFunctions.chi_squared_pdf(1.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()

# --- Student's t PDF ---
func test_t_pdf_scipy_validated() -> void:
	var test_data: Array = PmfPdfTestData.VALUES["students_t_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.t_pdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], 1e-7)

func test_t_pdf_special_cases(x: float, df_nu: float, expected: float, tolerance: float, test_parameters := [
	[0.0, 1000.0, 1.0 / sqrt(2.0 * PI), 1e-2],  # For large df, approaches standard normal
	[0.0, 1.0, 1.0 / PI, 1e-7],                  # t(1) is Cauchy distribution
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.t_pdf(x, df_nu)
	assert_float(result).is_equal_approx(expected, tolerance)

func test_t_pdf_symmetry(x: float, df_nu: float, test_parameters := [
	[1.0, 5.0],
	[2.0, 3.0],
	[0.5, 10.0],
]) -> void:
	var result1: float = StatMath.PmfPdfFunctions.t_pdf(x, df_nu)
	var result2: float = StatMath.PmfPdfFunctions.t_pdf(-x, df_nu)
	assert_float(result1).is_equal_approx(result2, 1e-7)

func test_t_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.t_pdf(0.0, -1.0)
	await assert_error(test_call1).is_push_error("Degrees of freedom (df_nu) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.t_pdf(0.0, 0.0)
	await assert_error(test_call2).is_push_error("Degrees of freedom (df_nu) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.t_pdf(0.0, -1.0)
	var result2: float = StatMath.PmfPdfFunctions.t_pdf(0.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()

# --- F PDF ---
func test_f_pdf_basic() -> void:
	# Test that F PDF gives reasonable values
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 3.0)
	assert_float(result).is_greater(0.0)
	assert_float(result).is_less(10.0) # Sanity check

func test_f_pdf_edge_cases(x: float, d1_df: float, d2_df: float, expected: float, test_parameters := [
	[0.0, 2.0, 3.0, 0.0],   # At x=0
	[-1.0, 2.0, 3.0, 0.0],  # Negative x
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.f_pdf(x, d1_df, d2_df)
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_f_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, -1.0, 3.0)
	await assert_error(test_call1).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=-1.0, d2_df=3.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, 0.0, 3.0)
	await assert_error(test_call2).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=0.0, d2_df=3.0")
	
	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, -1.0)
	await assert_error(test_call3).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=2.0, d2_df=-1.0")
	
	var test_call4: Callable = func():
		StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 0.0)
	await assert_error(test_call4).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=2.0, d2_df=0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.f_pdf(1.0, -1.0, 3.0)
	var result2: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 0.0, 3.0)
	var result3: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, -1.0)
	var result4: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()
	assert_bool(is_nan(result3)).is_true()
	assert_bool(is_nan(result4)).is_true()


# =============================================================================
# SPECIAL RELATIONSHIPS AND PROPERTY TESTS
# =============================================================================

# Test that Gamma PDF reduces to Exponential when shape=1
func test_gamma_pdf_exponential_special_case(x: float, test_parameters := [
	[0.5],
	[1.0],
	[2.0],
	[3.0],
]) -> void:
	var scale: float = 2.0  # theta
	var rate: float = 1.0 / scale  # lambda = 1/theta
	
	if x > 0:
		var gamma_result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, 1.0, scale)
		var exp_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, rate)
		assert_float(gamma_result).is_equal_approx(exp_result, 1e-6)

# Test PDF properties - all PDFs should be non-negative
func test_normal_pdf_non_negative(x: float, test_parameters := [
	[-2.0], [-1.0], [0.0], [1.0], [2.0], [5.0],
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, 1.0)
	assert_float(result).is_greater_equal(0.0)

func test_exponential_pdf_non_negative(x: float, test_parameters := [
	[-2.0], [-1.0], [0.0], [1.0], [2.0], [5.0],
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, 1.0)
	assert_float(result).is_greater_equal(0.0)

func test_uniform_pdf_non_negative(x: float, test_parameters := [
	[-2.0], [-1.0], [0.0], [1.0], [2.0], [5.0],
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.uniform_pdf(x, -1.0, 3.0)
	assert_float(result).is_greater_equal(0.0)

func test_beta_pdf_non_negative(x: float, test_parameters := [
	[0.1], [0.3], [0.5], [0.7], [0.9],
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.beta_pdf(x, 2.0, 2.0)
	if not is_nan(result):
		assert_float(result).is_greater_equal(0.0)

func test_chi_squared_pdf_non_negative(x: float, test_parameters := [
	[-2.0], [-1.0], [0.0], [1.0], [2.0], [5.0],
]) -> void:
	var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, 2.0)
	if not is_nan(result):
		assert_float(result).is_greater_equal(0.0)

# Test numerical stability for extreme values
func test_pdf_numerical_stability(x: float, distribution: String, test_parameters := [
	[100.0, "normal"],
	[100.0, "exponential"], 
	[100.0, "gamma"],
	[1000.0, "normal"],
	[1000.0, "exponential"],
]) -> void:
	var result: float
	
	match distribution:
		"normal":
			result = StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, 1.0)
		"exponential":
			result = StatMath.PmfPdfFunctions.exponential_pdf(x, 1.0)
		"gamma":
			result = StatMath.PmfPdfFunctions.gamma_pdf(x, 2.0, 1.0)
	
	# Should be very small but finite
	assert_bool(is_finite(result)).is_true()
	assert_float(result).is_greater_equal(0.0)

# Test PDF integration approximation (total probability ≈ 1)
func test_pdf_integration_approximation(distribution: String, test_parameters := [
	["normal"],
	["exponential"],
	["beta"],
]) -> void:
	var sum: float = 0.0
	var dx: float = 0.01
	
	match distribution:
		"normal":
			# Normal distribution (from -4σ to 4σ)
			for i in range(-400, 401):
				var x: float = float(i) * dx
				sum += StatMath.PmfPdfFunctions.normal_pdf(x) * dx
		"exponential":
			# Exponential distribution (from 0 to 10/λ)
			var lambda: float = 1.0
			for i in range(0, 1001):
				var x: float = float(i) * dx
				sum += StatMath.PmfPdfFunctions.exponential_pdf(x, lambda) * dx
		"beta":
			# Beta distribution (from 0 to 1)
			var alpha: float = 2.0
			var beta: float = 2.0
			for i in range(0, 101):
				var x: float = float(i) * 0.01
				sum += StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta) * 0.01
	
	assert_float(sum).is_equal_approx(1.0, 0.01)
