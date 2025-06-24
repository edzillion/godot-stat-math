# res://addons/godot-stat-math/tests/core/pmf_pdf_functions/parameter_validation_test.gd
class_name PmfPdfFunctionsParameterValidationTest extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

# --- PMF Parameter Validation ---

func test_binomial_pmf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, -1, 0.5)
	await assert_error(test_call1).is_push_error("Number of trials (n_trials) must be non-negative. Received: -1")

	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, 5, -0.1)
	await assert_error(test_call2).is_push_error("Success probability (p_prob) must be between 0.0 and 1.0. Received: -0.1")

	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, 5, 1.1)
	await assert_error(test_call3).is_push_error("Success probability (p_prob) must be between 0.0 and 1.0. Received: 1.1")

func test_poisson_pmf_invalid_parameters() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.poisson_pmf(2, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be non-negative. Received: -1.0")

func test_negative_binomial_pmf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 0, 0.5)
	await assert_error(test_call1).is_push_error("Number of required successes (r_successes) must be positive. Received: 0")

	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 0.0)
	await assert_error(test_call2).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")

# --- PDF Parameter Validation ---

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

func test_weibull_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.weibull_pdf(1.0, -1.0, 1.0)
	await assert_error(test_call1).is_push_error("Scale parameter (scale_param) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.weibull_pdf(1.0, 0.0, 1.0)
	await assert_error(test_call2).is_push_error("Scale parameter (scale_param) must be positive. Received: 0.0")
	
	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.weibull_pdf(1.0, 2.0, -1.0)
	await assert_error(test_call3).is_push_error("Shape parameter (shape_param) must be positive. Received: -1.0")
	
	var test_call4: Callable = func():
		StatMath.PmfPdfFunctions.weibull_pdf(1.0, 2.0, 0.0)
	await assert_error(test_call4).is_push_error("Shape parameter (shape_param) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.weibull_pdf(1.0, -1.0, 1.0)
	var result2: float = StatMath.PmfPdfFunctions.weibull_pdf(1.0, 0.0, 1.0)
	var result3: float = StatMath.PmfPdfFunctions.weibull_pdf(1.0, 2.0, -1.0)
	var result4: float = StatMath.PmfPdfFunctions.weibull_pdf(1.0, 2.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()
	assert_bool(is_nan(result3)).is_true()
	assert_bool(is_nan(result4)).is_true()

func test_lognormal_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.lognormal_pdf(1.0, 0.0, -1.0)
	await assert_error(test_call1).is_push_error("Standard deviation (sigma) must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.lognormal_pdf(1.0, 0.0, 0.0)
	await assert_error(test_call2).is_push_error("Standard deviation (sigma) must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.lognormal_pdf(1.0, 0.0, -1.0)
	var result2: float = StatMath.PmfPdfFunctions.lognormal_pdf(1.0, 0.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()


# =============================================================================
# NEW FUNCTIONS PARAMETER VALIDATION TESTS
# =============================================================================

func test_geometric_pmf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.geometric_pmf(1, 0.0)
	await assert_error(test_call1).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.geometric_pmf(1, -0.1)
	await assert_error(test_call2).is_push_error("Success probability (p_prob) must be in (0,1]. Received: -0.1")
	
	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.geometric_pmf(1, 1.1)
	await assert_error(test_call3).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 1.1")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.geometric_pmf(1, 0.0)
	var result2: float = StatMath.PmfPdfFunctions.geometric_pmf(1, -0.1)
	var result3: float = StatMath.PmfPdfFunctions.geometric_pmf(1, 1.1)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()
	assert_bool(is_nan(result3)).is_true()


func test_cauchy_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.cauchy_pdf(0.0, 0.0, -1.0)
	await assert_error(test_call1).is_push_error("Scale parameter must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.cauchy_pdf(0.0, 0.0, 0.0)
	await assert_error(test_call2).is_push_error("Scale parameter must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.cauchy_pdf(0.0, 0.0, -1.0)
	var result2: float = StatMath.PmfPdfFunctions.cauchy_pdf(0.0, 0.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()


func test_pareto_pdf_invalid_parameters() -> void:
	var test_call1: Callable = func():
		StatMath.PmfPdfFunctions.pareto_pdf(1.0, -1.0, 2.0)
	await assert_error(test_call1).is_push_error("Scale parameter must be positive. Received: -1.0")
	
	var test_call2: Callable = func():
		StatMath.PmfPdfFunctions.pareto_pdf(1.0, 0.0, 2.0)
	await assert_error(test_call2).is_push_error("Scale parameter must be positive. Received: 0.0")
	
	var test_call3: Callable = func():
		StatMath.PmfPdfFunctions.pareto_pdf(1.0, 1.0, -1.0)
	await assert_error(test_call3).is_push_error("Shape parameter must be positive. Received: -1.0")
	
	var test_call4: Callable = func():
		StatMath.PmfPdfFunctions.pareto_pdf(1.0, 1.0, 0.0)
	await assert_error(test_call4).is_push_error("Shape parameter must be positive. Received: 0.0")
	
	# Test return values are NAN
	var result1: float = StatMath.PmfPdfFunctions.pareto_pdf(1.0, -1.0, 2.0)
	var result2: float = StatMath.PmfPdfFunctions.pareto_pdf(1.0, 0.0, 2.0)
	var result3: float = StatMath.PmfPdfFunctions.pareto_pdf(1.0, 1.0, -1.0)
	var result4: float = StatMath.PmfPdfFunctions.pareto_pdf(1.0, 1.0, 0.0)
	assert_bool(is_nan(result1)).is_true()
	assert_bool(is_nan(result2)).is_true()
	assert_bool(is_nan(result3)).is_true()
	assert_bool(is_nan(result4)).is_true() 
