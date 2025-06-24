# res://addons/godot-stat-math/tests/core/pmf_pdf_functions/scipy_validation_test.gd
class_name PmfPdfFunctionsScipyValidationTest extends GdUnitTestSuite

const PMF_PDF_TEST_DATA = preload("res://addons/godot-stat-math/tables/pmf_pdf_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

# --- PMF Functions ---

func test_binomial_pmf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["binomial_pmf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.binomial_pmf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_poisson_pmf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["poisson_pmf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.poisson_pmf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_negative_binomial_pmf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["negative_binomial_pmf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

# --- PDF Functions ---

func test_normal_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["normal_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.normal_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_exponential_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["exponential_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.exponential_pdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_uniform_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["uniform_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.uniform_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_gamma_pdf_basic() -> void:
	# For Gamma(2, 1), PDF at x=1 - scipy.stats.gamma.pdf(1.0, 2.0, scale=1.0)
	var result: float = StatMath.PmfPdfFunctions.gamma_pdf(1.0, 2.0, 1.0)
	var expected: float = 0.36787944  # scipy.stats.gamma.pdf(1.0, 2.0, scale=1.0)
	assert_float(result).is_equal_approx(expected, StatMath.PROBABILITY_TOLERANCE)

func test_gamma_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["gamma_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.gamma_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_beta_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["beta_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.beta_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_chi_squared_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["chi_squared_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_t_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["students_t_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.t_pdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_f_pdf_basic() -> void:
	# Test that F PDF gives reasonable values
	var result: float = StatMath.PmfPdfFunctions.f_pdf(1.0, 2.0, 3.0)
	assert_float(result).is_greater(0.0)
	assert_float(result).is_less(10.0) # Sanity check

func test_f_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["f_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.f_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_weibull_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["weibull_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.weibull_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_lognormal_pdf_scipy_validated() -> void:
	var test_data: Array = PMF_PDF_TEST_DATA.VALUES["lognormal_pdf"]
	for case in test_data:
		var result: float = StatMath.PmfPdfFunctions.lognormal_pdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

# --- Edge Case Tests ---

func test_weibull_pdf_edge_cases() -> void:
	var test_cases: Array = [
		{"x": 0.0, "scale": 1.0, "shape": 2.0, "expected": 0.0},    # At x=0 for shape > 1
		{"x": -1.0, "scale": 1.0, "shape": 2.0, "expected": 0.0},   # Negative x
		{"x": 0.0, "scale": 1.0, "shape": 0.5, "expected": INF},    # At x=0 for shape < 1 (should be infinity)
		{"x": 0.0, "scale": 1.0, "shape": 1.0, "expected": 1.0},    # At x=0 for shape = 1 (exponential case)
	]
	
	for case in test_cases:
		var result: float = StatMath.PmfPdfFunctions.weibull_pdf(case["x"], case["scale"], case["shape"])
		if case["expected"] == INF:
			assert_bool(is_inf(result)).is_true()
		else:
			assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_lognormal_pdf_edge_cases() -> void:
	var test_cases: Array = [
		{"x": 0.0, "mu": 0.0, "sigma": 1.0, "expected": 0.0},    # At x=0
		{"x": -1.0, "mu": 0.0, "sigma": 1.0, "expected": 0.0},   # Negative x
	]
	
	for case in test_cases:
		var result: float = StatMath.PmfPdfFunctions.lognormal_pdf(case["x"], case["mu"], case["sigma"])
		assert_float(result).is_equal_approx(case["expected"], StatMath.PROBABILITY_TOLERANCE)

func test_beta_pdf_outside_range() -> void:
	# Test boundary conditions - PDF should be 0 outside [0,1]
	var result1: float = StatMath.PmfPdfFunctions.beta_pdf(-0.1, 2.0, 2.0)
	var result2: float = StatMath.PmfPdfFunctions.beta_pdf(1.1, 2.0, 2.0)
	assert_float(result1).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)
	assert_float(result2).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)

func test_chi_squared_pdf_edge_cases() -> void:
	# Test boundary conditions
	var result1: float = StatMath.PmfPdfFunctions.chi_squared_pdf(0.0, 2.0)
	var result2: float = StatMath.PmfPdfFunctions.chi_squared_pdf(-1.0, 2.0)
	assert_float(result1).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)
	assert_float(result2).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)

func test_f_pdf_edge_cases() -> void:
	# Test boundary conditions
	var result1: float = StatMath.PmfPdfFunctions.f_pdf(0.0, 2.0, 3.0)
	var result2: float = StatMath.PmfPdfFunctions.f_pdf(-1.0, 2.0, 3.0)
	assert_float(result1).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)
	assert_float(result2).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)

func test_weibull_pdf_boundary_conditions() -> void:
	var test_cases: Array = [
		{"x": 0.0, "scale": 1.0, "shape": 0.5},  # x=0, shape<1 (infinity)
		{"x": 0.0, "scale": 1.0, "shape": 1.0},  # x=0, shape=1 (finite)
		{"x": 0.0, "scale": 1.0, "shape": 2.0},  # x=0, shape>1 (zero)
		{"x": 1000.0, "scale": 1.0, "shape": 2.0},  # Large x (should approach 0)
	]
	
	for case in test_cases:
		var result: float = StatMath.PmfPdfFunctions.weibull_pdf(case["x"], case["scale"], case["shape"])
		
		if case["x"] == 0.0:
			if case["shape"] < 1.0:
				assert_bool(is_inf(result)).is_true()
			elif case["shape"] == 1.0:
				assert_float(result).is_equal_approx(1.0 / case["scale"], StatMath.PROBABILITY_TOLERANCE)
			else:  # shape > 1.0
				assert_float(result).is_equal_approx(0.0, StatMath.PROBABILITY_TOLERANCE)
		elif case["x"] == 1000.0:
			assert_float(result).is_less(StatMath.BOUNDARY_TOLERANCE)  # Should be very small

func test_weibull_pdf_deterministic_behavior() -> void:
	# Same inputs should always give same outputs
	var x: float = 1.5
	var shape: float = 2.0
	var scale: float = 3.0
	
	var result1: float = StatMath.PmfPdfFunctions.weibull_pdf(x, scale, shape)
	var result2: float = StatMath.PmfPdfFunctions.weibull_pdf(x, scale, shape)
	
	assert_float(result1).is_equal_approx(result2, StatMath.HIGH_PRECISION_TOLERANCE)

func test_beta_pdf_boundary_behavior() -> void:
	# Test values very close to boundaries
	var epsilon: float = StatMath.BOUNDARY_TOLERANCE
	
	# Values just inside the range should give finite results
	var result_near_zero: float = StatMath.PmfPdfFunctions.beta_pdf(epsilon, 2.0, 2.0)
	var result_near_one: float = StatMath.PmfPdfFunctions.beta_pdf(1.0 - epsilon, 2.0, 2.0)
	
	assert_bool(is_finite(result_near_zero)).is_true()
	assert_bool(is_finite(result_near_one)).is_true()
	assert_float(result_near_zero).is_greater(0.0)
	assert_float(result_near_one).is_greater(0.0)

# --- Non-negative Tests ---

func test_normal_pdf_non_negative() -> void:
	var x_values: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, 1.0)
		assert_float(result).is_greater_equal(0.0)

func test_exponential_pdf_non_negative() -> void:
	var x_values: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, 1.0)
		assert_float(result).is_greater_equal(0.0)

func test_uniform_pdf_non_negative() -> void:
	var x_values: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.uniform_pdf(x, -1.0, 3.0)
		assert_float(result).is_greater_equal(0.0)

func test_beta_pdf_non_negative() -> void:
	var x_values: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.beta_pdf(x, 2.0, 2.0)
		if not is_nan(result):
			assert_float(result).is_greater_equal(0.0)

func test_chi_squared_pdf_non_negative() -> void:
	var x_values: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, 2.0)
		if not is_nan(result):
			assert_float(result).is_greater_equal(0.0)

func test_weibull_pdf_non_negative() -> void:
	var x_values: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.weibull_pdf(x, 1.0, 2.0)
		if not is_inf(result):
			assert_float(result).is_greater_equal(0.0)

func test_lognormal_pdf_non_negative() -> void:
	var x_values: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.lognormal_pdf(x, 0.0, 1.0)
		assert_float(result).is_greater_equal(0.0)

# --- Numerical Stability Tests ---

func test_pdf_numerical_stability() -> void:
	var test_cases: Array = [
		{"x": 100.0, "distribution": StatMath.SupportedDistributions.NORMAL},
		{"x": 100.0, "distribution": StatMath.SupportedDistributions.EXPONENTIAL},
		{"x": 100.0, "distribution": StatMath.SupportedDistributions.GAMMA},
		{"x": 1000.0, "distribution": StatMath.SupportedDistributions.NORMAL},
		{"x": 1000.0, "distribution": StatMath.SupportedDistributions.EXPONENTIAL},
	]
	
	for case in test_cases:
		var result: float
		var x: float = case["x"]
		var distribution: StatMath.SupportedDistributions = case["distribution"]
		
		match distribution:
			StatMath.SupportedDistributions.NORMAL:
				result = StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, 1.0)
			StatMath.SupportedDistributions.EXPONENTIAL:
				result = StatMath.PmfPdfFunctions.exponential_pdf(x, 1.0)
			StatMath.SupportedDistributions.GAMMA:
				result = StatMath.PmfPdfFunctions.gamma_pdf(x, 2.0, 1.0)
		
		# Should be very small but finite
		assert_bool(is_finite(result)).is_true()
		assert_float(result).is_greater_equal(0.0)

# --- PDF Integration Tests ---

func test_pdf_integration_normal() -> void:
	# Test that normal PDF integrates to 1.0 over appropriate range
	# Using ±4σ covers 99.99% of the distribution
	var sigma: float = 1.0
	var integration_range: float = 4.0 * sigma
	var step_size: float = integration_range / 1000.0  # 1000 steps for numerical integration
	var sum: float = 0.0
	
	for i in range(-1000, 1001):
		var x: float = float(i) * step_size
		sum += StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, sigma) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_exponential() -> void:
	# Test that exponential PDF integrates to 1.0 over [0, +∞)
	# Using upper bound of 10/λ captures 99.995% of the distribution
	var lambda: float = 1.0
	var upper_bound: float = 10.0 / lambda
	var step_size: float = upper_bound / 1000.0
	var sum: float = 0.0
	
	for i in range(0, 1001):
		var x: float = float(i) * step_size
		sum += StatMath.PmfPdfFunctions.exponential_pdf(x, lambda) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_uniform() -> void:
	# Test that uniform PDF integrates to 1.0 over [a, b]
	var a: float = 1.0
	var b: float = 4.0
	var num_steps: int = 1000
	var step_size: float = (b - a) / float(num_steps)
	var sum: float = 0.0
	
	# Use midpoint rule for more accurate integration
	for i in range(num_steps):
		var x: float = a + (float(i) + 0.5) * step_size
		sum += StatMath.PmfPdfFunctions.uniform_pdf(x, a, b) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_TOLERANCE)

func test_pdf_integration_beta() -> void:
	# Test that beta PDF integrates to 1.0 over [0, 1]
	var alpha: float = 2.0
	var beta_param: float = 2.0
	var step_size: float = 0.001
	var sum: float = 0.0
	
	for i in range(1, 1000):  # Skip 0 and 1 to avoid boundary issues
		var x: float = float(i) * step_size
		sum += StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta_param) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_gamma() -> void:
	# Test that gamma PDF integrates to 1.0 over [0, +∞)
	# Using upper bound that captures 99.9% of the distribution
	var k_shape: float = 2.0
	var theta_scale: float = 1.0
	# For Gamma(k, θ), mean = kθ, std = √(kθ²)
	# Using mean + 6*std as upper bound captures >99.9%
	var mean: float = k_shape * theta_scale
	var std_dev: float = sqrt(k_shape) * theta_scale
	var upper_bound: float = mean + 6.0 * std_dev
	var step_size: float = upper_bound / 1000.0
	var sum: float = 0.0
	
	for i in range(1, 1001):  # Skip 0 to avoid boundary issues
		var x: float = float(i) * step_size
		sum += StatMath.PmfPdfFunctions.gamma_pdf(x, k_shape, theta_scale) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_weibull() -> void:
	# Test that Weibull PDF integrates to 1.0 over [0, +∞)
	var scale_param: float = 2.0
	var shape_param: float = 2.0
	# For Weibull, using 99.9% quantile as upper bound
	# Approximate 99.9% quantile: λ * (-ln(0.001))^(1/k)
	var upper_bound: float = scale_param * pow(-log(0.001), 1.0 / shape_param)
	var step_size: float = upper_bound / 1000.0
	var sum: float = 0.0
	
	for i in range(1, 1001):  # Skip 0 to avoid boundary issues for shape < 1
		var x: float = float(i) * step_size
		sum += StatMath.PmfPdfFunctions.weibull_pdf(x, scale_param, shape_param) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_lognormal() -> void:
	# Test that lognormal PDF integrates to 1.0 over (0, +∞)
	var mu: float = 0.0
	var sigma: float = 1.0
	# For lognormal, using quantiles to determine integration bounds
	# Approximate 0.1% to 99.9% quantiles cover most of the distribution
	var lower_bound: float = exp(mu - 3.0 * sigma)  # Approximate 0.1% quantile
	var upper_bound: float = exp(mu + 3.0 * sigma)  # Approximate 99.9% quantile
	var num_steps: int = 1000
	var step_size: float = (upper_bound - lower_bound) / float(num_steps)
	var sum: float = 0.0
	
	for i in range(1, num_steps):
		var x: float = lower_bound + float(i) * step_size
		sum += StatMath.PmfPdfFunctions.lognormal_pdf(x, mu, sigma) * step_size
	
	assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_parametrized() -> void:
	var distributions: Array[StatMath.SupportedDistributions] = [
		StatMath.SupportedDistributions.NORMAL,
		StatMath.SupportedDistributions.EXPONENTIAL, 
		StatMath.SupportedDistributions.BETA,
	]
	
	for distribution in distributions:
		var sum: float = 0.0
		var dx: float = 0.01
		
		match distribution:
			StatMath.SupportedDistributions.NORMAL:
				# Normal distribution (from -10 to 10)
				for i in range(-1000, 1001):
					var x: float = float(i) * dx
					sum += StatMath.PmfPdfFunctions.normal_pdf(x) * dx
			StatMath.SupportedDistributions.EXPONENTIAL:
				# Exponential distribution (from 0 to 10/λ)
				var lambda: float = 1.0
				for i in range(0, 1001):
					var x: float = float(i) * dx
					sum += StatMath.PmfPdfFunctions.exponential_pdf(x, lambda) * dx
			StatMath.SupportedDistributions.BETA:
				# Beta distribution (from 0 to 1)
				var alpha: float = 2.0
				var beta: float = 2.0
				for i in range(0, 101):
					var x: float = float(i) * 0.01
					sum += StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta) * 0.01
		
		assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

func test_pdf_integration_approximation() -> void:
	var distributions: Array[StatMath.SupportedDistributions] = [
		StatMath.SupportedDistributions.NORMAL,
		StatMath.SupportedDistributions.EXPONENTIAL,
		StatMath.SupportedDistributions.BETA,
	]
	
	for distribution in distributions:
		var sum: float = 0.0
		var dx: float = 0.01
		
		match distribution:
			StatMath.SupportedDistributions.NORMAL:
				# Normal distribution (from -4σ to 4σ)
				for i in range(-400, 401):
					var x: float = float(i) * dx
					sum += StatMath.PmfPdfFunctions.normal_pdf(x) * dx
			StatMath.SupportedDistributions.EXPONENTIAL:
				# Exponential distribution (from 0 to 10/λ)
				var lambda: float = 1.0
				for i in range(0, 1001):
					var x: float = float(i) * dx
					sum += StatMath.PmfPdfFunctions.exponential_pdf(x, lambda) * dx
			StatMath.SupportedDistributions.BETA:
				# Beta distribution (from 0 to 1)
				var alpha: float = 2.0
				var beta: float = 2.0
				for i in range(0, 101):
					var x: float = float(i) * 0.01
					sum += StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta) * 0.01

		assert_float(sum).is_equal_approx(1.0, StatMath.NUMERICAL_INTEGRATION_TOLERANCE) 
