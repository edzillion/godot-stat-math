# res://addons/godot-stat-math/tests/core/integration/cdf_pdf_integration_test.gd
class_name CdfPdfIntegrationTest extends GdUnitTestSuite

const CDF_PDF_INTEGRATION_TEST_DATA = preload("res://addons/godot-stat-math/tables/cdf_pdf_integration_test_data.gd")

## This test suite verifies integration and end-to-end statistical workflows:
## • CDF and PDF consistency through probability calculations
## • CDFs should be monotonically increasing  
## • End-to-end statistical computation validation
## • Numerical stability under various conditions
## • Cross-distribution mathematical relationships

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

## Tests derivative relationship using scipy-validated data
func test_scipy_derivative_validation() -> void:
	# Test Normal distribution derivative relationship
	var normal_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["normal_derivative_tests"]
	for case in normal_data:
		var x: float = case["params"][0]
		var mu: float = case["params"][1] 
		var sigma: float = case["params"][2]
		
		# Validate CDF matches scipy
		var cdf_result: float = StatMath.CdfFunctions.normal_cdf(x, mu, sigma)
		assert_float(cdf_result).is_equal_approx(case["cdf_expected"], StatMath.FLOAT_TOLERANCE)
		
		# Validate PDF matches scipy
		var pdf_result: float = StatMath.PmfPdfFunctions.normal_pdf(x, mu, sigma)
		assert_float(pdf_result).is_equal_approx(case["pdf_expected"], StatMath.FLOAT_TOLERANCE)

## Tests monotonicity using scipy-validated data
func test_scipy_monotonicity_validation() -> void:
	# Test multiple distributions for monotonicity
	var distributions: Array[String] = ["normal_monotonicity", "exponential_monotonicity", "uniform_monotonicity", "beta_monotonicity", "gamma_monotonicity", "weibull_monotonicity"]
	
	for dist_name in distributions:
		var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES[dist_name]
		var prev_cdf: float = -1.0
		
		for case in test_data:
			var calculated_cdf: float
			
			match dist_name:
				"normal_monotonicity":
					calculated_cdf = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
				"exponential_monotonicity":
					calculated_cdf = StatMath.CdfFunctions.exponential_cdf(case["params"][0], case["params"][1])
				"uniform_monotonicity":
					calculated_cdf = StatMath.CdfFunctions.uniform_cdf(case["params"][0], case["params"][1], case["params"][2])
				"beta_monotonicity":
					calculated_cdf = StatMath.CdfFunctions.beta_cdf(case["params"][0], case["params"][1], case["params"][2])
				"gamma_monotonicity":
					calculated_cdf = StatMath.CdfFunctions.gamma_cdf(case["params"][0], case["params"][1], case["params"][2])
				"weibull_monotonicity":
					calculated_cdf = StatMath.CdfFunctions.weibull_cdf(case["params"][0], case["params"][1], case["params"][2])
			
			# Validate against scipy expected value
			assert_float(calculated_cdf).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)
			
			# Validate monotonicity property
			assert_float(calculated_cdf).is_greater_equal(prev_cdf)
			assert_float(calculated_cdf).is_greater_equal(0.0)
			assert_float(calculated_cdf).is_less_equal(1.0)
			
			prev_cdf = calculated_cdf

## Tests cross-function consistency using scipy-validated data
func test_scipy_cross_function_validation() -> void:
	# Test Normal CDF/PPF consistency
	var normal_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["normal_cdf_ppf_consistency"]
	for case in normal_data:
		var x: float = case["cdf_params"][0]
		var mu: float = case["cdf_params"][1]
		var sigma: float = case["cdf_params"][2]
		
		# Validate CDF matches scipy
		var cdf_result: float = StatMath.CdfFunctions.normal_cdf(x, mu, sigma)
		assert_float(cdf_result).is_equal_approx(case["cdf_expected"], StatMath.FLOAT_TOLERANCE)
		
		# Validate PPF round-trip consistency
		var ppf_result: float = StatMath.PpfFunctions.normal_ppf(cdf_result, mu, sigma)
		assert_float(ppf_result).is_equal_approx(case["ppf_expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Tests boundary behavior using scipy-validated data
func test_scipy_boundary_validation() -> void:
	# Test boundary conditions for multiple distributions
	var boundary_tests: Array[String] = ["normal_boundary_tests", "uniform_boundary_tests", "beta_boundary_tests", "exponential_boundary_tests"]
	
	for test_name in boundary_tests:
		var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES[test_name]
		
		for case in test_data:
			var calculated_cdf: float
			var calculated_pdf: float
			
			match test_name:
				"normal_boundary_tests":
					calculated_cdf = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
					calculated_pdf = StatMath.PmfPdfFunctions.normal_pdf(case["params"][0], case["params"][1], case["params"][2])
				"uniform_boundary_tests":
					calculated_cdf = StatMath.CdfFunctions.uniform_cdf(case["params"][0], case["params"][1], case["params"][2])
					calculated_pdf = StatMath.PmfPdfFunctions.uniform_pdf(case["params"][0], case["params"][1], case["params"][2])
				"beta_boundary_tests":
					calculated_cdf = StatMath.CdfFunctions.beta_cdf(case["params"][0], case["params"][1], case["params"][2])
					calculated_pdf = StatMath.PmfPdfFunctions.beta_pdf(case["params"][0], case["params"][1], case["params"][2])
				"exponential_boundary_tests":
					calculated_cdf = StatMath.CdfFunctions.exponential_cdf(case["params"][0], case["params"][1])
					calculated_pdf = StatMath.PmfPdfFunctions.exponential_pdf(case["params"][0], case["params"][1])
			
			# Validate against scipy expected values
			assert_float(calculated_cdf).is_equal_approx(case["cdf_expected"], StatMath.FLOAT_TOLERANCE)
			assert_float(calculated_pdf).is_equal_approx(case["pdf_expected"], StatMath.FLOAT_TOLERANCE)

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

## Tests that the numerical derivative of Normal CDF approximates Normal PDF
func test_normal_cdf_pdf_derivative_relationship() -> void:
	var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["normal_derivative_tests"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for case in test_data:
		var x: float = case["params"][0]
		var mu: float = case["params"][1]
		var sigma: float = case["params"][2]
		
		# Calculate numerical derivative: d/dx CDF(x) ≈ (CDF(x+h) - CDF(x-h)) / (2h)
		var cdf_plus: float = StatMath.CdfFunctions.normal_cdf(x + h, mu, sigma)
		var cdf_minus: float = StatMath.CdfFunctions.normal_cdf(x - h, mu, sigma)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		# Compare against scipy-validated PDF value
		assert_float(numerical_derivative).is_equal_approx(case["pdf_expected"], StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Exponential CDF approximates Exponential PDF
func test_exponential_cdf_pdf_derivative_relationship() -> void:
	var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["exponential_derivative_tests"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for case in test_data:
		var x: float = case["params"][0]
		var lambda_param: float = case["params"][1]
		
		var cdf_plus: float = StatMath.CdfFunctions.exponential_cdf(x + h, lambda_param)
		var cdf_minus: float = StatMath.CdfFunctions.exponential_cdf(x - h, lambda_param)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		assert_float(numerical_derivative).is_equal_approx(case["pdf_expected"], StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Uniform CDF approximates Uniform PDF
func test_uniform_cdf_pdf_derivative_relationship() -> void:
	var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["uniform_derivative_tests"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for case in test_data:
		var x: float = case["params"][0]
		var a: float = case["params"][1]
		var b: float = case["params"][2]
		
		var cdf_plus: float = StatMath.CdfFunctions.uniform_cdf(x + h, a, b)
		var cdf_minus: float = StatMath.CdfFunctions.uniform_cdf(x - h, a, b)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		assert_float(numerical_derivative).is_equal_approx(case["pdf_expected"], StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Beta CDF approximates Beta PDF
func test_beta_cdf_pdf_derivative_relationship() -> void:
	var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["beta_derivative_tests"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for case in test_data:
		var x: float = case["params"][0]
		var alpha: float = case["params"][1]
		var beta_param: float = case["params"][2]
		
		var cdf_plus: float = StatMath.CdfFunctions.beta_cdf(x + h, alpha, beta_param)
		var cdf_minus: float = StatMath.CdfFunctions.beta_cdf(x - h, alpha, beta_param)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		assert_float(numerical_derivative).is_equal_approx(case["pdf_expected"], StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Weibull CDF approximates Weibull PDF
func test_weibull_cdf_pdf_derivative_relationship() -> void:
	var test_data: Array = CDF_PDF_INTEGRATION_TEST_DATA.VALUES["weibull_derivative_tests"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for case in test_data:
		var x: float = case["params"][0]
		var scale_param: float = case["params"][1]
		var shape_param: float = case["params"][2]
		
		var cdf_plus: float = StatMath.CdfFunctions.weibull_cdf(x + h, scale_param, shape_param)
		var cdf_minus: float = StatMath.CdfFunctions.weibull_cdf(x - h, scale_param, shape_param)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		assert_float(numerical_derivative).is_equal_approx(case["pdf_expected"], StatMath.DERIVATIVE_TOLERANCE)

## Tests Gamma distribution probability consistency (Gamma(1,scale) = Exponential(1/scale))
func test_gamma_exponential_mathematical_relationship() -> void:
	# Mathematical relationship that Gamma(1, scale) = Exponential(1/scale)
	var shape: float = 1.0
	var scale: float = 2.0
	var lambda_equiv: float = 1.0 / scale
	var test_points: Array[float] = [0.5, 1.0, 2.0, 4.0]
	
	for x in test_points:
		var gamma_cdf: float = StatMath.CdfFunctions.gamma_cdf(x, shape, scale)
		var exponential_cdf: float = StatMath.CdfFunctions.exponential_cdf(x, lambda_equiv)
		
		assert_float(gamma_cdf).is_equal_approx(exponential_cdf, StatMath.PROBABILITY_TOLERANCE)

# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================

## Tests a complete statistical workflow: data generation → analysis → validation
func test_end_to_end_normal_distribution_workflow() -> void:
	# Use simple hardcoded workflow parameters for illustrative purposes
	var sample_size: int = 1000
	var mu: float = 5.0
	var sigma: float = 2.0
	var test_seed: int = 12345
	var percentile_value: float = 95.0
	var samples: Array[float] = []
	
	# Use a fixed seed for reproducible testing
	StatMath.set_global_seed(test_seed)
	
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_normal(mu, sigma))
	
	# Calculate sample statistics using BasicStats
	var sample_mean: float = StatMath.BasicStats.mean(samples)
	var sample_std: float = StatMath.BasicStats.standard_deviation(samples)
	
	# Validate that sample statistics are close to theoretical values
	assert_float(sample_mean).is_equal_approx(mu, StatMath.NEGATIVE_BINOMIAL_TOLERANCE)
	assert_float(sample_std).is_equal_approx(sigma, StatMath.HIGH_DISTRIBUTION_TOLERANCE)
	
	# Test that our CDF/PDF functions work with sample data
	samples.sort()
	var percentile_95: float = StatMath.BasicStats.percentile(samples, percentile_value)
	var theoretical_95: float = StatMath.PpfFunctions.normal_ppf(percentile_value / 100.0, mu, sigma)
	
	assert_float(percentile_95).is_equal_approx(theoretical_95, StatMath.DEFAULT_TOLERANCE_FACTOR)

# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

## Tests numerical stability under extreme parameter values
func test_numerical_stability_extreme_values() -> void:
	# Test with very small parameters
	var small_sigma: float = 1e-6
	var result_small: float = StatMath.CdfFunctions.normal_cdf(0.0, 0.0, small_sigma)
	assert_bool(is_finite(result_small)).is_true()
	assert_float(result_small).is_greater_equal(0.0)
	assert_float(result_small).is_less_equal(1.0)
	
	# Test with very large parameters  
	var large_x: float = 1e6
	var result_large: float = StatMath.CdfFunctions.exponential_cdf(large_x, 1.0)
	assert_bool(is_finite(result_large)).is_true()
	assert_float(result_large).is_equal_approx(1.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Test with extreme probability values
	var result_extreme_low: float = StatMath.PpfFunctions.normal_ppf(1e-10)
	var result_extreme_high: float = StatMath.PpfFunctions.normal_ppf(1.0 - 1e-10)
	assert_bool(is_finite(result_extreme_low)).is_true()
	assert_bool(is_finite(result_extreme_high)).is_true()

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

## Tests parameter validation for integration functions
func test_integration_parameter_validation() -> void:
	# Integration tests primarily focus on mathematical properties
	# Individual function parameter validation is tested in their respective test files
	pass
