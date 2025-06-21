# res://addons/godot-stat-math/tests/core/cdf_pdf_integration_test.gd
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

## Tests integration consistency using scipy-validated data
func test_scipy_integration_validation() -> void:
	# This is a placeholder for scipy-validated integration tests
	# Currently this file focuses on mathematical property testing
	pass

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- CDF ↔ PDF Derivative Relationship Tests ---

## Tests that the numerical derivative of Normal CDF approximates Normal PDF
func test_normal_cdf_pdf_derivative_relationship() -> void:
	var test_points: Array = CDF_PDF_INTEGRATION_TEST_DATA.DERIVATIVE_TEST_POINTS[StatMath.SupportedDistributions.NORMAL]
	var params: Dictionary = CDF_PDF_INTEGRATION_TEST_DATA.DISTRIBUTION_PARAMETERS["normal_standard"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		# Calculate numerical derivative: d/dx CDF(x) ≈ (CDF(x+h) - CDF(x-h)) / (2h)
		var cdf_plus: float = StatMath.CdfFunctions.normal_cdf(x + h, params["mu"], params["sigma"])
		var cdf_minus: float = StatMath.CdfFunctions.normal_cdf(x - h, params["mu"], params["sigma"])
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		# Calculate actual PDF value
		var pdf_value: float = StatMath.PmfPdfFunctions.normal_pdf(x, params["mu"], params["sigma"])
		
		assert_float(numerical_derivative).is_equal_approx(pdf_value, StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Exponential CDF approximates Exponential PDF
func test_exponential_cdf_pdf_derivative_relationship() -> void:
	var test_points: Array = CDF_PDF_INTEGRATION_TEST_DATA.DERIVATIVE_TEST_POINTS[StatMath.SupportedDistributions.EXPONENTIAL]
	var params: Dictionary = CDF_PDF_INTEGRATION_TEST_DATA.DISTRIBUTION_PARAMETERS["exponential_rate_2"]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		var cdf_plus: float = StatMath.CdfFunctions.exponential_cdf(x + h, params["lambda_param"])
		var cdf_minus: float = StatMath.CdfFunctions.exponential_cdf(x - h, params["lambda_param"])
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		var pdf_value: float = StatMath.PmfPdfFunctions.exponential_pdf(x, params["lambda_param"])
		
		assert_float(numerical_derivative).is_equal_approx(pdf_value, StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Uniform CDF approximates Uniform PDF
func test_uniform_cdf_pdf_derivative_relationship() -> void:
	var a: float = 1.0
	var b: float = 4.0
	var test_points: Array[float] = [1.5, 2.0, 2.5, 3.0, 3.5]  # Points strictly inside [a,b]
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		var cdf_plus: float = StatMath.CdfFunctions.uniform_cdf(x + h, a, b)
		var cdf_minus: float = StatMath.CdfFunctions.uniform_cdf(x - h, a, b)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		var pdf_value: float = StatMath.PmfPdfFunctions.uniform_pdf(x, a, b)
		
		assert_float(numerical_derivative).is_equal_approx(pdf_value, StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Beta CDF approximates Beta PDF
func test_beta_cdf_pdf_derivative_relationship() -> void:
	var alpha: float = 2.0
	var beta_param: float = 3.0
	var test_points: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]  # Points strictly inside (0,1)
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		var cdf_plus: float = StatMath.CdfFunctions.beta_cdf(x + h, alpha, beta_param)
		var cdf_minus: float = StatMath.CdfFunctions.beta_cdf(x - h, alpha, beta_param)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		var pdf_value: float = StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta_param)
		
		assert_float(numerical_derivative).is_equal_approx(pdf_value, StatMath.DERIVATIVE_TOLERANCE)

## Tests Gamma distribution probability consistency (Gamma(1,scale) = Exponential(1/scale))
func test_gamma_cdf_pdf_probability_consistency() -> void:
	# Test that Gamma(1, scale) is equivalent to Exponential(1/scale)
	var shape: float = 1.0
	var scale: float = 2.0
	var lambda_equiv: float = 1.0 / scale
	
	var test_points: Array[float] = [0.5, 1.0, 2.0, 4.0]
	
	for x in test_points:
		var gamma_cdf: float = StatMath.CdfFunctions.gamma_cdf(x, shape, scale)
		var exponential_cdf: float = StatMath.CdfFunctions.exponential_cdf(x, lambda_equiv)
		
		# They should be approximately equal due to mathematical relationship
		assert_float(gamma_cdf).is_equal_approx(exponential_cdf, StatMath.PROBABILITY_TOLERANCE)

## Tests that the numerical derivative of Weibull CDF approximates Weibull PDF
func test_weibull_cdf_pdf_derivative_relationship() -> void:
	var scale_param: float = 2.0
	var shape_param: float = 2.0
	var test_points: Array[float] = [0.5, 1.0, 1.5, 2.0, 3.0]  # Points > 0
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		var cdf_plus: float = StatMath.CdfFunctions.weibull_cdf(x + h, scale_param, shape_param)
		var cdf_minus: float = StatMath.CdfFunctions.weibull_cdf(x - h, scale_param, shape_param)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		var pdf_value: float = StatMath.PmfPdfFunctions.weibull_pdf(x, scale_param, shape_param)
		
		assert_float(numerical_derivative).is_equal_approx(pdf_value, StatMath.DERIVATIVE_TOLERANCE)

# --- CDF Monotonicity Tests ---

## Tests that CDFs are monotonically increasing for all continuous distributions
func test_cdf_monotonicity_all_distributions() -> void:
	var distributions: Array[Dictionary] = CDF_PDF_INTEGRATION_TEST_DATA.MONOTONICITY_TEST_DATA
	
	for dist in distributions:
		var points: Array = dist["points"]
		var prev_cdf: float = -1.0
		
		for i in range(points.size()):
			var x: float = points[i]
			var current_cdf: float
			
			# Direct function calls instead of terrible abstraction
			match dist["name"]:
				StatMath.SupportedDistributions.NORMAL:
					current_cdf = StatMath.CdfFunctions.normal_cdf(x, dist["params"][0], dist["params"][1])
				StatMath.SupportedDistributions.EXPONENTIAL:
					current_cdf = StatMath.CdfFunctions.exponential_cdf(x, dist["params"][0])
				StatMath.SupportedDistributions.UNIFORM:
					current_cdf = StatMath.CdfFunctions.uniform_cdf(x, dist["params"][0], dist["params"][1])
				StatMath.SupportedDistributions.BETA:
					current_cdf = StatMath.CdfFunctions.beta_cdf(x, dist["params"][0], dist["params"][1])
				StatMath.SupportedDistributions.GAMMA:
					current_cdf = StatMath.CdfFunctions.gamma_cdf(x, dist["params"][0], dist["params"][1])
				StatMath.SupportedDistributions.WEIBULL:
					current_cdf = StatMath.CdfFunctions.weibull_cdf(x, dist["params"][0], dist["params"][1])
				_:
					push_error("Unknown distribution: " + str(dist["name"]))
					current_cdf = NAN
			
			# CDF should be monotonically non-decreasing
			assert_float(current_cdf).is_greater_equal(prev_cdf)
			
			# CDF should be between 0 and 1
			assert_float(current_cdf).is_greater_equal(0.0)
			assert_float(current_cdf).is_less_equal(1.0)
			
			prev_cdf = current_cdf

# --- End-to-End Statistical Computation Tests ---

## Tests a complete statistical workflow: data generation → analysis → validation
func test_end_to_end_normal_distribution_workflow() -> void:
	# Generate sample from normal distribution using our Distributions module
	var workflow_params: Dictionary = CDF_PDF_INTEGRATION_TEST_DATA.WORKFLOW_TEST_PARAMETERS
	var sample_size: int = workflow_params["sample_size"]
	var mu: float = workflow_params["normal_params"]["mu"]
	var sigma: float = workflow_params["normal_params"]["sigma"]
	var samples: Array[float] = []
	
	# Use a fixed seed for reproducible testing
	StatMath.set_global_seed(workflow_params["test_seed"])
	
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_normal(mu, sigma))
	
	# Calculate sample statistics using BasicStats
	var sample_mean: float = StatMath.BasicStats.mean(samples)
	var sample_std: float = StatMath.BasicStats.standard_deviation(samples)
	
	# Validate that sample statistics are close to theoretical values
	# With 1000 samples, we expect good approximation
	assert_float(sample_mean).is_equal_approx(mu, StatMath.NEGATIVE_BINOMIAL_TOLERANCE)  # Within tolerance of true mean
	assert_float(sample_std).is_equal_approx(sigma, StatMath.HIGH_DISTRIBUTION_TOLERANCE)  # Within tolerance of true std
	
	# Test that our CDF/PDF functions work with sample data
	samples.sort()  # Sort the array before calculating percentile
	var percentile_95: float = StatMath.BasicStats.percentile(samples, workflow_params["percentile_value"])
	var theoretical_95: float = StatMath.PpfFunctions.normal_ppf(workflow_params["percentile_value"] / 100.0, mu, sigma)
	
	assert_float(percentile_95).is_equal_approx(theoretical_95, StatMath.DEFAULT_TOLERANCE_FACTOR)  # Within tolerance factor - increased tolerance for sampling variation

## Tests cross-function consistency in probability calculations
func test_cross_function_probability_consistency() -> void:
	# Test that CDF, PDF, and PPF are mathematically consistent
	var distributions: Array[Dictionary] = [
		{"name": StatMath.SupportedDistributions.NORMAL, "cdf_params": [1.5, 0.0, 1.0], "pdf_params": [1.5, 0.0, 1.0], "ppf_params": [0.0, 1.0]},
		{"name": StatMath.SupportedDistributions.EXPONENTIAL, "cdf_params": [2.0, 1.0], "pdf_params": [2.0, 1.0], "ppf_params": [1.0]},
		{"name": StatMath.SupportedDistributions.UNIFORM, "cdf_params": [2.5, 1.0, 4.0], "pdf_params": [2.5, 1.0, 4.0], "ppf_params": [1.0, 4.0]}
	]
	
	for dist in distributions:
		var name: StatMath.SupportedDistributions = dist["name"]
		
		# Calculate CDF value directly
		var cdf_val: float
		var x_val: float = dist["cdf_params"][0]
		
		match name:
			StatMath.SupportedDistributions.NORMAL:
				cdf_val = StatMath.CdfFunctions.normal_cdf(x_val, dist["cdf_params"][1], dist["cdf_params"][2])
			StatMath.SupportedDistributions.EXPONENTIAL:
				cdf_val = StatMath.CdfFunctions.exponential_cdf(x_val, dist["cdf_params"][1])
			StatMath.SupportedDistributions.UNIFORM:
				cdf_val = StatMath.CdfFunctions.uniform_cdf(x_val, dist["cdf_params"][1], dist["cdf_params"][2])
			_:
				push_error("Unknown distribution: " + str(name))
				cdf_val = NAN
		
		# Calculate corresponding PPF value directly
		var ppf_val: float
		match name:
			StatMath.SupportedDistributions.NORMAL:
				ppf_val = StatMath.PpfFunctions.normal_ppf(cdf_val, dist["ppf_params"][0], dist["ppf_params"][1])
			StatMath.SupportedDistributions.EXPONENTIAL:
				ppf_val = StatMath.PpfFunctions.exponential_ppf(cdf_val, dist["ppf_params"][0])
			StatMath.SupportedDistributions.UNIFORM:
				ppf_val = StatMath.PpfFunctions.uniform_ppf(cdf_val, dist["ppf_params"][0], dist["ppf_params"][1])
			_:
				push_error("PPF not implemented for distribution: " + str(name))
				ppf_val = NAN
		
		# PPF(CDF(x)) should equal x
		assert_float(ppf_val).is_equal_approx(dist["cdf_params"][0], StatMath.INVERSE_CONSISTENCY_TOLERANCE)

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
	assert_float(result_large).is_equal_approx(1.0, StatMath.BOUNDARY_TOLERANCE)  # Should be very close to 1
	
	# Test with extreme probability values
	var result_extreme_low: float = StatMath.PpfFunctions.normal_ppf(1e-10)
	var result_extreme_high: float = StatMath.PpfFunctions.normal_ppf(1.0 - 1e-10)
	assert_bool(is_finite(result_extreme_low)).is_true()
	assert_bool(is_finite(result_extreme_high)).is_true()

## Tests behavior at distribution boundaries and special points
func test_distribution_boundary_behavior() -> void:
	# Test uniform distribution at boundaries
	assert_float(StatMath.CdfFunctions.uniform_cdf(1.0, 1.0, 4.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.CdfFunctions.uniform_cdf(4.0, 1.0, 4.0)).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)
	
	# Test beta distribution at boundaries
	assert_float(StatMath.CdfFunctions.beta_cdf(0.0, 2.0, 3.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.CdfFunctions.beta_cdf(1.0, 2.0, 3.0)).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)
	
	# Test exponential distribution at x=0
	assert_float(StatMath.CdfFunctions.exponential_cdf(0.0, 2.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

## Tests parameter validation for integration functions
func test_integration_parameter_validation() -> void:
	# This is a placeholder for parameter validation tests
	# Integration tests primarily focus on mathematical properties
	# Individual function parameter validation is tested in their respective test files
	pass

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# TERRIBLE ABSTRACTION LAYERS ELIMINATED - USE DIRECT FUNCTION CALLS 
