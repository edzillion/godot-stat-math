# res://addons/godot-stat-math/tests/core/cdf_pdf_integration_test.gd
class_name CdfPdfIntegrationTest extends GdUnitTestSuite

## Phase 3: Integration Testing - Mathematical Relationships
##
## This test suite verifies integration and end-to-end statistical workflows:
## • CDF and PDF consistency through probability calculations
## • CDFs should be monotonically increasing  
## • End-to-end statistical computation validation
## • Numerical stability under various conditions
## • Cross-distribution mathematical relationships

# Using centralized tolerances from StatMath class
const FLOAT_TOLERANCE: float = StatMath.FLOAT_TOLERANCE

# =============================================================================
# CDF ↔ PDF DERIVATIVE RELATIONSHIP TESTS
# =============================================================================

## Tests that the numerical derivative of Normal CDF approximates Normal PDF
func test_normal_cdf_pdf_derivative_relationship() -> void:
	var test_points: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
	var mu: float = 0.0
	var sigma: float = 1.0
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		# Calculate numerical derivative: d/dx CDF(x) ≈ (CDF(x+h) - CDF(x-h)) / (2h)
		var cdf_plus: float = StatMath.CdfFunctions.normal_cdf(x + h, mu, sigma)
		var cdf_minus: float = StatMath.CdfFunctions.normal_cdf(x - h, mu, sigma)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		# Calculate actual PDF value
		var pdf_value: float = StatMath.PmfPdfFunctions.normal_pdf(x, mu, sigma)
		
		assert_float(numerical_derivative).is_equal_approx(pdf_value, StatMath.DERIVATIVE_TOLERANCE)

## Tests that the numerical derivative of Exponential CDF approximates Exponential PDF
func test_exponential_cdf_pdf_derivative_relationship() -> void:
	var test_points: Array[float] = [0.1, 0.5, 1.0, 2.0, 5.0]  # Avoid x=0 for stability
	var lambda_param: float = 2.0
	var h: float = StatMath.NUMERICAL_DIFFERENTIATION_H
	
	for x in test_points:
		var cdf_plus: float = StatMath.CdfFunctions.exponential_cdf(x + h, lambda_param)
		var cdf_minus: float = StatMath.CdfFunctions.exponential_cdf(x - h, lambda_param)
		var numerical_derivative: float = (cdf_plus - cdf_minus) / (2.0 * h)
		
		var pdf_value: float = StatMath.PmfPdfFunctions.exponential_pdf(x, lambda_param)
		
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

# =============================================================================
# CDF MONOTONICITY TESTS
# =============================================================================

## Tests that CDFs are monotonically increasing for all continuous distributions
func test_cdf_monotonicity_all_distributions() -> void:
	var distributions: Array[Dictionary] = [
		{"name": StatMath.SupportedDistributions.NORMAL, "params": [0.0, 1.0], "points": [-3.0, -1.0, 0.0, 1.0, 3.0]},
		{"name": StatMath.SupportedDistributions.EXPONENTIAL, "params": [1.0], "points": [0.1, 0.5, 1.0, 2.0, 5.0]},
		{"name": StatMath.SupportedDistributions.UNIFORM, "params": [1.0, 4.0], "points": [1.0, 1.5, 2.5, 3.5, 4.0]},
		{"name": StatMath.SupportedDistributions.BETA, "params": [2.0, 3.0], "points": [0.0, 0.25, 0.5, 0.75, 1.0]},
		{"name": StatMath.SupportedDistributions.GAMMA, "params": [2.0, 1.5], "points": [0.1, 1.0, 2.0, 4.0, 6.0]},
		{"name": StatMath.SupportedDistributions.WEIBULL, "params": [2.0, 2.0], "points": [0.1, 1.0, 2.0, 3.0, 4.0]}
	]
	
	for dist in distributions:
		var points: Array = dist["points"]
		var prev_cdf: float = -1.0
		
		for i in range(points.size()):
			var x: float = points[i]
			var current_cdf: float = _get_cdf_value(dist["name"], x, dist["params"])
			
			# CDF should be monotonically non-decreasing
			assert_float(current_cdf).is_greater_equal(prev_cdf)
			
			# CDF should be between 0 and 1
			assert_float(current_cdf).is_greater_equal(0.0)
			assert_float(current_cdf).is_less_equal(1.0)
			
			prev_cdf = current_cdf

# =============================================================================
# END-TO-END STATISTICAL COMPUTATION TESTS
# =============================================================================

## Tests a complete statistical workflow: data generation → analysis → validation
func test_end_to_end_normal_distribution_workflow() -> void:
	# Generate sample from normal distribution using our Distributions module
	var sample_size: int = 1000
	var mu: float = 5.0
	var sigma: float = 2.0
	var samples: Array[float] = []
	
	# Use a fixed seed for reproducible testing
	StatMath.set_global_seed(12345)
	
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
	var percentile_95: float = StatMath.BasicStats.percentile(samples, 95.0)
	var theoretical_95: float = StatMath.PpfFunctions.normal_ppf(0.95, mu, sigma)
	
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
		
		# Calculate CDF value
		var cdf_val: float = _get_cdf_value(name, dist["cdf_params"][0], dist["cdf_params"].slice(1))
		
		# Calculate corresponding PPF value (should return original x)
		var ppf_val: float = _get_ppf_value(name, cdf_val, dist["ppf_params"])
		
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
	assert_float(result_large).is_equal_approx(1.0, 1e-10)  # Should be very close to 1
	
	# Test with extreme probability values
	var result_extreme_low: float = StatMath.PpfFunctions.normal_ppf(1e-10)
	var result_extreme_high: float = StatMath.PpfFunctions.normal_ppf(1.0 - 1e-10)
	assert_bool(is_finite(result_extreme_low)).is_true()
	assert_bool(is_finite(result_extreme_high)).is_true()

## Tests behavior at distribution boundaries and special points
func test_distribution_boundary_behavior() -> void:
	# Test uniform distribution at boundaries
	assert_float(StatMath.CdfFunctions.uniform_cdf(1.0, 1.0, 4.0)).is_equal_approx(0.0, FLOAT_TOLERANCE)
	assert_float(StatMath.CdfFunctions.uniform_cdf(4.0, 1.0, 4.0)).is_equal_approx(1.0, FLOAT_TOLERANCE)
	
	# Test beta distribution at boundaries
	assert_float(StatMath.CdfFunctions.beta_cdf(0.0, 2.0, 3.0)).is_equal_approx(0.0, FLOAT_TOLERANCE)
	assert_float(StatMath.CdfFunctions.beta_cdf(1.0, 2.0, 3.0)).is_equal_approx(1.0, FLOAT_TOLERANCE)
	
	# Test exponential distribution at x=0
	assert_float(StatMath.CdfFunctions.exponential_cdf(0.0, 2.0)).is_equal_approx(0.0, FLOAT_TOLERANCE)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

## Helper function to get CDF values for different distributions
func _get_cdf_value(distribution: Variant, x: float, params: Array) -> float:
	# Handle both string and enum inputs during transition
	var dist_enum: StatMath.SupportedDistributions
	if distribution is String:
		dist_enum = _string_to_enum(distribution)
	else:
		dist_enum = distribution
	
	match dist_enum:
		StatMath.SupportedDistributions.NORMAL:
			return StatMath.CdfFunctions.normal_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.EXPONENTIAL:
			return StatMath.CdfFunctions.exponential_cdf(x, params[0])
		StatMath.SupportedDistributions.UNIFORM:
			return StatMath.CdfFunctions.uniform_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.BETA:
			return StatMath.CdfFunctions.beta_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.GAMMA:
			return StatMath.CdfFunctions.gamma_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.WEIBULL:
			return StatMath.CdfFunctions.weibull_cdf(x, params[0], params[1])
		_:
			push_error("Unknown distribution enum: " + str(dist_enum))
			return NAN

## Helper function to convert string distribution names to enum values
func _string_to_enum(distribution: String) -> StatMath.SupportedDistributions:
	match distribution:
		"normal": return StatMath.SupportedDistributions.NORMAL
		"exponential": return StatMath.SupportedDistributions.EXPONENTIAL
		"uniform": return StatMath.SupportedDistributions.UNIFORM
		"beta": return StatMath.SupportedDistributions.BETA
		"gamma": return StatMath.SupportedDistributions.GAMMA
		"weibull": return StatMath.SupportedDistributions.WEIBULL
		_:
			push_error("Unknown distribution: " + distribution)
			return StatMath.SupportedDistributions.NORMAL

## Helper function to get PPF values for different distributions
func _get_ppf_value(distribution: Variant, p: float, params: Array) -> float:
	# Handle both string and enum inputs during transition
	var dist_enum: StatMath.SupportedDistributions
	if distribution is String:
		dist_enum = _string_to_enum(distribution)
	else:
		dist_enum = distribution
	
	match dist_enum:
		StatMath.SupportedDistributions.NORMAL:
			return StatMath.PpfFunctions.normal_ppf(p, params[0], params[1])
		StatMath.SupportedDistributions.EXPONENTIAL:
			return StatMath.PpfFunctions.exponential_ppf(p, params[0])
		StatMath.SupportedDistributions.UNIFORM:
			return StatMath.PpfFunctions.uniform_ppf(p, params[0], params[1])
		_:
			push_error("PPF not implemented for distribution: " + str(dist_enum))
			return NAN 