# addons/godot-stat-math/tests/core/ppf_functions_test.gd
class_name PpfFunctionsTest extends GdUnitTestSuite

# =============================================================================
# TOLERANCE CONSTANTS
# =============================================================================

## Tolerance for floating point comparisons in PPF calculations  
const FLOAT_TOLERANCE: float = 1e-6

## Tolerance for scipy validation comparisons
const SCIPY_TOLERANCE: float = 2e-6

## Tolerance for numerical approximations  
const NUMERICAL_TOLERANCE: float = 1e-5

## Tolerance for CDF-PPF consistency (round-trip) tests
const CDF_PPF_CONSISTENCY_TOLERANCE: float = 1e-5

# =============================================================================
# PHASE 3: SCIPY VALIDATION TESTS
# =============================================================================

## Validates normal distribution PPF against known scipy values
func test_normal_ppf_scipy_validation_parametrized(p: float, mu: float, sigma: float, expected: float, test_parameters := [
	[0.025, 0.0, 1.0, -1.95996398],
	[0.5, 0.0, 1.0, 0.0],
	[0.975, 0.0, 1.0, 1.95996398],
	[0.5, 10.0, 2.0, 10.0],
	[0.84134475, 0.0, 1.0, 1.0]
	]) -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(p, mu, sigma)
	assert_float(result).is_equal_approx(expected, SCIPY_TOLERANCE)

## Validates exponential distribution PPF against known scipy values
## Note: scipy uses scale parameterization (scale = 1/lambda), we use rate parameterization (lambda)
func test_exponential_ppf_scipy_validation_parametrized(p: float, lambda_param: float, expected: float, test_parameters := [
	# Using test data from ppf_test_data.gd with parameter conversion:
	# scipy: expon.ppf(p, scale=scale) where scale = 1/lambda
	# ours: exponential_ppf(p, lambda) where lambda = 1/scale
	[0.5, 1.0, 0.69314718],        # scipy: p=0.5, scale=1.0 -> lambda=1.0, expected=0.69314718
	[0.632121, 1.0, 1.00000120],   # scipy: p=0.632121, scale=1.0 -> lambda=1.0, expected=1.00000120  
	[0.95, 2.0, 1.49786614],       # scipy: p=0.95, scale=2.0 -> lambda=0.5, expected=1.49786614
	[0.1, 0.5, 0.21072103],        # scipy: p=0.1, scale=0.5 -> lambda=2.0, expected=0.21072103
	]) -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(p, lambda_param)
	assert_float(result).is_equal_approx(expected, SCIPY_TOLERANCE)

## Validates uniform distribution PPF against known scipy values
func test_uniform_ppf_scipy_validation_parametrized(p: float, a: float, b: float, expected: float, test_parameters := [
	[0.25, 0.0, 4.0, 1.0],
	[0.5, 1.0, 5.0, 3.0],
	[0.75, 2.0, 6.0, 5.0],
	[0.0, 0.0, 1.0, 0.0],
	[1.0, 0.0, 1.0, 1.0]
	]) -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(p, a, b)
	assert_float(result).is_equal_approx(expected, SCIPY_TOLERANCE)

## Validates pareto distribution PPF against known scipy values
func test_pareto_ppf_scipy_validation_parametrized(p: float, scale: float, shape: float, expected: float, test_parameters := [
	[0.5, 1.0, 1.0, 2.0],
	[0.75, 2.0, 1.0, 8.0],
	[0.9, 3.0, 2.0, 9.48683298]
	]) -> void:
	var result: float = StatMath.PpfFunctions.pareto_ppf(p, scale, shape)
	assert_float(result).is_equal_approx(expected, SCIPY_TOLERANCE)

## Validates weibull distribution PPF against known scipy values
func test_weibull_ppf_scipy_validation_parametrized(p: float, scale: float, shape: float, expected: float, test_parameters := [
	[0.5, 1.0, 1.0, 0.69314718],
	[0.632121, 2.0, 2.0, 2.0],
	[0.25, 1.0, 2.0, 0.53636002]
	]) -> void:
	var result: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
	assert_float(result).is_equal_approx(expected, SCIPY_TOLERANCE)

# =============================================================================
# PHASE 3: CONSISTENCY TESTS (CDF-PPF INVERSE RELATIONSHIPS)
# =============================================================================

## Tests that CDF and PPF are inverse functions for all distributions
func test_cdf_ppf_round_trip_consistency_parametrized(distribution: StatMath.SupportedDistributions, test_x: float, params: Array, test_parameters := [
	# Normal distribution
	[StatMath.SupportedDistributions.NORMAL, 0.0, [0.0, 1.0]],
	[StatMath.SupportedDistributions.NORMAL, 1.5, [0.0, 1.0]],
	[StatMath.SupportedDistributions.NORMAL, -2.0, [0.0, 1.0]],
	[StatMath.SupportedDistributions.NORMAL, 5.0, [2.0, 1.5]],
	# Exponential distribution
	[StatMath.SupportedDistributions.EXPONENTIAL, 0.5, [1.0]],
	[StatMath.SupportedDistributions.EXPONENTIAL, 2.0, [0.5]],
	[StatMath.SupportedDistributions.EXPONENTIAL, 1.0, [2.0]],
	[StatMath.SupportedDistributions.EXPONENTIAL, 5.0, [0.2]],
	# Uniform distribution
	[StatMath.SupportedDistributions.UNIFORM, 1.5, [1.0, 4.0]],
	[StatMath.SupportedDistributions.UNIFORM, 2.5, [0.0, 5.0]],
	[StatMath.SupportedDistributions.UNIFORM, 3.0, [2.0, 6.0]],
	# Weibull distribution
	[StatMath.SupportedDistributions.WEIBULL, 1.0, [1.0, 1.0]],
	[StatMath.SupportedDistributions.WEIBULL, 2.0, [2.0, 2.0]],
	[StatMath.SupportedDistributions.WEIBULL, 0.5, [1.5, 3.0]],
	# Pareto distribution
	[StatMath.SupportedDistributions.PARETO, 2.5, [2.0, 1.5]],
	[StatMath.SupportedDistributions.PARETO, 5.0, [2.0, 1.0]],  # Fixed: scale=2.0, shape=1.0 
	[StatMath.SupportedDistributions.PARETO, 3.0, [2.5, 3.0]]
	]) -> void:
	# Forward: x -> CDF(x) -> PPF(CDF(x)) should equal x
	var cdf_value: float = _get_cdf_value(distribution, test_x, params)
	var ppf_result: float = _get_ppf_value(distribution, cdf_value, params)
	assert_float(ppf_result).is_equal_approx(test_x, CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests that PPF and CDF are inverse functions (p -> PPF(p) -> CDF(PPF(p)) = p)
func test_ppf_cdf_round_trip_consistency_parametrized(distribution: StatMath.SupportedDistributions, test_p: float, params: Array, test_parameters := [
	# Normal distribution
	[StatMath.SupportedDistributions.NORMAL, 0.1, [0.0, 1.0]],
	[StatMath.SupportedDistributions.NORMAL, 0.5, [0.0, 1.0]],
	[StatMath.SupportedDistributions.NORMAL, 0.9, [0.0, 1.0]],
	[StatMath.SupportedDistributions.NORMAL, 0.25, [5.0, 2.0]],
	# Exponential distribution  
	[StatMath.SupportedDistributions.EXPONENTIAL, 0.2, [1.0]],
	[StatMath.SupportedDistributions.EXPONENTIAL, 0.5, [2.0]],
	[StatMath.SupportedDistributions.EXPONENTIAL, 0.8, [0.5]],
	# Uniform distribution
	[StatMath.SupportedDistributions.UNIFORM, 0.3, [1.0, 4.0]],
	[StatMath.SupportedDistributions.UNIFORM, 0.7, [0.0, 10.0]],
	# Weibull distribution
	[StatMath.SupportedDistributions.WEIBULL, 0.4, [1.0, 1.0]],
	[StatMath.SupportedDistributions.WEIBULL, 0.6, [2.0, 2.0]],
	# Pareto distribution
	[StatMath.SupportedDistributions.PARETO, 0.3, [2.0, 1.0]],  # Fixed: scale=2.0, shape=1.0
	[StatMath.SupportedDistributions.PARETO, 0.8, [2.0, 1.5]]
	]) -> void:
	# Reverse: p -> PPF(p) -> CDF(PPF(p)) should equal p
	var ppf_value: float = _get_ppf_value(distribution, test_p, params)
	var cdf_result: float = _get_cdf_value(distribution, ppf_value, params)
	assert_float(cdf_result).is_equal_approx(test_p, CDF_PPF_CONSISTENCY_TOLERANCE)

# =============================================================================
# PHASE 3: MONOTONICITY AND MATHEMATICAL PROPERTIES
# =============================================================================

## Tests that PPF functions are monotonically increasing
func test_ppf_monotonicity_parametrized(distribution: StatMath.SupportedDistributions, params: Array, test_probabilities: Array[float], test_parameters := [
	[StatMath.SupportedDistributions.NORMAL, [0.0, 1.0], [0.1, 0.25, 0.5, 0.75, 0.9]],
	[StatMath.SupportedDistributions.NORMAL, [2.0, 0.5], [0.05, 0.3, 0.6, 0.8, 0.95]],
	[StatMath.SupportedDistributions.EXPONENTIAL, [1.0], [0.1, 0.3, 0.5, 0.7, 0.9]],
	[StatMath.SupportedDistributions.EXPONENTIAL, [2.0], [0.05, 0.25, 0.5, 0.75, 0.95]],
	[StatMath.SupportedDistributions.UNIFORM, [1.0, 5.0], [0.0, 0.25, 0.5, 0.75, 1.0]],
	[StatMath.SupportedDistributions.UNIFORM, [0.0, 10.0], [0.1, 0.4, 0.6, 0.8, 0.99]],
	[StatMath.SupportedDistributions.WEIBULL, [1.0, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]],
	[StatMath.SupportedDistributions.WEIBULL, [2.0, 2.0], [0.05, 0.25, 0.5, 0.75, 0.95]],
	[StatMath.SupportedDistributions.PARETO, [2.0, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]],  # Fixed: scale=2.0, shape=1.0
	[StatMath.SupportedDistributions.PARETO, [2.0, 1.5], [0.05, 0.25, 0.5, 0.75, 0.95]]
	]) -> void:
	var prev_ppf: float = -INF
	
	for i in range(test_probabilities.size()):
		var p: float = test_probabilities[i]
		var current_ppf: float = _get_ppf_value(distribution, p, params)
		
		# PPF should be monotonically non-decreasing
		assert_float(current_ppf).is_greater_equal(prev_ppf)
		
		# PPF should be finite for valid probabilities (except boundary cases)
		if p > 0.0 and p < 1.0:
			assert_bool(is_finite(current_ppf)).is_true()
		
		prev_ppf = current_ppf

# =============================================================================
# PHASE 3: BOUNDARY CONDITIONS AND SPECIAL VALUES
# =============================================================================

## Tests PPF behavior at probability boundaries (0, 1) and special values
func test_ppf_boundary_conditions_parametrized(distribution: StatMath.SupportedDistributions, boundary_type: String, params: Array, p_value: float, expected_behavior: String, test_parameters := [
	# Lower boundary (p = 0)
	[StatMath.SupportedDistributions.NORMAL, "p_zero", [0.0, 1.0], 0.0, "negative_infinity"],
	[StatMath.SupportedDistributions.EXPONENTIAL, "p_zero", [1.0], 0.0, "zero"],
	[StatMath.SupportedDistributions.UNIFORM, "p_zero", [2.0, 5.0], 0.0, "lower_bound"],
	[StatMath.SupportedDistributions.WEIBULL, "p_zero", [1.0, 2.0], 0.0, "zero"],
	[StatMath.SupportedDistributions.PARETO, "p_zero", [2.0, 1.5], 0.0, "scale_value"],
	# Upper boundary (p = 1) 
	[StatMath.SupportedDistributions.NORMAL, "p_one", [0.0, 1.0], 1.0, "positive_infinity"],
	[StatMath.SupportedDistributions.EXPONENTIAL, "p_one", [1.0], 1.0, "positive_infinity"],
	[StatMath.SupportedDistributions.UNIFORM, "p_one", [2.0, 5.0], 1.0, "upper_bound"],
	[StatMath.SupportedDistributions.WEIBULL, "p_one", [1.0, 2.0], 1.0, "positive_infinity"],
	[StatMath.SupportedDistributions.PARETO, "p_one", [2.0, 1.5], 1.0, "positive_infinity"],
	# Special values (p = 0.5, median)
	[StatMath.SupportedDistributions.NORMAL, "median", [0.0, 1.0], 0.5, "zero"],
	[StatMath.SupportedDistributions.NORMAL, "median", [5.0, 2.0], 0.5, "mean_value"],
	[StatMath.SupportedDistributions.UNIFORM, "median", [2.0, 8.0], 0.5, "midpoint"],
	[StatMath.SupportedDistributions.EXPONENTIAL, "median", [1.0], 0.5, "ln_2"],
	# Quartiles
	[StatMath.SupportedDistributions.UNIFORM, "first_quartile", [0.0, 4.0], 0.25, "one_quarter"],
	[StatMath.SupportedDistributions.UNIFORM, "third_quartile", [0.0, 4.0], 0.75, "three_quarters"]
	]) -> void:
	var result: float = _get_ppf_value(distribution, p_value, params)
	
	match expected_behavior:
		"negative_infinity":
			assert_bool(is_inf(result) and result < 0.0).is_true()
		"positive_infinity":
			assert_bool(is_inf(result) and result > 0.0).is_true()
		"zero":
			assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)
		"lower_bound":
			assert_float(result).is_equal_approx(params[0], FLOAT_TOLERANCE)
		"upper_bound":
			assert_float(result).is_equal_approx(params[1], FLOAT_TOLERANCE)
		"scale_value":
			assert_float(result).is_equal_approx(params[0], FLOAT_TOLERANCE)  # Pareto scale
		"mean_value":
			assert_float(result).is_equal_approx(params[0], FLOAT_TOLERANCE)  # Normal mean
		"midpoint":
			var expected_mid: float = (params[0] + params[1]) / 2.0
			assert_float(result).is_equal_approx(expected_mid, FLOAT_TOLERANCE)
		"ln_2":
			assert_float(result).is_equal_approx(0.6931472, SCIPY_TOLERANCE)
		"one_quarter":
			var expected_quarter: float = params[0] + 0.25 * (params[1] - params[0])
			assert_float(result).is_equal_approx(expected_quarter, FLOAT_TOLERANCE)
		"three_quarters":
			var expected_three_quarter: float = params[0] + 0.75 * (params[1] - params[0])
			assert_float(result).is_equal_approx(expected_three_quarter, FLOAT_TOLERANCE)

# =============================================================================
# PHASE 3: ENHANCED PARAMETER VALIDATION
# =============================================================================

## Enhanced parameter validation tests with detailed error messages
func test_ppf_parameter_validation_enhanced_parametrized(distribution: StatMath.SupportedDistributions, validation_type: String, params: Array, p_value: float, expected_error: String, test_parameters := [
	# Probability parameter validation
	[StatMath.SupportedDistributions.NORMAL, "p_negative", [0.0, 1.0], -0.1, "Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1"],
	[StatMath.SupportedDistributions.NORMAL, "p_greater_than_one", [0.0, 1.0], 1.5, "Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.5"],
	[StatMath.SupportedDistributions.EXPONENTIAL, "p_negative", [1.0], -0.01, "Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.01"],
	[StatMath.SupportedDistributions.UNIFORM, "p_out_of_range", [1.0, 5.0], 2.0, "Probability p must be between 0.0 and 1.0 (inclusive). Received: 2.0"],
	# Distribution parameter validation
	[StatMath.SupportedDistributions.NORMAL, "sigma_zero", [0.0, 0.0], 0.5, "Standard deviation sigma must be positive. Received: 0.0"],
	[StatMath.SupportedDistributions.NORMAL, "sigma_negative", [0.0, -1.0], 0.5, "Standard deviation sigma must be positive. Received: -1.0"],
	[StatMath.SupportedDistributions.EXPONENTIAL, "lambda_zero", [0.0], 0.5, "Rate lambda_param must be positive. Received: 0.0"],
	[StatMath.SupportedDistributions.EXPONENTIAL, "lambda_negative", [-1.0], 0.5, "Rate lambda_param must be positive. Received: -1.0"],
	[StatMath.SupportedDistributions.UNIFORM, "a_greater_than_b", [5.0, 2.0], 0.5, "Parameter b must be greater than or equal to a. Received a=5.0, b=2.0"],
	[StatMath.SupportedDistributions.WEIBULL, "scale_zero", [0.0, 2.0], 0.5, "Scale parameter must be positive. Received: 0.0"],
	[StatMath.SupportedDistributions.WEIBULL, "shape_negative", [1.0, -1.0], 0.5, "Shape parameter must be positive. Received: -1.0"],
	[StatMath.SupportedDistributions.PARETO, "scale_zero", [0.0, 1.0], 0.5, "Scale parameter must be positive. Received: 0.0"],
	[StatMath.SupportedDistributions.PARETO, "shape_negative", [1.0, -2.0], 0.5, "Shape parameter must be positive. Received: -2.0"]
	]) -> void:
	var test_call: Callable = func():
		_get_ppf_value(distribution, p_value, params)
	
	await assert_error(test_call).is_push_error(expected_error)

# =============================================================================
# PHASE 3: SPECIAL MATHEMATICAL RELATIONSHIPS
# =============================================================================

## Tests special mathematical relationships between distributions via PPF
func test_ppf_special_relationships_parametrized(relationship_type: String, params: Array, p_value: float, test_parameters := [
	# Weibull(scale=λ, shape=1) = Exponential(rate=1/λ)
	["weibull_exponential", [2.0, 1.0, 0.5], 0.5],  # [scale, shape, rate]
	["weibull_exponential", [1.5, 1.0, 2.0/3.0], 0.3],
	# Normal approximation for large parameters
	["normal_large_param", [0.0, 1.0], 0.025],
	["normal_large_param", [0.0, 1.0], 0.975],
	# Uniform special cases
	["uniform_degenerate", [5.0, 5.0], 0.5],  # a = b case
	# Pareto scale relationships
	["pareto_scale", [1.0, 1.0], 0.5]  # PPF(0.5) = 2*scale when shape=1
	]) -> void:
	match relationship_type:
		"weibull_exponential":
			var scale: float = params[0]
			var shape: float = params[1] 
			var rate: float = params[2]
			var weibull_ppf: float = StatMath.PpfFunctions.weibull_ppf(p_value, scale, shape)
			var exp_ppf: float = StatMath.PpfFunctions.exponential_ppf(p_value, rate)
			assert_float(weibull_ppf).is_equal_approx(exp_ppf, NUMERICAL_TOLERANCE)
		
		"normal_large_param":
			# Test that normal PPF gives expected z-scores
			var mu: float = params[0]
			var sigma: float = params[1]
			var result: float = StatMath.PpfFunctions.normal_ppf(p_value, mu, sigma)
			if p_value == 0.025:
				assert_float(result).is_equal_approx(-1.9599640, SCIPY_TOLERANCE)
			elif p_value == 0.975:
				assert_float(result).is_equal_approx(1.9599640, SCIPY_TOLERANCE)
		
		"uniform_degenerate":
			# When a = b, PPF should always return that value
			var a: float = params[0]
			var b: float = params[1]
			var result: float = StatMath.PpfFunctions.uniform_ppf(p_value, a, b)
			assert_float(result).is_equal_approx(a, FLOAT_TOLERANCE)
		
		"pareto_scale":
			# For Pareto(scale, 1), PPF(0.5) = 2*scale
			var scale: float = params[0]
			var shape: float = params[1]
			var result: float = StatMath.PpfFunctions.pareto_ppf(p_value, scale, shape)
			if p_value == 0.5 and shape == 1.0:
				assert_float(result).is_equal_approx(2.0 * scale, NUMERICAL_TOLERANCE)

# =============================================================================
# PHASE 3: HELPER FUNCTIONS
# =============================================================================

## Enhanced helper function to get PPF values for all supported distributions
func _get_ppf_value(distribution: StatMath.SupportedDistributions, p: float, params: Array) -> float:
	match distribution:
		StatMath.SupportedDistributions.NORMAL:
			return StatMath.PpfFunctions.normal_ppf(p, params[0], params[1])
		StatMath.SupportedDistributions.EXPONENTIAL:
			return StatMath.PpfFunctions.exponential_ppf(p, params[0])
		StatMath.SupportedDistributions.UNIFORM:
			return StatMath.PpfFunctions.uniform_ppf(p, params[0], params[1])
		StatMath.SupportedDistributions.WEIBULL:
			return StatMath.PpfFunctions.weibull_ppf(p, params[0], params[1])
		StatMath.SupportedDistributions.PARETO:
			return StatMath.PpfFunctions.pareto_ppf(p, params[0], params[1])
		_:
			push_error("PPF not implemented for distribution: " + str(distribution))
			return NAN

## Enhanced helper function to get CDF values for consistency testing
func _get_cdf_value(distribution: StatMath.SupportedDistributions, x: float, params: Array) -> float:
	match distribution:
		StatMath.SupportedDistributions.NORMAL:
			return StatMath.CdfFunctions.normal_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.EXPONENTIAL:
			return StatMath.CdfFunctions.exponential_cdf(x, params[0])
		StatMath.SupportedDistributions.UNIFORM:
			return StatMath.CdfFunctions.uniform_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.WEIBULL:
			return StatMath.CdfFunctions.weibull_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.PARETO:
			return StatMath.CdfFunctions.pareto_cdf(x, params[0], params[1])
		_:
			push_error("CDF not implemented for distribution: " + str(distribution))
			return NAN
