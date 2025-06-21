# res://addons/godot-stat-math/tests/core/ppf_functions_test.gd
class_name PpfFunctionsTest extends GdUnitTestSuite

const PPF_TEST_DATA = preload("res://addons/godot-stat-math/tables/ppf_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

## Validates normal distribution PPF against scipy values
func test_normal_ppf_scipy_validation() -> void:
	var test_data: Array = PPF_TEST_DATA.VALUES["normal_ppf"]
	for case in test_data:
		var result: float = StatMath.PpfFunctions.normal_ppf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Validates exponential distribution PPF against scipy values
func test_exponential_ppf_scipy_validation() -> void:
	var test_data: Array = PPF_TEST_DATA.VALUES["exponential_ppf"]
	for case in test_data:
		var result: float = StatMath.PpfFunctions.exponential_ppf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Validates uniform distribution PPF against scipy values
func test_uniform_ppf_scipy_validation() -> void:
	var test_data: Array = PPF_TEST_DATA.VALUES["uniform_ppf"]
	for case in test_data:
		var result: float = StatMath.PpfFunctions.uniform_ppf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Validates pareto distribution PPF against scipy values
func test_pareto_ppf_scipy_validation() -> void:
	var test_data: Array = PPF_TEST_DATA.VALUES["pareto_ppf"]
	for case in test_data:
		var result: float = StatMath.PpfFunctions.pareto_ppf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

## Validates weibull distribution PPF against scipy values
func test_weibull_ppf_scipy_validation() -> void:
	var test_data: Array = PPF_TEST_DATA.VALUES["weibull_ppf"]
	for case in test_data:
		var result: float = StatMath.PpfFunctions.weibull_ppf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)

# =============================================================================
# CDF-PPF CONSISTENCY TESTS - DIRECT FUNCTION CALLS
# =============================================================================

## Tests that CDF and PPF are inverse functions - Normal Distribution
func test_normal_cdf_ppf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"x": 0.0, "mu": 0.0, "sigma": 1.0},
		{"x": 1.5, "mu": 0.0, "sigma": 1.0},
		{"x": -2.0, "mu": 0.0, "sigma": 1.0},
		{"x": 5.0, "mu": 2.0, "sigma": 1.5}
	]
	
	for case in test_cases:
		# Forward: x -> CDF(x) -> PPF(CDF(x)) should equal x
		var cdf_value: float = StatMath.CdfFunctions.normal_cdf(case["x"], case["mu"], case["sigma"])
		var ppf_result: float = StatMath.PpfFunctions.normal_ppf(cdf_value, case["mu"], case["sigma"])
		assert_float(ppf_result).is_equal_approx(case["x"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests that CDF and PPF are inverse functions - Exponential Distribution
func test_exponential_cdf_ppf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"x": 0.5, "lambda": 1.0},
		{"x": 2.0, "lambda": 0.5},
		{"x": 1.0, "lambda": 2.0},
		{"x": 5.0, "lambda": 0.2}
	]
	
	for case in test_cases:
		var cdf_value: float = StatMath.CdfFunctions.exponential_cdf(case["x"], case["lambda"])
		var ppf_result: float = StatMath.PpfFunctions.exponential_ppf(cdf_value, case["lambda"])
		assert_float(ppf_result).is_equal_approx(case["x"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests that CDF and PPF are inverse functions - Uniform Distribution
func test_uniform_cdf_ppf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"x": 1.5, "a": 1.0, "b": 4.0},
		{"x": 2.5, "a": 0.0, "b": 5.0},
		{"x": 3.0, "a": 2.0, "b": 6.0}
	]
	
	for case in test_cases:
		var cdf_value: float = StatMath.CdfFunctions.uniform_cdf(case["x"], case["a"], case["b"])
		var ppf_result: float = StatMath.PpfFunctions.uniform_ppf(cdf_value, case["a"], case["b"])
		assert_float(ppf_result).is_equal_approx(case["x"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests that PPF and CDF are inverse functions - Normal Distribution
func test_normal_ppf_cdf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"p": 0.1, "mu": 0.0, "sigma": 1.0},
		{"p": 0.5, "mu": 0.0, "sigma": 1.0},
		{"p": 0.9, "mu": 0.0, "sigma": 1.0},
		{"p": 0.25, "mu": 5.0, "sigma": 2.0}
	]
	
	for case in test_cases:
		# Reverse: p -> PPF(p) -> CDF(PPF(p)) should equal p
		var ppf_value: float = StatMath.PpfFunctions.normal_ppf(case["p"], case["mu"], case["sigma"])
		var cdf_result: float = StatMath.CdfFunctions.normal_cdf(ppf_value, case["mu"], case["sigma"])
		assert_float(cdf_result).is_equal_approx(case["p"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests that PPF and CDF are inverse functions - Exponential Distribution
func test_exponential_ppf_cdf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"p": 0.2, "lambda": 1.0},
		{"p": 0.5, "lambda": 2.0},
		{"p": 0.8, "lambda": 0.5}
	]
	
	for case in test_cases:
		var ppf_value: float = StatMath.PpfFunctions.exponential_ppf(case["p"], case["lambda"])
		var cdf_result: float = StatMath.CdfFunctions.exponential_cdf(ppf_value, case["lambda"])
		assert_float(cdf_result).is_equal_approx(case["p"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

# =============================================================================
# MONOTONICITY TESTS - DIRECT FUNCTION CALLS
# =============================================================================

## Tests that PPF functions are monotonically increasing - Normal Distribution
func test_normal_ppf_monotonicity() -> void:
	var test_probabilities: Array[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
	var mu: float = 0.0
	var sigma: float = 1.0
	var prev_ppf: float = -INF
	
	for p in test_probabilities:
		var current_ppf: float = StatMath.PpfFunctions.normal_ppf(p, mu, sigma)
		assert_float(current_ppf).is_greater_equal(prev_ppf)
		assert_bool(is_finite(current_ppf)).is_true()
		prev_ppf = current_ppf

## Tests that PPF functions are monotonically increasing - Exponential Distribution
func test_exponential_ppf_monotonicity() -> void:
	var test_probabilities: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
	var lambda_param: float = 1.0
	var prev_ppf: float = -INF
	
	for p in test_probabilities:
		var current_ppf: float = StatMath.PpfFunctions.exponential_ppf(p, lambda_param)
		assert_float(current_ppf).is_greater_equal(prev_ppf)
		assert_bool(is_finite(current_ppf)).is_true()
		prev_ppf = current_ppf

# =============================================================================
# BOUNDARY CONDITIONS TESTS - DIRECT FUNCTION CALLS
# =============================================================================

## Tests PPF behavior at probability boundaries - Normal Distribution
func test_normal_ppf_boundary_conditions() -> void:
	var mu: float = 0.0
	var sigma: float = 1.0
	
	# Lower boundary (p = 0)
	var p_zero_result: float = StatMath.PpfFunctions.normal_ppf(0.0, mu, sigma)
	assert_bool(is_inf(p_zero_result) and p_zero_result < 0.0).is_true()
	
	# Upper boundary (p = 1)
	var p_one_result: float = StatMath.PpfFunctions.normal_ppf(1.0, mu, sigma)
	assert_bool(is_inf(p_one_result) and p_one_result > 0.0).is_true()
	
	# Median (p = 0.5)
	var median_result: float = StatMath.PpfFunctions.normal_ppf(0.5, mu, sigma)
	assert_float(median_result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

## Tests PPF behavior at probability boundaries - Uniform Distribution
func test_uniform_ppf_boundary_conditions() -> void:
	var a: float = 2.0
	var b: float = 5.0
	
	# Lower boundary (p = 0)
	var p_zero_result: float = StatMath.PpfFunctions.uniform_ppf(0.0, a, b)
	assert_float(p_zero_result).is_equal_approx(a, StatMath.FLOAT_TOLERANCE)
	
	# Upper boundary (p = 1)
	var p_one_result: float = StatMath.PpfFunctions.uniform_ppf(1.0, a, b)
	assert_float(p_one_result).is_equal_approx(b, StatMath.FLOAT_TOLERANCE)
	
	# Median (p = 0.5)
	var median_result: float = StatMath.PpfFunctions.uniform_ppf(0.5, a, b)
	var expected_midpoint: float = (a + b) / 2.0
	assert_float(median_result).is_equal_approx(expected_midpoint, StatMath.FLOAT_TOLERANCE)

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

## Tests parameter validation for normal PPF
func test_normal_ppf_parameter_validation() -> void:
	# Invalid probability values
	var test_p_negative: Callable = func():
		StatMath.PpfFunctions.normal_ppf(-0.1, 0.0, 1.0)
	await assert_error(test_p_negative).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")
	
	var test_p_greater_than_one: Callable = func():
		StatMath.PpfFunctions.normal_ppf(1.5, 0.0, 1.0)
	await assert_error(test_p_greater_than_one).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.5")
	
	# Invalid sigma values
	var test_sigma_zero: Callable = func():
		StatMath.PpfFunctions.normal_ppf(0.5, 0.0, 0.0)
	await assert_error(test_sigma_zero).is_push_error("Standard deviation sigma must be positive. Received: 0.0")

## Tests parameter validation for exponential PPF
func test_exponential_ppf_parameter_validation() -> void:
	# Invalid lambda values
	var test_lambda_zero: Callable = func():
		StatMath.PpfFunctions.exponential_ppf(0.5, 0.0)
	await assert_error(test_lambda_zero).is_push_error("Rate lambda_param must be positive. Received: 0.0")
	
	var test_lambda_negative: Callable = func():
		StatMath.PpfFunctions.exponential_ppf(0.5, -1.0)
	await assert_error(test_lambda_negative).is_push_error("Rate lambda_param must be positive. Received: -1.0")

# =============================================================================
# SPECIAL MATHEMATICAL RELATIONSHIPS
# =============================================================================

## Tests special mathematical relationships - Exponential Weibull Equivalence
func test_exponential_weibull_equivalence() -> void:
	# Weibull(scale=λ, shape=1) = Exponential(rate=1/λ)
	var scale: float = 2.0
	var shape: float = 1.0
	var rate: float = 1.0 / scale
	var p: float = 0.5
	
	var weibull_ppf: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
	var exp_ppf: float = StatMath.PpfFunctions.exponential_ppf(p, rate)
	assert_float(weibull_ppf).is_equal_approx(exp_ppf, StatMath.NUMERICAL_TOLERANCE)

## Tests special values - Exponential Median
func test_exponential_median_special_value() -> void:
	# Exponential PPF(0.5) = ln(2) / lambda
	var lambda_param: float = 1.0
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.5, lambda_param)
	assert_float(result).is_equal_approx(0.6931472, StatMath.SPECIAL_VALUES_TOLERANCE)
