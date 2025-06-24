# res://addons/godot-stat-math/tests/core/ppf_functions/mathematical_property_test.gd
class_name PpfFunctionsMathematicalPropertyTest extends GdUnitTestSuite

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- CDF-PPF Consistency Tests ---

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

# --- Monotonicity Tests ---

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

# --- Boundary Conditions Tests ---

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

# --- Special Mathematical Relationships ---

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

## Tests that CDF and PPF are inverse functions - Pareto Distribution
func test_pareto_cdf_ppf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"x": 1.5, "scale": 1.0, "shape": 2.0},
		{"x": 2.0, "scale": 1.0, "shape": 1.5},
		{"x": 3.0, "scale": 2.0, "shape": 1.0},
		{"x": 5.0, "scale": 1.5, "shape": 3.0}
	]
	
	for case in test_cases:
		# Forward: x -> CDF(x) -> PPF(CDF(x)) should equal x
		var cdf_value: float = StatMath.CdfFunctions.pareto_cdf(case["x"], case["scale"], case["shape"])
		var ppf_result: float = StatMath.PpfFunctions.pareto_ppf(cdf_value, case["scale"], case["shape"])
		assert_float(ppf_result).is_equal_approx(case["x"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests that PPF and CDF are inverse functions - Pareto Distribution  
func test_pareto_ppf_cdf_round_trip_consistency() -> void:
	var test_cases: Array[Dictionary] = [
		{"p": 0.1, "scale": 1.0, "shape": 2.0},
		{"p": 0.3, "scale": 1.0, "shape": 1.5},
		{"p": 0.7, "scale": 2.0, "shape": 1.0},
		{"p": 0.9, "scale": 1.5, "shape": 3.0}
	]
	
	for case in test_cases:
		# Reverse: p -> PPF(p) -> CDF(PPF(p)) should equal p
		var ppf_value: float = StatMath.PpfFunctions.pareto_ppf(case["p"], case["scale"], case["shape"])
		var cdf_result: float = StatMath.CdfFunctions.pareto_cdf(ppf_value, case["scale"], case["shape"])
		assert_float(cdf_result).is_equal_approx(case["p"], StatMath.CDF_PPF_CONSISTENCY_TOLERANCE)

## Tests PPF behavior at probability boundaries - Pareto Distribution
func test_pareto_ppf_boundary_conditions() -> void:
	var scale: float = 2.0
	var shape: float = 1.5
	
	# Lower boundary (p = 0)
	var p_zero_result: float = StatMath.PpfFunctions.pareto_ppf(0.0, scale, shape)
	assert_float(p_zero_result).is_equal_approx(scale, StatMath.FLOAT_TOLERANCE)
	
	# Upper boundary (p = 1)
	var p_one_result: float = StatMath.PpfFunctions.pareto_ppf(1.0, scale, shape)
	assert_bool(is_inf(p_one_result) and p_one_result > 0.0).is_true()
	
	# Test monotonicity - higher probabilities should give higher values
	var p_low: float = 0.3
	var p_high: float = 0.7
	var ppf_low: float = StatMath.PpfFunctions.pareto_ppf(p_low, scale, shape)
	var ppf_high: float = StatMath.PpfFunctions.pareto_ppf(p_high, scale, shape)
	assert_float(ppf_high).is_greater(ppf_low) 
