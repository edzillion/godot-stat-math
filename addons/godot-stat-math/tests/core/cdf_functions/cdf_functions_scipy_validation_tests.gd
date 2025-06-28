# res://addons/godot-stat-math/tests/core/cdf_functions/cdf_functions_scipy_validation_tests.gd
class_name CdfFunctionsScipyValidationTests extends GdUnitTestSuite


const CDF_TEST_DATA = preload("res://addons/godot-stat-math/tables/cdf_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

## Tests normal CDF function with comprehensive test data
func test_normal_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["normal_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)


## Tests exponential CDF function with comprehensive test data
func test_exponential_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["exponential_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.exponential_cdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)


## Tests gamma CDF function with comprehensive test data
func test_gamma_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["gamma_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.gamma_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)


## Tests beta CDF function with comprehensive test data
func test_beta_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["beta_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.beta_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)


## Tests Chi-Square CDF function with comprehensive test data
func test_chi_square_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["chi_square_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.chi_square_cdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)


## Tests t-distribution CDF function with comprehensive test data
func test_t_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["t_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.t_cdf(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.NUMERICAL_TOLERANCE)


## Tests F-distribution CDF function with comprehensive test data
func test_f_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["f_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.f_cdf(case["params"][0], case["params"][1], case["params"][2])
		# F-distribution requires higher tolerance due to incomplete beta function complexity
		assert_float(result).is_equal_approx(case["expected"], StatMath.NUMERICAL_INTEGRATION_TOLERANCE)


## Tests Weibull CDF function with comprehensive test data
func test_weibull_cdf_comprehensive() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["weibull_cdf"]
	for case in test_data:
		var result: float = StatMath.CdfFunctions.weibull_cdf(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE) 
