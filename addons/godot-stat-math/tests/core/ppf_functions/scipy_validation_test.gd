# res://addons/godot-stat-math/tests/core/ppf_functions/scipy_validation_test.gd
class_name PpfFunctionsScipyValidationTest extends GdUnitTestSuite

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
