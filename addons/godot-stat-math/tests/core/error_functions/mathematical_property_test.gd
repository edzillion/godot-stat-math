# res://addons/godot-stat-math/tests/core/error_functions/mathematical_property_test.gd
class_name ErrorFunctionsMathematicalPropertyTest extends GdUnitTestSuite

const ERROR_FUNCTIONS_TEST_DATA = preload("res://addons/godot-stat-math/tables/error_functions_test_data.gd")

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- Inverse Error Function (erfinv) ---
func test_error_function_inverse_round_trip() -> void:
	var x: float = 0.5
	var erf_x: float = StatMath.ErrorFunctions.erf(x)
	var result: float = StatMath.ErrorFunctions.erf_inv(erf_x)
	assert_float(result).is_equal_approx(x, StatMath.ERF_INV_TOLERANCE)

func test_error_function_inverse_zero() -> void:
	var result: float = StatMath.ErrorFunctions.erf_inv(0.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_error_function_inverse_one() -> void:
	var result: float = StatMath.ErrorFunctions.erf_inv(1.0)
	assert_float(result).is_equal_approx(INF, StatMath.FLOAT_TOLERANCE)

func test_error_function_inverse_minus_one() -> void:
	var result: float = StatMath.ErrorFunctions.erf_inv(-1.0)
	assert_float(result).is_equal_approx(-INF, StatMath.FLOAT_TOLERANCE)

# --- Inverse Complementary Error Function (erfcinv) ---
func test_complementary_error_function_inverse_round_trip() -> void:
	# Using scipy-validated test data for erfc_inv(0.5) -> 0.47693628
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["erfc_inv"]
	var case: Dictionary = test_data[0]  # [0.5] -> 0.47693628
	var result: float = StatMath.ErrorFunctions.erfc_inv(case["params"][0])
	# Note: Using larger tolerance due to iterative approximation limitations
	assert_float(result).is_equal_approx(case["expected"], StatMath.ERF_INV_TOLERANCE)

func test_complementary_error_function_inverse_one() -> void:
	var result: float = StatMath.ErrorFunctions.erfc_inv(1.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_complementary_error_function_inverse_zero() -> void:
	var result: float = StatMath.ErrorFunctions.erfc_inv(0.0)
	assert_float(result).is_equal_approx(INF, StatMath.FLOAT_TOLERANCE)

func test_complementary_error_function_inverse_two() -> void:
	var result: float = StatMath.ErrorFunctions.erfc_inv(2.0)
	assert_float(result).is_equal_approx(-INF, StatMath.FLOAT_TOLERANCE)

# --- Log Gamma Function Properties ---
func test_log_gamma_consistency() -> void:
	# log_gamma(x) should be log(gamma(x))
	var x: float = 2.5
	var log_gamma_val: float = StatMath.ErrorFunctions.log_gamma(x)
	var gamma_val: float = StatMath.ErrorFunctions.gamma(x)
	assert_float(log_gamma_val).is_equal_approx(log(gamma_val), StatMath.FLOAT_TOLERANCE) 
