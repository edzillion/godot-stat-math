# res://addons/godot-stat-math/tests/core/error_functions/scipy_validation_test.gd
class_name ErrorFunctionsScipyValidationTest extends GdUnitTestSuite

const ERROR_FUNCTIONS_TEST_DATA = preload("res://addons/godot-stat-math/tables/error_functions_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

# --- Error Function (erf) ---
func test_error_function_zero() -> void:
	var result: float = StatMath.ErrorFunctions.erf(0.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_error_function_positive() -> void:
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["erf"]
	var case: Dictionary = test_data[1]  # erf(1.0) -> 0.84270079
	var result: float = StatMath.ErrorFunctions.erf(case["params"][0])
	# Using larger tolerance for error function approximation precision
	assert_float(result).is_equal_approx(case["expected"], StatMath.ERF_APPROX_TOLERANCE)

func test_error_function_negative() -> void:
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["erf"]
	var case: Dictionary = test_data[1]  # erf(1.0) -> 0.84270079, so erf(-1.0) -> -0.84270079
	var result: float = StatMath.ErrorFunctions.erf(-case["params"][0])  # Test negative value
	# Using larger tolerance for error function approximation precision
	assert_float(result).is_equal_approx(-case["expected"], StatMath.ERF_APPROX_TOLERANCE) # Odd function

func test_error_function_large_positive() -> void:
	var result: float = StatMath.ErrorFunctions.erf(10.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE) # erf(10) ~ 1

func test_error_function_large_negative() -> void:
	var result: float = StatMath.ErrorFunctions.erf(-10.0)
	assert_float(result).is_equal_approx(-1.0, StatMath.FLOAT_TOLERANCE) # erf(-10) ~ -1

# --- Complementary Error Function (erfc) ---
func test_complementary_error_function_zero() -> void:
	var result: float = StatMath.ErrorFunctions.erfc(0.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_complementary_error_function_positive() -> void:
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["erfc"]
	var case: Dictionary = test_data[1]  # erfc(1.0) -> 0.15729921
	var result: float = StatMath.ErrorFunctions.erfc(case["params"][0])
	# Using larger tolerance for error function approximation precision
	assert_float(result).is_equal_approx(case["expected"], StatMath.ERF_APPROX_TOLERANCE)

func test_complementary_error_function_negative() -> void:
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["erfc"]
	var case: Dictionary = test_data[1]  # erfc(1.0) -> 0.15729921, so erfc(-1.0) -> 1 + erf(1.0)
	var result: float = StatMath.ErrorFunctions.erfc(-case["params"][0])  # Test negative value
	# Using larger tolerance for error function approximation precision
	assert_float(result).is_equal_approx(2.0 - case["expected"], StatMath.ERF_APPROX_TOLERANCE) # erfc(-x) = 2 - erfc(x)

# --- Gamma Functions ---
func test_gamma_integer() -> void:
	# Using scipy-generated test data for gamma function integer values
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["gamma_integer"]
	for case in test_data:
		var result: float = StatMath.ErrorFunctions.gamma(case["params"][0])
		assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_gamma_half_integer() -> void:
	# Using scipy-generated test data for gamma function half-integer values
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["gamma_half_integer"]
	for case in test_data:
		var result: float = StatMath.ErrorFunctions.gamma(case["params"][0])
		assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE) 