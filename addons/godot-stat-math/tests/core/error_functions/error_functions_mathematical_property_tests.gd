# res://addons/godot-stat-math/tests/core/error_functions/error_functions_mathematical_property_tests.gd
class_name ErrorFunctionsMathematicalPropertyTests extends GdUnitTestSuite

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

# --- Gamma Function Properties ---
func test_gamma_reflection_formula() -> void:
	# Test reflection formula: Γ(z)Γ(1-z) = π/sin(πz) for non-integer z
	var test_values: Array[float] = [0.3, 0.7, 0.25, 0.75, 0.1, 0.9]
	
	for z in test_values:
		var gamma_z: float = StatMath.ErrorFunctions.gamma(z)
		var gamma_1_minus_z: float = StatMath.ErrorFunctions.gamma(1.0 - z)
		var product: float = gamma_z * gamma_1_minus_z
		var expected: float = PI / sin(PI * z)
		assert_float(product).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

# --- Error Function Properties ---
func test_erf_odd_function_property() -> void:
	# Test that erf(-x) == -erf(x) for various values
	var test_values: Array[float] = [0.5, 1.0, 1.5, 2.0, 3.0]
	
	for x in test_values:
		var erf_x: float = StatMath.ErrorFunctions.erf(x)
		var erf_neg_x: float = StatMath.ErrorFunctions.erf(-x)
		assert_float(erf_neg_x).is_equal_approx(-erf_x, StatMath.FLOAT_TOLERANCE)

func test_erfc_erf_complementary_property() -> void:
	# Test that erfc(x) + erf(x) == 1 for various values
	var test_values: Array[float] = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5]
	
	for x in test_values:
		var erf_x: float = StatMath.ErrorFunctions.erf(x)
		var erfc_x: float = StatMath.ErrorFunctions.erfc(x)
		var sum: float = erf_x + erfc_x
		assert_float(sum).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE) 
