# res://addons/godot-stat-math/tests/core/error_functions_test.gd
class_name ErrorFunctionsTest extends GdUnitTestSuite

const ERROR_FUNCTIONS_TEST_DATA = preload("res://addons/godot-stat-math/tables/error_functions_test_data.gd")

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

func test_error_function_inverse_invalid_gt_one() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erf_inv(1.1)
	await assert_error(test_call).is_push_error("Input y for erfinv must be in the range [-1, 1]. Received: 1.1")

func test_error_function_inverse_invalid_lt_minus_one() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erf_inv(-1.1)
	await assert_error(test_call).is_push_error("Input y for erfinv must be in the range [-1, 1]. Received: -1.1")

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

func test_complementary_error_function_inverse_invalid_gt_two() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erfc_inv(2.1)
	await assert_error(test_call).is_push_error("Input y for erfcinv must be in the range [0, 2]. Received: 2.1")

func test_complementary_error_function_inverse_invalid_lt_zero() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erfc_inv(-0.1)
	await assert_error(test_call).is_push_error("Input y for erfcinv must be in the range [0, 2]. Received: -0.1")

# --- Gamma and Log Gamma Functions ---
func test_gamma_integer() -> void:
	# Gamma(n) = (n-1)!
	assert_float(StatMath.ErrorFunctions.gamma(4.0)).is_equal_approx(6.0, StatMath.FLOAT_TOLERANCE) # 3!
	assert_float(StatMath.ErrorFunctions.gamma(5.0)).is_equal_approx(24.0, StatMath.FLOAT_TOLERANCE) # 4!
	assert_float(StatMath.ErrorFunctions.gamma(1.0)).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_gamma_half_integer() -> void:
	# Gamma(0.5) = sqrt(PI)
	assert_float(StatMath.ErrorFunctions.gamma(0.5)).is_equal_approx(sqrt(PI), StatMath.FLOAT_TOLERANCE)
	# Gamma(1.5) = 0.5 * Gamma(0.5) = 0.5 * sqrt(PI)
	assert_float(StatMath.ErrorFunctions.gamma(1.5)).is_equal_approx(0.5 * sqrt(PI), StatMath.FLOAT_TOLERANCE)

func test_gamma_invalid_input() -> void:
	assert_that(is_nan(StatMath.ErrorFunctions.gamma(0.0))).is_true()
	assert_that(is_nan(StatMath.ErrorFunctions.gamma(-1.0))).is_true()

func test_log_gamma_consistency() -> void:
	# log_gamma(x) should be log(gamma(x))
	var x: float = 2.5
	var log_gamma_val: float = StatMath.ErrorFunctions.log_gamma(x)
	var gamma_val: float = StatMath.ErrorFunctions.gamma(x)
	assert_float(log_gamma_val).is_equal_approx(log(gamma_val), StatMath.FLOAT_TOLERANCE)

func test_log_gamma_invalid_input() -> void:
	assert_that(is_nan(StatMath.ErrorFunctions.log_gamma(0.0))).is_true()
	assert_that(is_nan(StatMath.ErrorFunctions.log_gamma(-1.5))).is_true() 
