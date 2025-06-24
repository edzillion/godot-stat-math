# res://addons/godot-stat-math/tests/core/error_functions/parameter_validation_test.gd
class_name ErrorFunctionsParameterValidationTest extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

# --- Inverse Error Function Input Validation ---
func test_error_function_inverse_invalid_gt_one() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erf_inv(1.1)
	await assert_error(test_call).is_push_error("Input y for erfinv must be in the range [-1, 1]. Received: 1.1")

func test_error_function_inverse_invalid_lt_minus_one() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erf_inv(-1.1)
	await assert_error(test_call).is_push_error("Input y for erfinv must be in the range [-1, 1]. Received: -1.1")

# --- Inverse Complementary Error Function Input Validation ---
func test_complementary_error_function_inverse_invalid_gt_two() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erfc_inv(2.1)
	await assert_error(test_call).is_push_error("Input y for erfcinv must be in the range [0, 2]. Received: 2.1")

func test_complementary_error_function_inverse_invalid_lt_zero() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.erfc_inv(-0.1)
	await assert_error(test_call).is_push_error("Input y for erfcinv must be in the range [0, 2]. Received: -0.1")

# --- Gamma Function Input Validation ---
func test_gamma_invalid_input() -> void:
	assert_that(is_nan(StatMath.ErrorFunctions.gamma(0.0))).is_true()
	assert_that(is_nan(StatMath.ErrorFunctions.gamma(-1.0))).is_true()

func test_log_gamma_invalid_input() -> void:
	assert_that(is_nan(StatMath.ErrorFunctions.log_gamma(0.0))).is_true()
	assert_that(is_nan(StatMath.ErrorFunctions.log_gamma(-1.5))).is_true()

# --- Error Function Special Input Validation ---
func test_erf_infinity_inputs() -> void:
	# Test that erf(∞) = 1
	var result_inf: float = StatMath.ErrorFunctions.erf(INF)
	assert_float(result_inf).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)
	
	# Test that erf(-∞) = -1  
	var result_neg_inf: float = StatMath.ErrorFunctions.erf(-INF)
	assert_float(result_neg_inf).is_equal_approx(-1.0, StatMath.FLOAT_TOLERANCE)

func test_erf_nan_input() -> void:
	# Test that erf(NAN) = NAN
	var result: float = StatMath.ErrorFunctions.erf(NAN)
	assert_that(is_nan(result)).is_true()

# --- Complementary Error Function Special Input Validation ---
func test_erfc_infinity_inputs() -> void:
	# Test that erfc(∞) = 0
	var result_inf: float = StatMath.ErrorFunctions.erfc(INF)
	assert_float(result_inf).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# Test that erfc(-∞) = 2
	var result_neg_inf: float = StatMath.ErrorFunctions.erfc(-INF)
	assert_float(result_neg_inf).is_equal_approx(2.0, StatMath.FLOAT_TOLERANCE)

func test_erfc_nan_input() -> void:
	# Test that erfc(NAN) = NAN
	var result: float = StatMath.ErrorFunctions.erfc(NAN)
	assert_that(is_nan(result)).is_true() 