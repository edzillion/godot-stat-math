# addons/godot-stat-math/tests/core/error_functions_test.gd
class_name ErrorFunctionsTest extends GdUnitTestSuite

# --- Error Function (erf) ---
func test_error_function_zero() -> void:
	var result: float = StatMath.ErrorFunctions.error_function(0.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_error_function_positive() -> void:
	var result: float = StatMath.ErrorFunctions.error_function(1.0)
	assert_float(result).is_equal_approx(0.84270079, 1e-6) # Reference value

func test_error_function_negative() -> void:
	var result: float = StatMath.ErrorFunctions.error_function(-1.0)
	assert_float(result).is_equal_approx(-0.84270079, 1e-6) # Odd function

func test_error_function_large_positive() -> void:
	var result: float = StatMath.ErrorFunctions.error_function(10.0)
	assert_float(result).is_equal_approx(1.0, 1e-7) # erf(10) ~ 1

func test_error_function_large_negative() -> void:
	var result: float = StatMath.ErrorFunctions.error_function(-10.0)
	assert_float(result).is_equal_approx(-1.0, 1e-7) # erf(-10) ~ -1

# --- Complementary Error Function (erfc) ---
func test_complementary_error_function_zero() -> void:
	var result: float = StatMath.ErrorFunctions.complementary_error_function(0.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_complementary_error_function_positive() -> void:
	var result: float = StatMath.ErrorFunctions.complementary_error_function(1.0)
	assert_float(result).is_equal_approx(0.15729921, 1e-6) # 1 - erf(1)

func test_complementary_error_function_negative() -> void:
	var result: float = StatMath.ErrorFunctions.complementary_error_function(-1.0)
	assert_float(result).is_equal_approx(1.84270079, 1e-6) # 1 - erf(-1)

# --- Inverse Error Function (erfinv) ---
func test_error_function_inverse_zero() -> void:
	var result: float = StatMath.ErrorFunctions.error_function_inverse(0.0)
	assert_float(result).is_equal_approx(0.0, 1e-6)

func test_error_function_inverse_one() -> void:
	var result: float = StatMath.ErrorFunctions.error_function_inverse(1.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_error_function_inverse_minus_one() -> void:
	var result: float = StatMath.ErrorFunctions.error_function_inverse(-1.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_error_function_inverse_invalid_gt_one() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.error_function_inverse(1.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Input y for erfinv must be in the range [-1, 1].")

func test_error_function_inverse_invalid_lt_minus_one() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.error_function_inverse(-1.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Input y for erfinv must be in the range [-1, 1].")

# --- Inverse Complementary Error Function (erfcinv) ---
func test_complementary_error_function_inverse_one() -> void:
	var result: float = StatMath.ErrorFunctions.complementary_error_function_inverse(1.0)
	assert_float(result).is_equal_approx(0.0, 1e-6)

func test_complementary_error_function_inverse_zero() -> void:
	var result: float = StatMath.ErrorFunctions.complementary_error_function_inverse(0.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_complementary_error_function_inverse_two() -> void:
	var result: float = StatMath.ErrorFunctions.complementary_error_function_inverse(2.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_complementary_error_function_inverse_invalid_gt_two() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.complementary_error_function_inverse(2.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Input y for erfcinv must be in the range [0, 2].")

func test_complementary_error_function_inverse_invalid_lt_zero() -> void:
	var test_call: Callable = func():
		StatMath.ErrorFunctions.complementary_error_function_inverse(-0.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Input y for erfcinv must be in the range [0, 2].") 
