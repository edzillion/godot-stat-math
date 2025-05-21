# addons/godot-stat-math/tests/core/helper_functions_test.gd
class_name HelperFunctionsTest extends GdUnitTestSuite

# --- Binomial Coefficient ---
func test_binomial_coefficient_basic() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(5, 2)
	assert_float(result).is_equal_approx(10.0, 1e-7)

func test_binomial_coefficient_r_zero() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(5, 0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_binomial_coefficient_r_equals_n() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(5, 5)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_binomial_coefficient_r_greater_than_n() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(3, 5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_binomial_coefficient_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(-1, 2)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter n must be non-negative for binomial coefficient.")

func test_binomial_coefficient_invalid_r_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(5, -1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter r must be non-negative for binomial coefficient.")

# --- Log Factorial ---
func test_log_factorial_basic() -> void:
	var result: float = StatMath.HelperFunctions.log_factorial(5)
	assert_float(result).is_equal_approx(log(120.0), 1e-7)

func test_log_factorial_zero() -> void:
	var result: float = StatMath.HelperFunctions.log_factorial(0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_log_factorial_invalid_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_factorial(-1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Factorial (and its log) is undefined for negative numbers.")

# --- Log Binomial Coefficient ---
func test_log_binomial_coef_basic() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(5, 2)
	assert_float(result).is_equal_approx(log(10.0), 1e-7)

func test_log_binomial_coef_k_zero() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(5, 0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_log_binomial_coef_k_equals_n() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(5, 5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_log_binomial_coef_k_greater_than_n() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(3, 5)
	assert_float(result).is_equal_approx(-INF, 1e-7)

func test_log_binomial_coef_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(-1, 2)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter n must be non-negative for binomial coefficient.")

func test_log_binomial_coef_invalid_k_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(5, -1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter k must be non-negative for binomial coefficient.")

# --- Gamma Function ---
func test_gamma_function_basic() -> void:
	var result: float = StatMath.HelperFunctions.gamma_function(5.0)
	assert_float(result).is_equal_approx(24.0, 1e-5) # Gamma(5) = 4!

func test_gamma_function_half() -> void:
	var result: float = StatMath.HelperFunctions.gamma_function(0.5)
	assert_float(result).is_equal_approx(sqrt(PI), 1e-5)

func test_gamma_function_negative_integer() -> void:
	var result: float = StatMath.HelperFunctions.gamma_function(-1.0)
	assert_float(result).is_equal_approx(INF, 1e-5)

# --- Log Gamma ---
func test_log_gamma_basic() -> void:
	var result: float = StatMath.HelperFunctions.log_gamma(5.0)
	assert_float(result).is_equal_approx(log(24.0), 1e-5)

func test_log_gamma_invalid_z_zero() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_gamma(0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Log Gamma function is typically defined for z > 0.")

# --- Beta Function ---
func test_beta_function_basic() -> void:
	var result: float = StatMath.HelperFunctions.beta_function(2.0, 3.0)
	assert_float(result).is_equal_approx(1.0 / 12.0, 1e-7)

func test_beta_function_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.beta_function(-1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameters a and b must be positive for Beta function.")

# --- Incomplete Beta (placeholder) ---
func test_incomplete_beta_x_zero() -> void:
	var result: float = StatMath.HelperFunctions.incomplete_beta(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_incomplete_beta_x_one() -> void:
	var result: float = StatMath.HelperFunctions.incomplete_beta(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_incomplete_beta_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(0.5, -1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape parameters a and b must be positive.")

func test_incomplete_beta_invalid_x_out_of_range() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(-0.1, 2.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter x_val must be between 0.0 and 1.0.")

# --- Log Beta Function Direct ---
func test_log_beta_function_direct_basic() -> void:
	var result: float = StatMath.HelperFunctions.log_beta_function_direct(2.0, 3.0)
	assert_float(result).is_equal_approx(log(1.0 / 12.0), 1e-7)

func test_log_beta_function_direct_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_beta_function_direct(-1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameters a and b must be positive for Beta function.")

# --- Lower Incomplete Gamma Regularized (placeholder) ---
func test_lower_incomplete_gamma_regularized_z_zero() -> void:
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, 0.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_lower_incomplete_gamma_regularized_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(-1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape parameter a must be positive for Incomplete Gamma function.")

func test_lower_incomplete_gamma_regularized_invalid_z_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, -1.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter z must be non-negative for Lower Incomplete Gamma.") 