# res://addons/godot-stat-math/tests/core/helper_functions/parameter_validation_test.gd
class_name HelperFunctionsParameterValidationTest extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

# --- Binomial Coefficient Parameter Validation ---
func test_binomial_coefficient_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(-1, 2)
	await assert_error(test_call).is_push_error("Parameter n must be non-negative for binomial coefficient. Received: -1")

func test_binomial_coefficient_invalid_r_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(5, -1)
	await assert_error(test_call).is_push_error("Parameter r must be non-negative for binomial coefficient. Received: -1")

# --- Log Factorial Parameter Validation ---
func test_log_factorial_invalid_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_factorial(-1)
	await assert_error(test_call).is_push_error("Factorial (and its log) is undefined for negative numbers. Received: -1")

# --- Log Binomial Coefficient Parameter Validation ---
func test_log_binomial_coef_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(-1, 2)
	await assert_error(test_call).is_push_error("Parameter n must be non-negative for binomial coefficient. Received: -1")

func test_log_binomial_coef_invalid_k_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(5, -1)
	await assert_error(test_call).is_push_error("Parameter k must be non-negative for binomial coefficient. Received: -1")

# --- Beta Function Parameter Validation ---
func test_beta_function_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.beta_function(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Parameters a and b must be positive for Beta function. Received a=-1.0, b=2.0")

# --- Incomplete Beta Parameter Validation ---
func test_incomplete_beta_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters a and b must be positive. Received a=-1.0, b=2.0")

func test_incomplete_beta_invalid_x_out_of_range() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(-0.1, 2.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter x_val must be between 0.0 and 1.0. Received: -0.1")

# --- Log Beta Function Direct Parameter Validation ---
func test_log_beta_function_direct_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_beta_function_direct(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Parameters a and b must be positive for Beta function. Received a=-1.0, b=2.0")

# --- Lower Incomplete Gamma Regularized Parameter Validation ---
func test_lower_incomplete_gamma_regularized_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameter a must be positive for Incomplete Gamma function. Received: -1.0")

func test_lower_incomplete_gamma_regularized_invalid_z_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, -1.0)
	await assert_error(test_call).is_push_error("Parameter z must be non-negative for Lower Incomplete Gamma. Received: -1.0") 
