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
	await assert_error(test_call).is_push_error("Parameter n must be non-negative for binomial coefficient.")

func test_binomial_coefficient_invalid_r_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(5, -1)
	await assert_error(test_call).is_push_error("Parameter r must be non-negative for binomial coefficient.")

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
	await assert_error(test_call).is_push_error("Factorial (and its log) is undefined for negative numbers.")

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
	await assert_error(test_call).is_push_error("Parameter n must be non-negative for binomial coefficient.")

func test_log_binomial_coef_invalid_k_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(5, -1)
	await assert_error(test_call).is_push_error("Parameter k must be non-negative for binomial coefficient.")

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
	await assert_error(test_call).is_push_error("Log Gamma function is typically defined for z > 0.")

# --- Beta Function ---
func test_beta_function_basic() -> void:
	var result: float = StatMath.HelperFunctions.beta_function(2.0, 3.0)
	assert_float(result).is_equal_approx(1.0 / 12.0, 1e-7)

func test_beta_function_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.beta_function(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Parameters a and b must be positive for Beta function.")

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
	await assert_error(test_call).is_push_error("Shape parameters a and b must be positive.")

func test_incomplete_beta_invalid_x_out_of_range() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(-0.1, 2.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter x_val must be between 0.0 and 1.0.")

# --- Additional Comprehensive Incomplete Beta Tests ---

func test_incomplete_beta_special_case_beta_2_2() -> void:
	# Test the special case Beta(2,2) which has an exact formula
	var x: float = 0.5
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, 2.0, 2.0)
	var expected: float = x * x * (3.0 - 2.0 * x) # Expected exact formula
	assert_float(result).is_equal_approx(expected, 1e-6)


func test_incomplete_beta_special_case_beta_2_2_quarter() -> void:
	# Test Beta(2,2) at x = 0.25
	var x: float = 0.25
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, 2.0, 2.0)
	var expected: float = x * x * (3.0 - 2.0 * x) # 0.25^2 * (3 - 2*0.25) = 0.0625 * 2.5 = 0.15625
	assert_float(result).is_equal_approx(expected, 1e-6)


func test_incomplete_beta_special_case_beta_2_2_three_quarters() -> void:
	# Test Beta(2,2) at x = 0.75
	var x: float = 0.75
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, 2.0, 2.0)
	var expected: float = x * x * (3.0 - 2.0 * x) # 0.75^2 * (3 - 2*0.75) = 0.5625 * 1.5 = 0.84375
	assert_float(result).is_equal_approx(expected, 1e-6)


func test_incomplete_beta_numerical_integration_beta_3_2() -> void:
	# Test numerical integration for Beta(3,2) at x = 0.5
	var x: float = 0.5
	var a: float = 3.0
	var b: float = 2.0
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, a, b)
	
	# For Beta(3,2), the exact incomplete beta at x=0.5 can be calculated
	# I(0.5; 3, 2) = integral from 0 to 0.5 of t^2 * (1-t)^1 dt
	# = integral from 0 to 0.5 of t^2 - t^3 dt
	# = [t^3/3 - t^4/4] from 0 to 0.5
	# = (0.5^3/3 - 0.5^4/4) - 0 = 0.125/3 - 0.0625/4 = 0.041667 - 0.015625 = 0.026042
	var expected_raw: float = 0.026042
	var beta_func: float = StatMath.HelperFunctions.beta_function(a, b)
	var expected_normalized: float = expected_raw / beta_func
	
	assert_float(result).is_equal_approx(expected_normalized, 1e-3) # Numerical integration tolerance


func test_incomplete_beta_symmetry() -> void:
	# Test the beta distribution symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
	var x: float = 0.3
	var a: float = 2.5
	var b: float = 3.5
	
	var left_side: float = StatMath.HelperFunctions.incomplete_beta(x, a, b)
	var right_side: float = 1.0 - StatMath.HelperFunctions.incomplete_beta(1.0 - x, b, a)
	
	assert_float(left_side).is_equal_approx(right_side, 1e-4) # Should be symmetric


func test_incomplete_beta_monotonicity() -> void:
	# Test that incomplete beta is monotonically increasing in x
	var a: float = 2.0
	var b: float = 3.0
	
	var x1: float = 0.2
	var x2: float = 0.4
	var x3: float = 0.6
	var x4: float = 0.8
	
	var result1: float = StatMath.HelperFunctions.incomplete_beta(x1, a, b)
	var result2: float = StatMath.HelperFunctions.incomplete_beta(x2, a, b)
	var result3: float = StatMath.HelperFunctions.incomplete_beta(x3, a, b)
	var result4: float = StatMath.HelperFunctions.incomplete_beta(x4, a, b)
	
	# Should be monotonically increasing
	assert_float(result1).is_less(result2)
	assert_float(result2).is_less(result3)
	assert_float(result3).is_less(result4)


func test_incomplete_beta_invalid_x_greater_than_one() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(1.1, 2.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter x_val must be between 0.0 and 1.0.")


func test_incomplete_beta_invalid_b_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(0.5, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameters a and b must be positive.")

# --- Log Beta Function Direct ---
func test_log_beta_function_direct_basic() -> void:
	var result: float = StatMath.HelperFunctions.log_beta_function_direct(2.0, 3.0)
	assert_float(result).is_equal_approx(log(1.0 / 12.0), 1e-7)

func test_log_beta_function_direct_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_beta_function_direct(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Parameters a and b must be positive for Beta function.")

# --- Lower Incomplete Gamma Regularized (comprehensive tests) ---
func test_lower_incomplete_gamma_regularized_z_zero() -> void:
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, 0.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_lower_incomplete_gamma_regularized_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameter a must be positive for Incomplete Gamma function.")

func test_lower_incomplete_gamma_regularized_invalid_z_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, -1.0)
	await assert_error(test_call).is_push_error("Parameter z must be non-negative for Lower Incomplete Gamma.")

# --- Additional Comprehensive Lower Incomplete Gamma Tests ---

func test_lower_incomplete_gamma_regularized_a_equals_one() -> void:
	# For a=1, the lower incomplete gamma regularized is 1 - exp(-z)
	var a: float = 1.0
	var z: float = 2.0
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
	var expected: float = 1.0 - exp(-z)
	assert_float(result).is_equal_approx(expected, 1e-6)


func test_lower_incomplete_gamma_regularized_small_z() -> void:
	# Test with small z values where series expansion should be accurate
	var a: float = 2.5
	var z: float = 0.1
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
	
	# Should be a small positive value
	assert_float(result).is_greater(0.0)
	assert_float(result).is_less(0.1) # Should be small for small z


func test_lower_incomplete_gamma_regularized_large_z() -> void:
	# Test with large z values where result should approach 1
	var a: float = 2.0
	var z: float = 10.0
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
	
	# Should approach 1 for large z
	assert_float(result).is_greater(0.9)
	assert_float(result).is_less_equal(1.0)


func test_lower_incomplete_gamma_regularized_monotonicity() -> void:
	# Test that the function is monotonically increasing in z
	var a: float = 3.0
	
	var z1: float = 0.5
	var z2: float = 1.0
	var z3: float = 1.5
	var z4: float = 2.0
	
	var result1: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z1)
	var result2: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z2)
	var result3: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z3)
	var result4: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z4)
	
	# Should be monotonically increasing
	assert_float(result1).is_less(result2)
	assert_float(result2).is_less(result3)
	assert_float(result3).is_less(result4)


func test_lower_incomplete_gamma_regularized_bounds() -> void:
	# Test that the function stays within [0, 1] bounds
	var test_params: Array[Array] = [
		[0.5, 0.1], [1.0, 1.0], [2.0, 2.0], [5.0, 10.0], [10.0, 5.0]
	]
	
	for params in test_params:
		var a: float = params[0]
		var z: float = params[1]
		var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
		
		assert_float(result).is_greater_equal(0.0)
		assert_float(result).is_less_equal(1.0)


func test_lower_incomplete_gamma_regularized_convergence_warning() -> void:
	# Test edge case where convergence might be slow (very small a, moderate z)
	var a: float = 0.1
	var z: float = 5.0
	
	# This might trigger convergence warnings but should still return a valid result
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
	
	assert_float(result).is_greater_equal(0.0)
	assert_float(result).is_less_equal(1.0)
	assert_bool(is_nan(result)).is_false()


func test_lower_incomplete_gamma_regularized_zero_z_different_a() -> void:
	# Test z=0 for different values of a
	var a_values: Array[float] = [0.5, 1.0, 2.0, 5.0, 10.0]
	
	for a in a_values:
		var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, 0.0)
		assert_float(result).is_equal_approx(0.0, 1e-10) # Should always be 0 for z=0


# --- Integration Tests for Incomplete Functions ---

func test_incomplete_functions_beta_cdf_integration() -> void:
	# Test that incomplete beta integrates properly with beta CDF
	var x: float = 0.3
	var alpha: float = 2.0
	var beta: float = 3.0
	
	var incomplete_result: float = StatMath.HelperFunctions.incomplete_beta(x, alpha, beta)
	
	# The incomplete beta should be between 0 and 1 for valid CDF
	assert_float(incomplete_result).is_between(0.0, 1.0)
	
	# Should be reasonable for the given parameters
	assert_float(incomplete_result).is_greater(0.0)
	assert_float(incomplete_result).is_less(1.0)


func test_incomplete_functions_gamma_cdf_integration() -> void:
	# Test that lower incomplete gamma integrates properly with gamma CDF calculations
	var z: float = 1.5
	var a: float = 2.5
	
	var incomplete_result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
	
	# Should be a valid probability
	assert_float(incomplete_result).is_between(0.0, 1.0)
	
	# For reasonable parameters, should be neither 0 nor 1
	assert_float(incomplete_result).is_greater(0.05)
	assert_float(incomplete_result).is_less(0.95)

# --- Sanitize Numeric Array ---
func test_sanitize_numeric_array_mixed_types() -> void:
	var input_array: Array = [1.5, "invalid", 2, null, 3.7, false, 4]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input_array)
	var expected: Array[float] = [1.5, 2.0, 3.7, 4.0]
	assert_array(result).is_equal(expected)

func test_sanitize_numeric_array_all_numeric() -> void:
	var input_array: Array = [3.1, 1, 2.5, 4]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input_array)
	var expected: Array[float] = [1.0, 2.5, 3.1, 4.0]
	assert_array(result).is_equal(expected)

func test_sanitize_numeric_array_all_invalid() -> void:
	var input_array: Array = ["text", null, false, {}]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input_array)
	assert_array(result).is_empty()

func test_sanitize_numeric_array_empty() -> void:
	var input_array: Array = []
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input_array)
	assert_array(result).is_empty()

func test_sanitize_numeric_array_sorting() -> void:
	var input_array: Array = [5, 1, 3, 2, 4]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input_array)
	var expected: Array[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
	assert_array(result).is_equal(expected) 
