# res://addons/godot-stat-math/tests/core/helper_functions_test.gd
class_name HelperFunctionsTest extends GdUnitTestSuite

const HELPER_FUNCTIONS_TEST_DATA = preload("res://addons/godot-stat-math/tables/helper_functions_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

# --- Binomial Coefficient ---
func test_binomial_coefficient_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["binomial_coefficient"]
	var case: Dictionary = test_data[0]  # [5, 2] -> 10.0
	var result: float = StatMath.HelperFunctions.binomial_coefficient(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_binomial_coefficient_r_zero() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(5, 0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_binomial_coefficient_r_equals_n() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(5, 5)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_binomial_coefficient_r_greater_than_n() -> void:
	var result: float = StatMath.HelperFunctions.binomial_coefficient(3, 5)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- Binomial Coefficient Mathematical Properties ---
func test_binomial_coefficient_symmetry() -> void:
	# Test symmetry property: C(n,r) = C(n,n-r)
	var n: int = 10
	var r: int = 3
	var result1: float = StatMath.HelperFunctions.binomial_coefficient(n, r)
	var result2: float = StatMath.HelperFunctions.binomial_coefficient(n, n - r)
	assert_float(result1).is_equal_approx(result2, StatMath.FLOAT_TOLERANCE)

func test_binomial_coefficient_pascals_identity() -> void:
	# Test Pascal's identity: C(n,r) = C(n-1,r-1) + C(n-1,r)
	var n: int = 8
	var r: int = 3
	var left_side: float = StatMath.HelperFunctions.binomial_coefficient(n, r)
	var right_side: float = StatMath.HelperFunctions.binomial_coefficient(n-1, r-1) + StatMath.HelperFunctions.binomial_coefficient(n-1, r)
	assert_float(left_side).is_equal_approx(right_side, StatMath.FLOAT_TOLERANCE)

# --- Log Factorial Mathematical Properties ---
func test_log_factorial_growth_property() -> void:
	# Test that log(n!) is monotonically increasing
	var values: Array[int] = [1, 2, 3, 4, 5, 10]
	var prev_result: float = -1.0
	
	for n in values:
		var current_result: float = StatMath.HelperFunctions.log_factorial(n)
		assert_float(current_result).is_greater(prev_result)
		prev_result = current_result

# --- Beta Function Mathematical Properties ---
func test_beta_function_symmetry() -> void:
	# Test symmetry property: B(a,b) = B(b,a)
	var a: float = 2.5
	var b: float = 3.7
	var result1: float = StatMath.HelperFunctions.beta_function(a, b)
	var result2: float = StatMath.HelperFunctions.beta_function(b, a)
	assert_float(result1).is_equal_approx(result2, StatMath.FLOAT_TOLERANCE)

func test_beta_function_gamma_relationship() -> void:
	# Test relationship: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
	# For integer values where we can compute gamma exactly
	var a: float = 3.0  # Γ(3) = 2! = 2
	var b: float = 4.0  # Γ(4) = 3! = 6
	# Γ(7) = 6! = 720, so B(3,4) = 2*6/720 = 12/720 = 1/60
	var result: float = StatMath.HelperFunctions.beta_function(a, b)
	var expected: float = 1.0 / 60.0
	assert_float(result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

# --- Incomplete Beta Mathematical Properties ---
func test_incomplete_beta_boundary_conditions() -> void:
	# Test boundary conditions for incomplete beta
	var a: float = 2.0
	var b: float = 3.0
	
	# I(0; a, b) = 0
	var result_zero: float = StatMath.HelperFunctions.incomplete_beta(0.0, a, b)
	assert_float(result_zero).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# I(1; a, b) = 1
	var result_one: float = StatMath.HelperFunctions.incomplete_beta(1.0, a, b)
	assert_float(result_one).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_incomplete_beta_monotonicity() -> void:
	# Test that incomplete beta is monotonically increasing in x
	var a: float = 2.0
	var b: float = 3.0
	var x_values: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
	var prev_result: float = -1.0
	
	for x in x_values:
		var current_result: float = StatMath.HelperFunctions.incomplete_beta(x, a, b)
		assert_float(current_result).is_greater(prev_result)
		prev_result = current_result

# --- Lower Incomplete Gamma Mathematical Properties ---
func test_lower_incomplete_gamma_boundary_conditions() -> void:
	# Test boundary conditions
	var a: float = 2.0
	
	# P(a, 0) = 0
	var result_zero: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, 0.0)
	assert_float(result_zero).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# P(a, ∞) = 1 (test with large value)
	var result_large: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, 100.0)
	assert_float(result_large).is_greater(0.99)

func test_lower_incomplete_gamma_monotonicity() -> void:
	# Test monotonicity in z
	var a: float = 2.5
	var z_values: Array[float] = [0.5, 1.0, 2.0, 3.0, 5.0]
	var prev_result: float = -1.0
	
	for z in z_values:
		var current_result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
		assert_float(current_result).is_greater(prev_result)
		prev_result = current_result

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

func test_binomial_coefficient_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(-1, 2)
	await assert_error(test_call).is_push_error("Parameter n must be non-negative for binomial coefficient. Received: -1")

func test_binomial_coefficient_invalid_r_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.binomial_coefficient(5, -1)
	await assert_error(test_call).is_push_error("Parameter r must be non-negative for binomial coefficient. Received: -1")

# --- Log Factorial ---
func test_log_factorial_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["log_factorial"]
	var case: Dictionary = test_data[1]  # log_factorial(5) -> 4.78749174
	var result: float = StatMath.HelperFunctions.log_factorial(case["params"][0])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_log_factorial_zero() -> void:
	var result: float = StatMath.HelperFunctions.log_factorial(0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_log_factorial_invalid_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_factorial(-1)
	await assert_error(test_call).is_push_error("Factorial (and its log) is undefined for negative numbers. Received: -1")

# --- Log Binomial Coefficient ---
func test_log_binomial_coef_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["log_binomial_coef"]
	var case: Dictionary = test_data[0]  # log_binomial_coef(5, 2) -> 2.30258509
	var result: float = StatMath.HelperFunctions.log_binomial_coef(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_log_binomial_coef_k_zero() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(5, 0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_log_binomial_coef_k_equals_n() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(5, 5)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_log_binomial_coef_k_greater_than_n() -> void:
	var result: float = StatMath.HelperFunctions.log_binomial_coef(3, 5)
	assert_float(result).is_equal_approx(-INF, StatMath.FLOAT_TOLERANCE)

func test_log_binomial_coef_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(-1, 2)
	await assert_error(test_call).is_push_error("Parameter n must be non-negative for binomial coefficient. Received: -1")

func test_log_binomial_coef_invalid_k_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_binomial_coef(5, -1)
	await assert_error(test_call).is_push_error("Parameter k must be non-negative for binomial coefficient. Received: -1")

# --- Beta Function ---
func test_beta_function_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["beta_function"]
	var case: Dictionary = test_data[0]  # [2.0, 3.0] -> 0.08333333
	var result: float = StatMath.HelperFunctions.beta_function(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_beta_function_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.beta_function(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Parameters a and b must be positive for Beta function. Received a=-1.0, b=2.0")

# --- Incomplete Beta (placeholder) ---
func test_incomplete_beta_x_zero() -> void:
	var result: float = StatMath.HelperFunctions.incomplete_beta(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_incomplete_beta_x_one() -> void:
	var result: float = StatMath.HelperFunctions.incomplete_beta(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

func test_incomplete_beta_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters a and b must be positive. Received a=-1.0, b=2.0")

func test_incomplete_beta_invalid_x_out_of_range() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.incomplete_beta(-0.1, 2.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter x_val must be between 0.0 and 1.0. Received: -0.1")

# --- Additional Comprehensive Incomplete Beta Tests ---

func test_incomplete_beta_special_case_beta_2_2() -> void:
	# Test the special case Beta(2,2) which has an exact formula
	var x: float = 0.5
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, 2.0, 2.0)
	var expected: float = x * x * (3.0 - 2.0 * x) # Expected exact formula
	assert_float(result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)


func test_incomplete_beta_special_case_beta_2_2_quarter() -> void:
	# Test Beta(2,2) at x = 0.25
	var x: float = 0.25
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, 2.0, 2.0)
	var expected: float = x * x * (3.0 - 2.0 * x) # 0.25^2 * (3 - 2*0.25) = 0.0625 * 2.5 = 0.15625
	assert_float(result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)


func test_incomplete_beta_special_case_beta_2_2_three_quarters() -> void:
	# Test Beta(2,2) at x = 0.75
	var x: float = 0.75
	var result: float = StatMath.HelperFunctions.incomplete_beta(x, 2.0, 2.0)
	var expected: float = x * x * (3.0 - 2.0 * x) # 0.75^2 * (3 - 2*0.75) = 0.5625 * 1.5 = 0.84375
	assert_float(result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)


func test_incomplete_beta_scipy_validation() -> void:
	# Using scipy-generated test data for incomplete beta function
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["incomplete_beta"]
	for case in test_data:
		var result: float = StatMath.HelperFunctions.incomplete_beta(case["params"][0], case["params"][1], case["params"][2])
		assert_float(result).is_equal_approx(case["expected"], StatMath.NUMERICAL_INTEGRATION_TOLERANCE)

# --- Log Beta Function Direct ---
func test_log_beta_function_direct_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["beta_function"]
	var case: Dictionary = test_data[0]  # [2.0, 3.0] -> 0.08333333
	var result: float = StatMath.HelperFunctions.log_beta_function_direct(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(log(case["expected"]), StatMath.FLOAT_TOLERANCE)

func test_log_beta_function_direct_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.log_beta_function_direct(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Parameters a and b must be positive for Beta function. Received a=-1.0, b=2.0")

# --- Lower Incomplete Gamma Regularized (comprehensive tests) ---
func test_lower_incomplete_gamma_regularized_z_zero() -> void:
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, 0.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_lower_incomplete_gamma_regularized_invalid_a_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(-1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameter a must be positive for Incomplete Gamma function. Received: -1.0")

func test_lower_incomplete_gamma_regularized_invalid_z_negative() -> void:
	var test_call: Callable = func():
		StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, -1.0)
	await assert_error(test_call).is_push_error("Parameter z must be non-negative for Lower Incomplete Gamma. Received: -1.0")

# --- Lower Incomplete Gamma Regularized Scipy Validation ---
func test_lower_incomplete_gamma_regularized_scipy_validation() -> void:
	# Using scipy-generated test data for lower incomplete gamma regularized function
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["lower_incomplete_gamma_regularized"]
	for case in test_data:
		var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.ERF_APPROX_TOLERANCE)

# --- Additional Comprehensive Lower Incomplete Gamma Tests ---

func test_lower_incomplete_gamma_regularized_a_equals_one() -> void:
	# For a=1, the lower incomplete gamma regularized is 1 - exp(-z)
	var a: float = 1.0
	var z: float = 2.0
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)
	var expected: float = 1.0 - exp(-z)
	assert_float(result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)


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
		assert_float(result).is_equal_approx(0.0, StatMath.BOUNDARY_TOLERANCE) # Should always be 0 for z=0


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
	var input: Array = [1, 2.5, "3", 4, "hello", 5.0]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input)
	var expected: Array[float] = [1.0, 2.5, 3.0, 4.0, 5.0]
	assert_array(result).is_equal(expected)

func test_sanitize_numeric_array_with_invalid_values() -> void:
	var input: Array = [1, INF, 2, NAN, 3, "test"]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input)
	var expected: Array[float] = [1.0, 2.0, 3.0]
	assert_array(result).is_equal(expected)

func test_sanitize_numeric_array_is_sorted() -> void:
	var input: Array[float] = [5.5, 1.1, 4.4, 2.2, 3.3]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input)
	var expected: Array[float] = [1.1, 2.2, 3.3, 4.4, 5.5]
	assert_array(result).is_equal(expected)

func test_sanitize_numeric_array_with_negative_values() -> void:
	var input: Array[float] = [-10.0, 0.0, 1.0, 2.5, -3.0]
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(input)
	var expected: Array[float] = [-10.0, -3.0, 0.0, 1.0, 2.5]
	assert_array(result).is_equal(expected)

func test_sanitize_numeric_array_empty_input() -> void:
	var result: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array([])
	assert_array(result).is_empty() 
