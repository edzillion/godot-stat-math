# res://addons/godot-stat-math/tests/core/helper_functions/scipy_validation_test.gd
class_name HelperFunctionsScipyValidationTest extends GdUnitTestSuite

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

# --- Log Factorial ---
func test_log_factorial_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["log_factorial"]
	var case: Dictionary = test_data[1]  # log_factorial(5) -> 4.78749174
	var result: float = StatMath.HelperFunctions.log_factorial(case["params"][0])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

func test_log_factorial_zero() -> void:
	var result: float = StatMath.HelperFunctions.log_factorial(0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

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

# --- Beta Function ---
func test_beta_function_basic() -> void:
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["beta_function"]
	var case: Dictionary = test_data[0]  # [2.0, 3.0] -> 0.08333333
	var result: float = StatMath.HelperFunctions.beta_function(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)

# --- Incomplete Beta ---
func test_incomplete_beta_x_zero() -> void:
	var result: float = StatMath.HelperFunctions.incomplete_beta(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_incomplete_beta_x_one() -> void:
	var result: float = StatMath.HelperFunctions.incomplete_beta(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

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

# --- Lower Incomplete Gamma Regularized ---
func test_lower_incomplete_gamma_regularized_z_zero() -> void:
	var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(2.0, 0.0)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_lower_incomplete_gamma_regularized_scipy_validation() -> void:
	# Using scipy-generated test data for lower incomplete gamma regularized function
	var test_data: Array = HELPER_FUNCTIONS_TEST_DATA.VALUES["lower_incomplete_gamma_regularized"]
	for case in test_data:
		var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.ERF_APPROX_TOLERANCE)

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

func test_lower_incomplete_gamma_regularized_zero_z_different_a() -> void:
	# Test z=0 for different values of a
	var a_values: Array[float] = [0.5, 1.0, 2.0, 5.0, 10.0]
	
	for a in a_values:
		var result: float = StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, 0.0)
		assert_float(result).is_equal_approx(0.0, StatMath.BOUNDARY_TOLERANCE) # Should always be 0 for z=0

# --- Integration Tests ---
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
