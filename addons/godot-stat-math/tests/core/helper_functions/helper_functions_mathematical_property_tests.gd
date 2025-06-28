# res://addons/godot-stat-math/tests/core/helper_functions/helper_functions_mathematical_property_tests.gd
class_name HelperFunctionsMathematicalPropertyTests extends GdUnitTestSuite

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

# --- Log Binomial Coefficient Mathematical Properties ---
func test_log_binomial_coef_symmetry() -> void:
	# Test symmetry property: log_binomial_coef(n, k) == log_binomial_coef(n, n-k)
	var n: int = 12
	var k: int = 4
	var result1: float = StatMath.HelperFunctions.log_binomial_coef(n, k)
	var result2: float = StatMath.HelperFunctions.log_binomial_coef(n, n - k)
	assert_float(result1).is_equal_approx(result2, StatMath.FLOAT_TOLERANCE)

func test_log_binomial_coef_relationship_to_binomial_coef() -> void:
	# Test that log_binomial_coef(n, k) == log(binomial_coefficient(n, k)) for moderate values
	var test_cases: Array[Array] = [[8, 3], [10, 4], [6, 2], [7, 0], [5, 5]]
	
	for case in test_cases:
		var n: int = case[0]
		var k: int = case[1]
		var log_result: float = StatMath.HelperFunctions.log_binomial_coef(n, k)
		var binomial_result: float = StatMath.HelperFunctions.binomial_coefficient(n, k)
		var expected: float = log(binomial_result)
		assert_float(log_result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

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

# --- Log Beta Function Mathematical Properties ---
func test_log_beta_function_direct_symmetry() -> void:
	# Test symmetry: log_beta(a, b) == log_beta(b, a)
	var test_params: Array[Array] = [
		[1.0, 2.0],
		[2.0, 3.0],
		[0.5, 1.5],
		[3.5, 2.7],
		[10.0, 5.0]
	]
	
	for params in test_params:
		var a: float = params[0]
		var b: float = params[1]
		
		var result_ab: float = StatMath.HelperFunctions.log_beta_function_direct(a, b)
		var result_ba: float = StatMath.HelperFunctions.log_beta_function_direct(b, a)
		
		assert_float(result_ab).is_equal_approx(result_ba, StatMath.FLOAT_TOLERANCE)

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

# --- Additional Lower Incomplete Gamma Mathematical Properties ---
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

# =============================================================================
# GAMMA FUNCTION MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- Gamma Function Mathematical Properties ---
func test_gamma_function_factorial_relationship() -> void:
	# Test Γ(n) = (n-1)! for positive integers
	
	# Γ(1) = 0! = 1
	var result_1: float = StatMath.HelperFunctions.gamma_function(1.0)
	assert_float(result_1).is_equal_approx(1.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Γ(2) = 1! = 1
	var result_2: float = StatMath.HelperFunctions.gamma_function(2.0)
	assert_float(result_2).is_equal_approx(1.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Γ(3) = 2! = 2
	var result_3: float = StatMath.HelperFunctions.gamma_function(3.0)
	assert_float(result_3).is_equal_approx(2.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Γ(4) = 3! = 6
	var result_4: float = StatMath.HelperFunctions.gamma_function(4.0)
	assert_float(result_4).is_equal_approx(6.0, StatMath.BOUNDARY_TOLERANCE)
	
	# Γ(5) = 4! = 24
	var result_5: float = StatMath.HelperFunctions.gamma_function(5.0)
	assert_float(result_5).is_equal_approx(24.0, StatMath.BOUNDARY_TOLERANCE)

func test_gamma_function_reflection_formula() -> void:
	# Test Γ(z)Γ(1-z) = π/sin(πz) for non-integer z
	var test_values: Array[float] = [0.25, 0.3, 0.7, 0.75]
	
	for z in test_values:
		var gamma_z: float = StatMath.HelperFunctions.gamma_function(z)
		var gamma_1_minus_z: float = StatMath.HelperFunctions.gamma_function(1.0 - z)
		var left_side: float = gamma_z * gamma_1_minus_z
		var right_side: float = PI / sin(PI * z)
		
		assert_float(left_side).is_equal_approx(right_side, StatMath.FLOAT_TOLERANCE)

func test_gamma_function_recursion_formula() -> void:
	# Test Γ(z+1) = z * Γ(z) for z > 0
	var test_values: Array[float] = [0.5, 1.0, 1.5, 2.5, 3.7]
	
	for z in test_values:
		var gamma_z_plus_1: float = StatMath.HelperFunctions.gamma_function(z + 1.0)
		var gamma_z: float = StatMath.HelperFunctions.gamma_function(z)
		var expected: float = z * gamma_z
		
		assert_float(gamma_z_plus_1).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

func test_gamma_function_half_integer_sqrt_pi_relationship() -> void:
	# Test specific half-integer relationships with √π
	
	# Γ(0.5) = √π
	var gamma_half: float = StatMath.HelperFunctions.gamma_function(0.5)
	assert_float(gamma_half).is_equal_approx(sqrt(PI), StatMath.FLOAT_TOLERANCE)
	
	# Γ(1.5) = 0.5 * √π
	var gamma_one_half: float = StatMath.HelperFunctions.gamma_function(1.5)
	assert_float(gamma_one_half).is_equal_approx(0.5 * sqrt(PI), StatMath.FLOAT_TOLERANCE)
	
	# Γ(2.5) = 1.5 * 0.5 * √π = 0.75 * √π
	var gamma_two_half: float = StatMath.HelperFunctions.gamma_function(2.5)
	assert_float(gamma_two_half).is_equal_approx(0.75 * sqrt(PI), StatMath.FLOAT_TOLERANCE)

# --- Log Gamma Mathematical Properties ---
func test_log_gamma_recursion_property() -> void:
	# Test log_gamma(z+1) = log_gamma(z) + log(z) for z > 0
	var test_values: Array[float] = [0.5, 1.0, 1.5, 2.5, 3.7, 10.0]
	
	for z in test_values:
		var log_gamma_z_plus_1: float = StatMath.HelperFunctions.log_gamma(z + 1.0)
		var log_gamma_z: float = StatMath.HelperFunctions.log_gamma(z)
		var expected: float = log_gamma_z + log(z)
		
		assert_float(log_gamma_z_plus_1).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

func test_log_gamma_gamma_relationship() -> void:
	# Test log_gamma(z) = log(gamma_function(z)) for moderate values
	var test_values: Array[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
	
	for z in test_values:
		var log_gamma_result: float = StatMath.HelperFunctions.log_gamma(z)
		var gamma_result: float = StatMath.HelperFunctions.gamma_function(z)
		var expected: float = log(gamma_result)
		
		assert_float(log_gamma_result).is_equal_approx(expected, StatMath.FLOAT_TOLERANCE)

func test_log_gamma_monotonicity() -> void:
	# Test that log_gamma is monotonically increasing for z > 1.461632...
	# (the minimum point of the gamma function)
	var z_values: Array[float] = [1.5, 2.0, 3.0, 5.0, 10.0, 50.0]
	var prev_result: float = StatMath.HelperFunctions.log_gamma(z_values[0])
	
	for i in range(1, z_values.size()):
		var current_result: float = StatMath.HelperFunctions.log_gamma(z_values[i])
		assert_float(current_result).is_greater(prev_result)
		prev_result = current_result 

# =============================================================================
# CONVERT TO FLOAT ARRAY PROPERTY TESTS
# =============================================================================

# --- Convert to Float Array Properties ---
func test_convert_to_float_array_preserves_order() -> void:
	# Test that element order is maintained during conversion
	var input_ordered: Array = [1, 2.5, 3, 4.7, 5]
	var result: Array[float] = StatMath.HelperFunctions.convert_to_float_array(input_ordered)
	
	# Check that order is preserved
	assert_float(result[0]).is_equal(1.0)
	assert_float(result[1]).is_equal(2.5) 
	assert_float(result[2]).is_equal(3.0)
	assert_float(result[3]).is_equal(4.7)
	assert_float(result[4]).is_equal(5.0)
	
	# Test with reversed order
	var input_reversed: Array = [5, 4.7, 3, 2.5, 1]
	var result_reversed: Array[float] = StatMath.HelperFunctions.convert_to_float_array(input_reversed)
	
	# Check that reversed order is preserved
	assert_float(result_reversed[0]).is_equal(5.0)
	assert_float(result_reversed[1]).is_equal(4.7) 
	assert_float(result_reversed[2]).is_equal(3.0)
	assert_float(result_reversed[3]).is_equal(2.5)
	assert_float(result_reversed[4]).is_equal(1.0)

func test_convert_to_float_array_valid_conversion() -> void:
	# Test Array to Array[float] conversion with various types
	var input_mixed: Array = [1, 2.5, 3, 4.0]  # Mix of int and float
	var result: Array[float] = StatMath.HelperFunctions.convert_to_float_array(input_mixed)
	var expected: Array[float] = [1.0, 2.5, 3.0, 4.0]
	
	# Verify the conversion worked correctly
	assert_array(result).is_equal(expected)
	
	# Verify types are correct - check that result is properly typed
	assert_bool(result is Array[float]).is_true()
	
	# Test with edge cases - zero and negative values
	var input_edge: Array = [0, -1.5, -2, 0.0]
	var result_edge: Array[float] = StatMath.HelperFunctions.convert_to_float_array(input_edge)
	var expected_edge: Array[float] = [0.0, -1.5, -2.0, 0.0]
	
	assert_array(result_edge).is_equal(expected_edge)
