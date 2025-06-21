# res://addons/godot-stat-math/tests/core/helper_functions/mathematical_property_test.gd
class_name HelperFunctionsMathematicalPropertyTest extends GdUnitTestSuite

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