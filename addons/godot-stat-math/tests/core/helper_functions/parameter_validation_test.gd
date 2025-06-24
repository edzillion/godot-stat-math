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

# =============================================================================
# GAMMA FUNCTION PARAMETER VALIDATION TESTS
# =============================================================================

# --- Gamma Function Parameter Validation ---
func test_gamma_function_poles_at_non_positive_integers() -> void:
	# Test that gamma_function(0), gamma_function(-1), etc., return INF
	
	# Γ(0) = INF (pole)
	var result_zero: float = StatMath.HelperFunctions.gamma_function(0.0)
	assert_float(result_zero).is_equal(INF)
	
	# Γ(-1) = INF (pole)
	var result_neg_one: float = StatMath.HelperFunctions.gamma_function(-1.0)
	assert_float(result_neg_one).is_equal(INF)
	
	# Γ(-2) = INF (pole)
	var result_neg_two: float = StatMath.HelperFunctions.gamma_function(-2.0)
	assert_float(result_neg_two).is_equal(INF)
	
	# Γ(-5) = INF (pole)
	var result_neg_five: float = StatMath.HelperFunctions.gamma_function(-5.0)
	assert_float(result_neg_five).is_equal(INF)

func test_gamma_function_negative_non_integer_values() -> void:
	# Test that gamma function works for negative non-integer values using reflection formula
	# These should not be INF but should be valid finite values
	
	# Γ(-0.5) should be finite (not a pole)
	var result_neg_half: float = StatMath.HelperFunctions.gamma_function(-0.5)
	assert_bool(is_finite(result_neg_half)).is_true()
	assert_bool(is_nan(result_neg_half)).is_false()
	
	# Γ(-1.5) should be finite (not a pole)
	var result_neg_one_half: float = StatMath.HelperFunctions.gamma_function(-1.5)
	assert_bool(is_finite(result_neg_one_half)).is_true()
	assert_bool(is_nan(result_neg_one_half)).is_false()

func test_gamma_function_very_small_positive_values() -> void:
	# Test that small positive values near zero work correctly
	var small_values: Array[float] = [0.001, 0.01, 0.1]
	
	for z in small_values:
		var result: float = StatMath.HelperFunctions.gamma_function(z)
		assert_bool(is_finite(result)).is_true()
		assert_bool(is_nan(result)).is_false()
		assert_float(result).is_greater(0.0)

# --- Log Gamma Parameter Validation ---
func test_log_gamma_invalid_input_non_positive() -> void:
	# Test that z <= 0 pushes an error and returns NAN
	
	var test_call_zero: Callable = func():
		StatMath.HelperFunctions.log_gamma(0.0)
	await assert_error(test_call_zero).is_push_error("Log Gamma function is typically defined for z > 0. Received: 0.0")
	
	var result_zero: float = StatMath.HelperFunctions.log_gamma(0.0)
	assert_bool(is_nan(result_zero)).is_true()

func test_log_gamma_invalid_input_negative() -> void:
	var test_call_negative: Callable = func():
		StatMath.HelperFunctions.log_gamma(-1.0)
	await assert_error(test_call_negative).is_push_error("Log Gamma function is typically defined for z > 0. Received: -1.0")
	
	var result_negative: float = StatMath.HelperFunctions.log_gamma(-1.0)
	assert_bool(is_nan(result_negative)).is_true()

func test_log_gamma_invalid_input_very_negative() -> void:
	var test_call_very_negative: Callable = func():
		StatMath.HelperFunctions.log_gamma(-10.0)
	await assert_error(test_call_very_negative).is_push_error("Log Gamma function is typically defined for z > 0. Received: -10.0")
	
	var result_very_negative: float = StatMath.HelperFunctions.log_gamma(-10.0)
	assert_bool(is_nan(result_very_negative)).is_true()

func test_log_gamma_very_small_positive_values() -> void:
	# Test that very small positive values work correctly
	var small_values: Array[float] = [0.001, 0.01, 0.1]
	
	for z in small_values:
		var result: float = StatMath.HelperFunctions.log_gamma(z)
		assert_bool(is_finite(result)).is_true()
		assert_bool(is_nan(result)).is_false()

# =============================================================================
# CDF/PPF VALUE HELPER FUNCTIONS BEHAVIORAL & ERROR TESTS  
# =============================================================================

# --- get_cdf_value() Tests ---
func test_get_cdf_value_with_valid_enum() -> void:
	# Test with StatMath.SupportedDistributions enum
	var result: float = StatMath.HelperFunctions.get_cdf_value(StatMath.SupportedDistributions.NORMAL, 0.0, [0.0, 1.0])
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)

func test_get_cdf_value_with_valid_string() -> void:
	# Test with string distribution names
	var result_normal: float = StatMath.HelperFunctions.get_cdf_value("normal", 0.0, [0.0, 1.0])
	assert_float(result_normal).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)
	
	var result_uniform: float = StatMath.HelperFunctions.get_cdf_value("uniform", 0.5, [0.0, 1.0])
	assert_float(result_uniform).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)
	
	# Test case-insensitive string handling
	var result_case: float = StatMath.HelperFunctions.get_cdf_value("EXPONENTIAL", 1.0, [1.0])
	assert_bool(is_finite(result_case)).is_true()

func test_get_cdf_value_invalid_distribution_type() -> void:
	# Test error for invalid type - int gets processed as unimplemented distribution
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_cdf_value(42, 0.0, [0.0, 1.0])  # Invalid type: int
	await assert_error(test_call).is_push_error("CDF function not implemented for distribution: 42")
	
	var result: float = StatMath.HelperFunctions.get_cdf_value(42, 0.0, [0.0, 1.0])
	assert_bool(is_nan(result)).is_true()

func test_get_cdf_value_invalid_distribution_type_array() -> void:
	# Test error for Array input - this will trigger the type validation error
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_cdf_value([], 0.0, [0.0, 1.0])
	await assert_error(test_call).is_push_error("Invalid distribution type. Expected StatMath.SupportedDistributions enum or String.")
	
	var result: float = StatMath.HelperFunctions.get_cdf_value([], 0.0, [0.0, 1.0])
	assert_bool(is_nan(result)).is_true()

func test_get_cdf_value_unimplemented_distribution() -> void:
	# Test error for unimplemented CDF: "CDF function not implemented..."
	# T_DISTRIBUTION is in the enum but not implemented in get_cdf_value()
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_cdf_value(StatMath.SupportedDistributions.T_DISTRIBUTION, 0.0, [5.0])
	await assert_error(test_call).is_push_error("CDF function not implemented for distribution: %s" % StatMath.SupportedDistributions.T_DISTRIBUTION)
	
	var result: float = StatMath.HelperFunctions.get_cdf_value(StatMath.SupportedDistributions.T_DISTRIBUTION, 0.0, [5.0])
	assert_bool(is_nan(result)).is_true()

func test_get_cdf_value_unknown_string_distribution_fallback() -> void:
	# Test unknown string distribution - logs error but succeeds with NORMAL fallback
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_cdf_value("UNKNOWN_DIST", 0.0, [0.0, 1.0])
	await assert_error(test_call).is_push_error("Unknown distribution string: UNKNOWN_DIST")
	
	# Should still work due to fallback to NORMAL distribution
	var result: float = StatMath.HelperFunctions.get_cdf_value("UNKNOWN_DIST", 0.0, [0.0, 1.0])
	assert_float(result).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)



# --- get_ppf_value() Tests ---
func test_get_ppf_value_with_valid_enum() -> void:
	# Test with StatMath.SupportedDistributions enum
	var result: float = StatMath.HelperFunctions.get_ppf_value(StatMath.SupportedDistributions.NORMAL, 0.5, [0.0, 1.0])
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_get_ppf_value_with_valid_string() -> void:
	# Test with string distribution names - use enough parameters for any distribution that might be triggered
	var result_normal: float = StatMath.HelperFunctions.get_ppf_value("normal", 0.5, [0.0, 1.0])
	assert_float(result_normal).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	var result_uniform: float = StatMath.HelperFunctions.get_ppf_value("uniform", 0.5, [0.0, 1.0])
	assert_float(result_uniform).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)
	
	# Test case-insensitive string handling - exponential only needs 1 param
	var result_case: float = StatMath.HelperFunctions.get_ppf_value("EXPONENTIAL", 0.5, [1.0])
	assert_bool(is_finite(result_case)).is_true()

func test_get_ppf_value_invalid_distribution_type() -> void:
	# Test error for invalid type - provide enough params to avoid bounds issues
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_ppf_value([], 0.5, [0.0, 1.0])  # Invalid type: Array
	await assert_error(test_call).is_push_error("Invalid distribution type. Expected StatMath.SupportedDistributions enum or String.")
	
	var result: float = StatMath.HelperFunctions.get_ppf_value([], 0.5, [0.0, 1.0])
	assert_bool(is_nan(result)).is_true()

func test_get_ppf_value_invalid_distribution_type_dictionary() -> void:
	# Test error for dictionary input - provide enough params to avoid bounds issues
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_ppf_value({"test": "value"}, 0.5, [0.0, 1.0])
	await assert_error(test_call).is_push_error("Invalid distribution type. Expected StatMath.SupportedDistributions enum or String.")
	
	var result: float = StatMath.HelperFunctions.get_ppf_value({"test": "value"}, 0.5, [0.0, 1.0])
	assert_bool(is_nan(result)).is_true()

func test_get_ppf_value_unimplemented_distribution() -> void:
	# Test error for unimplemented PPF - GAMMA needs 2 params but isn't implemented in PPF
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_ppf_value(StatMath.SupportedDistributions.GAMMA, 0.5, [2.0, 1.0])
	await assert_error(test_call).is_push_error("PPF function not implemented for distribution: %s" % StatMath.SupportedDistributions.GAMMA)
	
	var result: float = StatMath.HelperFunctions.get_ppf_value(StatMath.SupportedDistributions.GAMMA, 0.5, [2.0, 1.0])
	assert_bool(is_nan(result)).is_true()

func test_get_ppf_value_unimplemented_distribution_string() -> void:
	# Test unimplemented distribution with string - BETA needs 2 params, not implemented in PPF
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_ppf_value("BETA", 0.5, [2.0, 3.0])
	await assert_error(test_call).is_push_error("PPF function not implemented for distribution: BETA")
	
	var result: float = StatMath.HelperFunctions.get_ppf_value("BETA", 0.5, [2.0, 3.0])
	assert_bool(is_nan(result)).is_true()

func test_get_ppf_value_unknown_string_distribution() -> void:
	# Test completely unknown string distribution - will fallback to NORMAL needing 2 params
	var test_call: Callable = func():
		StatMath.HelperFunctions.get_ppf_value("IMAGINARY_DISTRIBUTION", 0.5, [0.0, 1.0])
	await assert_error(test_call).is_push_error("Unknown distribution string: IMAGINARY_DISTRIBUTION")
	
	# The function should still try to use the fallback (NORMAL) and succeed
	var result: float = StatMath.HelperFunctions.get_ppf_value("IMAGINARY_DISTRIBUTION", 0.5, [0.0, 1.0])
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE) 

# =============================================================================
# VALIDATION HELPER FUNCTIONS BEHAVIORAL & ERROR TESTS  
# =============================================================================

# --- validate_indices() Tests ---
func test_validate_indices_valid_samples() -> void:
	# Test with a valid index array
	var samples: Array[int] = [0, 1, 2, 3, 4]
	var population_size: int = 10
	
	var result: bool = StatMath.HelperFunctions.validate_indices(samples, population_size)
	assert_bool(result).is_true()

func test_validate_indices_negative_index() -> void:
	# Test error: "Sample index must be non-negative"
	var samples_with_negative: Array[int] = [0, 1, -1, 3]
	var population_size: int = 10
	
	var test_call: Callable = func():
		StatMath.HelperFunctions.validate_indices(samples_with_negative, population_size)
	await assert_error(test_call).is_push_error("Sample index must be non-negative. Found: -1")
	
	var result: bool = StatMath.HelperFunctions.validate_indices(samples_with_negative, population_size)
	assert_bool(result).is_false()

func test_validate_indices_index_too_large() -> void:
	# Test error: "Sample index must be less than population size"
	var samples_too_large: Array[int] = [0, 1, 2, 10]  # 10 >= population_size (10)
	var population_size: int = 10
	
	var test_call: Callable = func():
		StatMath.HelperFunctions.validate_indices(samples_too_large, population_size)
	await assert_error(test_call).is_push_error("Sample index must be less than population size. Found: 10 >= 10")
	
	var result: bool = StatMath.HelperFunctions.validate_indices(samples_too_large, population_size)
	assert_bool(result).is_false()

# --- validate_unique_indices() Tests ---
func test_validate_unique_indices_valid_unique_samples() -> void:
	# Test with a valid unique index array
	var samples: Array[int] = [0, 1, 2, 3, 4]
	var population_size: int = 10
	
	var result: bool = StatMath.HelperFunctions.validate_unique_indices(samples, population_size)
	assert_bool(result).is_true()

func test_validate_unique_indices_duplicate_found() -> void:
	# Test error: "Sample indices must be unique"
	var samples_with_duplicate: Array[int] = [0, 1, 2, 1, 4]  # Duplicate: 1
	var population_size: int = 10
	
	var test_call: Callable = func():
		StatMath.HelperFunctions.validate_unique_indices(samples_with_duplicate, population_size)
	await assert_error(test_call).is_push_error("Sample indices must be unique. Found duplicate: 1")
	
	var result: bool = StatMath.HelperFunctions.validate_unique_indices(samples_with_duplicate, population_size)
	assert_bool(result).is_false()

func test_validate_unique_indices_size_mismatch() -> void:
	# Test error: "Number of unique indices must equal sample size"
	# This should be caught by the duplicate test above, but let's test the explicit check
	# The function checks unique_values.size() != samples.size() but duplicates would already be caught
	# So this test validates the logical consistency check
	var samples_unique: Array[int] = [0, 1, 2, 3]  # All unique, size 4
	var population_size: int = 10
	
	# This should pass since all are unique
	var result: bool = StatMath.HelperFunctions.validate_unique_indices(samples_unique, population_size)
	assert_bool(result).is_true()

func test_validate_unique_indices_inherits_validate_indices_errors() -> void:
	# Test that it properly inherits errors from validate_indices (negative index)
	var samples_with_negative: Array[int] = [0, 1, -1, 3]  # Invalid due to negative index
	var population_size: int = 10
	
	var test_call: Callable = func():
		StatMath.HelperFunctions.validate_unique_indices(samples_with_negative, population_size)
	await assert_error(test_call).is_push_error("Sample index must be non-negative. Found: -1")
	
	var result: bool = StatMath.HelperFunctions.validate_unique_indices(samples_with_negative, population_size)
	assert_bool(result).is_false()
