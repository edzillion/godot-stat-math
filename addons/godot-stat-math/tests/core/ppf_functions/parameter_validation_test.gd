# res://addons/godot-stat-math/tests/core/ppf_functions/parameter_validation_test.gd
class_name PpfFunctionsParameterValidationTest extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

## Tests parameter validation for normal PPF
func test_normal_ppf_parameter_validation() -> void:
	# Invalid probability values
	var test_p_negative: Callable = func():
		StatMath.PpfFunctions.normal_ppf(-0.1, 0.0, 1.0)
	await assert_error(test_p_negative).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")
	
	var test_p_greater_than_one: Callable = func():
		StatMath.PpfFunctions.normal_ppf(1.5, 0.0, 1.0)
	await assert_error(test_p_greater_than_one).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.5")
	
	# Invalid sigma values
	var test_sigma_zero: Callable = func():
		StatMath.PpfFunctions.normal_ppf(0.5, 0.0, 0.0)
	await assert_error(test_sigma_zero).is_push_error("Standard deviation sigma must be positive. Received: 0.0")

## Tests parameter validation for exponential PPF
func test_exponential_ppf_parameter_validation() -> void:
	# Invalid lambda values
	var test_lambda_zero: Callable = func():
		StatMath.PpfFunctions.exponential_ppf(0.5, 0.0)
	await assert_error(test_lambda_zero).is_push_error("Rate lambda_param must be positive. Received: 0.0")
	
	var test_lambda_negative: Callable = func():
		StatMath.PpfFunctions.exponential_ppf(0.5, -1.0)
	await assert_error(test_lambda_negative).is_push_error("Rate lambda_param must be positive. Received: -1.0") 
