# res://addons/godot-stat-math/tests/performance/core/error_functions_perf_test.gd
class_name ErrorFunctionsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.ErrorFunctions Module
##
## Tests computationally intensive error function calculations and related
## mathematical functions like incomplete gamma and beta functions.

# Test parameters specific to this module
const TEST_VALUES: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
const INVERSE_ERROR_FUNCTION_VALUES: Array[float] = [-0.9, -0.5, 0.0, 0.5, 0.9]
const INVERSE_COMP_ERROR_FUNCTION_VALUES: Array[float] = [0.1, 0.5, 1.0, 1.5, 1.9]

# Override module name for result tracking
func get_module_name() -> String:
	return "ErrorFunctions"


func test_error_function_performance() -> void:
	var test_name: String = "error_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.ErrorFunctions.erf(x_val)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_complementary_error_function_performance() -> void:
	var test_name: String = "complementary_error_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.ErrorFunctions.erfc(x_val)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_error_function_inverse_performance() -> void:
	var test_name: String = "error_function_inverse"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for y_val in INVERSE_ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.erf_inv(y_val)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_complementary_error_function_inverse_performance() -> void:
	var test_name: String = "complementary_error_function_inverse"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for y_val in INVERSE_COMP_ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.erfc_inv(y_val)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_log_gamma_performance() -> void:
	var test_name: String = "log_gamma"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			# Test various input ranges including edge cases
			StatMath.ErrorFunctions.log_gamma(0.5)
			StatMath.ErrorFunctions.log_gamma(1.0)
			StatMath.ErrorFunctions.log_gamma(2.5)
			StatMath.ErrorFunctions.log_gamma(10.0)
			StatMath.ErrorFunctions.log_gamma(100.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_gamma_function_performance() -> void:
	var test_name: String = "gamma_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			# Test various input ranges - Lanczos approximation calculations
			StatMath.ErrorFunctions.gamma(0.5)
			StatMath.ErrorFunctions.gamma(1.0)
			StatMath.ErrorFunctions.gamma(2.5)
			StatMath.ErrorFunctions.gamma(5.0)
			StatMath.ErrorFunctions.gamma(10.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)

 
