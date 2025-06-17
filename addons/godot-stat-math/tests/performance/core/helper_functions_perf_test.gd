# res://addons/godot-stat-math/tests/performance/core/helper_functions_perf_test.gd
class_name HelperFunctionsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.HelperFunctions Module
##
## Tests computationally intensive mathematical helper functions
## to catch performance regressions during development.

# Override module name for result tracking
func get_module_name() -> String:
	return "HelperFunctions"


func test_gamma_function_performance() -> void:
	var test_name: String = "gamma_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):  # Gamma function uses Lanczos approximation
			StatMath.HelperFunctions.gamma_function(2.5)
			StatMath.HelperFunctions.gamma_function(0.5)
			StatMath.HelperFunctions.gamma_function(10.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_beta_function_performance() -> void:
	var test_name: String = "beta_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):  # Beta uses gamma ratio
			StatMath.HelperFunctions.beta_function(2.0, 3.0)
			StatMath.HelperFunctions.beta_function(0.5, 0.5)
			StatMath.HelperFunctions.beta_function(5.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_incomplete_beta_function_performance() -> void:
	var test_name: String = "incomplete_beta_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):  # Very expensive - uses series expansion
			StatMath.HelperFunctions.incomplete_beta(0.5, 2.0, 3.0)
			StatMath.HelperFunctions.incomplete_beta(0.2, 1.5, 2.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_binomial_coefficient_performance() -> void:
	var test_name: String = "binomial_coefficient"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):  # Combinatorial calculations
			StatMath.HelperFunctions.binomial_coefficient(20, 5)
			StatMath.HelperFunctions.binomial_coefficient(50, 10)
			StatMath.HelperFunctions.binomial_coefficient(100, 25)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_log_gamma_function_performance() -> void:
	var test_name: String = "log_gamma"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):
			StatMath.HelperFunctions.log_gamma(2.5)
			StatMath.HelperFunctions.log_gamma(10.5)
			StatMath.HelperFunctions.log_gamma(100.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_log_factorial_performance() -> void:
	var test_name: String = "log_factorial"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):
			StatMath.HelperFunctions.log_factorial(10)
			StatMath.HelperFunctions.log_factorial(20)
			StatMath.HelperFunctions.log_factorial(50)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_log_binomial_coefficient_performance() -> void:
	var test_name: String = "log_binomial_coef"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):
			StatMath.HelperFunctions.log_binomial_coef(20, 5)
			StatMath.HelperFunctions.log_binomial_coef(50, 10)
			StatMath.HelperFunctions.log_binomial_coef(100, 25)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_sanitize_numeric_array_performance() -> void:
	var test_name: String = "sanitize_numeric_array"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(PerfTestManager.FUNCTION_CALLS_PER_MEASUREMENT):
			# Test with mixed arrays including invalid values
			var mixed_array1: Array = [1.0, 2.5, "invalid", 3.2, null, 4.1, 5.8]
			var mixed_array2: Array = [10, 20.5, "test", 30, false, 40.2]
			var mixed_array3: Array = [1.1, 2.2, 3.3, 4.4, 5.5]  # Clean array
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array1)
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array2)
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array3)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)




 
