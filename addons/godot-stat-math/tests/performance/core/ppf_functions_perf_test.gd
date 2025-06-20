# res://addons/godot-stat-math/tests/performance/core/ppf_functions_perf_test.gd
class_name PpfFunctionsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.PpfFunctions Module
##
## Tests computationally intensive percent point functions (quantiles)
## which often use iterative methods like Newton-Raphson.

# Test parameters specific to this module
const PROBABILITY_VALUES: Array[float] = [0.1, 0.5, 0.9]

# Override module name for result tracking
func get_module_name() -> String:
	return "PpfFunctions"


func test_normal_ppf_performance() -> void:
	var test_name: String = "normal_ppf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for p_val in PROBABILITY_VALUES:
				StatMath.PpfFunctions.normal_ppf(p_val, 0.0, 1.0)
				StatMath.PpfFunctions.normal_ppf(p_val, 5.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_gamma_ppf_performance() -> void:
	var test_name: String = "gamma_ppf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for p_val in PROBABILITY_VALUES:
				StatMath.PpfFunctions.gamma_ppf(p_val, 2.0, 1.0)
				StatMath.PpfFunctions.gamma_ppf(p_val, 5.0, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_beta_ppf_performance() -> void:
	var test_name: String = "beta_ppf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for p_val in PROBABILITY_VALUES:
				StatMath.PpfFunctions.beta_ppf(p_val, 2.0, 3.0)
				StatMath.PpfFunctions.beta_ppf(p_val, 0.5, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_weibull_ppf_performance() -> void:
	var test_name: String = "weibull_ppf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for p_val in PROBABILITY_VALUES:
				StatMath.PpfFunctions.weibull_ppf(p_val, 2.0, 1.5)
				StatMath.PpfFunctions.weibull_ppf(p_val, 1.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)



 
