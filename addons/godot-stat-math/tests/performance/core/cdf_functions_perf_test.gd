# res://addons/godot-stat-math/tests/performance/core/cdf_functions_perf_test.gd
class_name CdfFunctionsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.CdfFunctions Module
##
## Tests cumulative distribution function calculations that use
## complex mathematical operations like error functions and incomplete functions.

# Test parameters specific to this module
const TEST_VALUES: Array[float] = [-2.0, 0.0, 1.0, 2.0]

# Override module name for result tracking
func get_module_name() -> String:
	return "CdfFunctions"


func test_normal_cdf_performance() -> void:
	var test_name: String = "normal_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.CdfFunctions.normal_cdf(x_val, 0.0, 1.0)
				StatMath.CdfFunctions.normal_cdf(x_val, 5.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_gamma_cdf_performance() -> void:
	var test_name: String = "gamma_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Gamma CDF only defined for x >= 0
					StatMath.CdfFunctions.gamma_cdf(x_val, 2.0, 1.0)
					StatMath.CdfFunctions.gamma_cdf(x_val, 5.0, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_beta_cdf_performance() -> void:
	var test_name: String = "beta_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		var beta_test_values: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
		for i in range(TEST_ITERATIONS):
			for x_val in beta_test_values:  # Beta CDF only defined for x in [0,1]
				StatMath.CdfFunctions.beta_cdf(x_val, 2.0, 3.0)
				StatMath.CdfFunctions.beta_cdf(x_val, 0.5, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_weibull_cdf_performance() -> void:
	var test_name: String = "weibull_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Weibull CDF only defined for x >= 0
					StatMath.CdfFunctions.weibull_cdf(x_val, 2.0, 1.5)
					StatMath.CdfFunctions.weibull_cdf(x_val, 1.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_exponential_cdf_performance() -> void:
	var test_name: String = "exponential_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Exponential CDF only defined for x >= 0
					StatMath.CdfFunctions.exponential_cdf(x_val, 1.0)
					StatMath.CdfFunctions.exponential_cdf(x_val, 2.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_chi_square_cdf_performance() -> void:
	var test_name: String = "chi_square_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Chi-square CDF only defined for x >= 0
					StatMath.CdfFunctions.chi_square_cdf(x_val, 2.0)
					StatMath.CdfFunctions.chi_square_cdf(x_val, 5.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_f_cdf_performance() -> void:
	var test_name: String = "f_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # F CDF only defined for x >= 0
					StatMath.CdfFunctions.f_cdf(x_val, 3.0, 5.0)
					StatMath.CdfFunctions.f_cdf(x_val, 2.0, 10.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_t_cdf_performance() -> void:
	var test_name: String = "t_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.CdfFunctions.t_cdf(x_val, 3.0)
				StatMath.CdfFunctions.t_cdf(x_val, 10.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_binomial_cdf_performance() -> void:
	var test_name: String = "binomial_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.binomial_cdf(5, 20, 0.3)
			StatMath.CdfFunctions.binomial_cdf(10, 30, 0.4)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_poisson_cdf_performance() -> void:
	var test_name: String = "poisson_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.poisson_cdf(3, 2.5)
			StatMath.CdfFunctions.poisson_cdf(8, 5.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_geometric_cdf_performance() -> void:
	var test_name: String = "geometric_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.geometric_cdf(5, 0.2)
			StatMath.CdfFunctions.geometric_cdf(10, 0.1)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_negative_binomial_cdf_performance() -> void:
	var test_name: String = "negative_binomial_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.negative_binomial_cdf(10, 3, 0.4)
			StatMath.CdfFunctions.negative_binomial_cdf(15, 5, 0.3)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_pareto_cdf_performance() -> void:
	var test_name: String = "pareto_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 1.0:  # Testing with scale=1.0, so x must be >= scale
					StatMath.CdfFunctions.pareto_cdf(x_val, 1.0, 2.0)
					StatMath.CdfFunctions.pareto_cdf(x_val, 1.0, 3.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)



 
