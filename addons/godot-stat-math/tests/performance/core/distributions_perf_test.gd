# res://addons/godot-stat-math/tests/performance/core/distributions_perf_test.gd
class_name DistributionsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.Distributions Module
##
## Tests the most computationally intensive random variate generation functions
## to catch performance regressions during development.

# Override module name for result tracking
func get_module_name() -> String:
	return "Distributions"


func test_normal_distribution_performance() -> void:
	var test_name: String = "randf_normal"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(100):
			StatMath.Distributions.randf_normal(0.0, 1.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_gamma_distribution_performance() -> void:
	var test_name: String = "randf_gamma"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(100):  # Gamma is slower
			StatMath.Distributions.randf_gamma(2.5, 1.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_beta_distribution_performance() -> void:
	var test_name: String = "randf_beta"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(100):  # Beta is slower (uses gamma ratio)
			StatMath.Distributions.randf_beta(2.0, 3.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_binomial_distribution_performance() -> void:
	var test_name: String = "randi_binomial"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randi_binomial(0.3, 20)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_poisson_distribution_performance() -> void:
	var test_name: String = "randi_poisson"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randi_poisson(5.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_weibull_distribution_performance() -> void:
	var test_name: String = "randf_weibull"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_weibull(2.0, 1.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_uniform_int_distribution_performance() -> void:
	var test_name: String = "randi_uniform"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randi_uniform(1, 100)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_uniform_float_distribution_performance() -> void:
	var test_name: String = "randf_uniform"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_uniform(0.0, 10.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_exponential_distribution_performance() -> void:
	var test_name: String = "randf_exponential"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_exponential(1.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_pareto_distribution_performance() -> void:
	var test_name: String = "randf_pareto"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_pareto(1.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_cauchy_distribution_performance() -> void:
	var test_name: String = "randf_cauchy"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_cauchy(0.0, 1.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_triangular_distribution_performance() -> void:
	var test_name: String = "randf_triangular"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_triangular(0.0, 10.0, 3.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


 
