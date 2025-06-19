# res://addons/godot-stat-math/tests/performance/core/basic_stats_perf_test.gd
class_name BasicStatsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.BasicStats Module
##
## Tests statistical analysis functions on various dataset sizes
## to catch performance regressions during development.

# Override module name for result tracking
func get_module_name() -> String:
	return "BasicStats"


func test_mean_variance_performance() -> void:
	var test_name: String = "mean_variance_calculation"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			
			# Test core statistical calculations
			StatMath.BasicStats.mean(test_data)
			StatMath.BasicStats.variance(test_data)
			StatMath.BasicStats.standard_deviation(test_data)
	)
	print("results here: ", current_results)
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_median_performance() -> void:
	var test_name: String = "median_calculation"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			test_data.sort()  # Median expects sorted data
			
			# Median calculation (single operation, but sorting-dependent)
			StatMath.BasicStats.median(test_data)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_median_absolute_deviation_performance() -> void:
	var test_name: String = "median_absolute_deviation"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			test_data.sort()  # MAD expects sorted data
			
			# MAD is expensive - requires median calculation + deviation sorting
			StatMath.BasicStats.median_absolute_deviation(test_data)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_sample_statistics_performance() -> void:
	var test_name: String = "sample_statistics"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			
			# Sample variance and standard deviation (N-1 denominator)
			StatMath.BasicStats.sample_variance(test_data)
			StatMath.BasicStats.sample_standard_deviation(test_data)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_min_max_range_performance() -> void:
	var test_name: String = "min_max_range_calculation"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			
			# Test min, max, and range calculations
			StatMath.BasicStats.minimum(test_data)
			StatMath.BasicStats.maximum(test_data)
			StatMath.BasicStats.range_spread(test_data)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data) 
