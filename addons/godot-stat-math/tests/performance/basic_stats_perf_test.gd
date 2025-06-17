# res://addons/godot-stat-math/tests/performance/basic_stats_perf_test.gd
class_name BasicStatsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.BasicStats Module
##
## Tests statistical analysis functions on various dataset sizes
## to catch performance regressions during development.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test parameters - dataset sizes that matter for game analytics
const DATASET_SIZES: Array[int] = [100, 1000, 5000]  # Keep these - they're meaningful for stats

# Test parameters  
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test
const SAMPLE_DATA: Array[float] = [1.0, 2.5, 3.2, 4.1, 5.8, 6.3, 7.9, 8.4, 9.1, 10.0]


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_summary_statistics_performance() -> void:
	var test_name: String = "summary_statistics"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			test_data.sort()  # Summary stats expects sorted data
			
			# Summary statistics - all calculations at once
			StatMath.BasicStats.summary_statistics(test_data)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


# --- Test Data Generation ---

func _generate_test_data(size: int) -> Array[float]:
	var data: Array[float] = []
	data.resize(size)
	
	# Generate reproducible test data using fixed seed
	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	rng.seed = 12345
	
	for i in range(size):
		data[i] = rng.randf_range(-100.0, 100.0)
	
	return data


# --- Performance Testing Infrastructure ---

func _measure_test(test_name: String, test_func: Callable) -> Dictionary:
	# Warmup runs
	for i in range(WARMUP_ITERATIONS):
		test_func.call()
	
	# Actual measurements
	var times: Array[float] = []
	for i in range(MEASUREMENT_ITERATIONS):
		var start_time: int = Time.get_ticks_usec()
		test_func.call()
		var end_time: int = Time.get_ticks_usec()
		
		var execution_time_ms: float = (end_time - start_time) / 1000.0
		times.append(execution_time_ms)
	
	# Return median time to reduce noise from outliers
	times.sort()
	var median_index: int = times.size() / 2
	return {"execution_time_ms": times[median_index]}


func _load_baseline() -> Dictionary:
	if not FileAccess.file_exists(BASELINE_FILE):
		push_warning("No baseline file found at: %s" % BASELINE_FILE)
		return {}
	
	var file: FileAccess = FileAccess.open(BASELINE_FILE, FileAccess.READ)
	if file == null:
		push_error("Failed to read baseline file: %s" % BASELINE_FILE)
		return {}
	
	var json_string: String = file.get_as_text()
	file.close()
	
	var json: JSON = JSON.new()
	var parse_result: Error = json.parse(json_string)
	if parse_result != OK:
		push_error("Failed to parse baseline JSON: %s" % BASELINE_FILE)
		return {}
	
	var data: Dictionary = json.data
	if not data.has("tests"):
		push_error("Baseline file missing 'tests' key: " + BASELINE_FILE)
		return {}
	
	return data["tests"]


func _check_performance_regression(test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> void:
	if baseline_data.is_empty():
		print("⚠️  No baseline data available for %s - skipping regression check" % test_name)
		return
	
	# Look for test with module prefix since all modules are in one baseline file
	var prefixed_test_name: String = "basicstats_%s" % test_name
	if not baseline_data.has(prefixed_test_name):
		print("⚠️  No baseline found for %s - skipping regression check" % prefixed_test_name)
		return
	
	var baseline_time: float = baseline_data[prefixed_test_name]
	var current_time: float = current_results.execution_time_ms
	var time_change: float = (current_time - baseline_time) / baseline_time
	
	print("📊 %s: %.2f ms vs baseline %.2f ms (%.1f%% change)" % [
		test_name, current_time, baseline_time, time_change * 100.0
	])
	
	# Check for performance regression
	if time_change > REGRESSION_THRESHOLD:
		assert_float(time_change).is_less_equal(REGRESSION_THRESHOLD)


## Public method for baseline generation - returns performance measurements
func collect_performance_measurements() -> Dictionary:
	var results: Dictionary = {}
	
	print("Measuring basic stats performance...")
	
	# Mean variance calculation tests
	var measurement: Dictionary = _measure_test("mean_variance_calculation", func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			
			# Test core statistical calculations
			StatMath.BasicStats.mean(test_data)
			StatMath.BasicStats.variance(test_data)
			StatMath.BasicStats.standard_deviation(test_data)
	)
	results["mean_variance_calculation"] = measurement.execution_time_ms
	print("  mean_variance_calculation: %.2f ms" % measurement.execution_time_ms)
	
	# Median calculation tests
	measurement = _measure_test("median_calculation", func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			test_data.sort()  # Median expects sorted data
			
			# Median calculation (single operation, but sorting-dependent)
			StatMath.BasicStats.median(test_data)
	)
	results["median_calculation"] = measurement.execution_time_ms
	print("  median_calculation: %.2f ms" % measurement.execution_time_ms)
	
	# Median absolute deviation tests
	measurement = _measure_test("median_absolute_deviation", func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			test_data.sort()  # MAD expects sorted data
			
			# MAD is expensive - requires median calculation + deviation sorting
			StatMath.BasicStats.median_absolute_deviation(test_data)
	)
	results["median_absolute_deviation"] = measurement.execution_time_ms
	print("  median_absolute_deviation: %.2f ms" % measurement.execution_time_ms)
	
	# Summary statistics tests
	measurement = _measure_test("summary_statistics", func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			test_data.sort()  # Summary stats expects sorted data
			
			# Summary statistics - all calculations at once
			StatMath.BasicStats.summary_statistics(test_data)
	)
	results["summary_statistics"] = measurement.execution_time_ms
	print("  summary_statistics: %.2f ms" % measurement.execution_time_ms)
	
	# Sample statistics tests
	measurement = _measure_test("sample_statistics", func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			
			# Sample variance and standard deviation (N-1 denominator)
			StatMath.BasicStats.sample_variance(test_data)
			StatMath.BasicStats.sample_standard_deviation(test_data)
	)
	results["sample_statistics"] = measurement.execution_time_ms
	print("  sample_statistics: %.2f ms" % measurement.execution_time_ms)

	# NEW TEST: Min, max, and range calculations
	measurement = _measure_test("min_max_range_calculation", func():
		for dataset_size in DATASET_SIZES:
			var test_data: Array[float] = _generate_test_data(dataset_size)
			
			# Test min, max, and range calculations
			StatMath.BasicStats.minimum(test_data)
			StatMath.BasicStats.maximum(test_data)
			StatMath.BasicStats.range_spread(test_data)
	)
	results["min_max_range_calculation"] = measurement.execution_time_ms
	print("  min_max_range_calculation: %.2f ms" % measurement.execution_time_ms)
	
	return results 
