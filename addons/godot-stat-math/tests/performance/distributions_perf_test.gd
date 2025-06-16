# res://addons/godot-stat-math/tests/performance/distributions_perf_test.gd
class_name DistributionsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.Distributions Module
##
## Tests the most computationally intensive random variate generation functions
## to catch performance regressions during development.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5
const PERFORMANCE_THRESHOLD: float = 1.2  # 20% slower than baseline

# Test parameters
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test

# Baseline cache
var _baseline_cache: Dictionary = {}


func test_normal_distribution_performance() -> void:
	var baseline: Dictionary = _load_baseline()
	if baseline.is_empty():
		assert_that(false).is_true()
		# Baseline file not found. Run generate_all_baselines.gd first.
		return
	
	# Run performance measurements
	var current_results: Dictionary = collect_performance_measurements()
	
	# Check for regressions
	for test_name in current_results:
		var current_time: float = current_results[test_name]
		
		# Look for test with module prefix (since all modules are in one baseline file)
		var prefixed_test_name: String = "distributions_%s" % test_name
		if not baseline.has(prefixed_test_name):
			assert_that(false).is_true()
			# Baseline missing test: %s % prefixed_test_name
			continue
		
		var baseline_time: float = baseline[prefixed_test_name]
		var regression_ratio: float = current_time / baseline_time
		
		assert_float(regression_ratio).is_less_equal(PERFORMANCE_THRESHOLD)
		# Performance regression in %s: %.2fms vs baseline %.2fms (%.1fx slower) % [test_name, current_time, baseline_time, regression_ratio]


func test_gamma_distribution_performance() -> void:
	var test_name: String = "randf_gamma"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):  # Gamma is slower
			StatMath.Distributions.randf_gamma(2.5, 1.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_beta_distribution_performance() -> void:
	var test_name: String = "randf_beta"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):  # Beta is slower (uses gamma ratio)
			StatMath.Distributions.randf_beta(2.0, 3.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_binomial_distribution_performance() -> void:
	var test_name: String = "randi_binomial"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randi_binomial(0.3, 20)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_poisson_distribution_performance() -> void:
	var test_name: String = "randi_poisson"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randi_poisson(5.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_weibull_distribution_performance() -> void:
	var test_name: String = "randf_weibull"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_weibull(2.0, 1.5)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	if not _baseline_cache.is_empty():
		return _baseline_cache["tests"]
	
	var file: FileAccess = FileAccess.open(BASELINE_FILE, FileAccess.READ)
	if file == null:
		return {}
	
	var json_string: String = file.get_as_text()
	file.close()
	
	var json: JSON = JSON.new()
	var parse_result: Error = json.parse(json_string)
	if parse_result != OK:
		push_error("Failed to parse baseline JSON: " + BASELINE_FILE)
		return {}
	
	_baseline_cache = json.data
	
	if not _baseline_cache.has("tests"):
		push_error("Baseline file missing 'tests' key: " + BASELINE_FILE)
		return {}
	
	return _baseline_cache["tests"]


func _check_performance_regression(test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> void:
	if baseline_data.is_empty() or not baseline_data.has("tests"):
		print("⚠️  No baseline data available for %s - skipping regression check" % test_name)
		return
	
	if not baseline_data.tests.has(test_name):
		print("⚠️  No baseline found for %s - skipping regression check" % test_name)
		return
	
	var baseline_time: float
	var baseline_result = baseline_data.tests[test_name]
	
	# Handle both old (float) and new (dict) baseline formats
	if baseline_result is float:
		baseline_time = baseline_result
	else:
		baseline_time = baseline_result.execution_time_ms
	
	var current_time: float = current_results.execution_time_ms
	var time_change: float = (current_time - baseline_time) / baseline_time
	
	print("📊 %s: %.2f ms vs baseline %.2f ms (%.1f%% change)" % [
		test_name, current_time, baseline_time, time_change * 100.0
	])
	
	# Check for performance regression
	if time_change > (PERFORMANCE_THRESHOLD - 1.0):
		assert_float(time_change).is_less_equal(PERFORMANCE_THRESHOLD - 1.0)


## Public method for baseline generation - returns performance measurements
func collect_performance_measurements() -> Dictionary:
	var results: Dictionary = {}
	
	print("Measuring distributions performance...")
	
	# Normal distribution tests
	var measurement: Dictionary = _measure_test("randf_normal", func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_normal(0.0, 1.0)
	)
	results["randf_normal"] = measurement.execution_time_ms
	print("  randf_normal: %.2f ms" % measurement.execution_time_ms)
	
	# Gamma distribution tests  
	measurement = _measure_test("randf_gamma", func():
		for i in range(TEST_ITERATIONS):  # Gamma is slower
			StatMath.Distributions.randf_gamma(2.5, 1.0)
	)
	results["randf_gamma"] = measurement.execution_time_ms
	print("  randf_gamma: %.2f ms" % measurement.execution_time_ms)
	
	# Beta distribution tests
	measurement = _measure_test("randf_beta", func():
		for i in range(TEST_ITERATIONS):  # Beta is slow
			StatMath.Distributions.randf_beta(2.0, 3.0)
	)
	results["randf_beta"] = measurement.execution_time_ms
	print("  randf_beta: %.2f ms" % measurement.execution_time_ms)
	
	# Binomial distribution tests
	measurement = _measure_test("randi_binomial", func():
		for i in range(TEST_ITERATIONS):  # Binomial can be slower
			StatMath.Distributions.randi_binomial(20, 0.3)
	)
	results["randi_binomial"] = measurement.execution_time_ms
	print("  randi_binomial: %.2f ms" % measurement.execution_time_ms)
	
	# Poisson distribution tests
	measurement = _measure_test("randi_poisson", func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randi_poisson(5.0)
	)
	results["randi_poisson"] = measurement.execution_time_ms
	print("  randi_poisson: %.2f ms" % measurement.execution_time_ms)
	
	# Weibull distribution tests
	measurement = _measure_test("randf_weibull", func():
		for i in range(TEST_ITERATIONS):
			StatMath.Distributions.randf_weibull(2.0, 1.0)
	)
	results["randf_weibull"] = measurement.execution_time_ms
	print("  randf_weibull: %.2f ms" % measurement.execution_time_ms)
	
	return results 
