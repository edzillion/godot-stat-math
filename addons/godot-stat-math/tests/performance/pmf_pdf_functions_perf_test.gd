# res://addons/godot-stat-math/tests/performance/pmf_pdf_functions_perf_test.gd
class_name PmfPdfFunctionsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.PmfPdfFunctions Module
##
## Tests probability mass function (PMF) calculations to catch performance 
## regressions during development. PDF functions will be added when implemented.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test parameters
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test


func test_binomial_pmf_performance() -> void:
	var test_name: String = "binomial_pmf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			# Test various parameter combinations
			StatMath.PmfPdfFunctions.binomial_pmf(5, 20, 0.3)
			StatMath.PmfPdfFunctions.binomial_pmf(15, 50, 0.25)
			StatMath.PmfPdfFunctions.binomial_pmf(8, 30, 0.4)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_poisson_pmf_performance() -> void:
	var test_name: String = "poisson_pmf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			# Test various lambda values
			StatMath.PmfPdfFunctions.poisson_pmf(3, 5.0)
			StatMath.PmfPdfFunctions.poisson_pmf(10, 12.5)
			StatMath.PmfPdfFunctions.poisson_pmf(0, 2.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_negative_binomial_pmf_performance() -> void:
	var test_name: String = "negative_binomial_pmf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.PmfPdfFunctions.negative_binomial_pmf(10, 3, 0.4)
			StatMath.PmfPdfFunctions.negative_binomial_pmf(15, 5, 0.3)
			StatMath.PmfPdfFunctions.negative_binomial_pmf(8, 2, 0.6)
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
	var prefixed_test_name: String = "pmfpdffunctions_%s" % test_name
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
	
	print("Measuring PMF/PDF functions performance...")
	
	# Binomial PMF tests
	var measurement: Dictionary = _measure_test("binomial_pmf", func():
		for i in range(TEST_ITERATIONS):
			# Test various parameter combinations
			StatMath.PmfPdfFunctions.binomial_pmf(5, 20, 0.3)
			StatMath.PmfPdfFunctions.binomial_pmf(15, 50, 0.25)
			StatMath.PmfPdfFunctions.binomial_pmf(8, 30, 0.4)
	)
	results["binomial_pmf"] = measurement.execution_time_ms
	print("  binomial_pmf: %.2f ms" % measurement.execution_time_ms)
	
	# Poisson PMF tests
	measurement = _measure_test("poisson_pmf", func():
		for i in range(TEST_ITERATIONS):
			# Test various lambda values
			StatMath.PmfPdfFunctions.poisson_pmf(3, 5.0)
			StatMath.PmfPdfFunctions.poisson_pmf(10, 12.5)
			StatMath.PmfPdfFunctions.poisson_pmf(0, 2.0)
	)
	results["poisson_pmf"] = measurement.execution_time_ms
	print("  poisson_pmf: %.2f ms" % measurement.execution_time_ms)

	# NEW TEST: Negative binomial PMF tests
	measurement = _measure_test("negative_binomial_pmf", func():
		for i in range(TEST_ITERATIONS):
			StatMath.PmfPdfFunctions.negative_binomial_pmf(10, 3, 0.4)
			StatMath.PmfPdfFunctions.negative_binomial_pmf(15, 5, 0.3)
			StatMath.PmfPdfFunctions.negative_binomial_pmf(8, 2, 0.6)
	)
	results["negative_binomial_pmf"] = measurement.execution_time_ms
	print("  negative_binomial_pmf: %.2f ms" % measurement.execution_time_ms)
	
	return results 
