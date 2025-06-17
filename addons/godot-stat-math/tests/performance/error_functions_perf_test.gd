# res://addons/godot-stat-math/tests/performance/error_functions_perf_test.gd
class_name ErrorFunctionsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.ErrorFunctions Module
##
## Tests error function calculations that use approximation algorithms
## (Abramowitz-Stegun) to catch performance regressions during development.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/results/baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test parameters
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test
const ERROR_FUNCTION_VALUES: Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
const INVERSE_ERROR_FUNCTION_VALUES: Array[float] = [-0.8, -0.5, 0.0, 0.5, 0.8]
const INVERSE_COMP_ERROR_FUNCTION_VALUES: Array[float] = [0.2, 0.5, 1.0, 1.5, 1.8]


# GDUnit4 lifecycle methods for test run collection
func before() -> void:
	TestRunCollector.start_test_run()


func after() -> void:
	await TestRunCollector.finish_test_run()


func test_error_function_performance() -> void:
	var test_name: String = "error_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.error_function(x_val)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_complementary_error_function_performance() -> void:
	var test_name: String = "complementary_error_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.complementary_error_function(x_val)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_error_function_inverse_performance() -> void:
	var test_name: String = "error_function_inverse"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for y_val in INVERSE_ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.error_function_inverse(y_val)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_complementary_error_function_inverse_performance() -> void:
	var test_name: String = "complementary_error_function_inverse"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for y_val in INVERSE_COMP_ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.complementary_error_function_inverse(y_val)
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
	var prefixed_test_name: String = "errorfunctions_%s" % test_name
	if not baseline_data.has(prefixed_test_name):
		print("⚠️  No baseline found for %s - skipping regression check" % prefixed_test_name)
		return
	
	var baseline_time: float = baseline_data[prefixed_test_name].result_ms
	var current_time: float = current_results.execution_time_ms
	var time_change: float = (current_time - baseline_time) / baseline_time
	
	print("📊 %s: %.2f ms vs baseline %.2f ms (%.1f%% change)" % [
		test_name, current_time, baseline_time, time_change * 100.0
	])
	
	# Always collect the result (pass or fail)
	var is_failure: bool = time_change > REGRESSION_THRESHOLD
	TestRunCollector.add_test_result("ErrorFunctions", test_name, current_time, baseline_time, is_failure)
	
	# Check for performance regression
	if is_failure:
		assert_float(time_change).is_less_equal(REGRESSION_THRESHOLD)


## Public method for baseline generation - returns performance measurements
func collect_performance_measurements() -> Dictionary:
	var results: Dictionary = {}
	
	print("Measuring error functions performance...")
	
	# Error function tests
	var measurement: Dictionary = _measure_test("error_function", func():
		for i in range(TEST_ITERATIONS):
			for x_val in ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.error_function(x_val)
	)
	results["error_function"] = measurement.execution_time_ms
	print("  error_function: %.2f ms" % measurement.execution_time_ms)
	
	# Complementary error function tests
	measurement = _measure_test("complementary_error_function", func():
		for i in range(TEST_ITERATIONS):
			for x_val in ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.complementary_error_function(x_val)
	)
	results["complementary_error_function"] = measurement.execution_time_ms
	print("  complementary_error_function: %.2f ms" % measurement.execution_time_ms)

	# NEW TESTS: Error function inverses
	
	# Error function inverse tests
	measurement = _measure_test("error_function_inverse", func():
		for i in range(TEST_ITERATIONS):
			for y_val in INVERSE_ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.error_function_inverse(y_val)
	)
	results["error_function_inverse"] = measurement.execution_time_ms
	print("  error_function_inverse: %.2f ms" % measurement.execution_time_ms)
	
	# Complementary error function inverse tests
	measurement = _measure_test("complementary_error_function_inverse", func():
		for i in range(TEST_ITERATIONS):
			for y_val in INVERSE_COMP_ERROR_FUNCTION_VALUES:
				StatMath.ErrorFunctions.complementary_error_function_inverse(y_val)
	)
	results["complementary_error_function_inverse"] = measurement.execution_time_ms
	print("  complementary_error_function_inverse: %.2f ms" % measurement.execution_time_ms)
	
	return results 
