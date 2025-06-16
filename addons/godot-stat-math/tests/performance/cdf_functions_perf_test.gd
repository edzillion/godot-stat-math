# res://addons/godot-stat-math/tests/performance/cdf_functions_perf_test.gd
class_name CdfFunctionsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.CdfFunctions Module
##
## Tests cumulative distribution function calculations that use
## complex mathematical operations like error functions and incomplete functions.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/cdf_functions_baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test parameters
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test
const TEST_VALUES: Array[float] = [-2.0, 0.0, 1.0, 2.0]


func test_normal_cdf_performance() -> void:
	var test_name: String = "normal_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.CdfFunctions.normal_cdf(x_val, 0.0, 1.0)
				StatMath.CdfFunctions.normal_cdf(x_val, 5.0, 2.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	return json.data


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
	if time_change > REGRESSION_THRESHOLD:
		assert_float(time_change).is_less_equal(REGRESSION_THRESHOLD)


## Public method for baseline generation - returns performance measurements
func collect_performance_measurements() -> Dictionary:
	var results: Dictionary = {}
	
	print("Measuring CDF functions performance...")
	
	# Normal CDF tests
	var measurement: Dictionary = _measure_test("normal_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.CdfFunctions.normal_cdf(x_val, 0.0, 1.0)
				StatMath.CdfFunctions.normal_cdf(x_val, 5.0, 2.0)
	)
	results["normal_cdf"] = measurement.execution_time_ms
	print("  normal_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Gamma CDF tests
	measurement = _measure_test("gamma_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Gamma CDF only defined for x >= 0
					StatMath.CdfFunctions.gamma_cdf(x_val, 2.0, 1.0)
					StatMath.CdfFunctions.gamma_cdf(x_val, 5.0, 0.5)
	)
	results["gamma_cdf"] = measurement.execution_time_ms
	print("  gamma_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Beta CDF tests
	measurement = _measure_test("beta_cdf", func():
		var beta_test_values: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
		for i in range(TEST_ITERATIONS):
			for x_val in beta_test_values:  # Beta CDF only defined for x in [0,1]
				StatMath.CdfFunctions.beta_cdf(x_val, 2.0, 3.0)
				StatMath.CdfFunctions.beta_cdf(x_val, 0.5, 0.5)
	)
	results["beta_cdf"] = measurement.execution_time_ms
	print("  beta_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Weibull CDF tests
	measurement = _measure_test("weibull_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Weibull CDF only defined for x >= 0
					StatMath.CdfFunctions.weibull_cdf(x_val, 2.0, 1.5)
					StatMath.CdfFunctions.weibull_cdf(x_val, 1.0, 2.0)
	)
	results["weibull_cdf"] = measurement.execution_time_ms
	print("  weibull_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Exponential CDF tests
	measurement = _measure_test("exponential_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Exponential CDF only defined for x >= 0
					StatMath.CdfFunctions.exponential_cdf(x_val, 1.0)
					StatMath.CdfFunctions.exponential_cdf(x_val, 2.5)
	)
	results["exponential_cdf"] = measurement.execution_time_ms
	print("  exponential_cdf: %.2f ms" % measurement.execution_time_ms)
	
	return results 
