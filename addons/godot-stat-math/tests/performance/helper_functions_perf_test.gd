# res://addons/godot-stat-math/tests/performance/helper_functions_perf_test.gd
class_name HelperFunctionsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.HelperFunctions Module
##
## Tests computationally intensive mathematical helper functions
## to catch performance regressions during development.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test parameters
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test


func test_gamma_function_performance() -> void:
	var test_name: String = "gamma_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):  # Gamma function uses Lanczos approximation
			StatMath.HelperFunctions.gamma_function(2.5)
			StatMath.HelperFunctions.gamma_function(0.5)
			StatMath.HelperFunctions.gamma_function(10.5)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_beta_function_performance() -> void:
	var test_name: String = "beta_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):  # Beta uses gamma ratio
			StatMath.HelperFunctions.beta_function(2.0, 3.0)
			StatMath.HelperFunctions.beta_function(0.5, 0.5)
			StatMath.HelperFunctions.beta_function(5.0, 2.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_incomplete_beta_function_performance() -> void:
	var test_name: String = "incomplete_beta_function"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):  # Very expensive - uses series expansion
			StatMath.HelperFunctions.incomplete_beta(0.5, 2.0, 3.0)
			StatMath.HelperFunctions.incomplete_beta(0.2, 1.5, 2.5)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_binomial_coefficient_performance() -> void:
	var test_name: String = "binomial_coefficient"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):  # Combinatorial calculations
			StatMath.HelperFunctions.binomial_coefficient(20, 5)
			StatMath.HelperFunctions.binomial_coefficient(50, 10)
			StatMath.HelperFunctions.binomial_coefficient(100, 25)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_log_gamma_function_performance() -> void:
	var test_name: String = "log_gamma"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.HelperFunctions.log_gamma(2.5)
			StatMath.HelperFunctions.log_gamma(10.5)
			StatMath.HelperFunctions.log_gamma(100.5)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_log_factorial_performance() -> void:
	var test_name: String = "log_factorial"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.HelperFunctions.log_factorial(10)
			StatMath.HelperFunctions.log_factorial(20)
			StatMath.HelperFunctions.log_factorial(50)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_log_binomial_coefficient_performance() -> void:
	var test_name: String = "log_binomial_coef"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.HelperFunctions.log_binomial_coef(20, 5)
			StatMath.HelperFunctions.log_binomial_coef(50, 10)
			StatMath.HelperFunctions.log_binomial_coef(100, 25)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_sanitize_numeric_array_performance() -> void:
	var test_name: String = "sanitize_numeric_array"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			# Test with mixed arrays including invalid values
			var mixed_array1: Array = [1.0, 2.5, "invalid", 3.2, null, 4.1, 5.8]
			var mixed_array2: Array = [10, 20.5, "test", 30, false, 40.2]
			var mixed_array3: Array = [1.1, 2.2, 3.3, 4.4, 5.5]  # Clean array
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array1)
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array2)
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array3)
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
	var prefixed_test_name: String = "helperfunctions_%s" % test_name
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
	
	print("Measuring helper functions performance...")
	
	# Gamma function tests
	var measurement: Dictionary = _measure_test("gamma_function", func():
		for i in range(TEST_ITERATIONS):  # Gamma function uses Lanczos approximation
			StatMath.HelperFunctions.gamma_function(2.5)
			StatMath.HelperFunctions.gamma_function(0.5)
			StatMath.HelperFunctions.gamma_function(10.5)
	)
	results["gamma_function"] = measurement.execution_time_ms
	print("  gamma_function: %.2f ms" % measurement.execution_time_ms)
	
	# Beta function tests
	measurement = _measure_test("beta_function", func():
		for i in range(TEST_ITERATIONS):  # Beta uses gamma ratio
			StatMath.HelperFunctions.beta_function(2.0, 3.0)
			StatMath.HelperFunctions.beta_function(0.5, 0.5)
			StatMath.HelperFunctions.beta_function(5.0, 2.0)
	)
	results["beta_function"] = measurement.execution_time_ms
	print("  beta_function: %.2f ms" % measurement.execution_time_ms)
	
	# Incomplete beta function tests
	measurement = _measure_test("incomplete_beta_function", func():
		for i in range(TEST_ITERATIONS):  # Very expensive - uses series expansion
			StatMath.HelperFunctions.incomplete_beta(0.5, 2.0, 3.0)
			StatMath.HelperFunctions.incomplete_beta(0.2, 1.5, 2.5)
	)
	results["incomplete_beta_function"] = measurement.execution_time_ms
	print("  incomplete_beta_function: %.2f ms" % measurement.execution_time_ms)
	
	# Binomial coefficient tests
	measurement = _measure_test("binomial_coefficient", func():
		for i in range(TEST_ITERATIONS):  # Combinatorial calculations
			StatMath.HelperFunctions.binomial_coefficient(20, 5)
			StatMath.HelperFunctions.binomial_coefficient(50, 10)
			StatMath.HelperFunctions.binomial_coefficient(100, 25)
	)
	results["binomial_coefficient"] = measurement.execution_time_ms
	print("  binomial_coefficient: %.2f ms" % measurement.execution_time_ms)
	
	# Log gamma function tests
	measurement = _measure_test("log_gamma", func():
		for i in range(TEST_ITERATIONS):
			StatMath.HelperFunctions.log_gamma(2.5)
			StatMath.HelperFunctions.log_gamma(10.5)
			StatMath.HelperFunctions.log_gamma(100.5)
	)
	results["log_gamma"] = measurement.execution_time_ms
	print("  log_gamma: %.2f ms" % measurement.execution_time_ms)

	# NEW TESTS: Additional helper functions
	
	# Log factorial tests
	measurement = _measure_test("log_factorial", func():
		for i in range(TEST_ITERATIONS):
			StatMath.HelperFunctions.log_factorial(10)
			StatMath.HelperFunctions.log_factorial(20)
			StatMath.HelperFunctions.log_factorial(50)
	)
	results["log_factorial"] = measurement.execution_time_ms
	print("  log_factorial: %.2f ms" % measurement.execution_time_ms)
	
	# Log binomial coefficient tests
	measurement = _measure_test("log_binomial_coef", func():
		for i in range(TEST_ITERATIONS):
			StatMath.HelperFunctions.log_binomial_coef(20, 5)
			StatMath.HelperFunctions.log_binomial_coef(50, 10)
			StatMath.HelperFunctions.log_binomial_coef(100, 25)
	)
	results["log_binomial_coef"] = measurement.execution_time_ms
	print("  log_binomial_coef: %.2f ms" % measurement.execution_time_ms)
	
	# Sanitize numeric array tests
	measurement = _measure_test("sanitize_numeric_array", func():
		for i in range(TEST_ITERATIONS):
			# Test with mixed arrays including invalid values
			var mixed_array1: Array = [1.0, 2.5, "invalid", 3.2, null, 4.1, 5.8]
			var mixed_array2: Array = [10, 20.5, "test", 30, false, 40.2]
			var mixed_array3: Array = [1.1, 2.2, 3.3, 4.4, 5.5]  # Clean array
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array1)
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array2)
			StatMath.HelperFunctions.sanitize_numeric_array(mixed_array3)
	)
	results["sanitize_numeric_array"] = measurement.execution_time_ms
	print("  sanitize_numeric_array: %.2f ms" % measurement.execution_time_ms)
	
	return results 
