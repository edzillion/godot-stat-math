# res://addons/godot-stat-math/tests/performance/cdf_functions_perf_test.gd
class_name CdfFunctionsPerfTest extends GdUnitTestSuite

## Performance Test Suite for StatMath.CdfFunctions Module
##
## Tests cumulative distribution function calculations that use
## complex mathematical operations like error functions and incomplete functions.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
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
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_t_cdf_performance() -> void:
	var test_name: String = "t_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.CdfFunctions.t_cdf(x_val, 3.0)
				StatMath.CdfFunctions.t_cdf(x_val, 10.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_binomial_cdf_performance() -> void:
	var test_name: String = "binomial_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.binomial_cdf(5, 20, 0.3)
			StatMath.CdfFunctions.binomial_cdf(10, 30, 0.4)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_poisson_cdf_performance() -> void:
	var test_name: String = "poisson_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.poisson_cdf(3, 2.5)
			StatMath.CdfFunctions.poisson_cdf(8, 5.0)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_geometric_cdf_performance() -> void:
	var test_name: String = "geometric_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.geometric_cdf(5, 0.2)
			StatMath.CdfFunctions.geometric_cdf(10, 0.1)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


func test_negative_binomial_cdf_performance() -> void:
	var test_name: String = "negative_binomial_cdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.negative_binomial_cdf(10, 3, 0.4)
			StatMath.CdfFunctions.negative_binomial_cdf(15, 5, 0.3)
	)
	
	_check_performance_regression(test_name, current_results, baseline_data)


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
	var prefixed_test_name: String = "cdffunctions_%s" % test_name
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

	# NEW TESTS: Additional CDF functions
	
	# Chi-square CDF tests
	measurement = _measure_test("chi_square_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Chi-square CDF only defined for x >= 0
					StatMath.CdfFunctions.chi_square_cdf(x_val, 2.0)
					StatMath.CdfFunctions.chi_square_cdf(x_val, 5.0)
	)
	results["chi_square_cdf"] = measurement.execution_time_ms
	print("  chi_square_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# F CDF tests
	measurement = _measure_test("f_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # F CDF only defined for x >= 0
					StatMath.CdfFunctions.f_cdf(x_val, 3.0, 5.0)
					StatMath.CdfFunctions.f_cdf(x_val, 2.0, 10.0)
	)
	results["f_cdf"] = measurement.execution_time_ms
	print("  f_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Student's t CDF tests
	measurement = _measure_test("t_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.CdfFunctions.t_cdf(x_val, 3.0)
				StatMath.CdfFunctions.t_cdf(x_val, 10.0)
	)
	results["t_cdf"] = measurement.execution_time_ms
	print("  t_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Binomial CDF tests
	measurement = _measure_test("binomial_cdf", func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.binomial_cdf(5, 20, 0.3)
			StatMath.CdfFunctions.binomial_cdf(10, 30, 0.4)
	)
	results["binomial_cdf"] = measurement.execution_time_ms
	print("  binomial_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Poisson CDF tests
	measurement = _measure_test("poisson_cdf", func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.poisson_cdf(3, 2.5)
			StatMath.CdfFunctions.poisson_cdf(8, 5.0)
	)
	results["poisson_cdf"] = measurement.execution_time_ms
	print("  poisson_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Geometric CDF tests
	measurement = _measure_test("geometric_cdf", func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.geometric_cdf(5, 0.2)
			StatMath.CdfFunctions.geometric_cdf(10, 0.1)
	)
	results["geometric_cdf"] = measurement.execution_time_ms
	print("  geometric_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Negative binomial CDF tests
	measurement = _measure_test("negative_binomial_cdf", func():
		for i in range(TEST_ITERATIONS):
			StatMath.CdfFunctions.negative_binomial_cdf(10, 3, 0.4)
			StatMath.CdfFunctions.negative_binomial_cdf(15, 5, 0.3)
	)
	results["negative_binomial_cdf"] = measurement.execution_time_ms
	print("  negative_binomial_cdf: %.2f ms" % measurement.execution_time_ms)
	
	# Pareto CDF tests
	measurement = _measure_test("pareto_cdf", func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 1.0:  # Testing with scale=1.0, so x must be >= scale
					StatMath.CdfFunctions.pareto_cdf(x_val, 1.0, 2.0)
					StatMath.CdfFunctions.pareto_cdf(x_val, 1.0, 3.0)
	)
	results["pareto_cdf"] = measurement.execution_time_ms
	print("  pareto_cdf: %.2f ms" % measurement.execution_time_ms)
	
	return results 
