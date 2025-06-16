# res://addons/godot-stat-math/tests/performance/sampling_gen_perf_test.gd
class_name SamplingGenPerfTest extends GdUnitTestSuite

## Simple GDUnit4 Performance Test Suite for SamplingGen
## 
## Compares current performance against baseline file in repo.
## Use generate_baseline.gd tool script to create/update baselines.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test matrices - keep reasonable for test speed
const BATCH_SIZES: Array[int] = [256, 1024, 4096]
const DIMENSIONS: Array[int] = [1, 2, 3]
const METHODS: Array[StatMath.SamplingGen.SamplingMethod] = [
	StatMath.SamplingGen.SamplingMethod.RANDOM,
	StatMath.SamplingGen.SamplingMethod.SOBOL,
	StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM,
	StatMath.SamplingGen.SamplingMethod.HALTON
]

# Cached baseline data
var _baseline_cache: Dictionary = {}


func before():
	# Load baseline data once
	if _baseline_cache.is_empty():
		if FileAccess.file_exists(BASELINE_FILE):
			var file: FileAccess = FileAccess.open(BASELINE_FILE, FileAccess.READ)
			var json: JSON = JSON.new()
			var parse_result: Error = json.parse(file.get_as_text())
			file.close()
			
			if parse_result == OK:
				var data: Dictionary = json.data
				_baseline_cache = data.get("tests", {})
			else:
				push_warning("Failed to parse baseline file: %s" % json.error_string)
		else:
			push_warning("No baseline file found at: %s" % BASELINE_FILE)


## Test basic generate_samples performance across different configurations
func test_generate_samples():
	_run_performance_test("generate_samples", _test_generate_samples_impl)


func _test_generate_samples_impl():
	# Test all combinations of methods, dimensions, and batch sizes
	for method in METHODS:
		for dim in DIMENSIONS:
			for batch_size in BATCH_SIZES:
				var test_name: String = "generate_samples_%s_%dd_%d" % [
					StatMath.SamplingGen.SamplingMethod.keys()[method], dim, batch_size
				]
				
				var measurement: float = _measure_test(test_name, func():
					return StatMath.SamplingGen.generate_samples(method, batch_size, dim)
				)
				
				# Check against baseline if available
				_check_performance_regression(test_name, measurement)


func _run_performance_test(category: String, test_func: Callable) -> void:
	print("\nMeasuring %s..." % category)
	test_func.call()


func _measure_test(test_name: String, test_func: Callable) -> float:
	# Warmup iterations
	for i in WARMUP_ITERATIONS:
		test_func.call()
	
	# Measurement iterations
	var total_time: int = 0
	for i in MEASUREMENT_ITERATIONS:
		var start_time: int = Time.get_ticks_usec()
		var result = test_func.call()
		var end_time: int = Time.get_ticks_usec()
		total_time += (end_time - start_time)
	
	var avg_time_ms: float = (total_time / MEASUREMENT_ITERATIONS) / 1000.0
	print("  %s: %.2f ms" % [test_name, avg_time_ms])
	
	return avg_time_ms


func _check_performance_regression(test_name: String, current_time_ms: float) -> void:
	if _baseline_cache.has(test_name):
		var baseline_data = _baseline_cache[test_name]
		var baseline_time: float
		
		# Handle both old format (float) and new format (Dictionary)
		if baseline_data is Dictionary:
			baseline_time = baseline_data.execution_time_ms
		else:
			baseline_time = baseline_data
		
		var time_change: float = (current_time_ms - baseline_time) / baseline_time
		
		# Performance should not regress more than threshold
		if time_change > REGRESSION_THRESHOLD:
			assert_float(time_change).is_less_equal(REGRESSION_THRESHOLD)

func test_coordinated_shuffle_performance() -> void:
	print("\n=== Testing coordinated_shuffle Performance ===")
	
	for method in METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		print("\n--- Method: %s ---" % method_name)
		
		for deck_size in [52, 256, 512]:  # Card deck sizes
			var test_name: String = "coordinated_shuffle_%s_%d" % [method_name, deck_size]
			
			var results: Dictionary = _measure_performance(test_name, func():
				return StatMath.SamplingGen.coordinated_shuffle(deck_size, method, 0, -1)
			)
			
			_check_performance(test_name, results)

func test_coordinated_batch_shuffles_performance() -> void:
	print("\n=== Testing coordinated_batch_shuffles Performance ===")
	
	for method in METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		print("\n--- Method: %s ---" % method_name)
		
		for config in [{"deck": 52, "batch": 10}, {"deck": 256, "batch": 5}]:
			var test_name: String = "batch_shuffles_%s_%d_%d" % [method_name, config.deck, config.batch]
			
			var results: Dictionary = _measure_performance(test_name, func():
				return StatMath.SamplingGen.coordinated_batch_shuffles(config.deck, config.batch, method, 0, -1)
			)
			
			_check_performance(test_name, results)

func test_sample_indices_performance() -> void:
	print("\n=== Testing sample_indices Performance ===")
	
	var test_cases: Array = [
		{"strategy": StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT, "pop": 1000, "draw": 100},
		{"strategy": StatMath.SamplingGen.SelectionStrategy.FISHER_YATES, "pop": 1000, "draw": 100},
		{"strategy": StatMath.SamplingGen.SelectionStrategy.RESERVOIR, "pop": 10000, "draw": 1000}
	]
	
	# Test with RANDOM method (baseline)
	for test_case in test_cases:
		var strategy_name: String = StatMath.SamplingGen.SelectionStrategy.keys()[test_case.strategy]
		var test_name: String = "sample_indices_%s_RANDOM_%d_%d" % [strategy_name, test_case.pop, test_case.draw]
		
		var results: Dictionary = _measure_performance(test_name, func():
			return StatMath.SamplingGen.sample_indices(
				test_case.pop, 
				test_case.draw, 
				test_case.strategy, 
				StatMath.SamplingGen.SamplingMethod.RANDOM, 
				-1
			)
		)
		
		_check_performance(test_name, results)
	
	# Test FISHER_YATES with different sampling methods
	print("\n--- Testing sample_indices with different sampling methods ---")
	for method in [StatMath.SamplingGen.SamplingMethod.SOBOL, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE]:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		var test_name: String = "sample_indices_FISHER_YATES_%s_1000_100" % method_name
		
		var results: Dictionary = _measure_performance(test_name, func():
			return StatMath.SamplingGen.sample_indices(1000, 100, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES, method, -1)
		)
		
		_check_performance(test_name, results) 
