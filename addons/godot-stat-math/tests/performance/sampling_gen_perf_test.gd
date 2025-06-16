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

const SAMPLING_METHODS: Array = [
	StatMath.SamplingGen.SamplingMethod.RANDOM,
	StatMath.SamplingGen.SamplingMethod.SOBOL,
	StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM,
	StatMath.SamplingGen.SamplingMethod.HALTON,
	StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE
]

var baseline_data: Dictionary = {}

func before_all() -> void:
	print("=== SamplingGen Performance Tests ===")
	_load_baseline()

func _load_baseline() -> void:
	if not FileAccess.file_exists(BASELINE_FILE):
		print("⚠️  No baseline file found at: %s" % BASELINE_FILE)
		print("   Run generate_baseline.gd tool script to create baseline")
		return
	
	var file: FileAccess = FileAccess.open(BASELINE_FILE, FileAccess.READ)
	if not file:
		print("❌ Failed to load baseline file")
		return
	
	var json_text: String = file.get_as_text()
	file.close()
	
	var json: JSON = JSON.new()
	if json.parse(json_text) == OK:
		baseline_data = json.data
		print("✅ Loaded baseline with %d tests" % baseline_data.get("tests", {}).size())
	else:
		print("❌ Failed to parse baseline JSON")

func _measure_performance(test_name: String, test_func: Callable) -> float:
	# Warmup
	for i in range(WARMUP_ITERATIONS):
		var result = test_func.call()
		if result is Array:
			result.clear()
	
	# Measure
	var times: Array[float] = []
	for i in range(MEASUREMENT_ITERATIONS):
		var start_time: int = Time.get_ticks_usec()
		var result = test_func.call()
		var end_time: int = Time.get_ticks_usec()
		
		times.append((end_time - start_time) / 1000.0)  # Convert to ms
		
		if result is Array:
			result.clear()
	
	# Return median time
	times.sort()
	return times[times.size() / 2]

func _check_performance(test_name: String, current_time: float) -> void:
	if not baseline_data.has("tests") or not baseline_data.tests.has(test_name):
		print("    📊 No baseline for %s (%.2f ms)" % [test_name, current_time])
		return
	
	var baseline_time: float = baseline_data.tests[test_name]
	var change_ratio: float = (current_time - baseline_time) / baseline_time
	var change_percent: float = change_ratio * 100.0
	
	print("    📈 %s: %.2f ms (baseline: %.2f ms, %+.1f%%)" % [
		test_name, current_time, baseline_time, change_percent
	])
	
	if change_ratio > REGRESSION_THRESHOLD:
		assert_fail(
			"Performance regression in '%s': %.2f ms vs baseline %.2f ms (%.1f%% slower)" % [
				test_name, current_time, baseline_time, change_percent * 100.0
			]
		)

# === Test Cases ===

func test_generate_samples_performance() -> void:
	print("\n=== Testing generate_samples Performance ===")
	
	for method in SAMPLING_METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		print("\n--- Method: %s ---" % method_name)
		
		for dimensions in DIMENSIONS:
			for batch_size in BATCH_SIZES:
				var test_name: String = "generate_samples_%s_%dd_%d" % [method_name, dimensions, batch_size]
				
				var time: float = _measure_performance(test_name, func():
					return StatMath.SamplingGen.generate_samples(batch_size, dimensions, method, 0, -1)
				)
				
				_check_performance(test_name, time)

func test_coordinated_shuffle_performance() -> void:
	print("\n=== Testing coordinated_shuffle Performance ===")
	
	for method in SAMPLING_METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		print("\n--- Method: %s ---" % method_name)
		
		for deck_size in [52, 256, 512]:  # Card deck sizes
			var test_name: String = "coordinated_shuffle_%s_%d" % [method_name, deck_size]
			
			var time: float = _measure_performance(test_name, func():
				return StatMath.SamplingGen.coordinated_shuffle(deck_size, method, 0, -1)
			)
			
			_check_performance(test_name, time)

func test_coordinated_batch_shuffles_performance() -> void:
	print("\n=== Testing coordinated_batch_shuffles Performance ===")
	
	for method in SAMPLING_METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		print("\n--- Method: %s ---" % method_name)
		
		for config in [{"deck": 52, "batch": 10}, {"deck": 256, "batch": 5}]:
			var test_name: String = "batch_shuffles_%s_%d_%d" % [method_name, config.deck, config.batch]
			
			var time: float = _measure_performance(test_name, func():
				return StatMath.SamplingGen.coordinated_batch_shuffles(config.deck, config.batch, method, 0, -1)
			)
			
			_check_performance(test_name, time)

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
		
		var time: float = _measure_performance(test_name, func():
			return StatMath.SamplingGen.sample_indices(
				test_case.pop, 
				test_case.draw, 
				test_case.strategy, 
				StatMath.SamplingGen.SamplingMethod.RANDOM, 
				-1
			)
		)
		
		_check_performance(test_name, time)
	
	# Test FISHER_YATES with different sampling methods
	print("\n--- Testing sample_indices with different sampling methods ---")
	for method in [StatMath.SamplingGen.SamplingMethod.SOBOL, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE]:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		var test_name: String = "sample_indices_FISHER_YATES_%s_1000_100" % method_name
		
		var time: float = _measure_performance(test_name, func():
			return StatMath.SamplingGen.sample_indices(1000, 100, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES, method, -1)
		)
		
		_check_performance(test_name, time) 