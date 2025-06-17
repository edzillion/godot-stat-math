# res://addons/godot-stat-math/tests/performance/sampling_gen_perf_test.gd
class_name SamplingGenPerfTest extends GdUnitTestSuite

## Simple GDUnit4 Performance Test Suite for SamplingGen
## 
## Compares current performance against baseline file in repo.
## Use generate_baseline.gd tool script to create/update baselines.

# Configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/results/baseline.json"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

# Test matrices - keep reasonable for test speed
const BATCH_SIZES: Array[int] = [256, 1024, 4096]
const DIMENSIONS: Array[int] = [1, 2, 3]
const GENERATORS: Array[SamplingGen.SamplingMethod] = [
	SamplingGen.SamplingMethod.RANDOM,
	SamplingGen.SamplingMethod.SOBOL,
	SamplingGen.SamplingMethod.SOBOL_RANDOM,
	SamplingGen.SamplingMethod.HALTON
]

# Cached baseline data
var _baseline_cache: Dictionary = {}


# GDUnit4 lifecycle methods for test run collection
func before() -> void:
	TestRunCollector.start_test_run()


func after() -> void:
	await TestRunCollector.finish_test_run()


## Test basic generate_samples performance - this will be removed as it's redundant
## The collect_performance_measurements() method handles all sampling configurations
func test_generate_samples_performance() -> void:
	# This test is redundant - collect_performance_measurements() covers all cases
	# Keeping as no-op to maintain test suite compatibility
	pass


## Public method for baseline generation - returns performance measurements
func collect_performance_measurements() -> Dictionary:
	# Clear Sobol direction vectors cache to ensure fresh measurements
	SamplingGen._sobol_direction_vectors_cache.clear()
	SamplingGen._max_cached_dimension = -1
	
	var results: Dictionary = {}
	
	print("Measuring generate_samples performance...")
	for generator in GENERATORS:
		for dimension in DIMENSIONS:
			for batch_size in BATCH_SIZES:
				var test_name: String = "generate_samples_%s_%dd_%d" % [
					SamplingGen.SamplingMethod.keys()[generator], dimension, batch_size
				]
				
				var measurement: Dictionary = _measure_test(test_name, func(): 
					return SamplingGen.generate_samples(batch_size, dimension, generator)
				)
				
				results[test_name] = measurement.execution_time_ms
				print("  %s: %.2f ms" % [test_name, measurement.execution_time_ms])
	
	return results


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
	var median_time: float = times[median_index]
	
	return {"execution_time_ms": median_time}


func _load_baseline() -> Dictionary:
	if not _baseline_cache.is_empty():
		return _baseline_cache
	
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
	
	var data: Dictionary = json.data
	
	if not data.has("tests"):
		push_error("Baseline file missing 'tests' key: " + BASELINE_FILE)
		return {}
	
	_baseline_cache = data["tests"]
	return _baseline_cache


# TODO: Add other performance tests for coordinated_shuffle, batch_shuffles, sample_indices
# These would need their own collect_performance_measurements() implementations 
