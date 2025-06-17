# res://addons/godot-stat-math/tests/performance/core/sampling_gen_perf_test.gd
class_name SamplingGenPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.SamplingGen Module
## 
## Tests random number generation and sampling functions to catch
## performance regressions in critical game random systems.

# Test matrices - keep reasonable for test speed
const BATCH_SIZES: Array[int] = [32, 256, 1024]
const DIMENSIONS: Array[int] = [1, 3, 10]
const GENERATORS: Array[SamplingGen.SamplingMethod] = [
	SamplingGen.SamplingMethod.RANDOM,
	SamplingGen.SamplingMethod.SOBOL,
	SamplingGen.SamplingMethod.SOBOL_RANDOM,
	SamplingGen.SamplingMethod.HALTON
]

# Override module name for result tracking
func get_module_name() -> String:
	return "SamplingGen"


## Test generate_samples performance across different generators, dimensions, and batch sizes
func test_generate_samples_performance() -> void:
	var test_name: String = "generate_samples_performance"
	var baseline_data: Dictionary = _load_baseline()
	
	# Clear Sobol direction vectors cache to ensure fresh measurements
	SamplingGen._sobol_direction_vectors_cache.clear()
	SamplingGen._max_cached_dimension = -1
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for generator in GENERATORS:
			for dimension in DIMENSIONS:
				for batch_size in BATCH_SIZES:
					SamplingGen.generate_samples(batch_size, dimension, generator)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_coordinated_shuffle_performance() -> void:
	var test_name: String = "coordinated_shuffle_performance"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for batch_size in BATCH_SIZES:
			for generator in GENERATORS:
				# Test coordinated shuffle with different deck sizes and methods
				SamplingGen.coordinated_shuffle(batch_size, generator, 0, -1)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_sample_indices_performance() -> void:
	var test_name: String = "sample_indices_performance"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for batch_size in BATCH_SIZES:
			var sample_size: int = mini(batch_size / 4, 256)  # Sample quarter of the data, max 256
			SamplingGen.sample_indices(batch_size, sample_size)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


 
