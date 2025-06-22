# res://addons/godot-stat-math/tests/core/sampling_gen/parameter_validation_test.gd
class_name SamplingGenParameterValidationTest extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

func test_sample_indices_parameter_validation() -> void:
	# Test negative draw_count
	var test_negative_draw: Callable = func():
		StatMath.SamplingGen.sample_indices(10, -1)
	await assert_error(test_negative_draw).is_push_error("draw_count cannot be negative. Received: -1")
	
	# Test negative population_size
	var test_negative_pop: Callable = func():
		StatMath.SamplingGen.sample_indices(-10, 5)
	await assert_error(test_negative_pop).is_push_error("population_size cannot be negative. Received: -10")
	
	# Test draw_count > population_size for without replacement
	var test_invalid_without_replacement: Callable = func():
		StatMath.SamplingGen.sample_indices(5, 10, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES)
	await assert_error(test_invalid_without_replacement).is_push_error("Without replacement, draw_count cannot exceed population_size. Received draw_count=10, population_size=5")
	
	# Test draw_count > population_size for with replacement (should work)
	var valid_with_replacement: Array[int] = StatMath.SamplingGen.sample_indices(
		5, 10, StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT
	)
	assert_int(valid_with_replacement.size()).is_equal(10)


func test_sample_indices_edge_cases() -> void:
	# Zero draws
	var zero_draws: Array[int] = StatMath.SamplingGen.sample_indices(10, 0)
	assert_int(zero_draws.size()).is_equal(0)
	
	# Draw all elements
	var draw_all: Array[int] = StatMath.SamplingGen.sample_indices(
		5, 5, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES
	)
	assert_int(draw_all.size()).is_equal(5)
	_assert_unique_indices(draw_all, 5)
	
	# Single element population
	var single_element: Array[int] = StatMath.SamplingGen.sample_indices(
		1, 1, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES
	)
	assert_int(single_element.size()).is_equal(1)
	assert_int(single_element[0]).is_equal(0)


# --- PERFORMANCE AND STRESS TESTS (updated) ---

func test_large_scale_sampling() -> void:
	# Test with larger datasets to ensure performance
	var large_pop: int = 1000
	var large_draws: int = 100
	
	var large_sample: Array[int] = StatMath.SamplingGen.sample_indices(
		large_pop, large_draws,
		StatMath.SamplingGen.SelectionStrategy.FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.RANDOM
	)
	
	assert_int(large_sample.size()).is_equal(large_draws)
	_assert_unique_indices(large_sample, large_pop)


func test_threading_performance_basic() -> void:
	# Test that high-dimensional generation completes in reasonable time
	var n_draws: int = 50
	var high_dims: int = 10
	
	var start_time: int = Time.get_ticks_msec()
	var samples: Array = StatMath.SamplingGen.generate_samples_nd(
		n_draws, high_dims, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	var elapsed: int = Time.get_ticks_msec() - start_time
	
	assert_int(samples.size()).is_equal(n_draws)
	# Should complete within reasonable time (threading should help)
	assert_int(elapsed).is_less(5000) # 5 seconds max


# --- GLOBAL RNG DETERMINISM TESTS (updated) ---

func test_global_rng_determinism() -> void:
	var test_seed: int = 888
	var ndraws: int = 5
	
	# Test continuous sampling determinism
	StatMath.set_global_seed(test_seed)
	var continuous_1: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_continuous_1: Array[float] = StatMath.HelperFunctions.convert_to_float_array(continuous_1)
	
	StatMath.set_global_seed(test_seed)
	var continuous_2: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_continuous_2: Array[float] = StatMath.HelperFunctions.convert_to_float_array(continuous_2)
	
	assert_int(typed_continuous_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_continuous_1[i]).is_equal(typed_continuous_2[i])
	
	# Test discrete sampling determinism
	StatMath.set_global_seed(test_seed)
	var discrete_1: Array[int] = StatMath.SamplingGen.sample_indices(20, 5)
	
	StatMath.set_global_seed(test_seed)
	var discrete_2: Array[int] = StatMath.SamplingGen.sample_indices(20, 5)
	
	assert_int(discrete_1.size()).is_equal(5)
	for i in range(5):
		assert_int(discrete_1[i]).is_equal(discrete_2[i])


## Tests parameter validation for sampling functions
func test_sampling_invalid_parameters() -> void:
	# Test invalid n_draws
	var test_negative_draws: Callable = func():
		StatMath.SamplingGen.generate_samples(-1, 1)
	await assert_error(test_negative_draws).is_push_error("n_draws must be non-negative. Received: -1")
	
	# Test invalid dimensions
	var test_zero_dimensions: Callable = func():
		StatMath.SamplingGen.generate_samples(5, 0)
	await assert_error(test_zero_dimensions).is_push_error("dimensions must be >= 1. Received: 0")

## Tests parameter validation for sample_indices
func test_sample_indices_invalid_parameters() -> void:
	# Test invalid population_size
	var test_zero_population: Callable = func():
		StatMath.SamplingGen.sample_indices(0, 5)
	await assert_error(test_zero_population).is_push_error("population_size must be positive. Received: 0")
	
	# Test sampling more than population without replacement
	var test_oversample: Callable = func():
		StatMath.SamplingGen.sample_indices(5, 10, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES)
	await assert_error(test_oversample).is_push_error("Without replacement, draw_count cannot exceed population_size. Received draw_count=10, population_size=5")


# --- HELPER FUNCTIONS ---

func _assert_valid_indices(samples: Array[int], population_size: int) -> void:
	for sample_val in samples:
		assert_int(sample_val).is_greater_equal(0)
		assert_int(sample_val).is_less(population_size)


func _assert_unique_indices(samples: Array[int], population_size: int) -> void:
	_assert_valid_indices(samples, population_size)
	
	var unique_values: Dictionary = {}
	for sample_val in samples:
		assert_bool(unique_values.has(sample_val)).is_false()
		unique_values[sample_val] = true
	
	assert_int(unique_values.size()).is_equal(samples.size()) 