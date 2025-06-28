# res://addons/godot-stat-math/tests/core/sampling_gen/sampling_gen_parameter_validation_tests.gd
class_name SamplingGenParameterValidationTests extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

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

## Tests parameter validation for coordinated_batch_shuffles
func test_coordinated_batch_shuffles_invalid_parameters() -> void:
	# Test invalid deck_size - negative value
	var result_negative_deck: Array = StatMath.SamplingGen.coordinated_batch_shuffles(-5, 3)
	assert_int(result_negative_deck.size()).is_equal(0)  # Should return empty array
	
	# Test invalid deck_size - zero value  
	var result_zero_deck: Array = StatMath.SamplingGen.coordinated_batch_shuffles(0, 3)
	assert_int(result_zero_deck.size()).is_equal(0)  # Should return empty array
	
	# Test invalid n_shuffles - negative value
	var result_negative_shuffles: Array = StatMath.SamplingGen.coordinated_batch_shuffles(10, -2)
	assert_int(result_negative_shuffles.size()).is_equal(0)  # Should return empty array
	
	# Test invalid n_shuffles - zero value
	var result_zero_shuffles: Array = StatMath.SamplingGen.coordinated_batch_shuffles(10, 0)
	assert_int(result_zero_shuffles.size()).is_equal(0)  # Should return empty array


# --- HELPER FUNCTIONS ---

 
