# res://addons/godot-stat-math/tests/core/sampling_gen/scipy_validation_test.gd
class_name SamplingGenScipyValidationTest extends GdUnitTestSuite

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

func test_generate_samples_unified_interface_dimensions() -> void:
	var n_draws: int = 5
	
	# Test 1D generation
	var samples_1d: Variant = StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	assert_bool(samples_1d is Array[float]).is_true()
	var typed_samples_1d: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples_1d)
	assert_int(typed_samples_1d.size()).is_equal(n_draws)
	
	# Test 2D generation  
	var samples_2d: Variant = StatMath.SamplingGen.generate_samples(n_draws, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
	assert_bool(samples_2d is Array[Vector2]).is_true()
	var typed_samples_2d: Array[Vector2] = samples_2d as Array[Vector2]
	assert_int(typed_samples_2d.size()).is_equal(n_draws)
	
	# Test N-dimensional generation (5D)
	var samples_5d: Variant = StatMath.SamplingGen.generate_samples(n_draws, 5, StatMath.SamplingGen.SamplingMethod.SOBOL)
	assert_bool(samples_5d is Array).is_true()
	var typed_samples_5d: Array = samples_5d as Array
	assert_int(typed_samples_5d.size()).is_equal(n_draws)
	# Each sample should have 5 dimensions
	for sample in typed_samples_5d:
		assert_int(sample.size()).is_equal(5)


func test_generate_samples_unified_interface_starting_index() -> void:
	var n_draws: int = 3
	
	# Test starting_index parameter with deterministic sequences
	var samples_start_0: Variant = StatMath.SamplingGen.generate_samples(
		n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var samples_start_3: Variant = StatMath.SamplingGen.generate_samples(
		n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, 3
	)
	
	var typed_start_0: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples_start_0)
	var typed_start_3: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples_start_3)
	
	# Get first 6 samples to verify starting_index works correctly
	var first_6: Variant = StatMath.SamplingGen.generate_samples(
		6, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var typed_first_6: Array[float] = StatMath.HelperFunctions.convert_to_float_array(first_6)
	
	# samples_start_3 should equal elements [3,4,5] from first_6
	for i in range(n_draws):
		assert_float(typed_start_3[i]).is_equal_approx(typed_first_6[i + 3], StatMath.DETERMINISM_TOLERANCE)


func test_generate_samples_unified_interface_edge_cases() -> void:
	# Test zero draws
	var zero_1d: Variant = StatMath.SamplingGen.generate_samples(0, 1)
	var zero_2d: Variant = StatMath.SamplingGen.generate_samples(0, 2) 
	var zero_nd: Variant = StatMath.SamplingGen.generate_samples(0, 5)
	
	assert_int(StatMath.HelperFunctions.convert_to_float_array(zero_1d).size()).is_equal(0)
	assert_int((zero_2d as Array[Vector2]).size()).is_equal(0)
	assert_int((zero_nd as Array).size()).is_equal(0)
	
	# Test invalid dimensions
	var test_invalid_dims: Callable = func():
		StatMath.SamplingGen.generate_samples(5, 0)
	await assert_error(test_invalid_dims).is_push_error("dimensions must be >= 1. Received: 0")


# --- N-DIMENSIONAL GENERATION TESTS ---

func test_generate_samples_nd_basic() -> void:
	var n_draws: int = 10
	var dimensions: int = 4
	
	var samples: Array = StatMath.SamplingGen.generate_samples_nd(
		n_draws, dimensions, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	
	assert_int(samples.size()).is_equal(n_draws)
	
	# Verify each sample has correct dimensions and valid values
	for sample in samples:
		assert_int(sample.size()).is_equal(dimensions)
		for dim_val in sample:
			assert_float(dim_val).is_between(0.0, 1.0)


func test_generate_samples_nd_high_dimensions() -> void:
	var n_draws: int = 5
	var high_dims: int = 20
	
	var samples: Array = StatMath.SamplingGen.generate_samples_nd(
		n_draws, high_dims, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	
	assert_int(samples.size()).is_equal(n_draws)
	
	# Test threading kicks in for high dimensions (>=3)
	for sample in samples:
		assert_int(sample.size()).is_equal(high_dims)


func test_generate_samples_nd_starting_index_determinism() -> void:
	var n_draws: int = 3
	var dimensions: int = 3
	
	# Generate samples with different starting indices
	var samples_0: Array = StatMath.SamplingGen.generate_samples_nd(
		n_draws, dimensions, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var samples_2: Array = StatMath.SamplingGen.generate_samples_nd(
		n_draws, dimensions, StatMath.SamplingGen.SamplingMethod.SOBOL, 2
	)
	
	# Verify they produce different but deterministic results
	assert_bool(samples_0[0] != samples_2[0]).is_true() # Different starting points
	
	# Verify reproducibility
	var samples_0_repeat: Array = StatMath.SamplingGen.generate_samples_nd(
		n_draws, dimensions, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	
	for i in range(n_draws):
		for d in range(dimensions):
			assert_float(samples_0[i][d]).is_equal_approx(samples_0_repeat[i][d], StatMath.DETERMINISM_TOLERANCE)


func test_generate_samples_nd_all_methods() -> void:
	var n_draws: int = 5
	var dimensions: int = 3
	var seed: int = 42
	
	var methods: Array[StatMath.SamplingGen.SamplingMethod] = [
		StatMath.SamplingGen.SamplingMethod.RANDOM,
		StatMath.SamplingGen.SamplingMethod.SOBOL,
		StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM,
		StatMath.SamplingGen.SamplingMethod.HALTON,
		StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM,
		StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE
	]
	
	for method in methods:
		var samples: Array = StatMath.SamplingGen.generate_samples_nd(
			n_draws, dimensions, method, 0, seed
		)
		
		assert_int(samples.size()).is_equal(n_draws)
		
		for sample in samples:
			assert_int(sample.size()).is_equal(dimensions)
			for dim_val in sample:
				assert_float(dim_val).is_between(0.0, 1.0)


# --- COORDINATED SHUFFLE TESTS ---

func test_coordinated_shuffle_basic() -> void:
	var deck_size: int = 10
	
	var shuffled: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		deck_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	
	assert_int(shuffled.size()).is_equal(deck_size)
	
	# Verify all cards are present exactly once
	var card_counts: Dictionary = {}
	for card in shuffled:
		card_counts[card] = card_counts.get(card, 0) + 1
	
	for card in range(deck_size):
		assert_int(card_counts.get(card, 0)).is_equal(1)


func test_coordinated_shuffle_deterministic() -> void:
	var deck_size: int = 5
	
	# Same point_index should produce same shuffle
	var shuffle1: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		deck_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 42
	)
	var shuffle2: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		deck_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 42
	)
	
	for i in range(deck_size):
		assert_int(shuffle1[i]).is_equal(shuffle2[i])
	
	# Different point_index should produce different shuffle
	var shuffle3: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		deck_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 43
	)
	
	var differences: int = 0
	for i in range(deck_size):
		if shuffle1[i] != shuffle3[i]:
			differences += 1
	
	# Should have at least some differences
	assert_int(differences).is_greater(0)


func test_coordinated_shuffle_edge_cases() -> void:
	# Empty deck
	var empty: Array[int] = StatMath.SamplingGen.coordinated_shuffle(0)
	assert_int(empty.size()).is_equal(0)
	
	# Single card deck
	var single: Array[int] = StatMath.SamplingGen.coordinated_shuffle(1)
	assert_int(single.size()).is_equal(1)
	assert_int(single[0]).is_equal(0)


func test_coordinated_batch_shuffles() -> void:
	var deck_size: int = 8
	var n_shuffles: int = 5
	
	var batch: Array = StatMath.SamplingGen.coordinated_batch_shuffles(
		deck_size, n_shuffles, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	
	assert_int(batch.size()).is_equal(n_shuffles)
	
	# Verify each shuffle is valid
	for i in range(n_shuffles):
		var shuffle: Array = batch[i]
		assert_int(shuffle.size()).is_equal(deck_size)
		
		# Check all cards present
		var card_counts: Dictionary = {}
		for card in shuffle:
			card_counts[card] = card_counts.get(card, 0) + 1
		
		for card in range(deck_size):
			assert_int(card_counts.get(card, 0)).is_equal(1)
	
	# Verify shuffles are different (systematic exploration)
	if n_shuffles > 1:
		var first_shuffle: Array = batch[0]
		var second_shuffle: Array = batch[1]
		var differences: int = 0
		
		for i in range(deck_size):
			if first_shuffle[i] != second_shuffle[i]:
				differences += 1
		
		# Should have some differences between shuffles
		assert_int(differences).is_greater(0)


func test_coordinated_batch_shuffles_starting_index() -> void:
	var deck_size: int = 6
	var n_shuffles: int = 3
	
	# Test deterministic starting index
	var batch1: Array = StatMath.SamplingGen.coordinated_batch_shuffles(
		deck_size, n_shuffles, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var batch2: Array = StatMath.SamplingGen.coordinated_batch_shuffles(
		deck_size, n_shuffles, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	
	# Should be identical
	for i in range(n_shuffles):
		var shuffle1: Array = batch1[i]
		var shuffle2: Array = batch2[i]
		
		for j in range(deck_size):
			assert_int(shuffle1[j]).is_equal(shuffle2[j])


# --- COORDINATED SAMPLING ALTERNATIVE APPROACH ---
# Note: Instead of COORDINATED_FISHER_YATES, use coordinated_shuffle + slice for better clarity

func test_coordinated_sampling_alternative() -> void:
	# This demonstrates the CORRECT way to do coordinated sampling
	# Instead of using the removed COORDINATED_FISHER_YATES selection strategy
	var population_size: int = 20
	var draw_count: int = 5
	
	# Use coordinated_shuffle and take first N elements
	var full_shuffle: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		population_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var indices: Array[int] = full_shuffle.slice(0, draw_count)
	
	assert_int(indices.size()).is_equal(draw_count)
	
	# Verify all indices are valid and unique
	var seen: Dictionary = {}
	for idx in indices:
		assert_int(idx).is_between(0, population_size - 1)
		assert_bool(seen.has(idx)).is_false() # Should be unique
		seen[idx] = true
	
	assert_int(seen.size()).is_equal(draw_count)


func test_coordinated_sampling_deterministic() -> void:
	# Demonstrates deterministic coordinated sampling using existing API
	var population_size: int = 15
	var draw_count: int = 4
	
	# Same point_index should produce same results
	var shuffle1: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		population_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 42
	)
	var shuffle2: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		population_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 42
	)
	
	var indices1: Array[int] = shuffle1.slice(0, draw_count)
	var indices2: Array[int] = shuffle2.slice(0, draw_count)
	
	# Should be deterministic
	for i in range(draw_count):
		assert_int(indices1[i]).is_equal(indices2[i])


func test_coordinated_sampling_performance_comparison() -> void:
	# Shows that coordinated_shuffle + slice is simple and effective
	var population_size: int = 100
	var draw_count: int = 10
	
	# Coordinated approach - generates systematic sample
	var coordinated_shuffle: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		population_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var coordinated_sample: Array[int] = coordinated_shuffle.slice(0, draw_count)
	
	# Regular Fisher-Yates approach - generates independent random sample  
	var fisher_yates_sample: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	
	# Both should be valid
	assert_int(coordinated_sample.size()).is_equal(draw_count)
	assert_int(fisher_yates_sample.size()).is_equal(draw_count)
	
	_assert_unique_indices(coordinated_sample, population_size)
	_assert_unique_indices(fisher_yates_sample, population_size)


# --- UPDATED EXISTING TESTS (following GDUnit4 rules) ---

func test_generate_samples_1d_random_basic() -> void:
	var ndraws: int = 10
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for sample_val in typed_samples:
		assert_float(sample_val).is_between(0.0, 1.0)


func test_generate_samples_1d_edge_cases() -> void:
	# Zero draws
	var zero_samples: Variant = StatMath.SamplingGen.generate_samples(0, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_zero: Array[float] = StatMath.HelperFunctions.convert_to_float_array(zero_samples)
	assert_int(typed_zero.size()).is_equal(0)
	
	# Negative draws
	var negative_samples: Variant = StatMath.SamplingGen.generate_samples(-5, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_negative: Array[float] = StatMath.HelperFunctions.convert_to_float_array(negative_samples)
	assert_int(typed_negative.size()).is_equal(0)


func test_generate_samples_1d_sobol_deterministic() -> void:
	var ndraws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	var expected_sobol: Array[float] = [0.0, 0.5, 0.75, 0.25, 0.375]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_samples[i]).is_equal_approx(expected_sobol[i], StatMath.DETERMINISM_TOLERANCE)


func test_generate_samples_1d_halton_deterministic() -> void:
	var ndraws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.HALTON)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	var expected_halton: Array[float] = [0.5, 0.25, 0.75, 0.125, 0.625]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_samples[i]).is_equal_approx(expected_halton[i], StatMath.DETERMINISM_TOLERANCE)


func test_generate_samples_1d_seeded_reproducibility() -> void:
	var ndraws: int = 5
	var seed: int = 12345
	
	# Test SOBOL_RANDOM reproducibility
	var sobol_1: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, 0, seed)
	var sobol_2: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, 0, seed)
	
	var typed_sobol_1: Array[float] = StatMath.HelperFunctions.convert_to_float_array(sobol_1)
	var typed_sobol_2: Array[float] = StatMath.HelperFunctions.convert_to_float_array(sobol_2)
	
	assert_int(typed_sobol_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_sobol_1[i]).is_equal_approx(typed_sobol_2[i], StatMath.DETERMINISM_TOLERANCE)
	
	# Test LATIN_HYPERCUBE reproducibility
	var lhs_1: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, seed)
	var lhs_2: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, seed)
	
	var typed_lhs_1: Array[float] = StatMath.HelperFunctions.convert_to_float_array(lhs_1)
	var typed_lhs_2: Array[float] = StatMath.HelperFunctions.convert_to_float_array(lhs_2)
	
	assert_int(typed_lhs_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_lhs_1[i]).is_equal_approx(typed_lhs_2[i], StatMath.DETERMINISM_TOLERANCE)


func test_generate_samples_1d_latin_hypercube_stratification() -> void:
	var ndraws: int = 20
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, 123)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	
	var sorted_samples: Array[float] = typed_samples.duplicate()
	sorted_samples.sort()
	
	for i in range(ndraws):
		var lower_bound: float = float(i) / float(ndraws)
		var upper_bound: float = float(i + 1) / float(ndraws)
		assert_float(sorted_samples[i]).is_greater_equal(lower_bound)
		assert_float(sorted_samples[i]).is_less(upper_bound)


func test_generate_samples_2d_basic() -> void:
	var ndraws: int = 10
	var methods: Array[StatMath.SamplingGen.SamplingMethod] = [
		StatMath.SamplingGen.SamplingMethod.RANDOM,
		StatMath.SamplingGen.SamplingMethod.SOBOL,
		StatMath.SamplingGen.SamplingMethod.HALTON,
		StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE
	]
	
	for method in methods:
		var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 2, method)
		var typed_samples: Array[Vector2] = samples as Array[Vector2]
		assert_int(typed_samples.size()).is_equal(ndraws)
		for sample_vec in typed_samples:
			assert_float(sample_vec.x).is_between(0.0, 1.0)
			assert_float(sample_vec.y).is_between(0.0, 1.0)


func test_generate_samples_2d_sobol_deterministic() -> void:
	var ndraws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var typed_samples: Array[Vector2] = samples as Array[Vector2]
	var expected_sobol_2d: Array[Vector2] = [
		Vector2(0.0, 0.0),
		Vector2(0.5, 0.5), 
		Vector2(0.75, 0.5),
		Vector2(0.25, 0.0),
		Vector2(0.375, 0.125)
	]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(typed_samples[i]).is_equal_approx(expected_sobol_2d[i], Vector2(StatMath.DETERMINISM_TOLERANCE, StatMath.DETERMINISM_TOLERANCE))


# --- DISCRETE INDEX SAMPLING TESTS (updated) ---

func test_sample_indices_with_replacement_basic() -> void:
	var population_size: int = 10
	var draw_count: int = 15  # More than population to test replacement
	var samples: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count, 
		StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT,
		StatMath.SamplingGen.SamplingMethod.RANDOM
	)
	
	assert_int(samples.size()).is_equal(draw_count)
	for sample_val in samples:
		assert_int(sample_val).is_between(0, population_size - 1)
	
	# Should allow duplicates
	var unique_values: Dictionary = {}
	for sample_val in samples:
		unique_values[sample_val] = true
	# With replacement, we might have fewer unique values than draws
	assert_int(unique_values.size()).is_less_equal(draw_count)


func test_sample_indices_without_replacement_basic() -> void:
	var population_size: int = 20
	var draw_count: int = 5
	var strategies: Array[StatMath.SamplingGen.SelectionStrategy] = [
		StatMath.SamplingGen.SelectionStrategy.FISHER_YATES,
		StatMath.SamplingGen.SelectionStrategy.RESERVOIR,
		StatMath.SamplingGen.SelectionStrategy.SELECTION_TRACKING
	]
	
	for strategy in strategies:
		var samples: Array[int] = StatMath.SamplingGen.sample_indices(
			population_size, draw_count, strategy, StatMath.SamplingGen.SamplingMethod.RANDOM
		)
		
		assert_int(samples.size()).is_equal(draw_count)
		
		# Check all samples are in valid range
		for sample_val in samples:
			assert_int(sample_val).is_between(0, population_size - 1)
		
		# Check all samples are unique
		var unique_values: Dictionary = {}
		for sample_val in samples:
			assert_bool(unique_values.has(sample_val)).is_false()
			unique_values[sample_val] = true
		assert_int(unique_values.size()).is_equal(draw_count)


func test_starting_index_sobol_sequence_continuity() -> void:
	# Test that starting_index produces continuous sequences
	var total_draws: int = 10
	var first_half: int = 5
	var second_half: int = 5
	
	# Generate full sequence
	var full_sequence: Variant = StatMath.SamplingGen.generate_samples(
		total_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	var typed_full_sequence: Array[float] = StatMath.HelperFunctions.convert_to_float_array(full_sequence)
	
	# Generate in two parts using starting_index
	var part1: Variant = StatMath.SamplingGen.generate_samples(
		first_half, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, 0, -1
	)
	var part2: Variant = StatMath.SamplingGen.generate_samples(
		second_half, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, 0, -1
	)
	
	# Test with explicit starting_index
	var part2_explicit: Variant = StatMath.SamplingGen.generate_samples(
		second_half, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, first_half
	)
	var part2_typed: Array[float] = StatMath.HelperFunctions.convert_to_float_array(part2_explicit)
	
	# part2_explicit should match the second half of full_sequence
	for i in range(second_half):
		assert_float(part2_typed[i]).is_equal_approx(typed_full_sequence[first_half + i], StatMath.DETERMINISM_TOLERANCE)


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