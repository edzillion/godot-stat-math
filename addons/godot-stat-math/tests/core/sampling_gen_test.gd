# addons/godot-stat-math/tests/core/sampling_gen_test.gd
class_name SamplingGenTest extends GdUnitTestSuite

# Member variables here if needed, e.g. for complex setups or shared resources.


# Called before each test.
func before_test() -> void:
	pass


# Called after each test.
func after_test() -> void:
	pass


# --- UNIFIED GENERATE_SAMPLES INTERFACE TESTS ---

func test_generate_samples_unified_interface_dimensions() -> void:
	var n_draws: int = 5
	
	# Test 1D generation
	var samples_1d: Variant = StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	assert_bool(samples_1d is Array[float]).is_true()
	var typed_samples_1d: Array[float] = samples_1d as Array[float]
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
	
	var typed_start_0: Array[float] = samples_start_0 as Array[float]
	var typed_start_3: Array[float] = samples_start_3 as Array[float]
	
	# Get first 6 samples to verify starting_index works correctly
	var first_6: Variant = StatMath.SamplingGen.generate_samples(
		6, 1, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var typed_first_6: Array[float] = first_6 as Array[float]
	
	# samples_start_3 should equal elements [3,4,5] from first_6
	for i in range(n_draws):
		assert_float(typed_start_3[i]).is_equal_approx(typed_first_6[i + 3], 0.00001)


func test_generate_samples_unified_interface_edge_cases() -> void:
	# Test zero draws
	var zero_1d: Variant = StatMath.SamplingGen.generate_samples(0, 1)
	var zero_2d: Variant = StatMath.SamplingGen.generate_samples(0, 2) 
	var zero_nd: Variant = StatMath.SamplingGen.generate_samples(0, 5)
	
	assert_int((zero_1d as Array[float]).size()).is_equal(0)
	assert_int((zero_2d as Array[Vector2]).size()).is_equal(0)
	assert_int((zero_nd as Array).size()).is_equal(0)
	
	# Test invalid dimensions
	var invalid_dims: Variant = StatMath.SamplingGen.generate_samples(5, 0)
	assert_bool(invalid_dims == null).is_true()


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
			assert_float(samples_0[i][d]).is_equal_approx(samples_0_repeat[i][d], 0.0000001)


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


# --- COORDINATED_FISHER_YATES SELECTION STRATEGY TESTS ---

func test_coordinated_fisher_yates_selection_strategy() -> void:
	var population_size: int = 20
	var draw_count: int = 5
	
	var indices: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	
	assert_int(indices.size()).is_equal(draw_count)
	
	# Verify all indices are valid and unique
	var seen: Dictionary = {}
	for idx in indices:
		assert_int(idx).is_between(0, population_size - 1)
		assert_bool(seen.has(idx)).is_false() # Should be unique
		seen[idx] = true
	
	assert_int(seen.size()).is_equal(draw_count)


func test_coordinated_fisher_yates_deterministic() -> void:
	var population_size: int = 15
	var draw_count: int = 4
	var seed: int = 123
	
	var indices1: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL, seed
	)
	var indices2: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL, seed
	)
	
	# Should be deterministic
	for i in range(draw_count):
		assert_int(indices1[i]).is_equal(indices2[i])


func test_coordinated_fisher_yates_small_deck_performance() -> void:
	# Test with smaller deck sizes to avoid potential performance issues
	var small_deck_sizes: Array[int] = [5, 10, 15]
	
	for deck_size in small_deck_sizes:
		var draw_count: int = min(3, deck_size)
		
		var indices: Array[int] = StatMath.SamplingGen.sample_indices(
			deck_size, draw_count,
			StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES,
			StatMath.SamplingGen.SamplingMethod.SOBOL
		)
		
		assert_int(indices.size()).is_equal(draw_count)
		_assert_unique_indices(indices, deck_size)


# --- UPDATED EXISTING TESTS (following GDUnit4 rules) ---

func test_generate_samples_1d_random_basic() -> void:
	var ndraws: int = 10
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_samples: Array[float] = samples as Array[float]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for sample_val in typed_samples:
		assert_float(sample_val).is_between(0.0, 1.0)


func test_generate_samples_1d_edge_cases() -> void:
	# Zero draws
	var zero_samples: Variant = StatMath.SamplingGen.generate_samples(0, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_zero: Array[float] = zero_samples as Array[float]
	assert_int(typed_zero.size()).is_equal(0)
	
	# Negative draws
	var negative_samples: Variant = StatMath.SamplingGen.generate_samples(-5, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_negative: Array[float] = negative_samples as Array[float]
	assert_int(typed_negative.size()).is_equal(0)


func test_generate_samples_1d_sobol_deterministic() -> void:
	var ndraws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var typed_samples: Array[float] = samples as Array[float]
	var expected_sobol: Array[float] = [0.0, 0.5, 0.75, 0.25, 0.375]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_samples[i]).is_equal_approx(expected_sobol[i], 0.00001)


func test_generate_samples_1d_halton_deterministic() -> void:
	var ndraws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.HALTON)
	var typed_samples: Array[float] = samples as Array[float]
	var expected_halton: Array[float] = [0.5, 0.25, 0.75, 0.125, 0.625]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_samples[i]).is_equal_approx(expected_halton[i], 0.00001)


func test_generate_samples_1d_seeded_reproducibility() -> void:
	var ndraws: int = 5
	var seed: int = 12345
	
	# Test SOBOL_RANDOM reproducibility
	var sobol_1: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, 0, seed)
	var sobol_2: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, 0, seed)
	
	var typed_sobol_1: Array[float] = sobol_1 as Array[float]
	var typed_sobol_2: Array[float] = sobol_2 as Array[float]
	
	assert_int(typed_sobol_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_sobol_1[i]).is_equal_approx(typed_sobol_2[i], 0.0000001)
	
	# Test LATIN_HYPERCUBE reproducibility
	var lhs_1: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, seed)
	var lhs_2: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, seed)
	
	var typed_lhs_1: Array[float] = lhs_1 as Array[float]
	var typed_lhs_2: Array[float] = lhs_2 as Array[float]
	
	assert_int(typed_lhs_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(typed_lhs_1[i]).is_equal_approx(typed_lhs_2[i], 0.0000001)


func test_generate_samples_1d_latin_hypercube_stratification() -> void:
	var ndraws: int = 20
	var samples: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, 123)
	var typed_samples: Array[float] = samples as Array[float]
	
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
		Vector2(0.75, 0.75),
		Vector2(0.25, 0.25),
		Vector2(0.375, 0.625)
	]
	
	assert_int(typed_samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(typed_samples[i]).is_equal_approx(expected_sobol_2d[i], Vector2(0.00001, 0.00001))


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
		StatMath.SamplingGen.SelectionStrategy.SELECTION_TRACKING,
		StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES
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


func test_sample_indices_hybrid_combinations() -> void:
	var population_size: int = 50
	var draw_count: int = 10
	
	# Test SOBOL + FISHER_YATES
	var sobol_fy: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.FISHER_YATES,
		StatMath.SamplingGen.SamplingMethod.SOBOL,
		42
	)
	assert_int(sobol_fy.size()).is_equal(draw_count)
	_assert_unique_indices(sobol_fy, population_size)
	
	# Test LATIN_HYPERCUBE + WITH_REPLACEMENT
	var lhs_wr: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT,
		StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE,
		42
	)
	assert_int(lhs_wr.size()).is_equal(draw_count)
	_assert_valid_indices(lhs_wr, population_size)
	
	# Test HALTON + RESERVOIR
	var halton_res: Array[int] = StatMath.SamplingGen.sample_indices(
		population_size, draw_count,
		StatMath.SamplingGen.SelectionStrategy.RESERVOIR,
		StatMath.SamplingGen.SamplingMethod.HALTON,
		42
	)
	assert_int(halton_res.size()).is_equal(draw_count)
	_assert_unique_indices(halton_res, population_size)


func test_sample_indices_seeded_reproducibility() -> void:
	var population_size: int = 30
	var draw_count: int = 8
	var seed: int = 98765
	
	# Test reproducibility with different strategy/method combinations
	var combinations: Array[Array] = [
		[StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT, StatMath.SamplingGen.SamplingMethod.SOBOL],
		[StatMath.SamplingGen.SelectionStrategy.FISHER_YATES, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE],
		[StatMath.SamplingGen.SelectionStrategy.RESERVOIR, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM],
		[StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES, StatMath.SamplingGen.SamplingMethod.SOBOL]
	]
	
	for combo in combinations:
		var strategy: StatMath.SamplingGen.SelectionStrategy = combo[0]
		var method: StatMath.SamplingGen.SamplingMethod = combo[1]
		
		var samples_1: Array[int] = StatMath.SamplingGen.sample_indices(population_size, draw_count, strategy, method, seed)
		var samples_2: Array[int] = StatMath.SamplingGen.sample_indices(population_size, draw_count, strategy, method, seed)
		
		assert_int(samples_1.size()).is_equal(draw_count)
		assert_int(samples_2.size()).is_equal(draw_count)
		
		for i in range(draw_count):
			assert_int(samples_1[i]).is_equal(samples_2[i])


func test_sample_indices_parameter_validation() -> void:
	# Test negative draw_count
	var invalid_draw: Array[int] = StatMath.SamplingGen.sample_indices(10, -1)
	assert_int(invalid_draw.size()).is_equal(0)
	
	# Test negative population_size
	var invalid_pop: Array[int] = StatMath.SamplingGen.sample_indices(-10, 5)
	assert_int(invalid_pop.size()).is_equal(0)
	
	# Test draw_count > population_size for without replacement
	var invalid_without_replacement: Array[int] = StatMath.SamplingGen.sample_indices(
		5, 10, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES
	)
	assert_int(invalid_without_replacement.size()).is_equal(0)
	
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


# --- CARD GAME SIMULATION TESTS (updated) ---

func test_card_game_dealing() -> void:
	var deck_size: int = 52
	var hand_size: int = 5
	
	# Test different dealing strategies for card games including new COORDINATED_FISHER_YATES
	var fisher_yates: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES
	)
	var reservoir: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.RESERVOIR
	)
	var selection_tracking: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.SELECTION_TRACKING
	)
	var coordinated: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.COORDINATED_FISHER_YATES
	)
	
	# All strategies should deal valid hands
	var all_hands: Array = [fisher_yates, reservoir, selection_tracking, coordinated]
	for hand in all_hands:
		assert_int(hand.size()).is_equal(hand_size)
		_assert_unique_indices(hand, deck_size)


func test_dice_rolling_simulation() -> void:
	# Test dice rolling with replacement (can roll same number multiple times)
	var dice_sides: int = 6
	var roll_count: int = 100
	
	var dice_rolls: Array[int] = StatMath.SamplingGen.sample_indices(
		dice_sides, roll_count,
		StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT,
		StatMath.SamplingGen.SamplingMethod.RANDOM
	)
	
	assert_int(dice_rolls.size()).is_equal(roll_count)
	for roll in dice_rolls:
		assert_int(roll).is_between(0, dice_sides - 1)  # 0-5 representing 1-6 on dice
	
	# Verify we can have duplicates (should be very likely with 100 rolls)
	var unique_values: Dictionary = {}
	for roll in dice_rolls:
		unique_values[roll] = true
	assert_int(unique_values.size()).is_less_equal(dice_sides)  # Should have 6 or fewer unique values


# --- ROYAL FLUSH SIMULATION TEST ---

func test_royal_flush_simulation_demo() -> void:
	# Test the coordinated shuffle approach for rare event simulation
	var deck_size: int = 52
	var n_trials: int = 100
	
	# Generate coordinated shuffles for consistent rare event analysis
	var batch_shuffles: Array = StatMath.SamplingGen.coordinated_batch_shuffles(
		deck_size, n_trials, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	
	assert_int(batch_shuffles.size()).is_equal(n_trials)
	
	# Verify systematic exploration
	var hand_frequencies: Dictionary = {}
	for trial in range(n_trials):
		var deck: Array = batch_shuffles[trial]
		var hand: Array = deck.slice(0, 5)  # First 5 cards
		var hand_key: String = str(hand)
		hand_frequencies[hand_key] = hand_frequencies.get(hand_key, 0) + 1
	
	# With Sobol sequences, we should get more systematic coverage
	# Should have mostly unique hands due to systematic exploration
	var unique_hands: int = hand_frequencies.size()
	assert_int(unique_hands).is_greater(n_trials * 0.8) # At least 80% unique hands


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


func test_bootstrap_sampling_pattern() -> void:
	# Test typical bootstrap sampling scenario
	var original_size: int = 100
	var bootstrap_size: int = 100
	
	var bootstrap_sample: Array[int] = StatMath.SamplingGen.sample_indices(
		original_size, bootstrap_size,
		StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT,
		StatMath.SamplingGen.SamplingMethod.RANDOM,
		42
	)
	
	assert_int(bootstrap_sample.size()).is_equal(bootstrap_size)
	_assert_valid_indices(bootstrap_sample, original_size)
	
	# Bootstrap should have some duplicates (very high probability)
	var unique_count: int = 0
	var seen: Dictionary = {}
	for idx in bootstrap_sample:
		if not seen.has(idx):
			seen[idx] = true
			unique_count += 1
	
	# Bootstrap should have fewer unique values than total samples (statistically almost certain)
	assert_int(unique_count).is_less(bootstrap_size)


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


# --- GLOBAL RNG DETERMINISM TESTS (updated) ---

func test_global_rng_determinism() -> void:
	var test_seed: int = 888
	var ndraws: int = 5
	
	# Test continuous sampling determinism
	StatMath.set_global_seed(test_seed)
	var continuous_1: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_continuous_1: Array[float] = continuous_1 as Array[float]
	
	StatMath.set_global_seed(test_seed)
	var continuous_2: Variant = StatMath.SamplingGen.generate_samples(ndraws, 1, StatMath.SamplingGen.SamplingMethod.RANDOM)
	var typed_continuous_2: Array[float] = continuous_2 as Array[float]
	
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


# --- STARTING INDEX COMPREHENSIVE TESTS ---

func test_starting_index_sobol_sequence_continuity() -> void:
	# Test that starting_index produces continuous sequences
	var total_draws: int = 10
	var first_half: int = 5
	var second_half: int = 5
	
	# Generate full sequence
	var full_sequence: Variant = StatMath.SamplingGen.generate_samples(
		total_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL
	)
	var typed_full_sequence: Array[float] = full_sequence as Array[float]
	
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
	var part2_typed: Array[float] = part2_explicit as Array[float]
	
	# part2_explicit should match the second half of full_sequence
	for i in range(second_half):
		assert_float(part2_typed[i]).is_equal_approx(typed_full_sequence[first_half + i], 0.00001)
