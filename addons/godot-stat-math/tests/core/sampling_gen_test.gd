# addons/godot-stat-math/tests/core/sampling_gen_test.gd
class_name SamplingGenTest extends GdUnitTestSuite

# Member variables here if needed, e.g. for complex setups or shared resources.


# Called before each test.
func before_test() -> void:
	pass


# Called after each test.
func after_test() -> void:
	pass


# --- CONTINUOUS SPACE SAMPLING TESTS (generate_samples_1d/2d) ---

func test_generate_samples_1d_random_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(ndraws)
	for sample_val in samples:
		assert_float(sample_val).is_between(0.0, 1.0)


func test_generate_samples_1d_edge_cases() -> void:
	# Zero draws
	var zero_samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(0, StatMath.SamplingGen.SamplingMethod.RANDOM)
	assert_int(zero_samples.size()).is_equal(0)
	
	# Negative draws
	var negative_samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(-5, StatMath.SamplingGen.SamplingMethod.RANDOM)
	assert_int(negative_samples.size()).is_equal(0)


func test_generate_samples_1d_sobol_deterministic() -> void:
	var ndraws: int = 5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var expected_sobol: Array[float] = [0.0, 0.5, 0.75, 0.25, 0.375]
	
	assert_int(samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(samples[i]).is_equal_approx(expected_sobol[i], 0.00001)


func test_generate_samples_1d_halton_deterministic() -> void:
	var ndraws: int = 5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON)
	var expected_halton: Array[float] = [0.5, 0.25, 0.75, 0.125, 0.625]
	
	assert_int(samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(samples[i]).is_equal_approx(expected_halton[i], 0.00001)


func test_generate_samples_1d_seeded_reproducibility() -> void:
	var ndraws: int = 5
	var seed: int = 12345
	
	# Test SOBOL_RANDOM reproducibility
	var sobol_1: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, seed)
	var sobol_2: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, seed)
	
	assert_int(sobol_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(sobol_1[i]).is_equal_approx(sobol_2[i], 0.0000001)
	
	# Test LATIN_HYPERCUBE reproducibility
	var lhs_1: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)
	var lhs_2: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)
	
	assert_int(lhs_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(lhs_1[i]).is_equal_approx(lhs_2[i], 0.0000001)


func test_generate_samples_1d_latin_hypercube_stratification() -> void:
	var ndraws: int = 20
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 123)
	
	var sorted_samples: Array[float] = samples.duplicate()
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
		var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, method)
		assert_int(samples.size()).is_equal(ndraws)
		for sample_vec in samples:
			assert_float(sample_vec.x).is_between(0.0, 1.0)
			assert_float(sample_vec.y).is_between(0.0, 1.0)


func test_generate_samples_2d_sobol_deterministic() -> void:
	var ndraws: int = 5
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var expected_sobol_2d: Array[Vector2] = [
		Vector2(0.0, 0.0),
		Vector2(0.5, 0.5), 
		Vector2(0.75, 0.75),
		Vector2(0.25, 0.25),
		Vector2(0.375, 0.625)
	]
	
	assert_int(samples.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(samples[i]).is_equal_approx(expected_sobol_2d[i], Vector2(0.00001, 0.00001))


# --- DISCRETE INDEX SAMPLING TESTS (sample_indices) ---

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
		[StatMath.SamplingGen.SelectionStrategy.RESERVOIR, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM]
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


# --- CARD GAME SIMULATION TESTS ---

func test_card_game_dealing() -> void:
	var deck_size: int = 52
	var hand_size: int = 5
	
	# Test different dealing strategies for card games
	var fisher_yates: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES
	)
	var reservoir: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.RESERVOIR
	)
	var selection_tracking: Array[int] = StatMath.SamplingGen.sample_indices(
		deck_size, hand_size, StatMath.SamplingGen.SelectionStrategy.SELECTION_TRACKING
	)
	
	# All strategies should deal valid hands
	var all_hands: Array = [fisher_yates, reservoir, selection_tracking]
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


# --- PERFORMANCE AND STRESS TESTS ---

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


# --- GLOBAL RNG DETERMINISM TESTS ---

func test_global_rng_determinism() -> void:
	var test_seed: int = 888
	var ndraws: int = 5
	
	# Test continuous sampling determinism
	StatMath.set_global_seed(test_seed)
	var continuous_1: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	StatMath.set_global_seed(test_seed)
	var continuous_2: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	assert_int(continuous_1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_float(continuous_1[i]).is_equal(continuous_2[i])
	
	# Test discrete sampling determinism
	StatMath.set_global_seed(test_seed)
	var discrete_1: Array[int] = StatMath.SamplingGen.sample_indices(20, 5)
	
	StatMath.set_global_seed(test_seed)
	var discrete_2: Array[int] = StatMath.SamplingGen.sample_indices(20, 5)
	
	assert_int(discrete_1.size()).is_equal(5)
	for i in range(5):
		assert_int(discrete_1[i]).is_equal(discrete_2[i])
