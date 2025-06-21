# res://addons/godot-stat-math/tests/core/sampling_gen/mathematical_property_test.gd
class_name SamplingGenMathematicalPropertyTest extends GdUnitTestSuite

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

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
		[StatMath.SamplingGen.SelectionStrategy.SELECTION_TRACKING, StatMath.SamplingGen.SamplingMethod.SOBOL]
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


# --- CARD GAME SIMULATION TESTS (updated) ---

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
	
	# For coordinated sampling, use coordinated_shuffle + slice
	var coordinated_shuffle: Array[int] = StatMath.SamplingGen.coordinated_shuffle(
		deck_size, StatMath.SamplingGen.SamplingMethod.SOBOL, 0
	)
	var coordinated: Array[int] = coordinated_shuffle.slice(0, hand_size)
	
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


## Tests mathematical properties of sampling methods
func test_sampling_uniformity_property() -> void:
	# Test that sampling methods produce uniform distributions
	var n_draws: int = 1000
	var samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(
		StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	)
	
	# Check that samples are uniformly distributed in [0,1]
	var min_val: float = StatMath.BasicStats.minimum(samples)
	var max_val: float = StatMath.BasicStats.maximum(samples)
	
	assert_float(min_val).is_greater_equal(0.0)
	assert_float(max_val).is_less_equal(1.0)
	
	# Check that samples are well-distributed (not clustered)
	var mean_val: float = StatMath.BasicStats.mean(samples)
	assert_float(mean_val).is_equal_approx(0.5, 0.1)  # Should be close to 0.5 for uniform distribution

## Tests deterministic properties of quasi-random sequences
func test_quasi_random_determinism() -> void:
	# Test that quasi-random sequences are deterministic
	var n_draws: int = 10
	var samples1: Array[float] = StatMath.HelperFunctions.convert_to_float_array(
		StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	)
	var samples2: Array[float] = StatMath.HelperFunctions.convert_to_float_array(
		StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	)
	
	# Should be identical
	for i in range(n_draws):
		assert_float(samples1[i]).is_equal_approx(samples2[i], StatMath.DETERMINISM_TOLERANCE)


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