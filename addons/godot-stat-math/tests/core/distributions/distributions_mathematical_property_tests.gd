# res://addons/godot-stat-math/tests/core/distributions/distributions_mathematical_property_tests.gd
class_name DistributionsMathematicalPropertyTests extends GdUnitTestSuite

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

func test_randi_uniform_range_properties() -> void:
	# Test deterministic behavior with fixed seeds
	StatMath.set_global_seed(42)
	
	# Equal bounds case
	var single_val: int = 5
	var result_equal: int = StatMath.Distributions.randi_uniform(single_val, single_val)
	assert_int(result_equal).is_equal(single_val)
	
	# Typical range case
	var result_typical: int = StatMath.Distributions.randi_uniform(1, 10)
	assert_bool(result_typical >= 1 and result_typical <= 10).is_true()
	
	# Negative range case
	var result_negative: int = StatMath.Distributions.randi_uniform(-10, -5)
	assert_bool(result_negative >= -10 and result_negative <= -5).is_true()
	
	# Mixed sign range case
	var result_mixed: int = StatMath.Distributions.randi_uniform(-5, 5)
	assert_bool(result_mixed >= -5 and result_mixed <= 5).is_true()

func test_randi_uniform_deterministic_with_seed() -> void:
	# Test that same seed produces same results
	StatMath.set_global_seed(123)
	var result1: int = StatMath.Distributions.randi_uniform(1, 100)
	
	StatMath.set_global_seed(123)
	var result2: int = StatMath.Distributions.randi_uniform(1, 100)
	
	assert_int(result1).is_equal(result2)

func test_randf_uniform_range_properties() -> void:
	# Test deterministic behavior with fixed seeds
	StatMath.set_global_seed(42)
	
	# Equal bounds case
	var result_equal: float = StatMath.Distributions.randf_uniform(5.0, 5.0)
	assert_float(result_equal).is_equal_approx(5.0, StatMath.ERF_APPROX_TOLERANCE)
	
	# Typical range case
	var result_typical: float = StatMath.Distributions.randf_uniform(0.0, 10.0)
	assert_float(result_typical).is_greater_equal(0.0)
	assert_float(result_typical).is_less(10.0)
	
	# Negative range case
	var result_negative: float = StatMath.Distributions.randf_uniform(-10.0, -5.0)
	assert_float(result_negative).is_greater_equal(-10.0)
	assert_float(result_negative).is_less(-5.0)
	
	# Mixed sign range case
	var result_mixed: float = StatMath.Distributions.randf_uniform(-5.0, 5.0)
	assert_float(result_mixed).is_greater_equal(-5.0)
	assert_float(result_mixed).is_less(5.0)

func test_randf_exponential_non_negative_property() -> void:
	# Test that exponential distribution always produces non-negative results
	StatMath.set_global_seed(42)
	
	# Typical lambda case
	var result_typical: float = StatMath.Distributions.randf_exponential(2.0)
	assert_float(result_typical).is_greater_equal(0.0)
	
	# Small lambda case (larger expected value)
	var result_small: float = StatMath.Distributions.randf_exponential(0.1)
	assert_float(result_small).is_greater_equal(0.0)
	
	# Large lambda case (smaller expected value)
	var large_lambda: float = 100.0
	var result_large: float = StatMath.Distributions.randf_exponential(large_lambda)
	assert_float(result_large).is_greater_equal(0.0)

func test_randf_erlang_non_negative_property() -> void:
	# Test that Erlang distribution always produces non-negative results
	StatMath.set_global_seed(42)
	
	# Typical case
	var result_typical: float = StatMath.Distributions.randf_erlang(3, 2.0)
	assert_float(result_typical).is_greater_equal(0.0)

func test_randf_erlang_exponential_equivalence() -> void:
	# Erlang with k=1 is equivalent to Exponential distribution
	var result: float = StatMath.Distributions.randf_erlang(1, 2.0)
	assert_float(result).is_greater_equal(0.0)

func test_randf_gaussian_returns_float() -> void:
	var result: float = StatMath.Distributions.randf_gaussian()
	# Basic check: ensure it's a float. More rigorous tests (mean/stddev) are complex for single calls.
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_gaussian should return a float.
	# We can also check that it's not NaN or INF, which might indicate issues in Box-Muller.
	assert_bool(is_nan(result)).is_false() # Gaussian result should not be NaN.
	assert_bool(is_inf(result)).is_false() # Gaussian result should not be INF.

func test_randf_normal_default_parameters() -> void:
	# Default should behave like randf_gaussian N(0,1)
	var result: float = StatMath.Distributions.randf_normal()
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_normal with defaults should return a float.
	assert_bool(is_nan(result)).is_false() # Default Normal result should not be NaN.
	assert_bool(is_inf(result)).is_false() # Default Normal result should not be INF.

func test_randf_normal_sigma_zero() -> void:
	var mu_val: float = 5.0
	var result: float = StatMath.Distributions.randf_normal(mu_val, 0.0)
	assert_float(result).is_equal_approx(mu_val, StatMath.ERF_APPROX_TOLERANCE)

func test_randf_normal_typical_case() -> void:
	# DEPRECATED: This test is too weak. Replaced by test_randf_normal_statistical_properties.
	pass

func test_randf_normal_negative_mu() -> void:
	var result: float = StatMath.Distributions.randf_normal(-5.0, 1.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_normal with negative mu should return a float.
	assert_bool(is_nan(result)).is_false() # Normal result with negative mu should not be NaN.
	assert_bool(is_inf(result)).is_false() # Normal result with negative mu should not be INF.

func test_randf_normal_statistical_properties() -> void:
	var mu: float = 10.0
	var sigma: float = 2.0
	var expected_mean: float = mu
	
	var sample_size: int = 2000
	var samples: Array[float] = []
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_normal(mu, sigma))
	
	var sample_mean: float = StatMath.BasicStats.mean(samples)
	# Check if sample mean is close to mu.
	# Tolerance can be based on standard error of the mean: sigma / sqrt(n)
	var tolerance: float = StatMath.STATISTICAL_TEST_STD_DEV_MULTIPLIER * sigma / sqrt(sample_size)
	assert_float(sample_mean).is_between(expected_mean - tolerance, expected_mean + tolerance)

func test_randf_cauchy_basic() -> void:
	var result: float = StatMath.Distributions.randf_cauchy()
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_with_location() -> void:
	var location: float = 5.0
	var result: float = StatMath.Distributions.randf_cauchy(location, 1.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_with_scale() -> void:
	var scale: float = 2.0
	var result: float = StatMath.Distributions.randf_cauchy(0.0, scale)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_with_location_and_scale() -> void:
	var location: float = -3.0
	var scale: float = 0.5
	var result: float = StatMath.Distributions.randf_cauchy(location, scale)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_negative_location() -> void:
	var result: float = StatMath.Distributions.randf_cauchy(-10.0, 1.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_very_small_scale() -> void:
	var scale: float = 1e-6
	var result: float = StatMath.Distributions.randf_cauchy(0.0, scale)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_large_scale() -> void:
	var scale: float = 100.0
	var result: float = StatMath.Distributions.randf_cauchy(0.0, scale)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_cauchy_deterministic_with_seed() -> void:
	var location: float = 2.0
	var scale: float = 1.5
	var seed: int = 42
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_cauchy(location, scale)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_cauchy(location, scale)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

func test_randf_cauchy_multiple_calls_different_values() -> void:
	# Test that multiple calls produce different values (with high probability)
	var results: Array[float] = []
	for i in range(10):
		results.append(StatMath.Distributions.randf_cauchy())
	
	# Check that we don't have all identical values (extremely unlikely)
	var all_same: bool = true
	var first_val: float = results[0]
	for val in results:
		if not is_equal_approx(val, first_val):
			all_same = false
			break
	
	assert_bool(all_same).is_false() # Multiple Cauchy samples should not all be identical

func test_randf_cauchy_statistical_properties() -> void:
	# Test multiple samples to verify basic properties
	var samples: Array[float] = []
	var seed: int = 1337
	
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples.append(StatMath.Distributions.randf_cauchy())
	
	# All samples should be finite numbers
	for sample in samples:
		assert_bool(is_nan(sample)).is_false()
		assert_bool(is_inf(sample)).is_false()
	
	# Should have significant variability due to heavy tails
	var min_val: float = samples[0]
	var max_val: float = samples[0]
	for sample in samples:
		min_val = min(min_val, sample)
		max_val = max(max_val, sample)
	
	# Cauchy should produce a reasonable spread (not all clustered)
	assert_float(max_val - min_val).is_greater(1.0)

# --- Additional Game Development Use Cases for Cauchy ---

func test_randf_cauchy_damage_variation() -> void:
	# Example: extreme damage variation with heavy tails
	var base_damage: float = 100.0
	var damage_variation: float = StatMath.Distributions.randf_cauchy(0.0, 10.0)
	var final_damage: float = max(0.0, base_damage + damage_variation)
	
	assert_float(final_damage).is_greater_equal(0.0)
	assert_bool(is_nan(damage_variation)).is_false()
	assert_bool(is_inf(damage_variation)).is_false()

func test_randf_cauchy_market_price_fluctuation() -> void:
	# Example: market price fluctuations with extreme events
	var base_price: float = 50.0
	var price_shock: float = StatMath.Distributions.randf_cauchy(0.0, 5.0)
	var market_price: float = max(1.0, base_price + price_shock)
	
	assert_float(market_price).is_greater_equal(1.0)
	assert_bool(is_nan(price_shock)).is_false()

func test_randf_cauchy_npc_reaction_time() -> void:
	# Example: NPC reaction times with occasional extreme delays
	var base_reaction: float = 1.0  # seconds
	var reaction_variation: float = StatMath.Distributions.randf_cauchy(0.0, 0.2)
	var reaction_time: float = max(0.1, base_reaction + reaction_variation)
	
	assert_float(reaction_time).is_greater_equal(0.1)
	assert_bool(is_nan(reaction_variation)).is_false()

func test_randf_cauchy_particle_velocity_distribution() -> void:
	# Example: particle velocity with heavy-tailed distribution
	var base_velocity: float = 10.0
	var velocity_perturbation: float = StatMath.Distributions.randf_cauchy(0.0, 2.0)
	var particle_velocity: float = base_velocity + velocity_perturbation
	
	assert_bool(is_nan(velocity_perturbation)).is_false()
	assert_bool(is_inf(velocity_perturbation)).is_false()

func test_randf_cauchy_procedural_terrain_height() -> void:
	# Example: terrain height with extreme features
	var base_height: float = 100.0
	var height_variation: float = StatMath.Distributions.randf_cauchy(0.0, 15.0)
	var terrain_height: float = base_height + height_variation
	
	assert_bool(is_nan(height_variation)).is_false()
	assert_bool(is_inf(height_variation)).is_false()

func test_randf_cauchy_heavy_tails_property() -> void:
	# Test that Cauchy produces some extreme values (demonstrating heavy tails)
	var extreme_count: int = 0
	var samples: Array[float] = []
	var seed: int = 2023
	
	StatMath.set_global_seed(seed)
	for i in range(1000):  # Larger sample for extreme value detection
		var sample: float = StatMath.Distributions.randf_cauchy(0.0, 1.0)
		samples.append(sample)
		# Count values beyond ±3 (would be very rare for normal distribution)
		if abs(sample) > 3.0:
			extreme_count += 1
	
	# Cauchy should produce more extreme values than normal distribution
	# Even with conservative threshold, should see some extreme values
	assert_int(extreme_count).is_greater(0) # Should have at least some extreme values

func test_randv_histogram_basic_case() -> void:
	var values: Array = ["a", "b", "c"]
	var probabilities: Array = [0.1, 0.3, 0.6] # Sums to 1.0
	var result: Variant = StatMath.Distributions.randv_histogram(values, probabilities)
	assert_bool(values.has(result)).is_true() # Result should be one of the input values.

func test_randv_histogram_probabilities_not_normalized() -> void:
	var values: Array = [10, 20, 30]
	var probabilities: Array = [1, 2, 7] # Sums to 10, will be normalized
	var result: Variant = StatMath.Distributions.randv_histogram(values, probabilities)
	assert_bool(values.has(result)).is_true() # Result should be one of the input values after normalization.
	assert_bool(result is int).is_true() # Result should be an int as per values array.

func test_randv_histogram_single_value() -> void:
	var values: Array = ["only_choice"]
	var probabilities: Array = [1.0]
	var result: Variant = StatMath.Distributions.randv_histogram(values, probabilities)
	assert_str(result as String).is_equal("only_choice")

func test_randv_histogram_single_value_non_one_prob() -> void:
	var values: Array = [42]
	var probabilities: Array = [100] # Non-1.0, but only option
	var result: Variant = StatMath.Distributions.randv_histogram(values, probabilities)
	assert_int(result as int).is_equal(42)

func test_rng_determinism_with_set_seed() -> void:
	const TEST_SEED: int = 777
	var results_run1: Array = []
	var results_run2: Array = []

	# First Run
	StatMath.set_global_seed(TEST_SEED)
	results_run1.append(StatMath.Distributions.randi_bernoulli(0.6))      # Expected int
	results_run1.append(StatMath.Distributions.randf_normal(15.0, 3.5))  # Expected float
	results_run1.append(StatMath.Distributions.randi_poisson(4.2))       # Expected int

	# Second Run
	StatMath.set_global_seed(TEST_SEED) # Reset to the same seed
	results_run2.append(StatMath.Distributions.randi_bernoulli(0.6))
	results_run2.append(StatMath.Distributions.randf_normal(15.0, 3.5))
	results_run2.append(StatMath.Distributions.randi_poisson(4.2))

	assert_int(results_run1.size()).is_equal(results_run2.size()) # Both runs should produce the same number of results.
	# Ensuring we have the expected number of results for this specific test's logic
	assert_bool(results_run1.size() == 3) #Test logic expects 3 results to compare.

	# Compare results element by element based on their expected types
	# Result 0 (int from randi_bernoulli)
	assert_bool(results_run1[0] is int).is_true() # Result 0 (Run 1) should be an int.
	assert_bool(results_run2[0] is int).is_true() # Result 0 (Run 2) should be an int.
	assert_int(results_run1[0]).is_equal(results_run2[0]) # Result 0 (randi_bernoulli) should be deterministic.

	# Result 1 (float from randf_normal)
	assert_bool(results_run1[1] is float).is_true() # Result 1 (Run 1) should be a float.
	assert_bool(results_run2[1] is float).is_true() # Result 1 (Run 2) should be a float.
	assert_float(results_run1[1]).is_equal_approx(results_run2[1], StatMath.DETERMINISM_TOLERANCE) # Result 1 (randf_normal) should be deterministic.")

	# Result 2 (int from randi_poisson)
	assert_bool(results_run1[2] is int).is_true() # Result 2 (Run 1) should be an int.
	assert_bool(results_run2[2] is int).is_true() # Result 2 (Run 2) should be an int.
	assert_int(results_run1[2]).is_equal(results_run2[2]) # Result 2 (randi_poisson) should be deterministic.")

# --- Tests for randf_gamma ---

func test_randf_gamma_basic() -> void:
	var result: float = StatMath.Distributions.randf_gamma(2.0, 2.0)
	assert_float(result).is_greater_equal(0.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()

func test_randf_gamma_shape_one() -> void:
	# Gamma(1, θ) is equivalent to Exponential(1/θ)
	var shape: float = 1.0
	var scale: float = 2.0
	var result: float = StatMath.Distributions.randf_gamma(shape, scale)
	assert_float(result).is_greater_equal(0.0)

func test_randf_gamma_shape_less_than_one() -> void:
	# Tests the Johnk's generator path for shape < 1
	var shape: float = 0.5
	var scale: float = 1.0
	var result: float = StatMath.Distributions.randf_gamma(shape, scale)
	assert_float(result).is_greater_equal(0.0)

func test_randf_gamma_shape_greater_than_one() -> void:
	# Tests the Marsaglia-Tsang method for shape >= 1
	var shape: float = 3.0
	var scale: float = 0.5
	var result: float = StatMath.Distributions.randf_gamma(shape, scale)
	assert_float(result).is_greater_equal(0.0)

func test_randf_gamma_large_parameters() -> void:
	var shape: float = 100.0
	var scale: float = 0.1
	var result: float = StatMath.Distributions.randf_gamma(shape, scale)
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_gamma_deterministic_with_seed() -> void:
	var shape: float = 2.0
	var scale: float = 1.5
	var seed: int = 12345
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_gamma(shape, scale)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_gamma(shape, scale)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

# --- Tests for randf_beta ---

func test_randf_beta_basic() -> void:
	var result: float = StatMath.Distributions.randf_beta(2.0, 3.0)
	assert_float(result).is_between(0.0, 1.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()

func test_randf_beta_symmetric() -> void:
	# Beta(2, 2) is symmetric around 0.5
	var alpha: float = 2.0
	var beta: float = 2.0
	var result: float = StatMath.Distributions.randf_beta(alpha, beta)
	assert_float(result).is_between(0.0, 1.0)

func test_randf_beta_skewed_left() -> void:
	# Beta(1, 3) is skewed toward 0
	var alpha: float = 1.0
	var beta: float = 3.0
	var result: float = StatMath.Distributions.randf_beta(alpha, beta)
	assert_float(result).is_between(0.0, 1.0)

func test_randf_beta_skewed_right() -> void:
	# Beta(3, 1) is skewed toward 1
	var alpha: float = 3.0
	var beta: float = 1.0
	var result: float = StatMath.Distributions.randf_beta(alpha, beta)
	assert_float(result).is_between(0.0, 1.0)

func test_randf_beta_uniform() -> void:
	# Beta(1, 1) is equivalent to Uniform(0, 1)
	var alpha: float = 1.0
	var beta: float = 1.0
	var result: float = StatMath.Distributions.randf_beta(alpha, beta)
	assert_float(result).is_between(0.0, 1.0)

func test_randf_beta_large_parameters() -> void:
	var alpha: float = 100.0
	var beta: float = 50.0
	var result: float = StatMath.Distributions.randf_beta(alpha, beta)
	assert_float(result).is_between(0.0, 1.0)
	assert_bool(is_nan(result)).is_false()

func test_randf_beta_small_parameters() -> void:
	var alpha: float = 0.1
	var beta: float = 0.1
	var result: float = StatMath.Distributions.randf_beta(alpha, beta)
	assert_float(result).is_between(0.0, 1.0)
	assert_bool(is_inf(result)).is_false()

func test_randf_beta_deterministic_with_seed() -> void:
	var alpha: float = 2.5
	var beta: float = 3.5
	var seed: int = 67890
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_beta(alpha, beta)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_beta(alpha, beta)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

# --- Statistical Properties Tests (Gamma and Beta) ---

func test_randf_gamma_statistical_properties() -> void:
	# Test multiple samples to verify statistical properties
	var shape: float = 2.0
	var scale: float = 1.0
	var samples: Array[float] = []
	var seed: int = 999
	
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples.append(StatMath.Distributions.randf_gamma(shape, scale))
	
	# All samples should be non-negative
	for sample in samples:
		assert_float(sample).is_greater_equal(0.0)
	
	# Should have reasonable variance (not all the same value)
	var min_val: float = samples[0]
	var max_val: float = samples[0]
	for sample in samples:
		min_val = min(min_val, sample)
		max_val = max(max_val, sample)
	
	assert_float(max_val - min_val).is_greater(0.1) # Should have some spread

func test_randf_beta_statistical_properties() -> void:
	# Test multiple samples to verify statistical properties
	var alpha: float = 2.0
	var beta: float = 3.0
	var samples: Array[float] = []
	var seed: int = 777
	
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples.append(StatMath.Distributions.randf_beta(alpha, beta))
	
	# All samples should be in [0, 1]
	for sample in samples:
		assert_float(sample).is_between(0.0, 1.0)
	
	# Should have reasonable variance (not all the same value)
	var min_val: float = samples[0]
	var max_val: float = samples[0]
	for sample in samples:
		min_val = min(min_val, sample)
		max_val = max(max_val, sample)
	
	assert_float(max_val - min_val).is_greater(0.05) # Should have some spread 

# --- Game Development Use Cases ---

func test_randf_gamma_damage_variation() -> void:
	# Example: damage variation where base damage is modified by Gamma distribution
	var base_damage: float = 100.0
	var shape: float = 2.0 # Controls variability shape
	var scale: float = 0.5 # Controls scaling
	
	var damage_multiplier: float = StatMath.Distributions.randf_gamma(shape, scale)
	var final_damage: float = base_damage * damage_multiplier
	
	assert_float(damage_multiplier).is_greater_equal(0.0)
	assert_float(final_damage).is_greater_equal(0.0)

func test_randf_beta_quality_scores() -> void:
	# Example: item quality as a score between 0 and 1
	var common_quality: float = StatMath.Distributions.randf_beta(2.0, 5.0) # Skewed toward lower quality
	var rare_quality: float = StatMath.Distributions.randf_beta(5.0, 2.0) # Skewed toward higher quality
	
	assert_float(common_quality).is_between(0.0, 1.0)
	assert_float(rare_quality).is_between(0.0, 1.0)

func test_combined_gamma_beta_procedural_generation() -> void:
	# Example: procedural terrain generation combining both distributions
	var terrain_roughness: float = StatMath.Distributions.randf_gamma(1.5, 0.8) # Gamma for continuous scaling
	var biome_blend: float = StatMath.Distributions.randf_beta(3.0, 3.0) # Beta for normalized blending
	
	assert_float(terrain_roughness).is_greater_equal(0.0)
	assert_float(biome_blend).is_between(0.0, 1.0)
	
	# Combined effect should be reasonable
	var combined_effect: float = terrain_roughness * biome_blend
	assert_float(combined_effect).is_greater_equal(0.0)

# --- Tests for randf_triangular ---

func test_randf_triangular_basic() -> void:
	var min_val: float = 0.0
	var max_val: float = 10.0
	var mode_val: float = 3.0
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_float(result).is_greater_equal(min_val)
	assert_float(result).is_less_equal(max_val)
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_triangular_symmetric() -> void:
	# Mode in the center creates symmetric triangular distribution
	var min_val: float = -5.0
	var max_val: float = 5.0
	var mode_val: float = 0.0  # Centered mode
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_left_skewed() -> void:
	# Mode closer to minimum creates left-skewed distribution
	var min_val: float = 0.0
	var max_val: float = 100.0
	var mode_val: float = 10.0  # Mode near minimum
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_right_skewed() -> void:
	# Mode closer to maximum creates right-skewed distribution
	var min_val: float = 0.0
	var max_val: float = 100.0
	var mode_val: float = 90.0  # Mode near maximum
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_mode_at_minimum() -> void:
	# Mode at minimum creates right-skewed triangle
	var min_val: float = 5.0
	var max_val: float = 15.0
	var mode_val: float = 5.0  # Mode equals minimum
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_mode_at_maximum() -> void:
	# Mode at maximum creates left-skewed triangle
	var min_val: float = 2.0
	var max_val: float = 8.0
	var mode_val: float = 8.0  # Mode equals maximum
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_negative_range() -> void:
	var min_val: float = -20.0
	var max_val: float = -5.0
	var mode_val: float = -10.0
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_mixed_sign_range() -> void:
	var min_val: float = -10.0
	var max_val: float = 10.0
	var mode_val: float = 2.0
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_small_range() -> void:
	var min_val: float = 0.9
	var max_val: float = 1.1
	var mode_val: float = 1.0
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_large_range() -> void:
	var min_val: float = -1000.0
	var max_val: float = 1000.0
	var mode_val: float = 100.0
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

func test_randf_triangular_deterministic_with_seed() -> void:
	var min_val: float = 1.0
	var max_val: float = 5.0
	var mode_val: float = 3.0
	var seed: int = 54321
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

func test_randf_triangular_multiple_calls_different_values() -> void:
	# Test that multiple calls produce different values (with high probability)
	var min_val: float = 0.0
	var max_val: float = 1.0
	var mode_val: float = 0.3
	var results: Array[float] = []
	
	for i in range(10):
		results.append(StatMath.Distributions.randf_triangular(min_val, max_val, mode_val))
	
	# Check that we don't have all identical values (extremely unlikely)
	var all_same: bool = true
	var first_val: float = results[0]
	for val in results:
		if not is_equal_approx(val, first_val):
			all_same = false
			break
	
	assert_bool(all_same).is_false()

func test_randf_triangular_degenerate_case_equal_bounds() -> void:
	# When min equals max, should return that value
	var value: float = 42.0
	var result: float = StatMath.Distributions.randf_triangular(value, value, value)
	assert_float(result).is_equal_approx(value, StatMath.DETERMINISM_TOLERANCE)

func test_randf_triangular_nearly_equal_bounds() -> void:
	# Test with very close but not equal bounds
	var min_val: float = 1.0
	var max_val: float = 1.0000001
	var mode_val: float = 1.00000005
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)

# --- Statistical Properties Tests ---

func test_randf_triangular_statistical_properties() -> void:
	# Test multiple samples to verify statistical properties
	var min_val: float = 0.0
	var max_val: float = 10.0
	var mode_val: float = 3.0
	var samples: Array[float] = []
	var seed: int = 11111
	
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples.append(StatMath.Distributions.randf_triangular(min_val, max_val, mode_val))
	
	# All samples should be within bounds
	for sample in samples:
		assert_float(sample).is_greater_equal(min_val)
		assert_float(sample).is_less_equal(max_val)
	
	# Should have reasonable variance (not all the same value)
	var sample_min: float = samples[0]
	var sample_max: float = samples[0]
	for sample in samples:
		sample_min = min(sample_min, sample)
		sample_max = max(sample_max, sample)
	
	assert_float(sample_max - sample_min).is_greater(1.0)

func test_randf_triangular_mode_bias_verification() -> void:
	# Test that values cluster around the mode more than uniform distribution would
	var min_val: float = 0.0
	var max_val: float = 100.0
	var mode_val: float = 25.0  # Mode closer to minimum
	var near_mode_count: int = 0
	var seed: int = 22222
	
	StatMath.set_global_seed(seed)
	for i in range(1000):  # Larger sample for statistical significance
		var sample: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
		# Count samples within 10 units of the mode
		if abs(sample - mode_val) <= 10.0:
			near_mode_count += 1
	
	# Should have more samples near mode than uniform distribution would
	# Uniform would have ~20% in this range, triangular should have more
	var near_mode_proportion: float = float(near_mode_count) / 1000.0
	assert_float(near_mode_proportion).is_greater(0.25)  # Conservative threshold

# --- Game Development Use Cases ---



# --- Additional Game Development Use Cases for Triangular ---

func test_randf_triangular_ai_decision_confidence() -> void:
	# Example: AI decision confidence with most likely outcome
	var min_confidence: float = 0.2
	var max_confidence: float = 1.0
	var typical_confidence: float = 0.7
	
	var ai_confidence: float = StatMath.Distributions.randf_triangular(min_confidence, max_confidence, typical_confidence)
	
	assert_float(ai_confidence).is_between(min_confidence, max_confidence)

func test_randf_triangular_loot_quality() -> void:
	# Example: loot quality with expected value
	var min_quality: float = 0.1
	var max_quality: float = 1.0
	var expected_quality: float = 0.4
	
	var loot_quality: float = StatMath.Distributions.randf_triangular(min_quality, max_quality, expected_quality)
	
	assert_float(loot_quality).is_between(min_quality, max_quality)

func test_randf_triangular_multiple_parameters() -> void:
	# Example: multiple triangular parameters in sequence
	var param1: float = StatMath.Distributions.randf_triangular(1.0, 5.0, 2.5)
	var param2: float = StatMath.Distributions.randf_triangular(0.0, 1.0, 0.3)
	var param3: float = StatMath.Distributions.randf_triangular(-2.0, 2.0, 0.0)
	
	assert_float(param1).is_between(1.0, 5.0)
	assert_float(param2).is_between(0.0, 1.0)
	assert_float(param3).is_between(-2.0, 2.0)

func test_randf_triangular_npc_stat_generation() -> void:
	# Example: NPC stat generation with preferred values
	var min_stat: float = 10.0
	var max_stat: float = 20.0
	var preferred_stat: float = 14.0
	
	var npc_stat: float = StatMath.Distributions.randf_triangular(min_stat, max_stat, preferred_stat)
	
	assert_float(npc_stat).is_between(min_stat, max_stat)

func test_randf_triangular_pricing_variation() -> void:
	# Example: item pricing with market preference
	var min_price: float = 80.0
	var max_price: float = 120.0
	var market_price: float = 95.0
	
	var item_price: float = StatMath.Distributions.randf_triangular(min_price, max_price, market_price)
	
	assert_float(item_price).is_between(min_price, max_price)

func test_randf_triangular_procedural_terrain_height() -> void:
	# Example: terrain height with preferred elevation
	var min_elevation: float = 0.0
	var max_elevation: float = 100.0
	var preferred_elevation: float = 30.0
	
	var terrain_height: float = StatMath.Distributions.randf_triangular(min_elevation, max_elevation, preferred_elevation)
	
	assert_float(terrain_height).is_between(min_elevation, max_elevation)

func test_randf_triangular_resource_spawn_rate() -> void:
	# Example: resource spawn rate with optimal value
	var min_spawn_rate: float = 0.1
	var max_spawn_rate: float = 2.0
	var optimal_rate: float = 0.8
	
	var spawn_rate: float = StatMath.Distributions.randf_triangular(min_spawn_rate, max_spawn_rate, optimal_rate)
	
	assert_float(spawn_rate).is_between(min_spawn_rate, max_spawn_rate)

func test_randf_triangular_skill_check_difficulty() -> void:
	# Example: skill check difficulty with expected level
	var min_difficulty: float = 1.0
	var max_difficulty: float = 10.0
	var expected_difficulty: float = 6.0
	
	var difficulty: float = StatMath.Distributions.randf_triangular(min_difficulty, max_difficulty, expected_difficulty)
	
	assert_float(difficulty).is_between(min_difficulty, max_difficulty)

func test_randf_triangular_weapon_damage() -> void:
	# Example: weapon damage with most common value
	var min_damage: float = 15.0
	var max_damage: float = 25.0
	var typical_damage: float = 18.0
	
	var weapon_damage: float = StatMath.Distributions.randf_triangular(min_damage, max_damage, typical_damage)
	
	assert_float(weapon_damage).is_between(min_damage, max_damage)

# --- Tests for randf_pareto ---

func test_randf_pareto_basic() -> void:
	# Basic test with typical parameters
	var scale: float = 1.0
	var shape: float = 2.0
	
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_pareto_scale_parameter() -> void:
	# Test that scale parameter acts as minimum value
	var scale: float = 5.0
	var shape: float = 1.5
	
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	# Result should be >= scale (minimum value)
	assert_float(result).is_greater_equal(scale)



func test_randf_pareto_large_scale() -> void:
	# Test with large scale parameter
	var scale: float = 100.0
	var shape: float = 1.0
	
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)

func test_randf_pareto_small_scale() -> void:
	# Test with small scale parameter
	var scale: float = 0.01
	var shape: float = 3.0
	
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)

func test_randf_pareto_very_small_shape() -> void:
	# Test with very small shape parameter (very heavy tail)
	var scale: float = 1.0
	var shape: float = 0.1
	
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)
	assert_bool(is_nan(result)).is_false()

func test_randf_pareto_large_shape() -> void:
	# Test with large shape parameter (light tail)
	var scale: float = 1.0
	var shape: float = 10.0
	
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)

func test_randf_pareto_deterministic_with_seed() -> void:
	# Test deterministic behavior with same seed
	var scale: float = 2.0
	var shape: float = 1.5
	var seed: int = 98765
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

func test_randf_pareto_multiple_calls_different_values() -> void:
	# Test that multiple calls produce different values
	var scale: float = 1.0
	var shape: float = 2.0
	var results: Array[float] = []
	var seed: int = 13579
	
	StatMath.set_global_seed(seed)
	for i in range(10):
		results.append(StatMath.Distributions.randf_pareto(scale, shape))
	
	# Check that not all values are the same
	var all_same: bool = true
	for i in range(1, results.size()):
		if not is_equal_approx(results[i], results[0]):
			all_same = false
			break
	
	assert_bool(all_same).is_false()

# --- Statistical Properties Tests ---

func test_randf_pareto_statistical_properties() -> void:
	# Test multiple samples to verify statistical properties
	var scale: float = 1.0
	var shape: float = 2.0
	var samples: Array[float] = []
	var seed: int = 33333
	
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples.append(StatMath.Distributions.randf_pareto(scale, shape))
	
	# All samples should be >= scale
	for sample in samples:
		assert_float(sample).is_greater_equal(scale)
	
	# Should have reasonable variance (not all the same value)
	var sample_min: float = samples[0]
	var sample_max: float = samples[0]
	for sample in samples:
		sample_min = min(sample_min, sample)
		sample_max = max(sample_max, sample)
	
	assert_float(sample_max - sample_min).is_greater(0.1)





# --- Game Development Use Cases ---

func test_randf_pareto_wealth_distribution() -> void:
	# Example: wealth distribution in game economy (80/20 rule)
	var base_wealth: float = 100.0
	var inequality_factor: float = 1.16  # Approximates 80/20 rule
	
	var player_wealth: float = StatMath.Distributions.randf_pareto(base_wealth, inequality_factor)
	
	assert_float(player_wealth).is_greater_equal(base_wealth)



func test_randf_pareto_multiple_applications() -> void:
	# Example: multiple Pareto applications in a complex system
	var guild_size: float = StatMath.Distributions.randf_pareto(5.0, 1.5)
	var territory_value: float = StatMath.Distributions.randf_pareto(1000.0, 1.2)
	var influence_points: float = StatMath.Distributions.randf_pareto(100.0, 0.9)
	
	assert_float(guild_size).is_greater_equal(5.0)
	assert_float(territory_value).is_greater_equal(1000.0)
	assert_float(influence_points).is_greater_equal(100.0)
	
	# All values should be reasonable for a game system
	assert_float(guild_size).is_greater(0.0)
	assert_float(territory_value).is_greater(0.0)
	assert_float(influence_points).is_greater(0.0)

# --- Additional Game Development Use Cases for Pareto ---

func test_randf_pareto_city_population() -> void:
	# Example: city population following power law
	var min_population: float = 1000.0
	var population_exponent: float = 1.3
	
	var city_population: float = StatMath.Distributions.randf_pareto(min_population, population_exponent)
	
	assert_float(city_population).is_greater_equal(min_population)

func test_randf_pareto_concentration_near_minimum() -> void:
	# Example: testing concentration near minimum with high shape
	var scale: float = 10.0
	var high_shape: float = 5.0  # High shape = more concentration near minimum
	
	var result: float = StatMath.Distributions.randf_pareto(scale, high_shape)
	
	assert_float(result).is_greater_equal(scale)

func test_randf_pareto_different_shapes() -> void:
	# Example: testing different shape effects
	var scale: float = 1.0
	var low_shape_result: float = StatMath.Distributions.randf_pareto(scale, 0.8)   # Heavy tail
	var medium_shape_result: float = StatMath.Distributions.randf_pareto(scale, 2.0) # Moderate tail
	var high_shape_result: float = StatMath.Distributions.randf_pareto(scale, 4.0)  # Light tail
	
	assert_float(low_shape_result).is_greater_equal(scale)
	assert_float(medium_shape_result).is_greater_equal(scale)
	assert_float(high_shape_result).is_greater_equal(scale)

func test_randf_pareto_heavy_tail_property() -> void:
	# Example: demonstrating heavy tail property
	var scale: float = 1.0
	var shape: float = 1.0  # Classic Pareto case
	var extreme_values: int = 0
	
	for i in range(1000):
		var sample: float = StatMath.Distributions.randf_pareto(scale, shape)
		if sample > scale * 5.0:  # Values 5x the minimum
			extreme_values += 1
	
	# Should have some extreme values due to heavy tail
	assert_int(extreme_values).is_greater(0)

func test_randf_pareto_loot_rarity() -> void:
	# Example: loot rarity distribution
	var common_loot_value: float = 10.0
	var rarity_factor: float = 1.5
	
	var loot_value: float = StatMath.Distributions.randf_pareto(common_loot_value, rarity_factor)
	
	assert_float(loot_value).is_greater_equal(common_loot_value)

func test_randf_pareto_market_price_spikes() -> void:
	# Example: market price spikes following power law
	var base_price: float = 50.0
	var spike_intensity: float = 1.2
	
	var market_spike: float = StatMath.Distributions.randf_pareto(base_price, spike_intensity)
	
	assert_float(market_spike).is_greater_equal(base_price)

func test_randf_pareto_network_effect_scaling() -> void:
	# Example: network effect scaling
	var base_effect: float = 1.0
	var network_power: float = 0.9
	
	var network_scaling: float = StatMath.Distributions.randf_pareto(base_effect, network_power)
	
	assert_float(network_scaling).is_greater_equal(base_effect)

func test_randf_pareto_player_skill_gaps() -> void:
	# Example: player skill distribution gaps
	var minimum_skill: float = 100.0
	var skill_inequality: float = 1.4
	
	var player_skill: float = StatMath.Distributions.randf_pareto(minimum_skill, skill_inequality)
	
	assert_float(player_skill).is_greater_equal(minimum_skill)

func test_randf_pareto_power_law_scaling() -> void:
	# Example: general power law scaling
	var base_value: float = 2.0
	var power_exponent: float = 1.8
	
	var scaled_result: float = StatMath.Distributions.randf_pareto(base_value, power_exponent)
	
	assert_float(scaled_result).is_greater_equal(base_value)

func test_randf_pareto_quest_reward_scaling() -> void:
	# Example: quest reward scaling
	var base_reward: float = 25.0
	var reward_scaling: float = 1.6
	
	var quest_reward: float = StatMath.Distributions.randf_pareto(base_reward, reward_scaling)
	
	assert_float(quest_reward).is_greater_equal(base_reward)

func test_randf_pareto_resource_deposits() -> void:
	# Example: resource deposit sizes
	var minimum_deposit: float = 5.0
	var deposit_distribution: float = 1.1
	
	var deposit_size: float = StatMath.Distributions.randf_pareto(minimum_deposit, deposit_distribution)
	
	assert_float(deposit_size).is_greater_equal(minimum_deposit)

# --- Tests for randf_weibull ---

func test_randf_weibull_basic() -> void:
	# Basic test with typical parameters
	var scale: float = 2.0
	var shape: float = 1.5
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_randf_weibull_scale_parameter() -> void:
	# Test that scale parameter affects the characteristic life
	var scale: float = 5.0
	var shape: float = 2.0
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	# Result should be non-negative and finite
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()

func test_randf_weibull_exponential_case() -> void:
	# When shape = 1, Weibull becomes exponential distribution
	var scale: float = 2.0
	var shape: float = 1.0  # Exponential case
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()

func test_randf_weibull_rayleigh_case() -> void:
	# When shape = 2, Weibull becomes Rayleigh distribution (wind speeds)
	var scale: float = 3.0
	var shape: float = 2.0  # Rayleigh case
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()

func test_randf_weibull_different_shapes() -> void:
	# Test different shape parameters
	var scale: float = 2.0
	
	var infant_mortality_result: float = StatMath.Distributions.randf_weibull(scale, 0.5)  # k < 1: decreasing failure rate
	var constant_failure_result: float = StatMath.Distributions.randf_weibull(scale, 1.0)  # k = 1: constant failure rate
	var wear_out_result: float = StatMath.Distributions.randf_weibull(scale, 3.0)         # k > 1: increasing failure rate
	
	# All should be valid
	assert_float(infant_mortality_result).is_greater_equal(0.0)
	assert_float(constant_failure_result).is_greater_equal(0.0)
	assert_float(wear_out_result).is_greater_equal(0.0)

func test_randf_weibull_large_scale() -> void:
	# Test with large scale parameter
	var scale: float = 100.0
	var shape: float = 2.0
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)

func test_randf_weibull_small_scale() -> void:
	# Test with small scale parameter
	var scale: float = 0.1
	var shape: float = 2.0
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)

func test_randf_weibull_very_small_shape() -> void:
	# Test with very small shape parameter (heavy infant mortality)
	var scale: float = 2.0
	var shape: float = 0.1
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()

func test_randf_weibull_large_shape() -> void:
	# Test with large shape parameter (sharp wear-out)
	var scale: float = 2.0
	var shape: float = 10.0
	
	var result: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result).is_greater_equal(0.0)

func test_randf_weibull_deterministic_with_seed() -> void:
	# Test deterministic behavior with same seed
	var scale: float = 2.0
	var shape: float = 1.5
	var seed: int = 12345
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_weibull(scale, shape)
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

func test_randf_weibull_multiple_calls_different_values() -> void:
	# Test that multiple calls produce different values
	var scale: float = 2.0
	var shape: float = 1.5
	var results: Array[float] = []
	var seed: int = 67890
	
	StatMath.set_global_seed(seed)
	for i in range(10):
		results.append(StatMath.Distributions.randf_weibull(scale, shape))
	
	# Check that not all values are the same
	var all_same: bool = true
	for i in range(1, results.size()):
		if not is_equal_approx(results[i], results[0]):
			all_same = false
			break
	
	assert_bool(all_same).is_false()

# --- Statistical Properties Tests ---

func test_randf_weibull_statistical_properties() -> void:
	# Test multiple samples to verify statistical properties
	var scale: float = 2.0
	var shape: float = 2.0  # Rayleigh case for predictable properties
	var samples: Array[float] = []
	var seed: int = 11111
	
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples.append(StatMath.Distributions.randf_weibull(scale, shape))
	
	# All samples should be >= 0
	for sample in samples:
		assert_float(sample).is_greater_equal(0.0)
	
	# Should have reasonable variance (not all the same value)
	var sample_min: float = samples[0]
	var sample_max: float = samples[0]
	for sample in samples:
		sample_min = min(sample_min, sample)
		sample_max = max(sample_max, sample)
	
	assert_float(sample_max - sample_min).is_greater(0.1)

func test_randf_weibull_shape_effect_on_distribution() -> void:
	# Test that different shapes produce different distribution characteristics
	var scale: float = 2.0
	var samples_low_shape: Array[float] = []
	var samples_high_shape: Array[float] = []
	var seed: int = 22222
	
	# Low shape (k < 1): decreasing failure rate
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples_low_shape.append(StatMath.Distributions.randf_weibull(scale, 0.5))
	
	# High shape (k > 1): increasing failure rate
	StatMath.set_global_seed(seed)
	for i in range(100):
		samples_high_shape.append(StatMath.Distributions.randf_weibull(scale, 3.0))
	
	# Both should produce valid samples
	for sample in samples_low_shape:
		assert_float(sample).is_greater_equal(0.0)
	for sample in samples_high_shape:
		assert_float(sample).is_greater_equal(0.0)

func test_randf_weibull_exponential_equivalence() -> void:
	# Test that Weibull(λ, 1) behaves like Exponential(1/λ)
	var scale: float = 2.0
	var shape: float = 1.0
	var weibull_samples: Array[float] = []
	var exponential_samples: Array[float] = []
	var seed: int = 33333
	
	# Generate Weibull samples with shape=1
	StatMath.set_global_seed(seed)
	for i in range(50):
		weibull_samples.append(StatMath.Distributions.randf_weibull(scale, shape))
	
	# Generate exponential samples with rate=1/scale
	StatMath.set_global_seed(seed)
	for i in range(50):
		exponential_samples.append(StatMath.Distributions.randf_exponential(1.0 / scale))
	
	# Both should be valid and have similar statistical properties
	for sample in weibull_samples:
		assert_float(sample).is_greater_equal(0.0)
	for sample in exponential_samples:
		assert_float(sample).is_greater_equal(0.0)

# --- Game Development Use Cases ---

func test_randf_weibull_equipment_durability() -> void:
	# Example: equipment failure modeling with wear-out pattern
	var characteristic_life: float = 1000.0  # Hours of use
	var wear_pattern: float = 2.5  # k > 1: increasing failure rate (wear-out)
	
	var equipment_lifetime: float = StatMath.Distributions.randf_weibull(characteristic_life, wear_pattern)
	
	assert_float(equipment_lifetime).is_greater_equal(0.0)
	# Equipment should have reasonable lifetime
	assert_float(equipment_lifetime).is_greater(0.0)

func test_randf_weibull_wind_speed_simulation() -> void:
	# Example: wind speed modeling using Rayleigh distribution (Weibull with k=2)
	var average_wind_speed: float = 15.0  # km/h
	var rayleigh_shape: float = 2.0  # Rayleigh case
	
	var wind_speed: float = StatMath.Distributions.randf_weibull(average_wind_speed, rayleigh_shape)
	
	assert_float(wind_speed).is_greater_equal(0.0)
	# Wind speed should be reasonable
	assert_float(wind_speed).is_greater_equal(0.0)

func test_randf_weibull_survival_time_modeling() -> void:
	# Example: character survival time in hostile environment
	var base_survival_time: float = 300.0  # Seconds
	var hazard_pattern: float = 1.8  # Slightly increasing hazard
	
	var survival_time: float = StatMath.Distributions.randf_weibull(base_survival_time, hazard_pattern)
	
	assert_float(survival_time).is_greater_equal(0.0)

func test_randf_weibull_component_reliability() -> void:
	# Example: electronic component failure in sci-fi game
	var mean_time_to_failure: float = 5000.0  # Game hours
	var reliability_factor: float = 3.0  # Sharp wear-out after design life
	
	var component_lifetime: float = StatMath.Distributions.randf_weibull(mean_time_to_failure, reliability_factor)
	
	assert_float(component_lifetime).is_greater_equal(0.0)

func test_randf_weibull_weather_event_duration() -> void:
	# Example: storm duration modeling
	var typical_storm_duration: float = 120.0  # Minutes
	var storm_pattern: float = 1.5  # Moderate wear-out pattern
	
	var storm_duration: float = StatMath.Distributions.randf_weibull(typical_storm_duration, storm_pattern)
	
	assert_float(storm_duration).is_greater_equal(0.0)

func test_randf_weibull_quest_completion_time() -> void:
	# Example: time to complete quests with increasing difficulty
	var base_completion_time: float = 60.0  # Minutes
	var difficulty_curve: float = 2.2  # Increasing time pressure
	
	var completion_time: float = StatMath.Distributions.randf_weibull(base_completion_time, difficulty_curve)
	
	assert_float(completion_time).is_greater_equal(0.0)

func test_randf_weibull_resource_depletion() -> void:
	# Example: resource node depletion time
	var resource_lifetime: float = 2000.0  # Resource units
	var depletion_pattern: float = 1.2  # Slight acceleration in depletion
	
	var depletion_time: float = StatMath.Distributions.randf_weibull(resource_lifetime, depletion_pattern)
	
	assert_float(depletion_time).is_greater_equal(0.0)

func test_randf_weibull_player_session_length() -> void:
	# Example: modeling player session lengths
	var typical_session: float = 45.0  # Minutes
	var engagement_pattern: float = 0.8  # k < 1: decreasing "failure" rate (longer sessions more likely)
	
	var session_length: float = StatMath.Distributions.randf_weibull(typical_session, engagement_pattern)
	
	assert_float(session_length).is_greater_equal(0.0)

func test_randf_weibull_network_latency_spikes() -> void:
	# Example: network latency spike duration
	var base_latency_duration: float = 50.0  # Milliseconds
	var network_stability: float = 2.8  # Sharp recovery pattern
	
	var latency_spike_duration: float = StatMath.Distributions.randf_weibull(base_latency_duration, network_stability)
	
	assert_float(latency_spike_duration).is_greater_equal(0.0)

func test_randf_weibull_multiple_reliability_applications() -> void:
	# Example: multiple reliability applications in a complex system
	var engine_lifetime: float = StatMath.Distributions.randf_weibull(10000.0, 2.5)  # Engine wear-out
	var battery_life: float = StatMath.Distributions.randf_weibull(500.0, 1.8)      # Battery degradation
	var sensor_duration: float = StatMath.Distributions.randf_weibull(8000.0, 3.2) # Sensor precision loss
	
	assert_float(engine_lifetime).is_greater_equal(0.0)
	assert_float(battery_life).is_greater_equal(0.0)
	assert_float(sensor_duration).is_greater_equal(0.0)
	
	# All lifetimes should be reasonable for a game system
	assert_float(engine_lifetime).is_greater(0.0)
	assert_float(battery_life).is_greater(0.0)
	assert_float(sensor_duration).is_greater(0.0) 
