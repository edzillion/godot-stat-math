# addons/godot-stat-math/core/distributions_test.gd
class_name DistributionsTest extends GdUnitTestSuite



func test_randi_bernoulli_p_zero() -> void:
	var result: int = StatMath.Distributions.randi_bernoulli(0.0)
	assert_int(result).is_equal(0)


func test_randi_bernoulli_p_one() -> void:
	var result: int = StatMath.Distributions.randi_bernoulli(1.0)
	assert_int(result).is_equal(1)


func test_randi_bernoulli_p_half() -> void:
	var result: int = StatMath.Distributions.randi_bernoulli(0.5)
	assert_bool(result == 0 or result == 1).is_true() # Result should be 0 or 1 for p=0.5 


func test_randi_bernoulli_invalid_p_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_bernoulli(-0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be between 0.0 and 1.0.")


func test_randi_bernoulli_invalid_p_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_bernoulli(1.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be between 0.0 and 1.0.")


# Tests for randi_binomial
func test_randi_binomial_p_zero() -> void:
	var result: int = StatMath.Distributions.randi_binomial(0.0, 10)
	assert_int(result).is_equal(0)


func test_randi_binomial_p_one() -> void:
	var n_trials: int = 5
	var result: int = StatMath.Distributions.randi_binomial(1.0, n_trials)
	assert_int(result).is_equal(n_trials)


func test_randi_binomial_n_zero() -> void:
	var result: int = StatMath.Distributions.randi_binomial(0.5, 0)
	assert_int(result).is_equal(0)


func test_randi_binomial_typical_case() -> void:
	var n_trials: int = 10
	var result: int = StatMath.Distributions.randi_binomial(0.5, n_trials)
	assert_bool(result >= 0 and result <= n_trials).is_true() # Result should be between 0 and %d % n_trials


func test_randi_binomial_invalid_p_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_binomial(-0.1, 5)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be between 0.0 and 1.0.")


func test_randi_binomial_invalid_p_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_binomial(1.1, 5)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be between 0.0 and 1.0.")


func test_randi_binomial_invalid_n_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_binomial(0.5, -1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Number of trials (n) must be non-negative.")


# Tests for randi_geometric
func test_randi_geometric_p_one() -> void:
	var result: int = StatMath.Distributions.randi_geometric(1.0)
	assert_int(result).is_equal(1)


func test_randi_geometric_typical_case() -> void:
	# For p=0.5, expected value is 1/0.5 = 2. Result must be >= 1.
	var result: int = StatMath.Distributions.randi_geometric(0.5)
	assert_bool(result >= 1).is_true() # Result should be at least 1 for p=0.5


func test_randi_geometric_p_very_small_expect_large_or_inf() -> void:
	# With a very small p, we expect a very large number of trials, possibly INF (int64.max).
	var p_very_small: float = 0.0000000000000001 # 1e-17
	var result: int = StatMath.Distributions.randi_geometric(p_very_small)
	# Check if it's a large positive number or int64.max if INF was cast.
	print("randi_geometric(1e-17) returned: %s (Expected large positive or int64.max)" % result)
	assert_bool(result > 1000 or result == StatMath.INT64_MAX_VAL).is_true() # Result for very small p should be very large or int64.max

	

func test_randi_geometric_invalid_p_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_geometric(0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be in (0,1].")


func test_randi_geometric_invalid_p_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_geometric(-0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be in (0,1].")


func test_randi_geometric_invalid_p_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_geometric(1.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Success probability (p) must be in (0,1].")


# Tests for randi_poisson
func test_randi_poisson_typical_case() -> void:
	# For a given lambda, the result should be non-negative.
	# For example, lambda = 3.0. Expected value is 3.
	var result: int = StatMath.Distributions.randi_poisson(3.0)
	assert_bool(result >= 0).is_true() # Result of Poisson distribution should be non-negative.


func test_randi_poisson_small_lambda() -> void:
	# Test with a small lambda, e.g., 0.1. Higher chance of getting 0.
	var result: int = StatMath.Distributions.randi_poisson(0.1)
	assert_bool(result >= 0).is_true() # Result of Poisson distribution should be non-negative, even for small lambda.


func test_randi_poisson_invalid_lambda_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_poisson(0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive.")


func test_randi_poisson_invalid_lambda_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_poisson(-1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive.")


# Tests for randi_pseudo
func test_randi_pseudo_c_param_one() -> void:
	var result: int = StatMath.Distributions.randi_pseudo(1.0)
	assert_int(result).is_equal(0) # "If c_param is 1.0, loop condition current_c < 1.0 is initially false, trials should be 0."


func test_randi_pseudo_typical_case() -> void:
	# For c_param = 0.3, max trials is 3 (0.3 -> 0.6 -> 0.9, then current_c becomes 1.2)
	# Trial can be 1, 2, or 3.
	var result: int = StatMath.Distributions.randi_pseudo(0.3)
	assert_bool(result >= 1 and result <= 3).is_true() # For c_param=0.3, result should be 1, 2, or 3.


func test_randi_pseudo_c_param_half() -> void:
	# c_param = 0.5. trial = 0. current_c = 0.5
	# Loop 1: 0.5 < 1.0. trial = 1. randi_bernoulli(0.5). 
	# If success, returns 1. 
	# If fail, current_c = 1.0. Loop 1.0 < 1.0 is false. Returns 1.
	# So, should always return 1.
	var result: int = StatMath.Distributions.randi_pseudo(0.5)
	assert_int(result).is_equal(1) # "For c_param=0.5, result should always be 1."


func test_randi_pseudo_invalid_c_param_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_pseudo(0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Probability increment (c_param) must be in (0.0, 1.0].")


func test_randi_pseudo_invalid_c_param_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_pseudo(-0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Probability increment (c_param) must be in (0.0, 1.0].")


func test_randi_pseudo_invalid_c_param_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_pseudo(1.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Probability increment (c_param) must be in (0.0, 1.0].")


# Tests for randi_seige
func test_randi_seige_initial_capture_guaranteed() -> void:
	# c_0 = 1.0, so capture should happen on the first trial.
	var result: int = StatMath.Distributions.randi_seige(0.5, 1.0, 0.1, -0.1)
	assert_int(result).is_equal(1) # "If c_0 is 1.0, trials should be 1."


func test_randi_seige_capture_after_one_guaranteed_win() -> void:
	# w=1.0 (always win), c_0=0.0, c_win=1.0. Should capture on trial 1.
	var result: int = StatMath.Distributions.randi_seige(1.0, 0.0, 1.0, 0.0)
	assert_int(result).is_equal(1) # "Guaranteed win leading to guaranteed capture should result in 1 trial."


func test_randi_seige_typical_case() -> void:
	# A general case, result should be >= 1.
	var result: int = StatMath.Distributions.randi_seige(0.5, 0.1, 0.2, -0.05)
	assert_bool(result >= 1).is_true() # Result of randi_seige should be at least 1 trial.


func test_randi_seige_no_change_eventually_captures() -> void:
	# If c_win and c_lose are 0, but c_0 is > 0, it should eventually capture.
	# This relies on randi_bernoulli(c_0) eventually returning 1.
	# This test might be flaky or long if c_0 is small. For c_0 = 0.1, it will take some trials.
	var result: int = StatMath.Distributions.randi_seige(0.5, 0.1, 0.0, 0.0)
	assert_bool(result >= 1).is_true() # With c_0 > 0 and no change, should eventually capture.


func test_randi_seige_invalid_w_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(-0.1, 0.5, 0.1, -0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Parameter w (win probability) must be between 0.0 and 1.0.")


func test_randi_seige_invalid_w_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(1.1, 0.5, 0.1, -0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Parameter w (win probability) must be between 0.0 and 1.0.")


func test_randi_seige_invalid_c0_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(0.5, -0.1, 0.1, -0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Parameter c_0 (initial capture probability) must be between 0.0 and 1.0.")


func test_randi_seige_invalid_c0_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(0.5, 1.1, 0.1, -0.1)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Parameter c_0 (initial capture probability) must be between 0.0 and 1.0.")


# Tests for randf_uniform
func test_randf_uniform_a_equals_b() -> void:
	var val: float = 5.0
	var result: float = StatMath.Distributions.randf_uniform(val, val)
	assert_float(result).is_equal_approx(val, 0.00001)


func test_randf_uniform_typical_case() -> void:
	var a: float = 2.0
	var b: float = 5.0
	var result: float = StatMath.Distributions.randf_uniform(a, b)
	assert_float(result).is_greater_equal(a)
	assert_float(result).is_less(b)


func test_randf_uniform_negative_range() -> void:
	var a: float = -5.0
	var b: float = -2.0
	var result: float = StatMath.Distributions.randf_uniform(a, b)
	assert_float(result).is_greater_equal(a)
	assert_float(result).is_less(b)


func test_randf_uniform_mixed_sign_range() -> void:
	var a: float = -3.0
	var b: float = 3.0
	var result: float = StatMath.Distributions.randf_uniform(a, b)
	assert_float(result).is_greater_equal(a)
	assert_float(result).is_less(b)


func test_randf_uniform_invalid_a_greater_than_b() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_uniform(5.0, 2.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Lower bound (a) must be less than or equal to upper bound (b) for Uniform distribution.")


# Tests for randf_exponential
func test_randf_exponential_typical_case() -> void:
	# Exponential distribution should produce non-negative results.
	var result: float = StatMath.Distributions.randf_exponential(2.0)
	assert_float(result).is_greater_equal(0.0)


func test_randf_exponential_small_lambda() -> void:
	# Small lambda means larger expected value, still non-negative.
	var result: float = StatMath.Distributions.randf_exponential(0.1)
	assert_float(result).is_greater_equal(0.0)


func test_randf_exponential_large_lambda() -> void:
	# Large lambda means smaller expected value, still non-negative.
	var result: float = StatMath.Distributions.randf_exponential(100.0)
	assert_float(result).is_greater_equal(0.0)


func test_randf_exponential_invalid_lambda_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_exponential(0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive for Exponential distribution.")


func test_randf_exponential_invalid_lambda_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_exponential(-1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive for Exponential distribution.")


# Tests for randf_erlang
func test_randf_erlang_typical_case() -> void:
	# Erlang distribution should produce non-negative results.
	var result: float = StatMath.Distributions.randf_erlang(3, 2.0)
	assert_float(result).is_greater_equal(0.0)


func test_randf_erlang_k_one() -> void:
	# Erlang with k=1 is equivalent to Exponential distribution.
	var result: float = StatMath.Distributions.randf_erlang(1, 2.0)
	assert_float(result).is_greater_equal(0.0)


func test_randf_erlang_invalid_k_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(0, 2.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Shape parameter (k) must be a positive integer for Erlang distribution.")


func test_randf_erlang_invalid_k_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(-1, 2.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Shape parameter (k) must be a positive integer for Erlang distribution.")


func test_randf_erlang_invalid_lambda_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(3, 0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive for Erlang distribution.")


func test_randf_erlang_invalid_lambda_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(3, -1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive for Erlang distribution.")


# Tests for randf_gaussian (Standard Normal N(0,1))
func test_randf_gaussian_returns_float() -> void:
	var result: float = StatMath.Distributions.randf_gaussian()
	# Basic check: ensure it's a float. More rigorous tests (mean/stddev) are complex for single calls.
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_gaussian should return a float.
	# We can also check that it's not NaN or INF, which might indicate issues in Box-Muller.
	assert_bool(is_nan(result)).is_false() # Gaussian result should not be NaN.
	assert_bool(is_inf(result)).is_false() # Gaussian result should not be INF.


# Tests for randf_normal(mu, sigma)
func test_randf_normal_default_parameters() -> void:
	# Default should behave like randf_gaussian N(0,1)
	var result: float = StatMath.Distributions.randf_normal()
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_normal with defaults should return a float.
	assert_bool(is_nan(result)).is_false() # Default Normal result should not be NaN.
	assert_bool(is_inf(result)).is_false() # Default Normal result should not be INF.


func test_randf_normal_sigma_zero() -> void:
	var mu_val: float = 5.0
	var result: float = StatMath.Distributions.randf_normal(mu_val, 0.0)
	assert_float(result).is_equal_approx(mu_val, 0.00001)


func test_randf_normal_typical_case() -> void:
	var result: float = StatMath.Distributions.randf_normal(10.0, 2.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_normal with specific mu/sigma should return a float.
	assert_bool(is_nan(result)).is_false() # Normal result should not be NaN.
	assert_bool(is_inf(result)).is_false() # Normal result should not be INF.


func test_randf_normal_negative_mu() -> void:
	var result: float = StatMath.Distributions.randf_normal(-5.0, 1.0)
	assert_bool(typeof(result) == TYPE_FLOAT).is_true() # randf_normal with negative mu should return a float.
	assert_bool(is_nan(result)).is_false() # Normal result with negative mu should not be NaN.
	assert_bool(is_inf(result)).is_false() # Normal result with negative mu should not be INF.


func test_randf_normal_invalid_sigma_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_normal(0.0, -1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Standard deviation (sigma) must be non-negative.")


# Tests for randf_cauchy
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
	
	assert_float(result1).is_equal_approx(result2, 0.0000001)


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


func test_randf_cauchy_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_cauchy(0.0, 0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Scale parameter must be positive for Cauchy distribution.")


func test_randf_cauchy_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_cauchy(0.0, -1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Scale parameter must be positive for Cauchy distribution.")


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


# --- Game Development Use Cases for Cauchy ---

func test_randf_cauchy_damage_variation() -> void:
	# Example: critical hit damage with occasional massive spikes
	var base_damage: float = 100.0
	var damage_multiplier: float = StatMath.Distributions.randf_cauchy(1.0, 0.1)
	
	# Clamp to reasonable game bounds while preserving the concept
	damage_multiplier = clamp(damage_multiplier, 0.1, 5.0)
	var final_damage: float = base_damage * damage_multiplier
	
	assert_float(damage_multiplier).is_between(0.1, 5.0)
	assert_float(final_damage).is_greater_equal(10.0)
	assert_float(final_damage).is_less_equal(500.0)


func test_randf_cauchy_market_price_fluctuation() -> void:
	# Example: economic simulation with market crashes/booms
	var current_price: float = 50.0
	var price_change_factor: float = StatMath.Distributions.randf_cauchy(1.0, 0.02)
	
	# Cap extreme price changes for game balance
	price_change_factor = clamp(price_change_factor, 0.5, 2.0)
	var new_price: float = current_price * price_change_factor
	
	assert_float(price_change_factor).is_between(0.5, 2.0)
	assert_float(new_price).is_greater_equal(25.0)
	assert_float(new_price).is_less_equal(100.0)


func test_randf_cauchy_procedural_terrain_height() -> void:
	# Example: terrain generation with dramatic peaks/valleys
	var base_height: float = 10.0
	var height_variation: float = StatMath.Distributions.randf_cauchy(0.0, 2.0)
	
	# Clamp for reasonable terrain bounds
	height_variation = clamp(height_variation, -20.0, 20.0)
	var final_height: float = base_height + height_variation
	
	assert_float(height_variation).is_between(-20.0, 20.0)
	assert_float(final_height).is_between(-10.0, 30.0)


func test_randf_cauchy_npc_reaction_time() -> void:
	# Example: NPC reaction times with occasional extreme delays/speed
	var base_reaction_time: float = 1.0  # seconds
	var time_multiplier: float = StatMath.Distributions.randf_cauchy(1.0, 0.05)
	
	# Reasonable bounds for gameplay
	time_multiplier = clamp(time_multiplier, 0.1, 3.0)
	var final_reaction_time: float = base_reaction_time * time_multiplier
	
	assert_float(time_multiplier).is_between(0.1, 3.0)
	assert_float(final_reaction_time).is_greater_equal(0.1)
	assert_float(final_reaction_time).is_less_equal(3.0)


func test_randf_cauchy_particle_velocity_distribution() -> void:
	# Example: explosion particles with heavy-tailed velocity distribution
	var base_speed: float = 100.0
	var speed_multiplier: float = StatMath.Distributions.randf_cauchy(1.0, 0.1)
	
	# Practical bounds for particle system
	speed_multiplier = clamp(speed_multiplier, 0.1, 5.0)
	var particle_speed: float = base_speed * speed_multiplier
	
	assert_float(speed_multiplier).is_between(0.1, 5.0)
	assert_float(particle_speed).is_between(10.0, 500.0)


# Tests for randv_histogram
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


# Assertion tests for randv_histogram
func test_randv_histogram_empty_values() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram([], [1.0])
	await assert_error(test_call).is_runtime_error("Assertion failed: Values array cannot be empty.")


func test_randv_histogram_empty_probabilities() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a"], [])
	await assert_error(test_call).is_runtime_error("Assertion failed: Values and probabilities arrays must have the same size.") # Also triggers size mismatch, but this one is fine.


func test_randv_histogram_mismatched_sizes() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a", "b"], [1.0])
	await assert_error(test_call).is_runtime_error("Assertion failed: Values and probabilities arrays must have the same size.")


func test_randv_histogram_non_numeric_probability() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a"], ["not_a_number"])
	await assert_error(test_call).is_runtime_error("Assertion failed: Probabilities must be numbers (int or float).")


func test_randv_histogram_negative_probability() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a"], [-0.5])
	await assert_error(test_call).is_runtime_error("Assertion failed: Probabilities must be non-negative.")


func test_randv_histogram_zero_sum_probabilities() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a", "b"], [0.0, 0.0])
	await assert_error(test_call).is_runtime_error("Assertion failed: Sum of probabilities must be positive for normalization.") 


# --- Test for RNG Determinism ---

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
	assert_float(results_run1[1]).is_equal_approx(results_run2[1], 0.0000001) # Result 1 (randf_normal) should be deterministic.")

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
	
	assert_float(result1).is_equal_approx(result2, 0.0000001)


func test_randf_gamma_invalid_shape_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(0.0, 1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Shape parameter must be positive for Gamma distribution.")


func test_randf_gamma_invalid_shape_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(-1.0, 1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Shape parameter must be positive for Gamma distribution.")


func test_randf_gamma_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(2.0, 0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Scale parameter must be positive for Gamma distribution.")


func test_randf_gamma_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(2.0, -1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Scale parameter must be positive for Gamma distribution.")


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
	
	assert_float(result1).is_equal_approx(result2, 0.0000001)


func test_randf_beta_invalid_alpha_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(0.0, 2.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Alpha parameter must be positive for Beta distribution.")


func test_randf_beta_invalid_alpha_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(-1.0, 2.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Alpha parameter must be positive for Beta distribution.")


func test_randf_beta_invalid_beta_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(2.0, 0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Beta parameter must be positive for Beta distribution.")


func test_randf_beta_invalid_beta_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(2.0, -1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Beta parameter must be positive for Beta distribution.")


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
	
	assert_float(result1).is_equal_approx(result2, 0.0000001)


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
	assert_float(result).is_equal_approx(value, 0.0000001)


func test_randf_triangular_nearly_equal_bounds() -> void:
	# Test with very close but not equal bounds
	var min_val: float = 1.0
	var max_val: float = 1.0000001
	var mode_val: float = 1.00000005
	var result: float = StatMath.Distributions.randf_triangular(min_val, max_val, mode_val)
	
	assert_float(result).is_between(min_val, max_val)


func test_randf_triangular_invalid_mode_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(5.0, 10.0, 3.0)  # mode < min
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Mode value must be greater than or equal to minimum value for Triangular distribution.")


func test_randf_triangular_invalid_mode_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(5.0, 10.0, 12.0)  # mode > max
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Mode value must be less than or equal to maximum value for Triangular distribution.")


func test_randf_triangular_invalid_max_less_than_min() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(10.0, 5.0, 7.0)  # max < min
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Maximum value must be greater than or equal to minimum value for Triangular distribution.")


func test_randf_triangular_invalid_max_equal_min_with_different_mode() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(5.0, 5.0, 7.0)  # max = min but mode ≠ min
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Mode value must be less than or equal to maximum value for Triangular distribution.")


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

func test_randf_triangular_weapon_damage() -> void:
	# Example: weapon damage with min/typical/max values
	var min_damage: float = 50.0
	var max_damage: float = 150.0
	var typical_damage: float = 80.0  # Most common damage value
	
	var damage: float = StatMath.Distributions.randf_triangular(min_damage, max_damage, typical_damage)
	
	assert_float(damage).is_between(min_damage, max_damage)


func test_randf_triangular_npc_stat_generation() -> void:
	# Example: NPC attribute generation
	var min_strength: float = 8.0
	var max_strength: float = 18.0
	var average_strength: float = 12.0
	
	var npc_strength: float = StatMath.Distributions.randf_triangular(min_strength, max_strength, average_strength)
	
	assert_float(npc_strength).is_between(min_strength, max_strength)


func test_randf_triangular_loot_quality() -> void:
	# Example: loot quality distribution
	var min_quality: float = 0.1  # Poor quality
	var max_quality: float = 1.0  # Perfect quality
	var common_quality: float = 0.3  # Most items are low-medium quality
	
	var item_quality: float = StatMath.Distributions.randf_triangular(min_quality, max_quality, common_quality)
	
	assert_float(item_quality).is_between(min_quality, max_quality)


func test_randf_triangular_skill_check_difficulty() -> void:
	# Example: skill check difficulty scaling
	var easy_threshold: float = 5.0
	var hard_threshold: float = 20.0
	var typical_threshold: float = 12.0
	
	var check_difficulty: float = StatMath.Distributions.randf_triangular(easy_threshold, hard_threshold, typical_threshold)
	
	assert_float(check_difficulty).is_between(easy_threshold, hard_threshold)


func test_randf_triangular_resource_spawn_rate() -> void:
	# Example: resource spawn timing
	var min_spawn_time: float = 5.0   # seconds
	var max_spawn_time: float = 30.0  # seconds  
	var typical_spawn_time: float = 15.0  # Most common spawn interval
	
	var next_spawn_time: float = StatMath.Distributions.randf_triangular(min_spawn_time, max_spawn_time, typical_spawn_time)
	
	assert_float(next_spawn_time).is_between(min_spawn_time, max_spawn_time)


func test_randf_triangular_procedural_terrain_height() -> void:
	# Example: terrain height generation with preferred elevations
	var sea_level: float = 0.0
	var mountain_peak: float = 1000.0
	var common_elevation: float = 200.0  # Hills and plains
	
	var terrain_height: float = StatMath.Distributions.randf_triangular(sea_level, mountain_peak, common_elevation)
	
	assert_float(terrain_height).is_between(sea_level, mountain_peak)


func test_randf_triangular_ai_decision_confidence() -> void:
	# Example: AI confidence in decisions
	var min_confidence: float = 0.0
	var max_confidence: float = 1.0
	var typical_confidence: float = 0.7  # AI is usually fairly confident
	
	var decision_confidence: float = StatMath.Distributions.randf_triangular(min_confidence, max_confidence, typical_confidence)
	
	assert_float(decision_confidence).is_between(min_confidence, max_confidence)


func test_randf_triangular_pricing_variation() -> void:
	# Example: market price fluctuation in economic simulation
	var min_price: float = 80.0   # 20% below base price
	var max_price: float = 120.0  # 20% above base price
	var fair_price: float = 95.0  # Slightly below base (buyer's market)
	
	var market_price: float = StatMath.Distributions.randf_triangular(min_price, max_price, fair_price)
	
	assert_float(market_price).is_between(min_price, max_price)


func test_randf_triangular_multiple_parameters() -> void:
	# Example: complex entity with multiple triangular attributes
	var health: float = StatMath.Distributions.randf_triangular(80.0, 120.0, 100.0)
	var speed: float = StatMath.Distributions.randf_triangular(3.0, 8.0, 5.0)
	var accuracy: float = StatMath.Distributions.randf_triangular(0.6, 0.95, 0.8)
	
	assert_float(health).is_between(80.0, 120.0)
	assert_float(speed).is_between(3.0, 8.0)
	assert_float(accuracy).is_between(0.6, 0.95)
	
	# All attributes should be reasonable for a game entity
	assert_float(health).is_greater(0.0)
	assert_float(speed).is_greater(0.0)
	assert_float(accuracy).is_greater(0.0)


# --- Tests for randf_pareto ---

func test_randf_pareto_basic() -> void:
	var scale: float = 1.0
	var shape: float = 1.0
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_bool(typeof(result) == TYPE_FLOAT).is_true()
	assert_float(result).is_greater_equal(scale)
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()


func test_randf_pareto_scale_parameter() -> void:
	var scale: float = 5.0
	var shape: float = 2.0
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	# Result should always be >= scale parameter
	assert_float(result).is_greater_equal(scale)


func test_randf_pareto_different_shapes() -> void:
	var scale: float = 1.0
	
	# Test different shape parameters
	var heavy_tail_result: float = StatMath.Distributions.randf_pareto(scale, 0.5)  # Heavy tail
	var medium_tail_result: float = StatMath.Distributions.randf_pareto(scale, 1.0)  # Medium tail
	var light_tail_result: float = StatMath.Distributions.randf_pareto(scale, 3.0)   # Light tail
	
	assert_float(heavy_tail_result).is_greater_equal(scale)
	assert_float(medium_tail_result).is_greater_equal(scale)
	assert_float(light_tail_result).is_greater_equal(scale)


func test_randf_pareto_large_scale() -> void:
	var scale: float = 100.0
	var shape: float = 1.5
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)


func test_randf_pareto_small_scale() -> void:
	var scale: float = 0.001
	var shape: float = 2.0
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)


func test_randf_pareto_very_small_shape() -> void:
	# Very small shape = very heavy tail
	var scale: float = 1.0
	var shape: float = 0.1
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)
	assert_bool(is_inf(result)).is_false()


func test_randf_pareto_large_shape() -> void:
	# Large shape = light tail, concentrated near minimum
	var scale: float = 1.0
	var shape: float = 10.0
	var result: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result).is_greater_equal(scale)


func test_randf_pareto_deterministic_with_seed() -> void:
	var scale: float = 2.0
	var shape: float = 1.5
	var seed: int = 98765
	
	StatMath.set_global_seed(seed)
	var result1: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	StatMath.set_global_seed(seed)
	var result2: float = StatMath.Distributions.randf_pareto(scale, shape)
	
	assert_float(result1).is_equal_approx(result2, 0.0000001)


func test_randf_pareto_multiple_calls_different_values() -> void:
	# Test that multiple calls produce different values (with high probability)
	var scale: float = 1.0
	var shape: float = 2.0
	var results: Array[float] = []
	
	for i in range(10):
		results.append(StatMath.Distributions.randf_pareto(scale, shape))
	
	# Check that we don't have all identical values (extremely unlikely)
	var all_same: bool = true
	var first_val: float = results[0]
	for val in results:
		if not is_equal_approx(val, first_val):
			all_same = false
			break
	
	assert_bool(all_same).is_false()


func test_randf_pareto_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(0.0, 1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Scale parameter must be positive for Pareto distribution.")


func test_randf_pareto_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(-1.0, 1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Scale parameter must be positive for Pareto distribution.")


func test_randf_pareto_invalid_shape_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(1.0, 0.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Shape parameter must be positive for Pareto distribution.")


func test_randf_pareto_invalid_shape_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(1.0, -1.0)
	await assert_error(test_invalid_input).is_runtime_error("Assertion failed: Shape parameter must be positive for Pareto distribution.")


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


func test_randf_pareto_heavy_tail_property() -> void:
	# Test that Pareto produces some extreme values (demonstrating heavy tails)
	var scale: float = 1.0
	var shape: float = 0.8  # Heavy tail
	var extreme_count: int = 0
	var seed: int = 44444
	
	StatMath.set_global_seed(seed)
	for i in range(1000):  # Larger sample for extreme value detection
		var sample: float = StatMath.Distributions.randf_pareto(scale, shape)
		# Count values beyond 10x the scale (would be rare for lighter distributions)
		if sample > scale * 10.0:
			extreme_count += 1
	
	# Pareto should produce some extreme values due to heavy tail
	assert_int(extreme_count).is_greater(0)


func test_randf_pareto_concentration_near_minimum() -> void:
	# Test that higher shape values concentrate more values near the minimum
	var scale: float = 1.0
	var shape: float = 5.0  # High shape = concentration near minimum
	var near_minimum_count: int = 0
	var seed: int = 55555
	
	StatMath.set_global_seed(seed)
	for i in range(1000):
		var sample: float = StatMath.Distributions.randf_pareto(scale, shape)
		# Count samples within 50% of scale
		if sample <= scale * 1.5:
			near_minimum_count += 1
	
	# High shape should concentrate most values near the minimum
	var near_minimum_proportion: float = float(near_minimum_count) / 1000.0
	assert_float(near_minimum_proportion).is_greater(0.6)  # Should be majority


# --- Game Development Use Cases ---

func test_randf_pareto_wealth_distribution() -> void:
	# Example: 80/20 wealth distribution in economic simulation
	var minimum_wealth: float = 100.0
	var wealth_inequality: float = 1.16  # Shape ~1.16 gives 80/20 distribution
	
	var player_wealth: float = StatMath.Distributions.randf_pareto(minimum_wealth, wealth_inequality)
	
	assert_float(player_wealth).is_greater_equal(minimum_wealth)


func test_randf_pareto_loot_rarity() -> void:
	# Example: loot drop rarity using Pareto for "common items common, rare items rare"
	var base_value: float = 1.0
	var rarity_factor: float = 0.8  # Lower = more extreme rare items
	
	var loot_value_multiplier: float = StatMath.Distributions.randf_pareto(base_value, rarity_factor)
	
	assert_float(loot_value_multiplier).is_greater_equal(base_value)


func test_randf_pareto_city_population() -> void:
	# Example: city size distribution (few large cities, many small towns)
	var minimum_population: float = 1000.0
	var urbanization_factor: float = 1.2
	
	var city_population: float = StatMath.Distributions.randf_pareto(minimum_population, urbanization_factor)
	
	assert_float(city_population).is_greater_equal(minimum_population)


func test_randf_pareto_resource_deposits() -> void:
	# Example: natural resource deposit sizes
	var minimum_deposit_size: float = 10.0
	var deposit_distribution: float = 1.5
	
	var resource_amount: float = StatMath.Distributions.randf_pareto(minimum_deposit_size, deposit_distribution)
	
	assert_float(resource_amount).is_greater_equal(minimum_deposit_size)


func test_randf_pareto_market_price_spikes() -> void:
	# Example: market price volatility with occasional massive spikes
	var base_price_multiplier: float = 1.0
	var volatility_factor: float = 0.9  # Heavy tail for price spikes
	
	var price_spike_multiplier: float = StatMath.Distributions.randf_pareto(base_price_multiplier, volatility_factor)
	
	assert_float(price_spike_multiplier).is_greater_equal(base_price_multiplier)


func test_randf_pareto_player_skill_gaps() -> void:
	# Example: modeling skill gaps in competitive games
	var minimum_skill_rating: float = 1000.0
	var skill_distribution: float = 1.3
	
	var player_skill: float = StatMath.Distributions.randf_pareto(minimum_skill_rating, skill_distribution)
	
	assert_float(player_skill).is_greater_equal(minimum_skill_rating)


func test_randf_pareto_quest_reward_scaling() -> void:
	# Example: quest reward distribution where most rewards are modest, few are huge
	var base_reward: float = 50.0
	var reward_scaling: float = 1.1  # Slight heavy tail
	
	var quest_reward: float = StatMath.Distributions.randf_pareto(base_reward, reward_scaling)
	
	assert_float(quest_reward).is_greater_equal(base_reward)


func test_randf_pareto_network_effect_scaling() -> void:
	# Example: network effects where popular content becomes extremely popular
	var minimum_views: float = 100.0
	var virality_factor: float = 0.7  # Heavy tail for viral content
	
	var content_popularity: float = StatMath.Distributions.randf_pareto(minimum_views, virality_factor)
	
	assert_float(content_popularity).is_greater_equal(minimum_views)


func test_randf_pareto_power_law_scaling() -> void:
	# Example: general power law scaling for various game mechanics
	var base_value: float = 1.0
	var power_law_exponent: float = 2.0
	
	var scaled_value: float = StatMath.Distributions.randf_pareto(base_value, power_law_exponent)
	
	assert_float(scaled_value).is_greater_equal(base_value)


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
