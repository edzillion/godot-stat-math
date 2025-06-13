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