# addons/godot-stat-math/tests/core/ppf_functions_test.gd
class_name PpfFunctionsTest extends GdUnitTestSuite

# --- Uniform PPF ---
func test_uniform_ppf_basic() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(0.5, 2.0, 4.0)
	assert_float(result).is_equal_approx(3.0, 1e-7)

func test_uniform_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(0.0, 2.0, 4.0)
	assert_float(result).is_equal_approx(2.0, 1e-7)

func test_uniform_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(1.0, 2.0, 4.0)
	assert_float(result).is_equal_approx(4.0, 1e-7)

func test_uniform_ppf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.uniform_ppf(-0.1, 2.0, 4.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

func test_uniform_ppf_invalid_b_less_than_a() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.uniform_ppf(0.5, 4.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter b must be greater than or equal to a. Received a=4.0, b=2.0")

# --- Normal PPF ---
func test_normal_ppf_standard_normal() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.5)
	assert_float(result).is_equal_approx(0.0, 1e-6)

func test_normal_ppf_mu_sigma() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.5, 2.0, 3.0)
	assert_float(result).is_equal_approx(2.0, 1e-6)

func test_normal_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_normal_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(1.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_normal_ppf_invalid_sigma() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.normal_ppf(0.5, 0.0, 0.0)
	await assert_error(test_call).is_push_error("Standard deviation sigma must be positive. Received: 0.0")

# --- Exponential PPF ---
func test_exponential_ppf_basic() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.5, 2.0)
	assert_float(result).is_equal_approx(-log(0.5)/2.0, 1e-7)

func test_exponential_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_exponential_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(1.0, 2.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_exponential_ppf_invalid_lambda() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.exponential_ppf(0.5, 0.0)
	await assert_error(test_call).is_push_error("Rate lambda_param must be positive. Received: 0.0")

# --- Beta PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_beta_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_beta_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_ppf_invalid_alpha() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.beta_ppf(0.5, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameter alpha_shape must be positive. Received: 0.0")

func test_beta_ppf_invalid_beta() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.beta_ppf(0.5, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Shape parameter beta_shape must be positive. Received: 0.0")

# --- Comprehensive Beta PPF Tests ---

func test_beta_ppf_symmetric_beta_2_2() -> void:
	# Beta(2,2) is symmetric, so PPF(0.5) should be 0.5
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.5, 1e-4) # Numerical tolerance

func test_beta_ppf_beta_2_2_quarter() -> void:
	# Test specific value for Beta(2,2) at p=0.25
	var p: float = 0.25
	var result: float = StatMath.PpfFunctions.beta_ppf(p, 2.0, 2.0)
	
	# Result should be between 0 and 0.5 (since p < 0.5)
	assert_float(result).is_between(0.0, 0.5)
	assert_float(result).is_greater(0.1) # Should be reasonably far from 0

func test_beta_ppf_beta_2_2_three_quarters() -> void:
	# Test specific value for Beta(2,2) at p=0.75
	var p: float = 0.75
	var result: float = StatMath.PpfFunctions.beta_ppf(p, 2.0, 2.0)
	
	# Result should be between 0.5 and 1 (since p > 0.5)
	assert_float(result).is_between(0.5, 1.0)
	assert_float(result).is_less(0.9) # Should be reasonably far from 1

func test_beta_ppf_monotonicity() -> void:
	# Test that beta PPF is monotonically increasing in p
	var alpha: float = 2.5
	var beta: float = 3.5
	
	var p1: float = 0.1
	var p2: float = 0.3
	var p3: float = 0.5
	var p4: float = 0.7
	var p5: float = 0.9
	
	var result1: float = StatMath.PpfFunctions.beta_ppf(p1, alpha, beta)
	var result2: float = StatMath.PpfFunctions.beta_ppf(p2, alpha, beta)
	var result3: float = StatMath.PpfFunctions.beta_ppf(p3, alpha, beta)
	var result4: float = StatMath.PpfFunctions.beta_ppf(p4, alpha, beta)
	var result5: float = StatMath.PpfFunctions.beta_ppf(p5, alpha, beta)
	
	# Should be monotonically increasing
	assert_float(result1).is_less(result2)
	assert_float(result2).is_less(result3)
	assert_float(result3).is_less(result4)
	assert_float(result4).is_less(result5)

func test_beta_ppf_bounds() -> void:
	# Test that beta PPF always returns values in [0, 1]
	var test_params: Array[Array] = [
		[0.1, 1.0, 1.0], [0.5, 0.5, 0.5], [0.9, 3.0, 2.0],
		[0.25, 5.0, 1.0], [0.75, 1.0, 5.0], [0.33, 10.0, 10.0]
	]
	
	for params in test_params:
		var p: float = params[0]
		var alpha: float = params[1]
		var beta: float = params[2]
		
		var result: float = StatMath.PpfFunctions.beta_ppf(p, alpha, beta)
		
		assert_float(result).is_greater_equal(0.0)
		assert_float(result).is_less_equal(1.0)
		assert_bool(is_nan(result)).is_false()

func test_beta_ppf_uniform_distribution() -> void:
	# Beta(1,1) is uniform distribution, so PPF should equal p
	var test_probs: Array[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
	
	for p in test_probs:
		var result: float = StatMath.PpfFunctions.beta_ppf(p, 1.0, 1.0)
		assert_float(result).is_equal_approx(p, 1e-4) # Should be approximately equal to p

func test_beta_ppf_extreme_parameters() -> void:
	# Test with extreme parameter values
	var p: float = 0.5
	
	# Large alpha, small beta (skewed toward 1)
	var result_large_alpha: float = StatMath.PpfFunctions.beta_ppf(p, 10.0, 1.0)
	assert_float(result_large_alpha).is_greater(0.5) # Should be greater than 0.5
	
	# Small alpha, large beta (skewed toward 0)
	var result_large_beta: float = StatMath.PpfFunctions.beta_ppf(p, 1.0, 10.0)
	assert_float(result_large_beta).is_less(0.5) # Should be less than 0.5

func test_beta_ppf_deterministic() -> void:
	# Test that same parameters give same results (deterministic)
	var p: float = 0.3
	var alpha: float = 2.5
	var beta: float = 3.0
	
	var result1: float = StatMath.PpfFunctions.beta_ppf(p, alpha, beta)
	var result2: float = StatMath.PpfFunctions.beta_ppf(p, alpha, beta)
	
	assert_float(result1).is_equal_approx(result2, 1e-10) # Should be exactly the same

func test_beta_ppf_coordinated_shuffle_use_case() -> void:
	# Test the specific use case from coordinated shuffle
	var raw_values: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
	var transformed_values: Array[float] = []
	
	for raw_val in raw_values:
		var beta_val: float = StatMath.PpfFunctions.beta_ppf(raw_val, 2.0, 2.0)
		transformed_values.append(beta_val)
	
	# All transformed values should be in [0, 1]
	for beta_val in transformed_values:
		assert_float(beta_val).is_between(0.0, 1.0)
	
	# Should maintain relative ordering (monotonicity)
	for i in range(transformed_values.size() - 1):
		assert_float(transformed_values[i]).is_less(transformed_values[i + 1])

func test_beta_ppf_edge_case_small_p() -> void:
	# Test very small p values
	var small_p: float = 0.001
	var result: float = StatMath.PpfFunctions.beta_ppf(small_p, 2.0, 2.0)
	
	assert_float(result).is_greater(0.0)
	assert_float(result).is_less(0.1) # Should be small but not zero
	assert_bool(is_nan(result)).is_false()

func test_beta_ppf_edge_case_large_p() -> void:
	# Test very large p values
	var large_p: float = 0.999
	var result: float = StatMath.PpfFunctions.beta_ppf(large_p, 2.0, 2.0)
	
	assert_float(result).is_greater(0.9) # Should be large but not 1
	assert_float(result).is_less(1.0)
	assert_bool(is_nan(result)).is_false()

func test_beta_ppf_invalid_p_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.beta_ppf(-0.1, 2.0, 2.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

func test_beta_ppf_invalid_p_greater_than_one() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.beta_ppf(1.1, 2.0, 2.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.1")

func test_beta_ppf_invalid_alpha_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.beta_ppf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameter alpha_shape must be positive. Received: -1.0")

func test_beta_ppf_invalid_beta_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.beta_ppf(0.5, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameter beta_shape must be positive. Received: -1.0")

func test_beta_ppf_consistency_with_cdf() -> void:
	# Test that PPF and CDF are inverse functions (round-trip test)
	var original_x: float = 0.3
	var alpha: float = 2.0
	var beta: float = 2.0
	
	# Calculate CDF of original_x, then PPF of that result
	var cdf_result: float = StatMath.CdfFunctions.beta_cdf(original_x, alpha, beta)
	var ppf_result: float = StatMath.PpfFunctions.beta_ppf(cdf_result, alpha, beta)
	
	# Should get back close to original_x
	assert_float(ppf_result).is_equal_approx(original_x, 1e-3) # Allowing for numerical errors

# --- Game Development Use Cases for Beta PPF ---

func test_beta_ppf_procedural_generation() -> void:
	# Example: using beta PPF for procedural generation with controlled distribution
	var uniform_inputs: Array[float] = [0.2, 0.4, 0.6, 0.8]
	var terrain_heights: Array[float] = []
	
	# Transform uniform inputs to beta distribution for more natural terrain
	for uniform_val in uniform_inputs:
		var height: float = StatMath.PpfFunctions.beta_ppf(uniform_val, 2.0, 5.0) # Skewed toward lower heights
		terrain_heights.append(height)
	
	# All heights should be valid
	for height in terrain_heights:
		assert_float(height).is_between(0.0, 1.0)
	
	# Should maintain ordering
	for i in range(terrain_heights.size() - 1):
		assert_float(terrain_heights[i]).is_less_equal(terrain_heights[i + 1])

func test_beta_ppf_item_quality_distribution() -> void:
	# Example: transforming uniform randomness to quality scores with desired distribution
	var random_seeds: Array[float] = [0.1, 0.3, 0.7, 0.9]
	var quality_scores: Array[float] = []
	
	# Use Beta(2, 8) for rare high-quality items (most items low quality)
	for seed in random_seeds:
		var quality: float = StatMath.PpfFunctions.beta_ppf(seed, 2.0, 8.0)
		quality_scores.append(quality)
	
	# All quality scores should be valid
	for quality in quality_scores:
		assert_float(quality).is_between(0.0, 1.0)
	
	# Most should be relatively low (due to Beta(2,8) distribution)
	var low_quality_count: int = 0
	for quality in quality_scores:
		if quality < 0.3:
			low_quality_count += 1
	
	# At least some should be low quality (statistically likely)
	assert_int(low_quality_count).is_greater_equal(1)

# --- Gamma PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_gamma_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_gamma_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_gamma_ppf_invalid_shape() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.gamma_ppf(0.5, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameter k_shape must be positive. Received: 0.0")

func test_gamma_ppf_invalid_scale() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.gamma_ppf(0.5, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Scale parameter theta_scale must be positive. Received: 0.0")

# --- Chi-Square PPF ---
func test_chi_square_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.chi_square_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_square_ppf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.chi_square_ppf(0.5, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom k_df must be positive. Received: 0.0")

# --- F PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_f_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.f_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_ppf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.f_ppf(0.5, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Degrees of freedom d1 must be positive. Received: 0.0")

# --- t PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_t_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_t_ppf_p_half() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.5, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_t_ppf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.t_ppf(0.5, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom df must be positive. Received: 0.0")

# --- Binomial PPF ---
func test_binomial_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(0.0, 5, 0.5)
	assert_int(result).is_equal(0)

func test_binomial_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(1.0, 5, 0.5)
	assert_int(result).is_equal(5)

func test_binomial_ppf_invalid_n() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.binomial_ppf(0.5, -1, 0.5)
	await assert_error(test_call).is_push_error("Number of trials n must be non-negative. Received: -1")

# --- Poisson PPF ---
func test_poisson_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.poisson_ppf(0.0, 3.0)
	assert_int(result).is_equal(0)

func test_poisson_ppf_invalid_lambda() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.poisson_ppf(0.5, -1.0)
	await assert_error(test_call).is_push_error("Rate lambda_param must be non-negative. Received: -1.0")

# --- Geometric PPF ---
func test_geometric_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(0.0, 0.5)
	assert_int(result).is_equal(1)

func test_geometric_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(1.0, 0.5)
	assert_int(result).is_equal(StatMath.INT_MAX_REPRESENTING_INF)

func test_geometric_ppf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.geometric_ppf(-0.1, 0.5)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

# --- Negative Binomial PPF ---
func test_negative_binomial_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(0.0, 2, 0.5)
	assert_int(result).is_equal(2)

func test_negative_binomial_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(1.0, 2, 0.5)
	assert_int(result).is_equal(StatMath.INT_MAX_REPRESENTING_INF)

func test_negative_binomial_ppf_invalid_r() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.negative_binomial_ppf(0.5, 0, 0.5)
	await assert_error(test_call).is_push_error("Number of successes r_successes must be a positive integer. Received: 0")

# --- Bernoulli PPF ---
func test_bernoulli_ppf_p_less_than_one_minus_prob_success() -> void:
	# CDF(0) = 1 - prob_success. If p <= CDF(0), result is 0.
	# prob_success = 0.7, 1 - prob_success = 0.3. p = 0.2. 0.2 <= 0.3, so expect 0.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.2, 0.7)
	assert_int(result).is_equal(0)

func test_bernoulli_ppf_p_greater_than_one_minus_prob_success() -> void:
	# prob_success = 0.7, 1 - prob_success = 0.3. p = 0.4. 0.4 > 0.3, so expect 1.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.4, 0.7)
	assert_int(result).is_equal(1)

func test_bernoulli_ppf_p_equals_one_minus_prob_success() -> void:
	# prob_success = 0.7, 1 - prob_success = 0.3. p = 0.3. 0.3 <= 0.3, so expect 0.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.3, 0.7)
	assert_int(result).is_equal(0)

func test_bernoulli_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.0, 0.7)
	assert_int(result).is_equal(0) # Smallest k is 0

func test_bernoulli_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(1.0, 0.7)
	assert_int(result).is_equal(1) # Smallest k to make CDF >= 1.0 is 1

func test_bernoulli_ppf_prob_success_zero() -> void:
	# prob_success = 0.0. CDF(0) = 1.0.
	# For p = 0.5, 0.5 <= 1.0, so expect 0.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, 0.0)
	assert_int(result).is_equal(0)

func test_bernoulli_ppf_prob_success_one() -> void:
	# prob_success = 1.0. CDF(0) = 0.0. CDF(1) = 1.0.
	# For p = 0.5, 0.5 > 0.0, so expect 1.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, 1.0)
	assert_int(result).is_equal(1)

func test_bernoulli_ppf_invalid_p_too_low() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.bernoulli_ppf(-0.1, 0.5)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

func test_bernoulli_ppf_invalid_p_too_high() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.bernoulli_ppf(1.1, 0.5)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.1")

func test_bernoulli_ppf_invalid_prob_success_too_low() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.bernoulli_ppf(0.5, -0.1)
	await assert_error(test_call).is_push_error("Success probability prob_success must be between 0.0 and 1.0. Received: -0.1")

func test_bernoulli_ppf_invalid_prob_success_too_high() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.bernoulli_ppf(0.5, 1.1)
	await assert_error(test_call).is_push_error("Success probability prob_success must be between 0.0 and 1.0. Received: 1.1")


# --- Discrete Histogram PPF ---
func test_discrete_histogram_ppf_basic_cases() -> void:
	var values: Array[String] = ["A", "B", "C"]
	var probabilities: Array[float] = [0.2, 0.5, 0.3] # CDF: A=0.2, B=0.7, C=1.0
	
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.1, values, probabilities)).is_equal("A")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.2, values, probabilities)).is_equal("A") # p == CDF(A)
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.20001, values, probabilities)).is_equal("B")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.69999, values, probabilities)).is_equal("B")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.7, values, probabilities)).is_equal("B") # p == CDF(B)
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.70001, values, probabilities)).is_equal("C")

func test_discrete_histogram_ppf_p_zero() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [0.5, 0.5]
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.0, values, probabilities)).is_equal("A")

func test_discrete_histogram_ppf_p_one_exact_sum() -> void:
	var values: Array[String] = ["A", "B", "C"]
	var probabilities: Array[float] = [0.2, 0.5, 0.3] # Sums to 1.0
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(1.0, values, probabilities)).is_equal("C")

func test_discrete_histogram_ppf_p_one_sum_less_than_one_fallback() -> void:
	var values: Array[String] = ["X", "Y"]
	var probabilities: Array[float] = [0.1, 0.1] # Sums to 0.2
	# Even if p=1.0, and sum_probs is less, it should return the last value due to the fallback logic.
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(1.0, values, probabilities)).is_equal("Y")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.15, values, probabilities)).is_equal("Y") # Should also fall into the last bucket here based on logic

func test_discrete_histogram_ppf_single_value() -> void:
	var values: Array[String] = ["OnlyOne"]
	var probabilities: Array[float] = [1.0]
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.0, values, probabilities)).is_equal("OnlyOne")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_equal("OnlyOne")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(1.0, values, probabilities)).is_equal("OnlyOne")

func test_discrete_histogram_ppf_values_can_be_numbers() -> void:
	var values: Array[int] = [10, 20, 30]
	var probabilities: Array[float] = [0.2, 0.5, 0.3]
	assert_int(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_equal(20)

func test_discrete_histogram_ppf_invalid_p_too_low() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = [1.0]
	var test_call: Callable = func():
		StatMath.PpfFunctions.discrete_histogram_ppf(-0.1, values, probabilities)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

func test_discrete_histogram_ppf_invalid_p_too_high() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = [1.0]
	var test_call: Callable = func():
		StatMath.PpfFunctions.discrete_histogram_ppf(1.1, values, probabilities)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.1")

func test_discrete_histogram_ppf_empty_values() -> void:
	var values: Array = []
	var probabilities: Array[float] = [1.0]
	var test_call: Callable = func():
		StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)
	await assert_error(test_call).is_push_error("Values array cannot be empty for discrete_histogram_ppf.")

func test_discrete_histogram_ppf_empty_probabilities() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = []
	var test_call: Callable = func():
		StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)
	await assert_error(test_call).is_push_error("Probabilities array cannot be empty for discrete_histogram_ppf.")

func test_discrete_histogram_ppf_size_mismatch() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [1.0]
	var test_call: Callable = func():
		StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)
	await assert_error(test_call).is_push_error("Values and probabilities arrays must have the same size. Received sizes 2 and 1.")

func test_discrete_histogram_ppf_negative_probability() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [-0.1, 1.1]
	var test_call: Callable = func():
		StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)
	await assert_error(test_call).is_push_error("All probabilities must be non-negative. Found: -0.1")

func test_discrete_histogram_ppf_probabilities_not_sum_to_one_warning() -> void:
	# This test mainly checks if the function still works correctly based on cumulative logic.
	# The warning itself is harder to check in a standard unit test without log capture.
	var values: Array[String] = ["Low", "High"]
	var probabilities: Array[float] = [0.1, 0.1] # Sums to 0.2, not 1.0
	# CDF: Low=0.1, High=0.2
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.05, values, probabilities)).is_equal("Low")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.1, values, probabilities)).is_equal("Low")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.15, values, probabilities)).is_equal("High")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.2, values, probabilities)).is_equal("High")
	# For p > sum of probabilities (0.2), it should return the last element
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_equal("High")


# --- Pareto PPF ---
func test_pareto_ppf_basic_calculation() -> void:
	# For p = 0.5, scale = 2, shape = 2: F⁻¹(0.5) = 2 / (1-0.5)^(1/2) = 2 / 0.707... ≈ 2.828
	var result: float = StatMath.PpfFunctions.pareto_ppf(0.5, 2.0, 2.0)
	var expected: float = 2.0 / pow(0.5, 1.0/2.0) # 2 / sqrt(0.5) ≈ 2.828
	assert_float(result).is_equal_approx(expected, 1e-6)

func test_pareto_ppf_precise_calculation() -> void:
	# For p = 0.875, scale = 2, shape = 3: F⁻¹(0.875) = 2 / (1-0.875)^(1/3) = 2 / 0.125^(1/3) = 2 / 0.5 = 4
	var result: float = StatMath.PpfFunctions.pareto_ppf(0.875, 2.0, 3.0)
	assert_float(result).is_equal_approx(4.0, 1e-6)

func test_pareto_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.pareto_ppf(0.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(2.0, 1e-7) # Should return scale parameter

func test_pareto_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.pareto_ppf(1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_pareto_ppf_monotonicity() -> void:
	# PPF should be monotonically increasing in p
	var scale: float = 2.0
	var shape: float = 3.0
	
	var p1: float = 0.1
	var p2: float = 0.3
	var p3: float = 0.5
	var p4: float = 0.7
	var p5: float = 0.9
	
	var result1: float = StatMath.PpfFunctions.pareto_ppf(p1, scale, shape)
	var result2: float = StatMath.PpfFunctions.pareto_ppf(p2, scale, shape)
	var result3: float = StatMath.PpfFunctions.pareto_ppf(p3, scale, shape)
	var result4: float = StatMath.PpfFunctions.pareto_ppf(p4, scale, shape)
	var result5: float = StatMath.PpfFunctions.pareto_ppf(p5, scale, shape)
	
	# Should be monotonically increasing
	assert_float(result1).is_less(result2)
	assert_float(result2).is_less(result3)
	assert_float(result3).is_less(result4)
	assert_float(result4).is_less(result5)

func test_pareto_ppf_different_shapes() -> void:
	var p: float = 0.5
	var scale: float = 2.0
	
	# Higher shape parameter = smaller values for same p (faster decay)
	var ppf_shape_1: float = StatMath.PpfFunctions.pareto_ppf(p, scale, 1.0)
	var ppf_shape_3: float = StatMath.PpfFunctions.pareto_ppf(p, scale, 3.0)
	var ppf_shape_5: float = StatMath.PpfFunctions.pareto_ppf(p, scale, 5.0)
	
	assert_float(ppf_shape_1).is_greater(ppf_shape_3)
	assert_float(ppf_shape_3).is_greater(ppf_shape_5)
	
	# All should be at least the scale parameter
	assert_float(ppf_shape_1).is_greater_equal(scale)
	assert_float(ppf_shape_3).is_greater_equal(scale)
	assert_float(ppf_shape_5).is_greater_equal(scale)

func test_pareto_ppf_bounds() -> void:
	# PPF should always return values >= scale parameter
	var test_cases: Array[Array] = [
		[0.1, 1.0, 0.5], [0.5, 3.0, 2.0], [0.9, 2.0, 4.0],
		[0.25, 10.0, 1.5], [0.75, 1.0, 10.0]
	]
	
	for case in test_cases:
		var p: float = case[0]
		var scale: float = case[1]
		var shape: float = case[2]
		
		var result: float = StatMath.PpfFunctions.pareto_ppf(p, scale, shape)
		
		assert_float(result).is_greater_equal(scale)
		assert_bool(is_nan(result)).is_false()
		assert_bool(is_inf(result)).is_false()

func test_pareto_ppf_deterministic() -> void:
	# Same parameters should give same results
	var p: float = 0.3
	var scale: float = 2.5
	var shape: float = 3.0
	
	var result1: float = StatMath.PpfFunctions.pareto_ppf(p, scale, shape)
	var result2: float = StatMath.PpfFunctions.pareto_ppf(p, scale, shape)
	
	assert_float(result1).is_equal_approx(result2, 1e-15)

func test_pareto_ppf_consistency_with_cdf() -> void:
	# Test that PPF and CDF are inverse functions (round-trip test)
	var original_p: float = 0.6
	var scale: float = 2.0
	var shape: float = 3.0
	
	# Calculate PPF of original_p, then CDF of that result
	var ppf_result: float = StatMath.PpfFunctions.pareto_ppf(original_p, scale, shape)
	var cdf_result: float = StatMath.CdfFunctions.pareto_cdf(ppf_result, scale, shape)
	
	# Should get back close to original_p
	assert_float(cdf_result).is_equal_approx(original_p, 1e-6)

func test_pareto_ppf_extreme_probabilities() -> void:
	var scale: float = 2.0
	var shape: float = 3.0
	
	# Very small p
	var small_p: float = 0.001
	var result_small: float = StatMath.PpfFunctions.pareto_ppf(small_p, scale, shape)
	assert_float(result_small).is_greater(scale)
	assert_float(result_small).is_less(scale * 2.0) # Should be close to scale
	
	# Very large p
	var large_p: float = 0.999
	var result_large: float = StatMath.PpfFunctions.pareto_ppf(large_p, scale, shape)
	assert_float(result_large).is_greater_equal(scale * 5.0) # Should be much larger than scale

func test_pareto_ppf_invalid_p_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.pareto_ppf(-0.1, 2.0, 3.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

func test_pareto_ppf_invalid_p_greater_than_one() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.pareto_ppf(1.1, 2.0, 3.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.1")

func test_pareto_ppf_invalid_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.pareto_ppf(0.5, 0.0, 3.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive. Received: 0.0")

func test_pareto_ppf_invalid_scale_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.pareto_ppf(0.5, -1.0, 3.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive. Received: -1.0")

func test_pareto_ppf_invalid_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.pareto_ppf(0.5, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive. Received: 0.0")

func test_pareto_ppf_invalid_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.pareto_ppf(0.5, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive. Received: -1.0")

# --- Game Development Use Cases for Pareto PPF ---

func test_pareto_ppf_wealth_generation() -> void:
	# Example: generating player wealth from uniform randomness
	var uniform_inputs: Array[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
	var min_wealth: float = 100.0
	var wealth_inequality: float = 2.0 # Lower shape = more inequality
	
	var wealth_values: Array[float] = []
	for uniform_val in uniform_inputs:
		var wealth: float = StatMath.PpfFunctions.pareto_ppf(uniform_val, min_wealth, wealth_inequality)
		wealth_values.append(wealth)
	
	# All wealth values should be valid
	for wealth in wealth_values:
		assert_float(wealth).is_greater_equal(min_wealth)
		assert_bool(is_nan(wealth)).is_false()
		assert_bool(is_inf(wealth)).is_false()
	
	# Should maintain ordering (monotonicity)
	for i in range(wealth_values.size() - 1):
		assert_float(wealth_values[i]).is_less(wealth_values[i + 1])

func test_pareto_ppf_loot_value_generation() -> void:
	# Example: generating loot values with Pareto distribution
	var random_seeds: Array[float] = [0.2, 0.4, 0.6, 0.8]
	var base_loot_value: float = 10.0
	var rarity_factor: float = 3.0 # Higher shape = less extreme values
	
	var loot_values: Array[float] = []
	for seed in random_seeds:
		var loot_value: float = StatMath.PpfFunctions.pareto_ppf(seed, base_loot_value, rarity_factor)
		loot_values.append(loot_value)
	
	# All loot values should be valid
	for loot_value in loot_values:
		assert_float(loot_value).is_greater_equal(base_loot_value)
		assert_bool(is_nan(loot_value)).is_false()
	
	# Should maintain ordering
	for i in range(loot_values.size() - 1):
		assert_float(loot_values[i]).is_less_equal(loot_values[i + 1])

func test_pareto_ppf_damage_scaling() -> void:
	# Example: scaling base damage using Pareto distribution for crits
	var crit_probability: float = 0.8 # High probability for significant crit
	var base_damage: float = 25.0
	var crit_scaling: float = 2.5
	
	var crit_damage: float = StatMath.PpfFunctions.pareto_ppf(crit_probability, base_damage, crit_scaling)
	
	# Should be at least base damage
	assert_float(crit_damage).is_greater_equal(base_damage)
	
	# Should be a reasonable multiplier for high probability
	assert_float(crit_damage).is_greater(base_damage * 1.5)
	assert_float(crit_damage).is_less(base_damage * 10.0) # Not too extreme

func test_pareto_ppf_npc_attribute_distribution() -> void:
	# Example: distributing NPC attributes with realistic inequality
	var attribute_probabilities: Array[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
	var min_attribute: float = 10.0
	var attribute_distribution: float = 1.8 # Lower shape for more variation
	
	var attribute_values: Array[float] = []
	for prob in attribute_probabilities:
		var attribute: float = StatMath.PpfFunctions.pareto_ppf(prob, min_attribute, attribute_distribution)
		attribute_values.append(attribute)
	
	# All attributes should be valid
	for attribute in attribute_values:
		assert_float(attribute).is_greater_equal(min_attribute)
		assert_bool(is_nan(attribute)).is_false()
	
	# Should show increasing values with increasing probability
	for i in range(attribute_values.size() - 1):
		assert_float(attribute_values[i]).is_less(attribute_values[i + 1])
	
	# Lower probabilities should give values closer to minimum
	assert_float(attribute_values[0]).is_less(attribute_values[-1] * 0.5)

func test_pareto_ppf_market_pricing() -> void:
	# Example: generating market prices with Pareto distribution
	var price_distribution_points: Array[float] = [0.3, 0.6, 0.9]
	var base_item_price: float = 50.0
	var market_concentration: float = 2.2
	
	var market_prices: Array[float] = []
	for point in price_distribution_points:
		var price: float = StatMath.PpfFunctions.pareto_ppf(point, base_item_price, market_concentration)
		market_prices.append(price)
	
	# All prices should be valid
	for price in market_prices:
		assert_float(price).is_greater_equal(base_item_price)
		assert_bool(is_nan(price)).is_false()
	
	# Should maintain ordering
	for i in range(market_prices.size() - 1):
		assert_float(market_prices[i]).is_less(market_prices[i + 1])
	
	# Prices should be reasonable for game economy
	assert_float(market_prices[0]).is_less(base_item_price * 5.0)

func test_pareto_ppf_procedural_generation_scaling() -> void:
	# Example: using Pareto PPF for procedural generation with realistic scaling
	var generation_seeds: Array[float] = [0.05, 0.15, 0.35, 0.65, 0.95]
	var base_scale: float = 1.0
	var scaling_factor: float = 3.0
	
	var scaled_values: Array[float] = []
	for seed in generation_seeds:
		var scaled_value: float = StatMath.PpfFunctions.pareto_ppf(seed, base_scale, scaling_factor)
		scaled_values.append(scaled_value)
	
	# All scaled values should be valid
	for value in scaled_values:
		assert_float(value).is_greater_equal(base_scale)
		assert_bool(is_nan(value)).is_false()
		assert_bool(is_inf(value)).is_false()
	
	# Should maintain ordering
	for i in range(scaled_values.size() - 1):
		assert_float(scaled_values[i]).is_less_equal(scaled_values[i + 1])
	
	# Lower seeds should give values closer to base scale (more reasonable ratio)
	assert_float(scaled_values[0]).is_less(scaled_values[-1] * 0.8) 

# --- Weibull PPF ---
func test_weibull_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.weibull_ppf(0.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_weibull_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.weibull_ppf(1.0, 2.0, 3.0)
	assert_bool(is_inf(result)).is_true()

func test_weibull_ppf_basic_calculation() -> void:
	# For p = 1 - exp(-1) ≈ 0.632, scale = 2, shape = 2: PPF should be 2
	var p: float = 1.0 - exp(-1.0)  # ≈ 0.632
	var result: float = StatMath.PpfFunctions.weibull_ppf(p, 2.0, 2.0)
	assert_float(result).is_equal_approx(2.0, 1e-6)

func test_weibull_ppf_exponential_case() -> void:
	# When shape = 1, Weibull becomes exponential
	var p: float = 0.5
	var scale: float = 2.0
	var shape: float = 1.0
	
	var weibull_result: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
	var exponential_result: float = StatMath.PpfFunctions.exponential_ppf(p, 1.0 / scale)
	
	assert_float(weibull_result).is_equal_approx(exponential_result, 1e-6)

func test_weibull_ppf_rayleigh_case() -> void:
	# When shape = 2, Weibull becomes Rayleigh distribution
	var p: float = 0.7
	var scale: float = 3.0
	var shape: float = 2.0
	
	var result: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
	
	# Should be valid and positive
	assert_float(result).is_greater_equal(0.0)
	assert_bool(is_nan(result)).is_false()
	assert_bool(is_inf(result)).is_false()

func test_weibull_ppf_different_shapes() -> void:
	var p: float = 0.5
	var scale: float = 2.0
	
	# Different shape parameters should give different PPF values
	var ppf_shape_05: float = StatMath.PpfFunctions.weibull_ppf(p, scale, 0.5)  # Decreasing failure rate
	var ppf_shape_1: float = StatMath.PpfFunctions.weibull_ppf(p, scale, 1.0)   # Constant failure rate
	var ppf_shape_3: float = StatMath.PpfFunctions.weibull_ppf(p, scale, 3.0)   # Increasing failure rate
	
	# All should be valid and positive
	assert_float(ppf_shape_05).is_greater_equal(0.0)
	assert_float(ppf_shape_1).is_greater_equal(0.0)
	assert_float(ppf_shape_3).is_greater_equal(0.0)
	
	# They should be different values
	assert_bool(is_equal_approx(ppf_shape_05, ppf_shape_1)).is_false()
	assert_bool(is_equal_approx(ppf_shape_1, ppf_shape_3)).is_false()

func test_weibull_ppf_monotonicity() -> void:
	# PPF should be monotonically increasing
	var scale: float = 2.0
	var shape: float = 2.0
	
	var p1: float = 0.1
	var p2: float = 0.3
	var p3: float = 0.6
	var p4: float = 0.9
	
	var ppf1: float = StatMath.PpfFunctions.weibull_ppf(p1, scale, shape)
	var ppf2: float = StatMath.PpfFunctions.weibull_ppf(p2, scale, shape)
	var ppf3: float = StatMath.PpfFunctions.weibull_ppf(p3, scale, shape)
	var ppf4: float = StatMath.PpfFunctions.weibull_ppf(p4, scale, shape)
	
	assert_float(ppf1).is_less_equal(ppf2)
	assert_float(ppf2).is_less_equal(ppf3)
	assert_float(ppf3).is_less_equal(ppf4)

func test_weibull_ppf_consistency_with_cdf() -> void:
	# PPF(CDF(x)) should equal x
	var x_values: Array[float] = [0.5, 1.0, 2.0, 5.0]
	var scale: float = 2.0
	var shape: float = 2.5
	
	for x in x_values:
		var cdf_val: float = StatMath.CdfFunctions.weibull_cdf(x, scale, shape)
		var ppf_val: float = StatMath.PpfFunctions.weibull_ppf(cdf_val, scale, shape)
		
		assert_float(ppf_val).is_equal_approx(x, 1e-6)

func test_weibull_ppf_bounds() -> void:
	# PPF should always be non-negative for valid inputs
	var p_values: Array[float] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
	var scale: float = 3.0
	var shape: float = 1.8
	
	for p in p_values:
		var result: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
		
		assert_float(result).is_greater_equal(0.0)
		assert_bool(is_nan(result)).is_false()

func test_weibull_ppf_deterministic() -> void:
	# Same parameters should give same results
	var p: float = 0.6
	var scale: float = 2.0
	var shape: float = 2.5
	
	var result1: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
	var result2: float = StatMath.PpfFunctions.weibull_ppf(p, scale, shape)
	
	assert_float(result1).is_equal_approx(result2, 1e-15)

func test_weibull_ppf_extreme_probabilities() -> void:
	var scale: float = 2.0
	var shape: float = 3.0
	
	# Very small p
	var small_p: float = 0.001
	var result_small: float = StatMath.PpfFunctions.weibull_ppf(small_p, scale, shape)
	assert_float(result_small).is_greater_equal(0.0)
	assert_float(result_small).is_less(scale * 0.5)  # Should be small for small p
	
	# Very large p
	var large_p: float = 0.999
	var result_large: float = StatMath.PpfFunctions.weibull_ppf(large_p, scale, shape)
	assert_float(result_large).is_greater(scale * 1.5)  # Should be reasonably large for large p

func test_weibull_ppf_invalid_p_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.weibull_ppf(-0.1, 2.0, 3.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: -0.1")

func test_weibull_ppf_invalid_p_greater_than_one() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.weibull_ppf(1.1, 2.0, 3.0)
	await assert_error(test_call).is_push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: 1.1")

func test_weibull_ppf_invalid_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.weibull_ppf(0.5, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive. Received: 0.0")

func test_weibull_ppf_invalid_scale_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.weibull_ppf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive. Received: -1.0")

func test_weibull_ppf_invalid_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.weibull_ppf(0.5, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive. Received: 0.0")

func test_weibull_ppf_invalid_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.PpfFunctions.weibull_ppf(0.5, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive. Received: -1.0")

# --- Game Development Use Cases for Weibull PPF ---

func test_weibull_ppf_equipment_lifetime_planning() -> void:
	# Example: determining equipment replacement schedules
	var reliability_targets: Array[float] = [0.5, 0.8, 0.95]  # Survival probabilities
	var characteristic_life: float = 1000.0  # Hours
	var wear_pattern: float = 2.5  # Increasing failure rate
	
	var replacement_times: Array[float] = []
	for target in reliability_targets:
		# PPF gives time when failure probability = (1 - target)
		var failure_time: float = StatMath.PpfFunctions.weibull_ppf(1.0 - target, characteristic_life, wear_pattern)
		replacement_times.append(failure_time)
	
	# Times should increase with higher reliability targets
	for i in range(replacement_times.size() - 1):
		assert_float(replacement_times[i]).is_greater_equal(replacement_times[i + 1])
	
	# All should be valid
	for time in replacement_times:
		assert_float(time).is_greater_equal(0.0)

func test_weibull_ppf_survival_time_quantiles() -> void:
	# Example: calculating survival time quantiles for game balancing
	var survival_quantiles: Array[float] = [0.25, 0.5, 0.75, 0.9]  # 25th, 50th, 75th, 90th percentiles
	var base_survival: float = 300.0  # Seconds
	var hazard_pattern: float = 1.8  # Increasing hazard
	
	var quantile_times: Array[float] = []
	for quantile in survival_quantiles:
		var survival_time: float = StatMath.PpfFunctions.weibull_ppf(quantile, base_survival, hazard_pattern)
		quantile_times.append(survival_time)
	
	# Should be monotonically increasing
	for i in range(quantile_times.size() - 1):
		assert_float(quantile_times[i]).is_less_equal(quantile_times[i + 1])
	
	# All should be valid
	for time in quantile_times:
		assert_float(time).is_greater_equal(0.0)

func test_weibull_ppf_wind_speed_thresholds() -> void:
	# Example: determining wind speed thresholds for weather effects (Rayleigh case)
	var wind_probabilities: Array[float] = [0.1, 0.5, 0.9]  # Low, medium, high wind events
	var characteristic_wind: float = 15.0  # km/h
	var rayleigh_shape: float = 2.0  # Rayleigh distribution
	
	var wind_thresholds: Array[float] = []
	for prob in wind_probabilities:
		var wind_speed: float = StatMath.PpfFunctions.weibull_ppf(prob, characteristic_wind, rayleigh_shape)
		wind_thresholds.append(wind_speed)
	
	# Should be increasing
	for i in range(wind_thresholds.size() - 1):
		assert_float(wind_thresholds[i]).is_less_equal(wind_thresholds[i + 1])
	
	# All should be reasonable wind speeds
	for speed in wind_thresholds:
		assert_float(speed).is_greater_equal(0.0)

func test_weibull_ppf_component_design_life() -> void:
	# Example: determining component design life for reliability targets
	var reliability_levels: Array[float] = [0.9, 0.95, 0.99]  # 90%, 95%, 99% reliability
	var design_life: float = 5000.0  # Hours
	var reliability_shape: float = 3.0  # Sharp wear-out
	
	var design_times: Array[float] = []
	for reliability in reliability_levels:
		# Time when failure probability = (1 - reliability)
		var design_time: float = StatMath.PpfFunctions.weibull_ppf(1.0 - reliability, design_life, reliability_shape)
		design_times.append(design_time)
	
	# Higher reliability should require shorter design times (more conservative)
	for i in range(design_times.size() - 1):
		assert_float(design_times[i]).is_greater_equal(design_times[i + 1])
	
	# All should be valid
	for time in design_times:
		assert_float(time).is_greater_equal(0.0)

func test_weibull_ppf_quest_difficulty_scaling() -> void:
	# Example: scaling quest completion times based on difficulty percentiles
	var difficulty_percentiles: Array[float] = [0.2, 0.5, 0.8]  # Easy, medium, hard
	var base_completion_time: float = 60.0  # Minutes
	var difficulty_curve: float = 2.2  # Increasing difficulty
	
	var completion_targets: Array[float] = []
	for percentile in difficulty_percentiles:
		var target_time: float = StatMath.PpfFunctions.weibull_ppf(percentile, base_completion_time, difficulty_curve)
		completion_targets.append(target_time)
	
	# Should be increasing with difficulty
	for i in range(completion_targets.size() - 1):
		assert_float(completion_targets[i]).is_less_equal(completion_targets[i + 1])
	
	# All should be reasonable times
	for time in completion_targets:
		assert_float(time).is_greater_equal(0.0)

func test_weibull_ppf_resource_management_planning() -> void:
	# Example: planning resource extraction based on depletion quantiles
	var depletion_quantiles: Array[float] = [0.1, 0.5, 0.9]  # Conservative, expected, optimistic
	var resource_lifetime: float = 2000.0  # Time units
	var depletion_pattern: float = 1.2  # Slight acceleration
	
	var extraction_plans: Array[float] = []
	for quantile in depletion_quantiles:
		var extraction_time: float = StatMath.PpfFunctions.weibull_ppf(quantile, resource_lifetime, depletion_pattern)
		extraction_plans.append(extraction_time)
	
	# Should be increasing
	for i in range(extraction_plans.size() - 1):
		assert_float(extraction_plans[i]).is_less_equal(extraction_plans[i + 1])
	
	# All should be valid
	for time in extraction_plans:
		assert_float(time).is_greater_equal(0.0)

func test_weibull_ppf_network_performance_sla() -> void:
	# Example: setting network performance SLA based on latency spike durations
	var sla_levels: Array[float] = [0.95, 0.99, 0.999]  # 95%, 99%, 99.9% of spikes
	var typical_spike_duration: float = 50.0  # Milliseconds
	var recovery_pattern: float = 2.8  # Sharp recovery
	
	var sla_thresholds: Array[float] = []
	for sla in sla_levels:
		var threshold: float = StatMath.PpfFunctions.weibull_ppf(sla, typical_spike_duration, recovery_pattern)
		sla_thresholds.append(threshold)
	
	# Higher SLA should have higher thresholds
	for i in range(sla_thresholds.size() - 1):
		assert_float(sla_thresholds[i]).is_less_equal(sla_thresholds[i + 1])
	
	# All should be reasonable latency values
	for threshold in sla_thresholds:
		assert_float(threshold).is_greater_equal(0.0)

func test_weibull_ppf_procedural_generation_scaling() -> void:
	# Example: using Weibull PPF for procedural generation with realistic scaling
	var generation_seeds: Array[float] = [0.1, 0.3, 0.7, 0.9]
	var base_scale: float = 10.0
	var scaling_factor: float = 2.0
	
	var scaled_values: Array[float] = []
	for seed in generation_seeds:
		var scaled_value: float = StatMath.PpfFunctions.weibull_ppf(seed, base_scale, scaling_factor)
		scaled_values.append(scaled_value)
	
	# All scaled values should be valid
	for value in scaled_values:
		assert_float(value).is_greater_equal(0.0)
		assert_bool(is_nan(value)).is_false()
		assert_bool(is_inf(value)).is_false()
	
	# Should maintain ordering
	for i in range(scaled_values.size() - 1):
		assert_float(scaled_values[i]).is_less_equal(scaled_values[i + 1])
	
	# Values should be reasonable for procedural generation
	assert_float(scaled_values[0]).is_greater(0.0)
	assert_float(scaled_values[-1]).is_less(base_scale * 5.0)  # Reasonable upper bound
