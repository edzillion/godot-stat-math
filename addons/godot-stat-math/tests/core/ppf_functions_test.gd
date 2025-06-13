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
	var result: float = StatMath.PpfFunctions.uniform_ppf(-0.1, 2.0, 4.0)
	assert_bool(is_nan(result)).is_true()

func test_uniform_ppf_invalid_b_less_than_a() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(0.5, 4.0, 2.0)
	assert_bool(is_nan(result)).is_true()

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
	var result: float = StatMath.PpfFunctions.normal_ppf(0.5, 0.0, 0.0)
	assert_bool(is_nan(result)).is_true()

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
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.5, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Beta PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_beta_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_beta_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_ppf_invalid_alpha() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_ppf_invalid_beta() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()

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
	var result: float = StatMath.PpfFunctions.beta_ppf(-0.1, 2.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_ppf_invalid_p_greater_than_one() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(1.1, 2.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_ppf_invalid_alpha_negative() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, -1.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_ppf_invalid_beta_negative() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, 2.0, -1.0)
	assert_bool(is_nan(result)).is_true()

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
	var result: float = StatMath.PpfFunctions.gamma_ppf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_gamma_ppf_invalid_scale() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(0.5, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Chi-Square PPF ---
func test_chi_square_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.chi_square_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_square_ppf_invalid_df() -> void:
	var result: float = StatMath.PpfFunctions.chi_square_ppf(0.5, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- F PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_f_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.f_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_ppf_invalid_df() -> void:
	var result: float = StatMath.PpfFunctions.f_ppf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

# --- t PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_t_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_t_ppf_p_half() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.5, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_t_ppf_invalid_df() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.5, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Binomial PPF ---
func test_binomial_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(0.0, 5, 0.5)
	assert_int(result).is_equal(0)

func test_binomial_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(1.0, 5, 0.5)
	assert_int(result).is_equal(5)

func test_binomial_ppf_invalid_n() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(0.5, -1, 0.5)
	assert_int(result).is_equal(-1)

# --- Poisson PPF ---
func test_poisson_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.poisson_ppf(0.0, 3.0)
	assert_int(result).is_equal(0)

func test_poisson_ppf_invalid_lambda() -> void:
	var result: int = StatMath.PpfFunctions.poisson_ppf(0.5, -1.0)
	assert_int(result).is_equal(-1)

# --- Geometric PPF ---
func test_geometric_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(0.0, 0.5)
	assert_int(result).is_equal(1)

func test_geometric_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(1.0, 0.5)
	assert_int(result).is_equal(StatMath.INT_MAX_REPRESENTING_INF)

func test_geometric_ppf_invalid_p() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(-0.1, 0.5)
	assert_int(result).is_equal(-1)

# --- Negative Binomial PPF ---
func test_negative_binomial_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(0.0, 2, 0.5)
	assert_int(result).is_equal(2)

func test_negative_binomial_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(1.0, 2, 0.5)
	assert_int(result).is_equal(StatMath.INT_MAX_REPRESENTING_INF)

func test_negative_binomial_ppf_invalid_r() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(0.5, 0, 0.5)
	assert_int(result).is_equal(-1)

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
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(-0.1, 0.5)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters

func test_bernoulli_ppf_invalid_p_too_high() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(1.1, 0.5)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters

func test_bernoulli_ppf_invalid_prob_success_too_low() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, -0.1)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters

func test_bernoulli_ppf_invalid_prob_success_too_high() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, 1.1)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters


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
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(-0.1, values, probabilities)).is_null()

func test_discrete_histogram_ppf_invalid_p_too_high() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(1.1, values, probabilities)).is_null()

func test_discrete_histogram_ppf_empty_values() -> void:
	var values: Array = []
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_empty_probabilities() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = []
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_size_mismatch() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_negative_probability() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [-0.1, 1.1]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

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
