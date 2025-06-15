# addons/godot-stat-math/tests/core/cdf_functions_test.gd
class_name CdfFunctionsTest extends GdUnitTestSuite

# --- Uniform CDF ---
func test_uniform_cdf_basic_range() -> void:
	var a: float = 2.0
	var b: float = 5.0
	var x: float = 3.0
	var result: float = StatMath.CdfFunctions.uniform_cdf(x, a, b)
	assert_float(result).is_equal_approx((x - a) / (b - a), 1e-7)

func test_uniform_cdf_x_below_a() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(1.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_uniform_cdf_x_above_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(6.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_uniform_cdf_a_equals_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(2.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_uniform_cdf_invalid_a_greater_than_b() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.uniform_cdf(2.0, 5.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Parameter a must be less than or equal to b for Uniform CDF.")

# --- Normal CDF ---
func test_normal_cdf_standard_normal() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(0.0)
	assert_float(result).is_equal_approx(0.5, 1e-6)

func test_normal_cdf_mu_sigma() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(2.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.5, 1e-6)

func test_normal_cdf_invalid_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Standard deviation (sigma) must be positive for Normal CDF.")

# --- Exponential CDF ---
func test_exponential_cdf_typical() -> void:
	var result: float = StatMath.CdfFunctions.exponential_cdf(1.0, 2.0)
	assert_float(result).is_greater_equal(0.0)
	assert_float(result).is_less_equal(1.0)

func test_exponential_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.exponential_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_exponential_cdf_invalid_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be positive for Exponential CDF.")

# --- Beta CDF ---
func test_beta_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_beta_cdf_x_one() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_cdf_invalid_alpha_beta() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape parameters (alpha, beta_param) must be positive for Beta CDF.")

# --- Gamma CDF ---
func test_gamma_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.gamma_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_gamma_cdf_invalid_shape_scale() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF.")

# --- Chi-Square CDF ---
func test_chi_square_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.chi_square_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_square_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Degrees of freedom (k_df) must be positive for Chi-Square CDF.")

# --- F-Distribution CDF ---
func test_f_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.f_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF.")

# --- Student's t-Distribution CDF ---
func test_t_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.t_cdf(0.0, 2.0)
	assert_float(result).is_greater_equal(0.0)
	assert_float(result).is_less_equal(1.0)

func test_t_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF.")

# --- Binomial CDF ---
func test_binomial_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(-1, 5, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_binomial_cdf_k_ge_n() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(5, 5, 0.5)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_binomial_cdf_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, -1, 0.5)
	await assert_error(test_call).is_runtime_error("Assertion failed: Number of trials (n_trials) must be non-negative.")

func test_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, 5, -0.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Probability (p_prob) must be between 0.0 and 1.0.")

# --- Poisson CDF ---
func test_poisson_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.poisson_cdf(-1, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_poisson_cdf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.poisson_cdf(2, -1.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be non-negative for Poisson CDF.")

# --- Geometric CDF ---
func test_geometric_cdf_k_less_than_1() -> void:
	var result: float = StatMath.CdfFunctions.geometric_cdf(0, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_geometric_cdf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.geometric_cdf(2, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Success probability (p_prob) must be in (0,1].")

# --- Negative Binomial CDF ---
func test_negative_binomial_cdf_k_less_than_r() -> void:
	var result: float = StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_negative_binomial_cdf_invalid_r() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 0, 0.5)
	await assert_error(test_call).is_runtime_error("Assertion failed: Number of successes (r_successes) must be positive.")

func test_negative_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Success probability (p_prob) must be in (0,1].")

# --- Pareto CDF ---
func test_pareto_cdf_x_equals_scale() -> void:
	var result: float = StatMath.CdfFunctions.pareto_cdf(2.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_pareto_cdf_basic_calculation() -> void:
	# For x = 4, scale = 2, shape = 3: F(4) = 1 - (2/4)^3 = 1 - 0.125 = 0.875
	var result: float = StatMath.CdfFunctions.pareto_cdf(4.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.875, 1e-7)

func test_pareto_cdf_x_below_scale() -> void:
	var result: float = StatMath.CdfFunctions.pareto_cdf(1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_pareto_cdf_large_x() -> void:
	# For very large x, CDF should approach 1
	var result: float = StatMath.CdfFunctions.pareto_cdf(1000.0, 2.0, 3.0)
	assert_float(result).is_greater(0.99)
	assert_float(result).is_less_equal(1.0)

func test_pareto_cdf_different_shapes() -> void:
	var x: float = 4.0
	var scale: float = 2.0
	
	# Higher shape parameter = faster decay, higher CDF for same x
	var cdf_shape_1: float = StatMath.CdfFunctions.pareto_cdf(x, scale, 1.0)
	var cdf_shape_3: float = StatMath.CdfFunctions.pareto_cdf(x, scale, 3.0)
	var cdf_shape_5: float = StatMath.CdfFunctions.pareto_cdf(x, scale, 5.0)
	
	assert_float(cdf_shape_1).is_less(cdf_shape_3)
	assert_float(cdf_shape_3).is_less(cdf_shape_5)

func test_pareto_cdf_monotonicity() -> void:
	# CDF should be monotonically increasing
	var scale: float = 2.0
	var shape: float = 3.0
	
	var x1: float = 2.5
	var x2: float = 3.0
	var x3: float = 4.0
	var x4: float = 6.0
	
	var cdf1: float = StatMath.CdfFunctions.pareto_cdf(x1, scale, shape)
	var cdf2: float = StatMath.CdfFunctions.pareto_cdf(x2, scale, shape)
	var cdf3: float = StatMath.CdfFunctions.pareto_cdf(x3, scale, shape)
	var cdf4: float = StatMath.CdfFunctions.pareto_cdf(x4, scale, shape)
	
	assert_float(cdf1).is_less_equal(cdf2)
	assert_float(cdf2).is_less_equal(cdf3)
	assert_float(cdf3).is_less_equal(cdf4)

func test_pareto_cdf_bounds() -> void:
	# CDF should always be between 0 and 1
	var test_cases: Array[Array] = [
		[2.0, 1.0, 0.5], [5.0, 3.0, 2.0], [10.0, 2.0, 4.0],
		[100.0, 10.0, 1.5], [1.5, 1.0, 10.0]
	]
	
	for case in test_cases:
		var x: float = case[0]
		var scale: float = case[1]
		var shape: float = case[2]
		
		var result: float = StatMath.CdfFunctions.pareto_cdf(x, scale, shape)
		
		assert_float(result).is_greater_equal(0.0)
		assert_float(result).is_less_equal(1.0)
		assert_bool(is_nan(result)).is_false()

func test_pareto_cdf_deterministic() -> void:
	# Same parameters should give same results
	var x: float = 5.0
	var scale: float = 2.0
	var shape: float = 3.0
	
	var result1: float = StatMath.CdfFunctions.pareto_cdf(x, scale, shape)
	var result2: float = StatMath.CdfFunctions.pareto_cdf(x, scale, shape)
	
	assert_float(result1).is_equal_approx(result2, 1e-15)

func test_pareto_cdf_invalid_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 0.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Scale parameter must be positive for Pareto CDF.")

func test_pareto_cdf_invalid_scale_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, -1.0, 2.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Scale parameter must be positive for Pareto CDF.")

func test_pareto_cdf_invalid_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 2.0, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape parameter must be positive for Pareto CDF.")

func test_pareto_cdf_invalid_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 2.0, -1.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Shape parameter must be positive for Pareto CDF.")

# --- Game Development Use Cases for Pareto CDF ---

func test_pareto_cdf_wealth_distribution_probability() -> void:
	# Example: probability that a player's wealth is below a certain threshold
	var wealth_threshold: float = 500.0
	var min_wealth: float = 100.0 # scale parameter
	var inequality_factor: float = 2.0 # shape parameter (lower = more inequality)
	
	var prob_below_threshold: float = StatMath.CdfFunctions.pareto_cdf(wealth_threshold, min_wealth, inequality_factor)
	
	# Should be a valid probability
	assert_float(prob_below_threshold).is_between(0.0, 1.0)
	
	# With shape=2 and threshold=5*scale, should be significant probability
	assert_float(prob_below_threshold).is_greater(0.5)

func test_pareto_cdf_loot_rarity_distribution() -> void:
	# Example: probability of getting loot below certain value
	var loot_values: Array[float] = [10.0, 50.0, 100.0, 500.0]
	var min_loot_value: float = 10.0
	var rarity_shape: float = 3.0 # Higher shape = less extreme values
	
	var probabilities: Array[float] = []
	for loot_value in loot_values:
		var prob: float = StatMath.CdfFunctions.pareto_cdf(loot_value, min_loot_value, rarity_shape)
		probabilities.append(prob)
	
	# Probabilities should increase with loot value
	for i in range(probabilities.size() - 1):
		assert_float(probabilities[i]).is_less_equal(probabilities[i + 1])
	
	# At minimum value, probability should be 0
	assert_float(probabilities[0]).is_equal_approx(0.0, 1e-7)

func test_pareto_cdf_damage_resistance_calculation() -> void:
	# Example: probability that damage dealt is below player's resistance
	var player_resistance: float = 75.0
	var min_damage: float = 25.0
	var damage_scaling: float = 2.5
	
	var prob_resist: float = StatMath.CdfFunctions.pareto_cdf(player_resistance, min_damage, damage_scaling)
	
	# Should be valid probability
	assert_float(prob_resist).is_between(0.0, 1.0)
	
	# Can use this probability for resist chance calculations
	assert_bool(prob_resist > 0.0).is_true()

func test_pareto_cdf_market_price_analysis() -> void:
	# Example: analyzing probability of items being below market price
	var market_price: float = 200.0
	var base_item_value: float = 50.0
	var market_volatility: float = 1.5 # Lower shape = more price volatility
	
	var prob_below_market: float = StatMath.CdfFunctions.pareto_cdf(market_price, base_item_value, market_volatility)
	
	# Should be valid probability for economic calculations
	assert_float(prob_below_market).is_between(0.0, 1.0)
	
	# With low shape parameter, most items should be near base value
	assert_float(prob_below_market).is_greater(0.3) 
