# res://addons/godot-stat-math/tests/core/distributions/distributions_parameter_validation_tests.gd
class_name DistributionsParameterValidationTests extends GdUnitTestSuite

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

# Tests for randi_bernoulli
func test_randi_bernoulli_invalid_p_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_bernoulli(-0.1)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be between 0.0 and 1.0. Received: -0.1")
	
	# Test that sentinel value is returned
	var result: int = StatMath.Distributions.randi_bernoulli(-0.1)
	assert_int(result).is_equal(-1)

func test_randi_bernoulli_invalid_p_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_bernoulli(1.1)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be between 0.0 and 1.0. Received: 1.1")
	
	# Test that sentinel value is returned
	var result: int = StatMath.Distributions.randi_bernoulli(1.1)
	assert_int(result).is_equal(-1)

# Tests for randi_binomial
func test_randi_binomial_invalid_p_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_binomial(-0.1, 5)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be between 0.0 and 1.0. Received: -0.1")

func test_randi_binomial_invalid_p_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_binomial(1.1, 5)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be between 0.0 and 1.0. Received: 1.1")

func test_randi_binomial_invalid_n_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_binomial(0.5, -1)
	await assert_error(test_invalid_input).is_push_error("Number of trials (n) must be non-negative. Received: -1")

# Tests for randi_geometric
func test_randi_geometric_invalid_p_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_geometric(0.0)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be in (0,1]. Received: 0.0")

func test_randi_geometric_invalid_p_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_geometric(-0.1)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be in (0,1]. Received: -0.1")

func test_randi_geometric_invalid_p_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_geometric(1.1)
	await assert_error(test_invalid_input).is_push_error("Success probability (p) must be in (0,1]. Received: 1.1")

# Tests for randi_poisson
func test_randi_poisson_invalid_lambda_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_poisson(0.0)
	await assert_error(test_invalid_input).is_push_error("Rate parameter (lambda_param) must be positive. Received: 0.0")

func test_randi_poisson_invalid_lambda_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_poisson(-1.0)
	await assert_error(test_invalid_input).is_push_error("Rate parameter (lambda_param) must be positive. Received: -1.0")

# Tests for randi_pseudo
func test_randi_pseudo_invalid_c_param_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_pseudo(0.0)
	await assert_error(test_invalid_input).is_push_error("Probability increment (c_param) must be in (0.0, 1.0]. Received: 0.0")

func test_randi_pseudo_invalid_c_param_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_pseudo(-0.1)
	await assert_error(test_invalid_input).is_push_error("Probability increment (c_param) must be in (0.0, 1.0]. Received: -0.1")

func test_randi_pseudo_invalid_c_param_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_pseudo(1.1)
	await assert_error(test_invalid_input).is_push_error("Probability increment (c_param) must be in (0.0, 1.0]. Received: 1.1")

# Tests for randi_seige
func test_randi_seige_invalid_w_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(-0.1, 0.5, 0.1, -0.1)
	await assert_error(test_invalid_input).is_push_error("Parameter w (win probability) must be between 0.0 and 1.0. Received: -0.1")

func test_randi_seige_invalid_w_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(1.1, 0.5, 0.1, -0.1)
	await assert_error(test_invalid_input).is_push_error("Parameter w (win probability) must be between 0.0 and 1.0. Received: 1.1")

func test_randi_seige_invalid_c0_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(0.5, -0.1, 0.1, -0.1)
	await assert_error(test_invalid_input).is_push_error("Parameter c_0 (initial capture probability) must be between 0.0 and 1.0. Received: -0.1")

func test_randi_seige_invalid_c0_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randi_seige(0.5, 1.1, 0.1, -0.1)
	await assert_error(test_invalid_input).is_push_error("Parameter c_0 (initial capture probability) must be between 0.0 and 1.0. Received: 1.1")

# Tests for randi_uniform
func test_randi_uniform_invalid_min_greater_than_max() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randi_uniform(10, 5)
	
	# Test that error is logged
	await assert_error(test_call).is_push_error("Minimum value must be less than or equal to maximum value for integer uniform distribution. Received min=10, max=5")
	
	# Test that sentinel value is returned
	var result: int = StatMath.Distributions.randi_uniform(10, 5)
	assert_int(result).is_equal(-1)

# Tests for randf_uniform
func test_randf_uniform_invalid_a_greater_than_b() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_uniform(5.0, 2.0)
	await assert_error(test_invalid_input).is_push_error("Lower bound (a) must be less than or equal to upper bound (b) for Uniform distribution. Received a=5.0, b=2.0")

# Tests for randf_exponential
func test_randf_exponential_invalid_lambda_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_exponential(0.0)
	await assert_error(test_invalid_input).is_push_error("Rate parameter (lambda_param) must be positive for Exponential distribution. Received: 0.0")

func test_randf_exponential_invalid_lambda_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_exponential(-1.0)
	await assert_error(test_invalid_input).is_push_error("Rate parameter (lambda_param) must be positive for Exponential distribution. Received: -1.0")

# Tests for randf_erlang
func test_randf_erlang_invalid_k_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(0, 2.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter (k) must be a positive integer for Erlang distribution. Received: 0")

func test_randf_erlang_invalid_k_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(-1, 2.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter (k) must be a positive integer for Erlang distribution. Received: -1")

func test_randf_erlang_invalid_lambda_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(3, 0.0)
	await assert_error(test_invalid_input).is_push_error("Rate parameter (lambda_param) must be positive for Erlang distribution. Received: 0.0")

func test_randf_erlang_invalid_lambda_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_erlang(3, -1.0)
	await assert_error(test_invalid_input).is_push_error("Rate parameter (lambda_param) must be positive for Erlang distribution. Received: -1.0")

# Tests for randf_normal
func test_randf_normal_invalid_sigma_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_normal(0.0, -1.0)
	await assert_error(test_invalid_input).is_push_error("Standard deviation (sigma) must be positive. Received: -1.0")

func test_randf_normal_invalid_sigma_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_normal(0.0, 0.0)
	await assert_error(test_invalid_input).is_push_error("Standard deviation (sigma) must be positive. Received: 0.0")

# Tests for randf_cauchy
func test_randf_cauchy_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_cauchy(0.0, 0.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Cauchy distribution. Received: 0.0")

func test_randf_cauchy_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_cauchy(0.0, -1.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Cauchy distribution. Received: -1.0")

# Tests for randv_histogram
func test_randv_histogram_empty_values() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram([], [1.0])
	await assert_error(test_call).is_push_error("Values array cannot be empty.")

func test_randv_histogram_empty_probabilities() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a"], [])
	await assert_error(test_call).is_push_error("Values and probabilities arrays must have the same size. Received values size=1, probabilities size=0")

func test_randv_histogram_mismatched_sizes() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a", "b"], [1.0])
	await assert_error(test_call).is_push_error("Values and probabilities arrays must have the same size. Received values size=2, probabilities size=1")

func test_randv_histogram_non_numeric_probability() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a"], ["not_a_number"])
	await assert_error(test_call).is_push_error("Probabilities must be numbers (int or float). Received: not_a_number")

func test_randv_histogram_negative_probability() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a"], [-0.5])
	await assert_error(test_call).is_push_error("Probabilities must be non-negative. Received: -0.5")

func test_randv_histogram_zero_sum_probabilities() -> void:
	var test_call: Callable = func():
		StatMath.Distributions.randv_histogram(["a", "b"], [0.0, 0.0])
	await assert_error(test_call).is_push_error("Sum of probabilities must be positive for normalization. Received sum: 0.0")

# Tests for randf_gamma
func test_randf_gamma_invalid_shape_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(0.0, 1.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter must be positive for Gamma distribution. Received: 0.0")

func test_randf_gamma_invalid_shape_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(-1.0, 1.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter must be positive for Gamma distribution. Received: -1.0")

func test_randf_gamma_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(2.0, 0.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Gamma distribution. Received: 0.0")

func test_randf_gamma_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_gamma(2.0, -1.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Gamma distribution. Received: -1.0")

# Tests for randf_beta
func test_randf_beta_invalid_alpha_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(0.0, 2.0)
	await assert_error(test_invalid_input).is_push_error("Alpha parameter must be positive for Beta distribution. Received: 0.0")

func test_randf_beta_invalid_alpha_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(-1.0, 2.0)
	await assert_error(test_invalid_input).is_push_error("Alpha parameter must be positive for Beta distribution. Received: -1.0")

func test_randf_beta_invalid_beta_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(2.0, 0.0)
	await assert_error(test_invalid_input).is_push_error("Beta parameter must be positive for Beta distribution. Received: 0.0")

func test_randf_beta_invalid_beta_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_beta(2.0, -1.0)
	await assert_error(test_invalid_input).is_push_error("Beta parameter must be positive for Beta distribution. Received: -1.0")

# Tests for randf_triangular
func test_randf_triangular_invalid_mode_too_low() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(5.0, 10.0, 3.0)  # mode < min
	await assert_error(test_invalid_input).is_push_error("Mode value must be greater than or equal to minimum value for Triangular distribution. Received min=5.0, mode=3.0")

func test_randf_triangular_invalid_mode_too_high() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(5.0, 10.0, 12.0)  # mode > max
	await assert_error(test_invalid_input).is_push_error("Mode value must be less than or equal to maximum value for Triangular distribution. Received mode=12.0, max=10.0")

func test_randf_triangular_invalid_max_less_than_min() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(10.0, 5.0, 7.0)  # max < min
	await assert_error(test_invalid_input).is_push_error("Maximum value must be greater than or equal to minimum value for Triangular distribution. Received min=10.0, max=5.0")

func test_randf_triangular_invalid_max_equal_min_with_different_mode() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_triangular(5.0, 5.0, 7.0)  # max = min but mode ≠ min
	await assert_error(test_invalid_input).is_push_error("Mode value must be less than or equal to maximum value for Triangular distribution. Received mode=7.0, max=5.0")

# Tests for randf_pareto
func test_randf_pareto_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(0.0, 1.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Pareto distribution. Received: 0.0")

func test_randf_pareto_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(-1.0, 1.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Pareto distribution. Received: -1.0")

func test_randf_pareto_invalid_shape_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(1.0, 0.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter must be positive for Pareto distribution. Received: 0.0")

func test_randf_pareto_invalid_shape_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_pareto(1.0, -1.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter must be positive for Pareto distribution. Received: -1.0")

# Tests for randf_weibull
func test_randf_weibull_invalid_scale_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_weibull(0.0, 1.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Weibull distribution. Received: 0.0")

func test_randf_weibull_invalid_scale_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_weibull(-1.0, 1.0)
	await assert_error(test_invalid_input).is_push_error("Scale parameter must be positive for Weibull distribution. Received: -1.0")

func test_randf_weibull_invalid_shape_zero() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_weibull(1.0, 0.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter must be positive for Weibull distribution. Received: 0.0")

func test_randf_weibull_invalid_shape_negative() -> void:
	var test_invalid_input: Callable = func():
		StatMath.Distributions.randf_weibull(1.0, -1.0)
	await assert_error(test_invalid_input).is_push_error("Shape parameter must be positive for Weibull distribution. Received: -1.0") 
