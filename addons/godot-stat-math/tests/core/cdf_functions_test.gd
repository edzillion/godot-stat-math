# res://addons/godot-stat-math/tests/core/cdf_functions_test.gd
class_name CdfFunctionsTest extends GdUnitTestSuite

const FLOAT_TOLERANCE: float = StatMath.FLOAT_TOLERANCE
# Using centralized tolerance constants from StatMath class
# Using centralized BOUNDARY_TOLERANCE from StatMath class

var simple_data: Array[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
const CDF_TEST_DATA = preload("res://addons/godot-stat-math/tables/cdf_test_data.gd")

## Builds test parameters from generated scipy-validated data instead of magic numbers
static func _build_cdf_test_parameters() -> Array:
	var test_params: Array = []
	var data: Dictionary = CDF_TEST_DATA.VALUES
	
	# Normal CDF tests
	if data.has("normal_cdf"):
		for case in data["normal_cdf"]:
			test_params.append(["normal", case.params[0], [case.params[1], case.params[2]], case.expected])
	
	# Exponential CDF tests
	if data.has("exponential_cdf"):
		for case in data["exponential_cdf"]:
			test_params.append(["exponential", case.params[0], [case.params[1]], case.expected])
	
	# Gamma CDF tests
	if data.has("gamma_cdf"):
		for case in data["gamma_cdf"]:
			test_params.append(["gamma", case.params[0], [case.params[1], case.params[2]], case.expected])
	
	# Beta CDF tests
	if data.has("beta_cdf"):
		for case in data["beta_cdf"]:
			test_params.append(["beta", case.params[0], [case.params[1], case.params[2]], case.expected])
	
	# Chi-square CDF tests
	if data.has("chi_square_cdf"):
		for case in data["chi_square_cdf"]:
			test_params.append(["chi_square", case.params[0], [case.params[1]], case.expected])
	
	# Weibull CDF tests
	if data.has("weibull_cdf"):
		for case in data["weibull_cdf"]:
			test_params.append(["weibull", case.params[0], [case.params[1], case.params[2]], case.expected])
	
	return test_params

# =============================================================================
# PHASE 3: ENHANCED CDF TESTING - SCIPY VALIDATED DATA
# =============================================================================

## Tests CDF functions against scipy-validated reference values
func test_cdf_scipy_validation_parametrized(distribution: String, x: float, params: Array, expected: float, test_parameters := _build_cdf_test_parameters()) -> void:
	var result: float = _get_cdf_value(distribution, x, params)
	assert_float(result).is_equal_approx(expected, StatMath.INVERSE_FUNCTION_TOLERANCE)

# =============================================================================
# PHASE 3: MONOTONICITY TESTING FOR ALL DISTRIBUTIONS  
# =============================================================================

## Tests that all CDF functions are monotonically non-decreasing
func test_cdf_monotonicity_parametrized(distribution: StatMath.SupportedDistributions, params: Array, test_points: Array[float], test_parameters := [
	[StatMath.SupportedDistributions.NORMAL, [0.0, 1.0], [-3.0, -1.0, 0.0, 1.0, 3.0]],
	[StatMath.SupportedDistributions.NORMAL, [2.0, 0.5], [1.0, 1.5, 2.0, 2.5, 3.0]],
	[StatMath.SupportedDistributions.EXPONENTIAL, [1.0], [0.0, 0.5, 1.0, 2.0, 5.0]],
	[StatMath.SupportedDistributions.EXPONENTIAL, [2.0], [0.0, 0.25, 0.5, 1.0, 2.5]],
	[StatMath.SupportedDistributions.UNIFORM, [1.0, 4.0], [0.5, 1.0, 2.0, 3.0, 4.0, 4.5]],
	[StatMath.SupportedDistributions.GAMMA, [2.0, 1.0], [0.0, 0.5, 1.0, 2.0, 4.0]],
	[StatMath.SupportedDistributions.BETA, [2.0, 3.0], [0.0, 0.2, 0.5, 0.8, 1.0]],
	[StatMath.SupportedDistributions.CHI_SQUARE, [2.0], [0.0, 1.0, 2.0, 5.0, 10.0]],
	[StatMath.SupportedDistributions.T_DISTRIBUTION, [5.0], [-3.0, -1.0, 0.0, 1.0, 3.0]],
	[StatMath.SupportedDistributions.F_DISTRIBUTION, [2.0, 3.0], [0.0, 0.5, 1.0, 2.0, 5.0]],
	[StatMath.SupportedDistributions.WEIBULL, [1.0, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0]],
	[StatMath.SupportedDistributions.PARETO, [1.0, 2.0], [0.5, 1.0, 1.5, 2.0, 3.0]]
	]) -> void:
	var prev_cdf: float = -1.0
	
	for i in range(test_points.size()):
		var x: float = test_points[i]
		var current_cdf: float = _get_cdf_value(distribution, x, params)
		
		# CDF should be monotonically non-decreasing
		assert_float(current_cdf).is_greater_equal(prev_cdf)
		# CDF should be between 0 and 1
		assert_float(current_cdf).is_greater_equal(0.0)
		assert_float(current_cdf).is_less_equal(1.0)
		
		prev_cdf = current_cdf

# =============================================================================
# PHASE 3: BOUNDARY VALUE TESTING
# =============================================================================

## Tests CDF behavior at domain boundaries and special values
func test_cdf_boundary_values_parametrized(distribution: String, boundary_type: String, params: Array, x_value: float, expected: float, test_parameters := [
	# Lower bounds
	["normal", "negative_infinity", [0.0, 1.0], -1000.0, 0.0],
	["exponential", "zero", [1.0], 0.0, 0.0],
	["exponential", "negative", [1.0], -1.0, 0.0], 
	["gamma", "zero", [2.0, 1.0], 0.0, 0.0],
	["gamma", "negative", [2.0, 1.0], -1.0, 0.0],
	["beta", "zero", [2.0, 3.0], 0.0, 0.0],
	["beta", "negative", [2.0, 3.0], -0.1, 0.0],
	["chi_square", "zero", [2.0], 0.0, 0.0],
	["chi_square", "negative", [2.0], -1.0, 0.0],
	["f", "zero", [2.0, 3.0], 0.0, 0.0],
	["f", "negative", [2.0, 3.0], -1.0, 0.0],
	["weibull", "zero", [1.0, 2.0], 0.0, 0.0],
	["weibull", "negative", [1.0, 2.0], -1.0, 0.0],
	["pareto", "below_scale", [2.0, 1.0], 1.0, 0.0],
	# Upper bounds  
	["normal", "positive_infinity", [0.0, 1.0], 1000.0, 1.0],
	["exponential", "large_x", [1.0], 100.0, 1.0],
	["gamma", "large_x", [2.0, 1.0], 100.0, 1.0],
	["beta", "one", [2.0, 3.0], 1.0, 1.0],
	["beta", "above_one", [2.0, 3.0], 1.1, 1.0],
	["chi_square", "large_x", [2.0], 100.0, 1.0],
	["t", "positive_infinity", [5.0], 1000.0, 1.0],
	["f", "large_x", [2.0, 3.0], 100.0, 1.0],
	["weibull", "large_x", [1.0, 2.0], 100.0, 1.0],
	["pareto", "large_x", [1.0, 2.0], 1000.0, 1.0],
	# Special values
	["uniform", "left_bound", [2.0, 5.0], 2.0, 0.0],
	["uniform", "right_bound", [2.0, 5.0], 5.0, 1.0],
	["uniform", "midpoint", [2.0, 5.0], 3.5, 0.5],
	["t", "zero", [5.0], 0.0, 0.5],
	["t", "zero", [1.0], 0.0, 0.5]
	]) -> void:
	var result: float = _get_cdf_value(distribution, x_value, params)
	assert_float(result).is_equal_approx(expected, StatMath.BOUNDARY_TOLERANCE)

# =============================================================================
# PHASE 3: SPECIAL VALUES AND QUANTILES
# =============================================================================

## Tests CDF values at standard quantiles (25th, 50th, 75th percentiles)
func test_cdf_standard_quantiles_parametrized(distribution: String, quantile_type: String, params: Array, expected_range_min: float, expected_range_max: float, test_parameters := [
	# Testing that median (50th percentile) gives CDF ≈ 0.5
	["normal", "median_check", [0.0, 1.0], 0.49, 0.51],  # x=0 should give ~0.5
	["exponential", "median_check", [1.0], 0.49, 0.51],  # x=ln(2) ≈ 0.693 should give ~0.5
	["uniform", "median_check", [2.0, 6.0], 0.49, 0.51],  # x=4 should give ~0.5
	["gamma", "first_quartile", [2.0, 1.0], 0.20, 0.30],  # x around 0.7 should give ~0.25
	["beta", "first_quartile", [2.0, 2.0], 0.20, 0.30],  # x around 0.3 should give ~0.25  
	["chi_square", "third_quartile", [1.0], 0.70, 0.80],  # x around 1.3 should give ~0.75
	["t", "median_check", [10.0], 0.49, 0.51],  # x=0 should give ~0.5
	["f", "median_check", [10.0, 10.0], 0.49, 0.51],  # x around 1 should give ~0.5
	["weibull", "median_check", [1.0, 1.0], 0.49, 0.51],  # x=ln(2) should give ~0.5
	["pareto", "median_check", [1.0, 1.0], 0.49, 0.51]   # x=2 should give ~0.5
	]) -> void:
	# Use specific test points that should be near the expected quantiles
	var test_x: float
	match [distribution, quantile_type]:
		["normal", "median_check"]: test_x = 0.0
		["exponential", "median_check"]: test_x = 0.693147
		["uniform", "median_check"]: test_x = 4.0  # midpoint of [2,6]
		["gamma", "first_quartile"]: test_x = 0.96  # 25th percentile for Gamma(2.0, 1.0)
		["beta", "first_quartile"]: test_x = 0.3   # approximate 25th percentile
		["chi_square", "third_quartile"]: test_x = 1.3  # approximate 75th percentile
		["t", "median_check"]: test_x = 0.0
		["f", "median_check"]: test_x = 1.0
		["weibull", "median_check"]: test_x = 0.693147
		["pareto", "median_check"]: test_x = 2.0
		_: test_x = 1.0
	
	var result: float = _get_cdf_value(distribution, test_x, params)
	assert_float(result).is_greater_equal(expected_range_min)
	assert_float(result).is_less_equal(expected_range_max)

# =============================================================================
# PHASE 3: PARAMETER VALIDATION ENHANCEMENT
# =============================================================================

## Enhanced parameter validation tests for probability bounds and edge cases
func test_cdf_parameter_validation_enhanced_parametrized(distribution: String, validation_type: String, params: Array, expected_error: String, test_parameters := [
	# Continuous distributions
	["normal", "sigma_zero", [0.0, 0.0], "Standard deviation (sigma) must be positive for Normal CDF. Received: 0.0"],
	["normal", "sigma_negative", [0.0, -1.0], "Standard deviation (sigma) must be positive for Normal CDF. Received: -1.0"],
	["exponential", "lambda_zero", [0.0], "Rate parameter (lambda_param) must be positive for Exponential CDF. Received: 0.0"],
	["exponential", "lambda_negative", [-1.0], "Rate parameter (lambda_param) must be positive for Exponential CDF. Received: -1.0"],
	["gamma", "shape_zero", [0.0, 1.0], "Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=0.0, theta_scale=1.0"],
	["gamma", "scale_zero", [1.0, 0.0], "Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=1.0, theta_scale=0.0"],
	["beta", "alpha_zero", [0.0, 1.0], "Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=0.0, beta_param=1.0"],
	["beta", "beta_negative", [1.0, -1.0], "Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=1.0, beta_param=-1.0"],
	["uniform", "a_greater_than_b", [5.0, 2.0], "Parameter a must be less than or equal to b for Uniform CDF. Received a=5.0, b=2.0"],
	# Special distributions
	["chi_square", "df_zero", [0.0], "Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: 0.0"],
	["chi_square", "df_negative", [-1.0], "Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: -1.0"],
	["t", "df_zero", [0.0], "Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: 0.0"],
	["t", "df_negative", [-5.0], "Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: -5.0"],
	["f", "d1_zero", [0.0, 5.0], "Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=0.0, d2_df=5.0"],
	["f", "d2_negative", [5.0, -1.0], "Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=5.0, d2_df=-1.0"],
	# Heavy-tailed distributions
	["weibull", "scale_zero", [0.0, 2.0], "Scale parameter must be positive for Weibull CDF. Received: 0.0"],
	["weibull", "shape_negative", [1.0, -1.0], "Shape parameter must be positive for Weibull CDF. Received: -1.0"],
	["pareto", "scale_zero", [0.0, 1.0], "Scale parameter must be positive for Pareto CDF. Received: 0.0"],
	["pareto", "shape_negative", [1.0, -2.0], "Shape parameter must be positive for Pareto CDF. Received: -2.0"],
	# Discrete distributions
	["binomial", "n_negative", [-1, 0.5], "Number of trials (n_trials) must be non-negative. Received: -1"],
	["binomial", "p_negative", [5, -0.1], "Probability (p_prob) must be between 0.0 and 1.0. Received: -0.1"],
	["binomial", "p_greater_than_one", [5, 1.5], "Probability (p_prob) must be between 0.0 and 1.0. Received: 1.5"],
	["poisson", "lambda_negative", [-1.0], "Rate parameter (lambda_param) must be non-negative for Poisson CDF. Received: -1.0"],
	["geometric", "p_zero", [0.0], "Success probability (p_prob) must be in (0,1]. Received: 0.0"],
	["geometric", "p_greater_than_one", [1.5], "Success probability (p_prob) must be in (0,1]. Received: 1.5"],
	["negative_binomial", "r_zero", [0, 0.5], "Number of successes (r_successes) must be positive. Received: 0"],
	["negative_binomial", "p_zero", [5, 0.0], "Success probability (p_prob) must be in (0,1]. Received: 0.0"]
	]) -> void:
	var test_call: Callable = func():
		match distribution:
			"normal": StatMath.CdfFunctions.normal_cdf(1.0, params[0], params[1])
			"exponential": StatMath.CdfFunctions.exponential_cdf(1.0, params[0])
			"gamma": StatMath.CdfFunctions.gamma_cdf(1.0, params[0], params[1])
			"beta": StatMath.CdfFunctions.beta_cdf(0.5, params[0], params[1])
			"uniform": StatMath.CdfFunctions.uniform_cdf(3.0, params[0], params[1])
			"chi_square": StatMath.CdfFunctions.chi_square_cdf(1.0, params[0])
			"t": StatMath.CdfFunctions.t_cdf(1.0, params[0])
			"f": StatMath.CdfFunctions.f_cdf(1.0, params[0], params[1])
			"weibull": StatMath.CdfFunctions.weibull_cdf(1.0, params[0], params[1])
			"pareto": StatMath.CdfFunctions.pareto_cdf(2.0, params[0], params[1])
			"binomial": StatMath.CdfFunctions.binomial_cdf(5, int(params[0]), params[1])
			"poisson": StatMath.CdfFunctions.poisson_cdf(5, params[0])
			"geometric": StatMath.CdfFunctions.geometric_cdf(5, params[0])
			"negative_binomial": StatMath.CdfFunctions.negative_binomial_cdf(10, int(params[0]), params[1])
	
	await assert_error(test_call).is_push_error(expected_error)

# =============================================================================
# HELPER FUNCTIONS FOR PHASE 3 TESTING
# =============================================================================

## Enhanced helper function to get CDF values for all supported distributions  
func _get_cdf_value(distribution: Variant, x: float, params: Array) -> float:
	# Handle both string and enum inputs during transition
	var dist_enum: StatMath.SupportedDistributions
	if distribution is String:
		dist_enum = _string_to_enum(distribution)
	else:
		dist_enum = distribution
	
	match dist_enum:
		StatMath.SupportedDistributions.NORMAL:
			return StatMath.CdfFunctions.normal_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.EXPONENTIAL:
			# Convert from scale parameter to rate parameter: λ = 1/scale
			return StatMath.CdfFunctions.exponential_cdf(x, 1.0 / params[0])
		StatMath.SupportedDistributions.UNIFORM:
			return StatMath.CdfFunctions.uniform_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.BETA:
			return StatMath.CdfFunctions.beta_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.GAMMA:
			return StatMath.CdfFunctions.gamma_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.WEIBULL:
			return StatMath.CdfFunctions.weibull_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.PARETO:
			return StatMath.CdfFunctions.pareto_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.CHI_SQUARE:
			return StatMath.CdfFunctions.chi_square_cdf(x, params[0])
		StatMath.SupportedDistributions.T_DISTRIBUTION:
			return StatMath.CdfFunctions.t_cdf(x, params[0])
		StatMath.SupportedDistributions.F_DISTRIBUTION:
			return StatMath.CdfFunctions.f_cdf(x, params[0], params[1])
		StatMath.SupportedDistributions.BINOMIAL:
			return StatMath.CdfFunctions.binomial_cdf(int(x), int(params[0]), params[1])
		StatMath.SupportedDistributions.POISSON:
			return StatMath.CdfFunctions.poisson_cdf(int(x), params[0])
		StatMath.SupportedDistributions.GEOMETRIC:
			return StatMath.CdfFunctions.geometric_cdf(int(x), params[0])
		StatMath.SupportedDistributions.NEGATIVE_BINOMIAL:
			return StatMath.CdfFunctions.negative_binomial_cdf(int(x), int(params[0]), params[1])
		_:
			push_error("Unknown distribution enum: " + str(dist_enum))
			return NAN

## Helper function to convert string distribution names to enum values
func _string_to_enum(distribution: String) -> StatMath.SupportedDistributions:
	match distribution:
		"normal": return StatMath.SupportedDistributions.NORMAL
		"exponential": return StatMath.SupportedDistributions.EXPONENTIAL
		"uniform": return StatMath.SupportedDistributions.UNIFORM
		"beta": return StatMath.SupportedDistributions.BETA
		"gamma": return StatMath.SupportedDistributions.GAMMA
		"weibull": return StatMath.SupportedDistributions.WEIBULL
		"pareto": return StatMath.SupportedDistributions.PARETO
		"chi_square": return StatMath.SupportedDistributions.CHI_SQUARE
		"t": return StatMath.SupportedDistributions.T_DISTRIBUTION
		"f": return StatMath.SupportedDistributions.F_DISTRIBUTION
		"binomial": return StatMath.SupportedDistributions.BINOMIAL
		"poisson": return StatMath.SupportedDistributions.POISSON
		"geometric": return StatMath.SupportedDistributions.GEOMETRIC
		"negative_binomial": return StatMath.SupportedDistributions.NEGATIVE_BINOMIAL
		_:
			push_error("Unknown distribution: " + distribution)
			return StatMath.SupportedDistributions.NORMAL  # Fallback value

# --- Uniform CDF ---
func test_uniform_cdf_basic_range() -> void:
	var a: float = 2.0
	var b: float = 5.0
	var x: float = 3.0
	var result: float = StatMath.CdfFunctions.uniform_cdf(x, a, b)
	assert_float(result).is_equal_approx((x - a) / (b - a), FLOAT_TOLERANCE)

func test_uniform_cdf_x_below_a() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(1.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_uniform_cdf_x_above_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(6.0, 2.0, 5.0)
	assert_float(result).is_equal_approx(1.0, FLOAT_TOLERANCE)

func test_uniform_cdf_a_equals_b() -> void:
	var result: float = StatMath.CdfFunctions.uniform_cdf(2.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, FLOAT_TOLERANCE)

func test_uniform_cdf_invalid_a_greater_than_b() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.uniform_cdf(2.0, 5.0, 2.0)
	await assert_error(test_call).is_push_error("Parameter a must be less than or equal to b for Uniform CDF. Received a=5.0, b=2.0")

# --- Normal CDF ---
func test_normal_cdf_standard_normal() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(0.0)
	assert_float(result).is_equal_approx(0.5, FLOAT_TOLERANCE)

func test_normal_cdf_mu_sigma() -> void:
	var result: float = StatMath.CdfFunctions.normal_cdf(2.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.5, FLOAT_TOLERANCE)

func test_normal_cdf_known_value() -> void:
	# Value from scipy.stats.norm.cdf(1.96)
	var result: float = StatMath.CdfFunctions.normal_cdf(1.96)
	assert_float(result).is_equal_approx(0.9750021, FLOAT_TOLERANCE)

func test_normal_cdf_invalid_sigma_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 0.0)
	await assert_error(test_call).is_push_error("Standard deviation (sigma) must be positive for Normal CDF. Received: 0.0")

# --- Exponential CDF ---
func test_exponential_cdf_typical() -> void:
	# Value from scipy.stats.expon.cdf(1.0, scale=1/2.0) -> 0.86466
	var result: float = StatMath.CdfFunctions.exponential_cdf(1.0, 2.0)
	assert_float(result).is_equal_approx(0.8646647, FLOAT_TOLERANCE)

func test_exponential_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.exponential_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_exponential_cdf_invalid_lambda_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.exponential_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be positive for Exponential CDF. Received: 0.0")

# --- Beta CDF ---
func test_beta_cdf_symmetric() -> void:
	# For symmetric alpha=beta, CDF at 0.5 should be 0.5
	var result: float = StatMath.CdfFunctions.beta_cdf(0.5, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.5, FLOAT_TOLERANCE)

func test_beta_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_beta_cdf_x_one() -> void:
	var result: float = StatMath.CdfFunctions.beta_cdf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, FLOAT_TOLERANCE)

func test_beta_cdf_invalid_alpha_beta() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.beta_cdf(0.5, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=-1.0, beta_param=2.0")

# --- Gamma CDF ---
func test_gamma_cdf_known_value() -> void:
	# Value from scipy.stats.gamma.cdf(2.0, a=2.0, scale=1.0) -> 0.59399415
	var result: float = StatMath.CdfFunctions.gamma_cdf(2.0, 2.0, 1.0)
	assert_float(result).is_equal_approx(0.59399415, FLOAT_TOLERANCE)

func test_gamma_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.gamma_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_gamma_cdf_invalid_shape_scale() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.gamma_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=0.0, theta_scale=2.0")

# --- Chi-Square CDF ---
func test_chi_square_cdf_known_value() -> void:
	# Value from scipy.stats.chi2.cdf(3.0, df=2.0) -> 0.77687
	var result: float = StatMath.CdfFunctions.chi_square_cdf(3.0, 2.0)
	assert_float(result).is_equal_approx(0.7768698, FLOAT_TOLERANCE)

func test_chi_square_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.chi_square_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_chi_square_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.chi_square_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: 0.0")

# --- F-Distribution CDF ---
func test_f_cdf_known_value() -> void:
	# Value from scipy.stats.f.cdf(1.5, dfn=2.0, dfd=2.0) -> 0.598
	var result: float = StatMath.CdfFunctions.f_cdf(1.5, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.598, FLOAT_TOLERANCE)

func test_f_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.f_cdf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_f_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.f_cdf(1.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=0.0, d2_df=2.0")

# --- Student's t-Distribution CDF ---
func test_t_cdf_x_zero() -> void:
	var result: float = StatMath.CdfFunctions.t_cdf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.5, FLOAT_TOLERANCE)

func test_t_cdf_known_value() -> void:
	# Value from scipy.stats.t.cdf(1.0, df=10)
	var result: float = StatMath.CdfFunctions.t_cdf(1.0, 10.0)
	assert_float(result).is_equal_approx(0.829553, StatMath.NUMERICAL_TOLERANCE) # Slightly lower tolerance for t-dist approximation

func test_t_cdf_invalid_df() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.t_cdf(1.0, 0.0)
	await assert_error(test_call).is_push_error("Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: 0.0")

# --- Binomial CDF ---
func test_binomial_cdf_known_value() -> void:
	# Value from scipy.stats.binom.cdf(k=2, n=5, p=0.5)
	# P(0) = 0.03125, P(1)=0.15625, P(2)=0.3125. Sum = 0.5
	var result: float = StatMath.CdfFunctions.binomial_cdf(2, 5, 0.5)
	assert_float(result).is_equal_approx(0.5, FLOAT_TOLERANCE)

func test_binomial_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(-1, 5, 0.5)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_binomial_cdf_k_ge_n() -> void:
	var result: float = StatMath.CdfFunctions.binomial_cdf(5, 5, 0.5)
	assert_float(result).is_equal_approx(1.0, FLOAT_TOLERANCE)

func test_binomial_cdf_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, -1, 0.5)
	await assert_error(test_call).is_push_error("Number of trials (n_trials) must be non-negative. Received: -1")

func test_binomial_cdf_invalid_p() -> void:
	var test_call_low: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, 5, -0.1)
	await assert_error(test_call_low).is_push_error("Probability (p_prob) must be between 0.0 and 1.0. Received: -0.1")
	
	var test_call_high: Callable = func():
		StatMath.CdfFunctions.binomial_cdf(2, 5, 1.1)
	await assert_error(test_call_high).is_push_error("Probability (p_prob) must be between 0.0 and 1.0. Received: 1.1")

# --- Poisson CDF ---
func test_poisson_cdf_known_value() -> void:
	# Value from scipy.stats.poisson.cdf(k=2, mu=2.0) -> 0.676676416183063
	var result: float = StatMath.CdfFunctions.poisson_cdf(2, 2.0)
	assert_float(result).is_equal_approx(0.676676416183063, FLOAT_TOLERANCE)

func test_poisson_cdf_k_negative() -> void:
	var result: float = StatMath.CdfFunctions.poisson_cdf(-1, 2.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_poisson_cdf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.poisson_cdf(2, -1.0)
	await assert_error(test_call).is_push_error("Rate parameter (lambda_param) must be non-negative for Poisson CDF. Received: -1.0")

# --- Geometric CDF ---
func test_geometric_cdf_known_value() -> void:
	# Value from 1 - (1-p)^k = 1 - (0.5)^3 = 0.875
	var result: float = StatMath.CdfFunctions.geometric_cdf(3, 0.5)
	assert_float(result).is_equal_approx(0.875, FLOAT_TOLERANCE)

func test_geometric_cdf_k_less_than_1() -> void:
	var result: float = StatMath.CdfFunctions.geometric_cdf(0, 0.5)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_geometric_cdf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.geometric_cdf(2, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")

# --- Negative Binomial CDF ---
func test_negative_binomial_cdf_known_value() -> void:
	# P(k<=5; r=3, p=0.5) = sum of PMF for k=3,4,5
	# PMF(k;r,p) = C(k-1,r-1) * p^r * (1-p)^(k-r)
	# P(3) = C(2,2)*0.5^3*0.5^0 = 0.125
	# P(4) = C(3,2)*0.5^3*0.5^1 = 3 * 0.125 * 0.5 = 0.1875
	# P(5) = C(4,2)*0.5^3*0.5^2 = 6 * 0.125 * 0.25 = 0.1875
	# Sum = 0.125 + 0.1875 + 0.1875 = 0.5
	var result: float = StatMath.CdfFunctions.negative_binomial_cdf(5, 3, 0.5)
	assert_float(result).is_equal_approx(0.5, FLOAT_TOLERANCE)

func test_negative_binomial_cdf_k_less_than_r() -> void:
	var result: float = StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.5)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_negative_binomial_cdf_invalid_r() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 0, 0.5)
	await assert_error(test_call).is_push_error("Number of successes (r_successes) must be positive. Received: 0")

func test_negative_binomial_cdf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.negative_binomial_cdf(2, 3, 0.0)
	await assert_error(test_call).is_push_error("Success probability (p_prob) must be in (0,1]. Received: 0.0")

# --- Pareto CDF ---
func test_pareto_cdf_x_equals_scale() -> void:
	var result: float = StatMath.CdfFunctions.pareto_cdf(2.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_pareto_cdf_basic_calculation() -> void:
	# For x = 4, scale = 2, shape = 3: F(4) = 1 - (2/4)^3 = 1 - 0.125 = 0.875
	var result: float = StatMath.CdfFunctions.pareto_cdf(4.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.875, FLOAT_TOLERANCE)

func test_pareto_cdf_x_below_scale() -> void:
	var result: float = StatMath.CdfFunctions.pareto_cdf(1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

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
	
	assert_float(result1).is_equal_approx(result2, StatMath.DETERMINISM_TOLERANCE)

func test_pareto_cdf_invalid_scale_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 0.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive for Pareto CDF. Received: 0.0")

func test_pareto_cdf_invalid_scale_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, -1.0, 2.0)
	await assert_error(test_call).is_push_error("Scale parameter must be positive for Pareto CDF. Received: -1.0")

func test_pareto_cdf_invalid_shape_zero() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 2.0, 0.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive for Pareto CDF. Received: 0.0")

func test_pareto_cdf_invalid_shape_negative() -> void:
	var test_call: Callable = func():
		StatMath.CdfFunctions.pareto_cdf(3.0, 2.0, -1.0)
	await assert_error(test_call).is_push_error("Shape parameter must be positive for Pareto CDF. Received: -1.0")

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
	assert_float(probabilities[0]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

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

# --- Weibull CDF ---
func test_weibull_cdf_known_value() -> void:
	# Using scipy-validated test data  
	var params: Array[float] = [1.5, 2.0, 1.0]  # x, scale, shape
	var expected: float = 0.527633447258985  # scipy.stats.weibull_min.cdf(1.5, c=1.0, scale=2.0)
	var result: float = StatMath.CdfFunctions.weibull_cdf(params[0], params[1], params[2])
	assert_float(result).is_equal_approx(expected, StatMath.INVERSE_FUNCTION_TOLERANCE)

func test_weibull_cdf_basic_calculation() -> void:
	# For x = 2, scale = 2, shape = 2: F(2) = 1 - exp(-(2/2)^2) = 1 - exp(-1) ≈ 0.632
	var result: float = StatMath.CdfFunctions.weibull_cdf(2.0, 2.0, 2.0)
	var expected: float = 1.0 - exp(-1.0)
	assert_float(result).is_equal_approx(expected, FLOAT_TOLERANCE)

func test_weibull_cdf_x_below_zero() -> void:
	var result: float = StatMath.CdfFunctions.weibull_cdf(-1.0, 2.0, 3.0)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_weibull_cdf_exponential_case() -> void:
	# When shape = 1, Weibull becomes exponential: F(x) = 1 - exp(-x/λ)
	var x: float = 3.0
	var scale: float = 2.0 # This is lambda_scale
	var shape: float = 1.0
	
	var weibull_result: float = StatMath.CdfFunctions.weibull_cdf(x, scale, shape)
	# For exponential, lambda_param is the rate, which is 1.0 / scale
	var exponential_result: float = StatMath.CdfFunctions.exponential_cdf(x, 1.0 / scale)
	
	assert_float(weibull_result).is_equal_approx(exponential_result, FLOAT_TOLERANCE)

func test_weibull_cdf_monotonicity() -> void:
	# CDF should be monotonically increasing
	var scale: float = 2.0
	var shape: float = 2.0
	
	var x1: float = 0.5
	var x2: float = 1.0
	var x3: float = 2.0
	var x4: float = 4.0
	
	var cdf1: float = StatMath.CdfFunctions.weibull_cdf(x1, scale, shape)
	var cdf2: float = StatMath.CdfFunctions.weibull_cdf(x2, scale, shape)
	var cdf3: float = StatMath.CdfFunctions.weibull_cdf(x3, scale, shape)
	var cdf4: float = StatMath.CdfFunctions.weibull_cdf(x4, scale, shape)
	
	assert_float(cdf1).is_less_equal(cdf2)
	assert_float(cdf2).is_less_equal(cdf3)
	assert_float(cdf3).is_less_equal(cdf4)

func test_weibull_cdf_bounds() -> void:
	# CDF should always be between 0 and 1
	var test_cases: Array[Array] = [
		[1.0, 1.0, 0.5], [2.0, 3.0, 2.0], [5.0, 2.0, 4.0],
		[10.0, 10.0, 1.5], [0.5, 1.0, 10.0]
	]
	
	for case in test_cases:
		var x: float = case[0]
		var shape: float = case[1]
		var scale: float = case[2]
		
		var result: float = StatMath.CdfFunctions.weibull_cdf(x, scale, shape)
		
		assert_float(result).is_greater_equal(0.0)
		assert_float(result).is_less_equal(1.0)
		assert_bool(is_nan(result)).is_false()

func test_weibull_cdf_invalid_shape() -> void:
	var test_call_zero: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 1.0, 0.0)
	await assert_error(test_call_zero).is_push_error("Shape parameter must be positive for Weibull CDF. Received: 0.0")

	var test_call_neg: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 1.0, -1.0)
	await assert_error(test_call_neg).is_push_error("Shape parameter must be positive for Weibull CDF. Received: -1.0")

func test_weibull_cdf_invalid_scale() -> void:
	var test_call_zero: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, 0.0, 1.0)
	await assert_error(test_call_zero).is_push_error("Scale parameter must be positive for Weibull CDF. Received: 0.0")

	var test_call_neg: Callable = func():
		StatMath.CdfFunctions.weibull_cdf(1.0, -1.0, 1.0)
	await assert_error(test_call_neg).is_push_error("Scale parameter must be positive for Weibull CDF. Received: -1.0")

# --- Game Development Use Cases for Weibull CDF ---

func test_weibull_cdf_equipment_failure_probability() -> void:
	# Example: probability that equipment fails before a certain time
	var time_threshold: float = 800.0  # Hours
	var characteristic_life: float = 1000.0  # Hours
	var wear_pattern: float = 2.5  # Increasing failure rate
	
	var failure_probability: float = StatMath.CdfFunctions.weibull_cdf(time_threshold, characteristic_life, wear_pattern)
	
	# Should be a valid probability
	assert_float(failure_probability).is_between(0.0, 1.0)
	
	# Should be reasonable for this scenario (some probability of early failure)
	assert_float(failure_probability).is_greater(0.1)
	assert_float(failure_probability).is_less(0.9)

func test_weibull_cdf_survival_analysis() -> void:
	# Example: probability of surviving less than a certain time
	var survival_times: Array[float] = [100.0, 300.0, 600.0, 1000.0]
	var base_survival: float = 500.0  # Base survival time
	var hazard_pattern: float = 1.8  # Increasing hazard
	
	var probabilities: Array[float] = []
	for time in survival_times:
		var prob: float = StatMath.CdfFunctions.weibull_cdf(time, base_survival, hazard_pattern)
		probabilities.append(prob)
	
	# Probabilities should increase with time
	for i in range(probabilities.size() - 1):
		assert_float(probabilities[i]).is_less_equal(probabilities[i + 1])
	
	# All should be valid probabilities
	for prob in probabilities:
		assert_float(prob).is_between(0.0, 1.0)

func test_weibull_cdf_wind_speed_distribution() -> void:
	# Example: probability of wind speed being below threshold (Rayleigh case)
	var wind_threshold: float = 20.0  # km/h
	var characteristic_wind: float = 15.0  # km/h
	var rayleigh_shape: float = 2.0  # Rayleigh distribution
	
	var prob_below_threshold: float = StatMath.CdfFunctions.weibull_cdf(wind_threshold, characteristic_wind, rayleigh_shape)
	
	# Should be valid probability
	assert_float(prob_below_threshold).is_between(0.0, 1.0)
	
	# Should be reasonable for wind speed analysis
	assert_float(prob_below_threshold).is_greater(0.3)

func test_weibull_cdf_component_reliability() -> void:
	# Example: reliability analysis for electronic components
	var operating_time: float = 4000.0  # Hours
	var design_life: float = 5000.0  # Hours
	var reliability_shape: float = 3.0  # Sharp wear-out
	
	var failure_probability: float = StatMath.CdfFunctions.weibull_cdf(operating_time, design_life, reliability_shape)
	
	# Should be valid probability
	assert_float(failure_probability).is_between(0.0, 1.0)
	
	# Before design life, failure probability should be relatively low
	assert_float(failure_probability).is_less(0.7)

func test_weibull_cdf_quest_completion_analysis() -> void:
	# Example: analyzing quest completion time distributions
	var completion_times: Array[float] = [30.0, 60.0, 120.0, 180.0]
	var typical_time: float = 90.0  # Minutes
	var difficulty_curve: float = 2.2  # Increasing difficulty
	
	var completion_probabilities: Array[float] = []
	for time in completion_times:
		var prob: float = StatMath.CdfFunctions.weibull_cdf(time, typical_time, difficulty_curve)
		completion_probabilities.append(prob)
	
	# Should maintain monotonicity
	for i in range(completion_probabilities.size() - 1):
		assert_float(completion_probabilities[i]).is_less_equal(completion_probabilities[i + 1])
	
	# All should be valid
	for prob in completion_probabilities:
		assert_float(prob).is_between(0.0, 1.0)

func test_weibull_cdf_resource_depletion_modeling() -> void:
	# Example: modeling resource node depletion probability
	var extraction_time: float = 1500.0  # Time units
	var resource_lifetime: float = 2000.0  # Expected lifetime
	var depletion_pattern: float = 1.2  # Slight acceleration
	
	var depletion_probability: float = StatMath.CdfFunctions.weibull_cdf(extraction_time, resource_lifetime, depletion_pattern)
	
	# Should be valid probability
	assert_float(depletion_probability).is_between(0.0, 1.0)
	
	# Should be reasonable for resource management
	assert_float(depletion_probability).is_greater(0.2)
	assert_float(depletion_probability).is_less(0.9)

func test_weibull_cdf_network_latency_analysis() -> void:
	# Example: network latency spike duration analysis
	var latency_threshold: float = 100.0  # Milliseconds
	var typical_spike_duration: float = 75.0  # Milliseconds
	var recovery_pattern: float = 2.8  # Sharp recovery
	
	var prob_short_spike: float = StatMath.CdfFunctions.weibull_cdf(latency_threshold, typical_spike_duration, recovery_pattern)
	
	# Should be valid probability
	assert_float(prob_short_spike).is_between(0.0, 1.0)
	
	# Most spikes should be relatively short
	assert_float(prob_short_spike).is_greater(0.4) 
