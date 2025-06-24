# res://addons/godot-stat-math/tests/core/pmf_pdf_functions/mathematical_property_test.gd
class_name PmfPdfFunctionsMathematicalPropertyTest extends GdUnitTestSuite

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- Beta PDF Mathematical Properties ---

func test_beta_pdf_uniform_special_case() -> void:
	# Beta(1,1) is uniform on [0,1] - testing mathematical property
	var x_values: Array[float] = [0.2, 0.4, 0.6, 0.8]
	for x in x_values:
		var result: float = StatMath.PmfPdfFunctions.beta_pdf(x, 1.0, 1.0)
		assert_float(result).is_equal_approx(1.0, StatMath.PROBABILITY_TOLERANCE)

func test_beta_pdf_symmetry_property() -> void:
	# For alpha = beta, PDF should be symmetric around x = 0.5
	var test_cases: Array[Array] = [
		[0.3, 2.0, 2.0],  # Basic symmetry test
		[0.2, 3.0, 3.0],  # Different parameters
		[0.4, 1.5, 1.5],  # Non-integer parameters
	]
	
	for case in test_cases:
		var x: float = case[0]
		var alpha: float = case[1]
		var beta_param: float = case[2]
		var x_mirror: float = 1.0 - x
		var pdf1: float = StatMath.PmfPdfFunctions.beta_pdf(x, alpha, beta_param)
		var pdf2: float = StatMath.PmfPdfFunctions.beta_pdf(x_mirror, alpha, beta_param)
		assert_float(pdf1).is_equal_approx(pdf2, StatMath.PROBABILITY_TOLERANCE)

# --- Chi-squared PDF Mathematical Properties ---

func test_chi_squared_pdf_exponential_relationship() -> void:
	# Chi-squared with df=2 is equivalent to exponential with rate=1/2
	var x_values: Array[float] = [1.0, 1.5, 2.0, 3.0, 5.0]
	for x in x_values:
		var chi_squared_result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, 2.0)
		var exponential_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, 0.5)
		assert_float(chi_squared_result).is_equal_approx(exponential_result, StatMath.PROBABILITY_TOLERANCE)

func test_chi_squared_pdf_gamma_relationship() -> void:
	# Chi-squared is Gamma(k/2, 2)
	var df: float = 4.0
	var x_values: Array[float] = [1.0, 2.0, 5.0]
	
	for x in x_values:
		if x > 0:
			var chi_sq_result: float = StatMath.PmfPdfFunctions.chi_squared_pdf(x, df)
			var gamma_result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, df / 2.0, 2.0)
			assert_float(chi_sq_result).is_equal_approx(gamma_result, StatMath.PROBABILITY_TOLERANCE)

# --- Student's t PDF Mathematical Properties ---

func test_t_pdf_special_cases() -> void:
	# For large df, approaches standard normal
	var result1: float = StatMath.PmfPdfFunctions.t_pdf(0.0, 1000.0)
	var expected1: float = 1.0 / sqrt(2.0 * PI)  # Standard normal PDF at x=0
	assert_float(result1).is_equal_approx(expected1, StatMath.ASYMPTOTIC_TOLERANCE)
	
	# t(1) is Cauchy distribution
	var result2: float = StatMath.PmfPdfFunctions.t_pdf(0.0, 1.0)
	var expected2: float = 1.0 / PI  # Cauchy PDF at x=0
	assert_float(result2).is_equal_approx(expected2, StatMath.PROBABILITY_TOLERANCE)

func test_t_pdf_symmetry() -> void:
	# t-distribution PDF should be symmetric: PDF(x) = PDF(-x)
	var test_cases: Array[Array] = [
		[1.0, 5.0],
		[2.0, 3.0],
		[0.5, 10.0],
	]
	
	for case in test_cases:
		var x: float = case[0]
		var df_nu: float = case[1]
		var result1: float = StatMath.PmfPdfFunctions.t_pdf(x, df_nu)
		var result2: float = StatMath.PmfPdfFunctions.t_pdf(-x, df_nu)
		assert_float(result1).is_equal_approx(result2, StatMath.PROBABILITY_TOLERANCE)

# --- F-distribution PDF Mathematical Properties ---

func test_f_pdf_beta_distribution_relationship() -> void:
	# If X ~ F(d1, d2), then Y = (d1*X)/(d1*X + d2) ~ Beta(d1/2, d2/2)
	# Testing the equivalence by verifying that transformation is consistent
	# For simple case: F(2,2) with x=1 should give y=0.5, both distributions symmetric
	
	var d1: float = 2.0
	var d2: float = 2.0
	var x: float = 1.0
	
	# Calculate F-distribution PDF
	var f_pdf_result: float = StatMath.PmfPdfFunctions.f_pdf(x, d1, d2)
	
	# Transform to Beta domain: y = (d1*x)/(d1*x + d2)
	var y: float = (d1 * x) / (d1 * x + d2)  # Should be 0.5 for symmetric case
	
	# Test the transformation point is correct
	assert_float(y).is_equal_approx(0.5, StatMath.BOUNDARY_TOLERANCE)
	
	# For F(2,2) at x=1, this is a special symmetric case
	# f_F(1; 2,2) = (sqrt((2*1)^2 * 2^2 / (2*1+2)^4)) / (1 * B(1,1))
	# = sqrt(16/256) / (1 * 1) = sqrt(1/16) = 1/4 = 0.25
	
	# Beta(1,1) at y=0.5 should be 1.0 (uniform distribution)
	var beta_pdf_result: float = StatMath.PmfPdfFunctions.beta_pdf(0.5, 1.0, 1.0)
	assert_float(beta_pdf_result).is_equal_approx(1.0, StatMath.PROBABILITY_TOLERANCE)
	
	# This is a known mathematical identity for this special case
	assert_float(f_pdf_result).is_equal_approx(0.25, StatMath.PROBABILITY_TOLERANCE)

# --- Gamma PDF Mathematical Properties ---

func test_gamma_pdf_exponential_special_case() -> void:
	# Test that Gamma PDF reduces to Exponential when shape=1
	var x_values: Array[float] = [0.5, 1.0, 2.0, 3.0]
	var scale: float = 2.0  # theta
	var rate: float = 1.0 / scale  # lambda = 1/theta
	
	for x in x_values:
		if x > 0:
			var gamma_result: float = StatMath.PmfPdfFunctions.gamma_pdf(x, 1.0, scale)
			var exp_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, rate)
			assert_float(gamma_result).is_equal_approx(exp_result, StatMath.PROBABILITY_TOLERANCE)

# --- Weibull PDF Mathematical Properties ---

func test_weibull_pdf_exponential_special_case() -> void:
	# When shape=1, Weibull becomes exponential
	var x_values: Array[float] = [0.5, 1.0, 2.0, 3.0]
	var scale: float = 2.0
	var shape: float = 1.0
	var rate: float = 1.0 / scale
	
	for x in x_values:
		if x >= 0:
			var weibull_result: float = StatMath.PmfPdfFunctions.weibull_pdf(x, scale, shape)
			var exp_result: float = StatMath.PmfPdfFunctions.exponential_pdf(x, rate)
			assert_float(weibull_result).is_equal_approx(exp_result, StatMath.PROBABILITY_TOLERANCE)

func test_weibull_pdf_rayleigh_special_case() -> void:
	# When shape=2, Weibull becomes Rayleigh distribution
	var x_values: Array[float] = [0.5, 1.0, 1.5, 2.0]
	var scale: float = 2.0
	var shape: float = 2.0
	
	for x in x_values:
		if x >= 0:
			var weibull_result: float = StatMath.PmfPdfFunctions.weibull_pdf(x, scale, shape)
			# Rayleigh PDF: f(x) = (x/σ²) * exp(-(x²)/(2σ²)) where σ = scale/sqrt(2)
			var sigma: float = scale / sqrt(2.0)
			var rayleigh_expected: float = (x / (sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma))
			assert_float(weibull_result).is_equal_approx(rayleigh_expected, StatMath.PROBABILITY_TOLERANCE)

func test_weibull_pdf_monotonicity() -> void:
	# For different shape parameters, test monotonic behavior
	var scale: float = 2.0
	
	# Shape < 1: Decreasing function
	var shape_decreasing: float = 0.5
	var x1: float = 0.1
	var x2: float = 1.0
	var pdf1: float = StatMath.PmfPdfFunctions.weibull_pdf(x1, scale, shape_decreasing)
	var pdf2: float = StatMath.PmfPdfFunctions.weibull_pdf(x2, scale, shape_decreasing)
	assert_float(pdf1).is_greater(pdf2)  # Should decrease
	
	# Shape > 1: First increases then decreases
	var shape_unimodal: float = 2.0
	var x_small: float = 0.1
	var x_mode: float = scale * pow((shape_unimodal - 1.0) / shape_unimodal, 1.0 / shape_unimodal)
	var x_large: float = 5.0
	
	var pdf_small: float = StatMath.PmfPdfFunctions.weibull_pdf(x_small, scale, shape_unimodal)
	var pdf_mode: float = StatMath.PmfPdfFunctions.weibull_pdf(x_mode, scale, shape_unimodal)
	var pdf_large: float = StatMath.PmfPdfFunctions.weibull_pdf(x_large, scale, shape_unimodal)
	
	assert_float(pdf_small).is_less(pdf_mode)  # Increases to mode
	assert_float(pdf_mode).is_greater(pdf_large)  # Decreases after mode

# --- Lognormal PDF Mathematical Properties ---

func test_lognormal_pdf_relationship_to_normal() -> void:
	# If X ~ Lognormal(μ, σ), then ln(X) ~ Normal(μ, σ)
	var x: float = 2.0
	var mu: float = 0.5
	var sigma: float = 1.0
	
	var lognormal_result: float = StatMath.PmfPdfFunctions.lognormal_pdf(x, mu, sigma)
	var normal_result: float = StatMath.PmfPdfFunctions.normal_pdf(log(x), mu, sigma)
	
	# Lognormal PDF = Normal PDF of log(x) divided by x
	var expected: float = normal_result / x
	assert_float(lognormal_result).is_equal_approx(expected, StatMath.PROBABILITY_TOLERANCE) 
