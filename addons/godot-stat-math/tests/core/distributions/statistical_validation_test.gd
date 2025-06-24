# res://addons/godot-stat-math/tests/core/distributions/statistical_validation_test.gd
class_name DistributionsStatisticalValidationTest extends GdUnitTestSuite

## Statistical Validation Tests for Distribution Functions
##
## These tests use a "lean statistical validation" approach:
## - Small sample sizes (200-1000 samples) for fast execution
## - Focus on catching algorithmic flaws rather than deep statistical auditing
## - Rule of thumb: Sample size = Expected frequency * 20 for discrete distributions
## - Chi-squared goodness-of-fit tests for discrete distributions
## - Mean/variance validation for continuous distributions

# Test constants
const SMALL_SAMPLE_SIZE: int = 200
const MEDIUM_SAMPLE_SIZE: int = 1000
const CHI_SQUARED_CRITICAL_VALUE_005: float = 3.841  # p=0.05, df=1 for Bernoulli
const CHI_SQUARED_CRITICAL_VALUE_005_DF2: float = 5.991  # p=0.05, df=2
const STATISTICAL_TOLERANCE_MULTIPLIER: float = 3.0  # 3 standard deviations

# =============================================================================
# DISCRETE DISTRIBUTION STATISTICAL VALIDATION
# =============================================================================

func test_randi_bernoulli_statistical_properties() -> void:
	# Test Bernoulli distribution produces correct success rate
	StatMath.set_global_seed(42)
	
	var p: float = 0.3
	var sample_size: int = SMALL_SAMPLE_SIZE
	var successes: int = 0
	
	# Generate samples
	for i in range(sample_size):
		if StatMath.Distributions.randi_bernoulli(p) == 1:
			successes += 1
	
	var failures: int = sample_size - successes
	var expected_successes: float = sample_size * p
	var expected_failures: float = sample_size * (1.0 - p)
	
	# Chi-squared goodness-of-fit test
	var chi_squared: float = _chi_squared_test([successes, failures], [expected_successes, expected_failures])
	
	# Should pass chi-squared test (reject null hypothesis if chi_squared > critical value)
	assert_bool(chi_squared < CHI_SQUARED_CRITICAL_VALUE_005).is_true()

func test_randi_binomial_statistical_properties() -> void:
	# Test Binomial distribution produces correct mean
	StatMath.set_global_seed(123)
	
	var p: float = 0.4
	var n: int = 10
	var sample_size: int = SMALL_SAMPLE_SIZE
	var total_successes: int = 0
	
	# Generate samples
	for i in range(sample_size):
		total_successes += StatMath.Distributions.randi_binomial(p, n)
	
	var sample_mean: float = float(total_successes) / float(sample_size)
	var expected_mean: float = n * p  # 4.0
	var variance: float = n * p * (1.0 - p)  # 2.4
	var standard_error: float = sqrt(variance / sample_size)
	
	# Test that sample mean is within 3 standard errors of expected mean
	var tolerance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error
	assert_float(sample_mean).is_between(expected_mean - tolerance, expected_mean + tolerance)

func test_randi_geometric_statistical_properties() -> void:
	# Test Geometric distribution produces correct mean
	StatMath.set_global_seed(456)
	
	var p: float = 0.2
	var sample_size: int = SMALL_SAMPLE_SIZE
	var total_trials: int = 0
	
	# Generate samples
	for i in range(sample_size):
		total_trials += StatMath.Distributions.randi_geometric(p)
	
	var sample_mean: float = float(total_trials) / float(sample_size)
	var expected_mean: float = 1.0 / p  # 5.0
	var variance: float = (1.0 - p) / (p * p)  # 20.0
	var standard_error: float = sqrt(variance / sample_size)
	
	# Test that sample mean is within 3 standard errors of expected mean
	var tolerance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error
	assert_float(sample_mean).is_between(expected_mean - tolerance, expected_mean + tolerance)

func test_randi_poisson_statistical_properties() -> void:
	# Test Poisson distribution produces correct mean and variance
	StatMath.set_global_seed(789)
	
	var lambda_param: float = 3.0
	var sample_size: int = SMALL_SAMPLE_SIZE
	var samples: Array[int] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randi_poisson(lambda_param))
	
	var sample_mean: float = _calculate_mean_int(samples)
	var sample_variance: float = _calculate_variance_int(samples, sample_mean)
	
	# For Poisson: E[X] = Var(X) = lambda
	var expected_mean: float = lambda_param
	var expected_variance: float = lambda_param
	
	# Standard error for mean
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	
	# Standard error for variance (approximately)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)

# =============================================================================
# CONTINUOUS DISTRIBUTION STATISTICAL VALIDATION
# =============================================================================

func test_randf_normal_statistical_properties() -> void:
	# Test Normal distribution produces correct mean and variance
	StatMath.set_global_seed(101)
	
	var mu: float = 5.0
	var sigma: float = 2.0
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_normal(mu, sigma))
	
	var sample_mean: float = _calculate_mean_float(samples)
	var sample_variance: float = _calculate_variance_float(samples, sample_mean)
	
	# Expected values
	var expected_mean: float = mu
	var expected_variance: float = sigma * sigma
	
	# Standard errors
	var standard_error_mean: float = sigma / sqrt(sample_size)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)

func test_randf_exponential_statistical_properties() -> void:
	# Test Exponential distribution produces correct mean and variance
	StatMath.set_global_seed(202)
	
	var lambda_param: float = 2.0
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_exponential(lambda_param))
	
	var sample_mean: float = _calculate_mean_float(samples)
	var sample_variance: float = _calculate_variance_float(samples, sample_mean)
	
	# For Exponential: E[X] = 1/λ, Var(X) = 1/λ²
	var expected_mean: float = 1.0 / lambda_param
	var expected_variance: float = 1.0 / (lambda_param * lambda_param)
	
	# Standard errors
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)

func test_randf_uniform_statistical_properties() -> void:
	# Test Uniform distribution produces correct mean and variance
	StatMath.set_global_seed(303)
	
	var a: float = 1.0
	var b: float = 5.0
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_uniform(a, b))
	
	var sample_mean: float = _calculate_mean_float(samples)
	var sample_variance: float = _calculate_variance_float(samples, sample_mean)
	
	# For Uniform[a,b]: E[X] = (a+b)/2, Var(X) = (b-a)²/12
	var expected_mean: float = (a + b) / 2.0
	var expected_variance: float = (b - a) * (b - a) / 12.0
	
	# Standard errors
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)

func test_randf_beta_statistical_properties() -> void:
	# Test Beta distribution produces correct mean
	StatMath.set_global_seed(404)
	
	var alpha: float = 2.0
	var beta_param: float = 3.0
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_beta(alpha, beta_param))
	
	var sample_mean: float = _calculate_mean_float(samples)
	
	# For Beta(α,β): E[X] = α/(α+β)
	var expected_mean: float = alpha / (alpha + beta_param)
	var expected_variance: float = (alpha * beta_param) / ((alpha + beta_param) * (alpha + beta_param) * (alpha + beta_param + 1.0))
	
	# Standard error for mean
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	
	# Also test that all values are in [0,1]
	for sample in samples:
		assert_float(sample).is_between(0.0, 1.0)

func test_randf_gamma_statistical_properties() -> void:
	# Test Gamma distribution produces correct mean and variance
	StatMath.set_global_seed(505)
	
	var alpha: float = 2.0  # shape parameter
	var theta: float = 1.5  # scale parameter
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_gamma(alpha, theta))
	
	var sample_mean: float = _calculate_mean_float(samples)
	var sample_variance: float = _calculate_variance_float(samples, sample_mean)
	
	# For Gamma(α,θ): E[X] = α×θ, Var(X) = α×θ²
	var expected_mean: float = alpha * theta  # 3.0
	var expected_variance: float = alpha * theta * theta  # 4.5
	
	# Standard errors
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)

func test_randf_erlang_statistical_properties() -> void:
	# Test Erlang distribution produces correct mean and variance
	StatMath.set_global_seed(606)
	
	var k: int = 3  # shape parameter (integer)
	var lambda_param: float = 2.0  # rate parameter
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_erlang(k, lambda_param))
	
	var sample_mean: float = _calculate_mean_float(samples)
	var sample_variance: float = _calculate_variance_float(samples, sample_mean)
	
	# For Erlang(k,λ): E[X] = k/λ, Var(X) = k/λ²
	var expected_mean: float = float(k) / lambda_param  # 1.5
	var expected_variance: float = float(k) / (lambda_param * lambda_param)  # 0.75
	
	# Standard errors
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)

func test_randi_uniform_statistical_properties() -> void:
	# Test discrete uniform distribution using chi-squared goodness-of-fit
	StatMath.set_global_seed(707)
	
	var min_val: int = 1
	var max_val: int = 10
	var range_size: int = max_val - min_val + 1  # 10 possible values
	var sample_size: int = range_size * 20  # 200 samples (20 per expected outcome)
	var observed_counts: Array[int] = []
	var expected_counts: Array[float] = []
	
	# Initialize counts array
	for i in range(range_size):
		observed_counts.append(0)
		expected_counts.append(float(sample_size) / float(range_size))  # Equal probability for each outcome
	
	# Generate samples and count occurrences
	for i in range(sample_size):
		var sample: int = StatMath.Distributions.randi_uniform(min_val, max_val)
		var index: int = sample - min_val  # Convert to 0-based index
		observed_counts[index] += 1
	
	# Perform chi-squared goodness-of-fit test
	var chi_squared: float = _chi_squared_test(observed_counts, expected_counts)
	
	# Critical value for χ² test with df=9 (10 categories - 1) at p=0.05 is 16.919
	var critical_value_df9: float = 16.919
	
	# Should pass chi-squared test (uniform distribution hypothesis)
	assert_bool(chi_squared < critical_value_df9).is_true()

func test_randf_triangular_statistical_properties() -> void:
	# Test Triangular distribution produces correct mean
	StatMath.set_global_seed(808)
	
	var min_val: float = 1.0
	var max_val: float = 5.0
	var mode_val: float = 3.0
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_triangular(min_val, max_val, mode_val))
	
	var sample_mean: float = _calculate_mean_float(samples)
	
	# For Triangular(a,b,c): E[X] = (a+b+c)/3
	var expected_mean: float = (min_val + max_val + mode_val) / 3.0  # 3.0
	var expected_variance: float = (min_val*min_val + max_val*max_val + mode_val*mode_val - min_val*max_val - min_val*mode_val - max_val*mode_val) / 18.0
	
	# Standard error for mean
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	
	# Also test that all values are in [min_val, max_val]
	for sample in samples:
		assert_float(sample).is_between(min_val, max_val)

func test_randf_pareto_statistical_properties() -> void:
	# Test Pareto distribution produces correct mean (when shape > 1)
	StatMath.set_global_seed(909)
	
	var scale: float = 2.0  # scale parameter
	var shape: float = 3.0  # shape parameter (> 1 for finite mean)
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_pareto(scale, shape))
	
	var sample_mean: float = _calculate_mean_float(samples)
	
	# For Pareto(scale,shape): E[X] = scale×shape/(shape-1) when shape > 1
	var expected_mean: float = scale * shape / (shape - 1.0)  # 3.0
	
	# Pareto has high variance, so use a more generous tolerance
	# Theoretical variance = scale²×shape/[(shape-1)²×(shape-2)] when shape > 2
	var expected_variance: float = (scale * scale * shape) / ((shape - 1.0) * (shape - 1.0) * (shape - 2.0))  # 4.5
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	
	# Also test that all values are >= scale (support is [scale, ∞))
	for sample in samples:
		assert_float(sample).is_greater_equal(scale)

func test_randf_weibull_statistical_properties() -> void:
	# Test Weibull distribution produces correct mean
	StatMath.set_global_seed(1010)
	
	var scale: float = 2.0  # scale parameter (λ)
	var shape: float = 2.0  # shape parameter (k)  
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_weibull(scale, shape))
	
	var sample_mean: float = _calculate_mean_float(samples)
	
	# For Weibull(λ,k): E[X] = λ × Γ(1 + 1/k)
	var gamma_arg: float = 1.0 + 1.0 / shape  # 1.5
	var gamma_value: float = StatMath.HelperFunctions.gamma_function(gamma_arg)
	var expected_mean: float = scale * gamma_value
	
	# Weibull variance calculation: Var(X) = λ² × [Γ(1 + 2/k) - Γ²(1 + 1/k)]
	var gamma_arg2: float = 1.0 + 2.0 / shape  # 2.0
	var gamma_value2: float = StatMath.HelperFunctions.gamma_function(gamma_arg2)
	var expected_variance: float = scale * scale * (gamma_value2 - gamma_value * gamma_value)
	
	# Standard error for mean
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	
	# Also test that all values are non-negative (support is [0, ∞))
	for sample in samples:
		assert_float(sample).is_greater_equal(0.0)

func test_randf_lognormal_statistical_properties() -> void:
	# Test Lognormal distribution produces correct mean and variance
	StatMath.set_global_seed(1111)
	
	var mu: float = 0.5  # location parameter
	var sigma: float = 0.8  # scale parameter  
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_lognormal(mu, sigma))
	
	var sample_mean: float = _calculate_mean_float(samples)
	var sample_variance: float = _calculate_variance_float(samples, sample_mean)
	
	# For Lognormal(μ,σ): E[X] = exp(μ + σ²/2), Var(X) = [exp(σ²) - 1] × exp(2μ + σ²)
	var expected_mean: float = exp(mu + sigma * sigma / 2.0)  # exp(0.5 + 0.64/2) = exp(0.82)
	var expected_variance: float = (exp(sigma * sigma) - 1.0) * exp(2.0 * mu + sigma * sigma)
	
	# Standard errors
	var standard_error_mean: float = sqrt(expected_variance / sample_size)
	var standard_error_variance: float = sqrt(2.0 * expected_variance * expected_variance / sample_size)
	
	var tolerance_mean: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_mean
	var tolerance_variance: float = STATISTICAL_TOLERANCE_MULTIPLIER * standard_error_variance
	
	assert_float(sample_mean).is_between(expected_mean - tolerance_mean, expected_mean + tolerance_mean)
	assert_float(sample_variance).is_between(expected_variance - tolerance_variance, expected_variance + tolerance_variance)
	
	# Also test that all values are positive (support is (0, ∞))
	for sample in samples:
		assert_float(sample).is_greater(0.0)

func test_randf_cauchy_statistical_properties() -> void:
	# Test Cauchy distribution properties (cannot test mean/variance - undefined!)
	StatMath.set_global_seed(1212)
	
	var location: float = 3.0  # location parameter
	var scale: float = 2.0     # scale parameter
	var sample_size: int = MEDIUM_SAMPLE_SIZE
	var samples: Array[float] = []
	
	# Generate samples
	for i in range(sample_size):
		samples.append(StatMath.Distributions.randf_cauchy(location, scale))
	
	# Sort samples to find median
	samples.sort()
	
	# Test 1: Median should approximate the location parameter
	var sample_median: float = _calculate_median_float(samples)
	var median_tolerance: float = 0.5  # Generous tolerance for Cauchy's heavy tails
	assert_float(sample_median).is_between(location - median_tolerance, location + median_tolerance)
	
	# Test 2: Test symmetry around location parameter
	var below_location: int = 0
	var above_location: int = 0
	
	for sample in samples:
		if sample < location:
			below_location += 1
		elif sample > location:
			above_location += 1
		# Equal values are ignored for symmetry test
	
	# Should be approximately 50/50 split around location
	var total_counted: int = below_location + above_location
	var expected_half: float = float(total_counted) / 2.0
	var symmetry_tolerance: float = float(total_counted) * 0.1  # 10% tolerance
	
	assert_float(float(below_location)).is_between(expected_half - symmetry_tolerance, expected_half + symmetry_tolerance)
	assert_float(float(above_location)).is_between(expected_half - symmetry_tolerance, expected_half + symmetry_tolerance)
	
	# Test 3: Scale parameter affects spread (compare IQR for different scales)
	StatMath.set_global_seed(1213)  # Reset for consistent comparison
	var scale_small: float = 1.0
	var samples_small: Array[float] = []
	
	for i in range(SMALL_SAMPLE_SIZE):
		samples_small.append(StatMath.Distributions.randf_cauchy(location, scale_small))
	
	samples_small.sort()
	var q1_small: float = samples_small[samples_small.size() / 4]
	var q3_small: float = samples_small[3 * samples_small.size() / 4]
	var iqr_small: float = q3_small - q1_small
	
	# Reset and test larger scale
	StatMath.set_global_seed(1213)  # Same seed for fair comparison
	var scale_large: float = 3.0
	var samples_large: Array[float] = []
	
	for i in range(SMALL_SAMPLE_SIZE):
		samples_large.append(StatMath.Distributions.randf_cauchy(location, scale_large))
	
	samples_large.sort()
	var q1_large: float = samples_large[samples_large.size() / 4]
	var q3_large: float = samples_large[3 * samples_large.size() / 4]
	var iqr_large: float = q3_large - q1_large
	
	# Larger scale should result in larger IQR (more spread)
	assert_bool(iqr_large > iqr_small).is_true()

func test_randi_pseudo_behavioral_properties() -> void:
	# Test that pseudo distribution increases success probability correctly
	StatMath.set_global_seed(1314)
	
	var c_param: float = 0.2  # Probability increment per trial
	var num_tests: int = 50   # Number of independent tests
	var success_trial_counts: Array[int] = []
	
	# Run multiple independent tests
	for test in range(num_tests):
		var trial_count: int = StatMath.Distributions.randi_pseudo(c_param)
		success_trial_counts.append(trial_count)
	
	# Test 1: All results should be >= 1 (always takes at least 1 trial)
	for trial_count in success_trial_counts:
		assert_bool(trial_count >= 1).is_true()
	
	# Test 2: With c_param = 0.2, theoretical mean is around 2.3 trials
	# (sum of 1/0.2 + 1/0.4 + 1/0.6 + 1/0.8 + 1/1.0) / 5 ≈ 2.28
	var sample_mean: float = _calculate_mean_int(success_trial_counts)
	assert_float(sample_mean).is_between(1.5, 3.5)  # Reasonable range for c_param=0.2
	
	# Test 3: Most results should be small numbers (heavily skewed toward early success)
	var early_success_count: int = 0
	for trial_count in success_trial_counts:
		if trial_count <= 3:
			early_success_count += 1
	
	# At least 60% should succeed within 3 trials with c_param = 0.2
	var early_success_rate: float = float(early_success_count) / float(num_tests)
	assert_bool(early_success_rate >= 0.6).is_true()

func test_randi_seige_behavioral_properties() -> void:
	# Test that siege distribution mechanics work as expected
	StatMath.set_global_seed(1415)
	
	# Test scenario: high win rate, gradual capture probability increase
	var w: float = 0.8        # High win probability
	var c_0: float = 0.1      # Low initial capture probability  
	var c_win: float = 0.2    # Good capture increase on win
	var c_lose: float = -0.05 # Small decrease on loss
	
	var num_tests: int = 50
	var trial_counts: Array[int] = []
	
	# Run multiple independent tests
	for test in range(num_tests):
		var trial_count: int = StatMath.Distributions.randi_seige(w, c_0, c_win, c_lose)
		trial_counts.append(trial_count)
	
	# Test 1: All results should be >= 1 (always takes at least 1 trial)
	for trial_count in trial_counts:
		assert_bool(trial_count >= 1).is_true()
	
	# Test 2: With high win rate and positive capture increases, should be reasonably fast
	var sample_mean: float = _calculate_mean_int(trial_counts)
	assert_float(sample_mean).is_between(1.0, 10.0)  # Should typically resolve quickly
	
	# Test 3: Compare different win rates - higher win rate should lead to faster resolution
	StatMath.set_global_seed(1416)
	var low_w: float = 0.3
	var low_w_trials: Array[int] = []
	
	for test in range(25):  # Smaller sample for comparison
		var trial_count: int = StatMath.Distributions.randi_seige(low_w, c_0, c_win, c_lose)
		low_w_trials.append(trial_count)
	
	StatMath.set_global_seed(1416)  # Same seed for fair comparison
	var high_w: float = 0.9
	var high_w_trials: Array[int] = []
	
	for test in range(25):
		var trial_count: int = StatMath.Distributions.randi_seige(high_w, c_0, c_win, c_lose)
		high_w_trials.append(trial_count)
	
	var low_w_mean: float = _calculate_mean_int(low_w_trials)
	var high_w_mean: float = _calculate_mean_int(high_w_trials)
	
	# Higher win rate should generally lead to faster resolution
	assert_bool(high_w_mean <= low_w_mean * 1.5).is_true()  # Allow some variance

# =============================================================================
# HELPER FUNCTIONS FOR STATISTICAL CALCULATIONS
# =============================================================================

func _chi_squared_test(observed: Array[int], expected: Array[float]) -> float:
	if observed.size() != expected.size():
		push_error("Observed and expected arrays must have the same size")
		return INF
	
	var chi_squared: float = 0.0
	for i in range(observed.size()):
		if expected[i] <= 0.0:
			push_error("Expected frequency must be positive")
			return INF
		var diff: float = float(observed[i]) - expected[i]
		chi_squared += (diff * diff) / expected[i]
	
	return chi_squared

func _calculate_mean_int(samples: Array[int]) -> float:
	var sum: int = 0
	for sample in samples:
		sum += sample
	return float(sum) / float(samples.size())

func _calculate_mean_float(samples: Array[float]) -> float:
	var sum: float = 0.0
	for sample in samples:
		sum += sample
	return sum / float(samples.size())

func _calculate_variance_int(samples: Array[int], mean: float) -> float:
	var sum_squared_diff: float = 0.0
	for sample in samples:
		var diff: float = float(sample) - mean
		sum_squared_diff += diff * diff
	return sum_squared_diff / float(samples.size() - 1)  # Sample variance (n-1)

func _calculate_variance_float(samples: Array[float], mean: float) -> float:
	var sum_squared_diff: float = 0.0
	for sample in samples:
		var diff: float = sample - mean
		sum_squared_diff += diff * diff
	return sum_squared_diff / float(samples.size() - 1)  # Sample variance (n-1)

func _calculate_median_float(sorted_samples: Array[float]) -> float:
	var n: int = sorted_samples.size()
	if n == 0:
		return NAN
	if n % 2 == 1:
		# Odd number of samples - return middle value
		return sorted_samples[n / 2]
	else:
		# Even number of samples - return average of two middle values
		var mid1: float = sorted_samples[n / 2 - 1]
		var mid2: float = sorted_samples[n / 2]
		return (mid1 + mid2) / 2.0 