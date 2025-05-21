extends RefCounted

# Cumulative Distribution Functions (CDF)
# This script provides static methods to calculate the CDF for various
# common statistical distributions. The CDF, F(x), gives the probability
# that a random variable X will take a value less than or equal to x.

# Uniform Distribution CDF: F(x; a, b)
# Calculates the probability that a random variable from a uniform distribution
# on the interval [a, b] is less than or equal to x.
static func uniform_cdf(x: float, a: float, b: float) -> float:
	assert(a <= b, "Parameter a must be less than or equal to b for Uniform CDF.")
	if x < a:
		return 0.0
	if x >= b: # If x is b or greater, CDF is 1.0
		return 1.0
	return (x - a) / (b - a)


# Normal Distribution CDF: F(x; μ, σ)
# Calculates the probability that a random variable from a normal (Gaussian)
# distribution with mean μ and standard deviation σ is less than or equal to x.
# Utilizes the error function (erf) for computation.
static func normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
	assert(sigma > 0.0, "Standard deviation (sigma) must be positive for Normal CDF.")
	var z: float = (x - mu) / sigma
	return 0.5 * (1.0 + StatMath.ErrorFunctions.error_function(z / sqrt(2.0)))


# Exponential Distribution CDF: F(x; λ)
# Calculates the probability that a random variable from an exponential
# distribution with rate parameter λ is less than or equal to x.
# Defined for x >= 0.
static func exponential_cdf(x: float, lambda_param: float) -> float:
	assert(lambda_param > 0.0, "Rate parameter (lambda_param) must be positive for Exponential CDF.")
	if x <= 0.0:
		return 0.0
	return 1.0 - exp(-lambda_param * x)


# Beta Distribution CDF: F(x; α, β)
# Calculates the probability that a random variable from a beta distribution
# with shape parameters α and β is less than or equal to x.
# Relies on the regularized incomplete beta function.
static func beta_cdf(x: float, alpha: float, beta_param: float) -> float:
	assert(alpha > 0.0 and beta_param > 0.0, "Shape parameters (alpha, beta_param) must be positive for Beta CDF.")
	if x <= 0.0:
		return 0.0
	if x >= 1.0:
		return 1.0
	return StatMath.HelperFunctions.incomplete_beta(x, alpha, beta_param)


# Gamma Distribution CDF: F(x; k, θ)
# Calculates the probability that a random variable from a gamma distribution
# with shape parameter k and scale parameter θ is less than or equal to x.
# Relies on the regularized lower incomplete gamma function.
static func gamma_cdf(x: float, k_shape: float, theta_scale: float) -> float: # Renamed k, theta
	assert(k_shape > 0.0 and theta_scale > 0.0, "Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF.")
	if x <= 0.0:
		return 0.0
	
	var a: float = k_shape
	var z_val: float = x / theta_scale # Renamed z to z_val
	
	# Approximation for large z_val to prevent overflow/performance issues
	# if StatMath.lower_incomplete_gamma_regularized struggles.
	# Thresholds may need tuning based on the specific implementation.
	if z_val > 200.0 and a < z_val:
		return 1.0
	
	return StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z_val)


# Chi-Square Distribution CDF: F(x; k)
# Calculates the probability that a random variable from a chi-square distribution
# with k degrees of freedom is less than or equal to x.
# This is a special case of the gamma distribution.
static func chi_square_cdf(x: float, k_df: float) -> float:
	assert(k_df > 0.0, "Degrees of freedom (k_df) must be positive for Chi-Square CDF.")
	if x <= 0.0: # Chi-square variable must be non-negative
		return 0.0
	# Chi-square with k_df degrees of freedom is Gamma(shape=k_df/2, scale=2)
	return gamma_cdf(x, k_df / 2.0, 2.0)


# F-Distribution CDF: F(x; d1, d2)
# Calculates the probability that a random variable from an F-distribution
# with d1 and d2 degrees of freedom is less than or equal to x.
# Relies on the regularized incomplete beta function.
static func f_cdf(x: float, d1_df: float, d2_df: float) -> float: # Renamed d1, d2
	assert(d1_df > 0.0 and d2_df > 0.0, "Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF.")
	if x <= 0.0:
		return 0.0
	
	var val_z: float = (d1_df * x) / (d1_df * x + d2_df) # Renamed intermediate var from z_val
	return StatMath.HelperFunctions.incomplete_beta(val_z, d1_df / 2.0, d2_df / 2.0)


# Student's t-Distribution CDF: F(x; ν)
# Calculates the probability that a random variable from a Student's t-distribution
# with ν (nu) degrees of freedom is less than or equal to x.
# Relies on the regularized incomplete beta function.
static func t_cdf(x_val: float, df_nu: float) -> float: # Renamed x, df
	assert(df_nu > 0.0, "Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF.")
	
	var t_squared: float = x_val * x_val
	var z_val: float = df_nu / (df_nu + t_squared)
	
	if x_val < 0.0:
		return 0.5 * StatMath.HelperFunctions.incomplete_beta(z_val, df_nu / 2.0, 0.5)
	else:
		return 1.0 - 0.5 * StatMath.HelperFunctions.incomplete_beta(z_val, df_nu / 2.0, 0.5)


# Binomial Distribution CDF: F(k; n, p)
# Calculates the probability of observing k or fewer successes in n independent
# Bernoulli trials, each with a success probability of p.
# This is a sum of PMF values.
static func binomial_cdf(k_successes: int, n_trials: int, p_prob: float) -> float:
	assert(n_trials >= 0, "Number of trials (n_trials) must be non-negative.")
	assert(p_prob >= 0.0 and p_prob <= 1.0, "Probability (p_prob) must be between 0.0 and 1.0.")
	
	if k_successes < 0:
		return 0.0
	if k_successes >= n_trials:
		return 1.0
	
	var cumulative_prob: float = 0.0 # Renamed cum_prob
	for i in range(k_successes + 1):
		cumulative_prob += StatMath.PmfPdfFunctions.binomial_pmf(i, n_trials, p_prob)
	return cumulative_prob


# Poisson Distribution CDF: F(k; λ)
# Calculates the probability of observing k or fewer events in a fixed interval
# of time or space, given an average rate λ of events.
# This is a sum of PMF values.
static func poisson_cdf(k_events: int, lambda_param: float) -> float:
	assert(lambda_param >= 0.0, "Rate parameter (lambda_param) must be non-negative for Poisson CDF.")
	
	if k_events < 0:
		return 0.0
	
	var cumulative_prob: float = 0.0 # Renamed cum_prob
	for i in range(k_events + 1):
		cumulative_prob += StatMath.PmfPdfFunctions.poisson_pmf(i, lambda_param)
	return cumulative_prob


# Geometric Distribution CDF: F(k; p)
# Calculates the probability that the first success in a series of independent
# Bernoulli trials occurs on or before the k-th trial.
# Assumes k is the number of trials (k >= 1).
static func geometric_cdf(k_trials: int, p_prob: float) -> float:
	assert(p_prob > 0.0 and p_prob <= 1.0, "Success probability (p_prob) must be in (0,1].")
	
	if k_trials < 1: # First success cannot occur before the 1st trial.
		return 0.0
	return 1.0 - pow(1.0 - p_prob, float(k_trials)) # Ensure k_trials is float for pow if base is float


# Negative Binomial Distribution CDF: F(k; r, p)
# Calculates the probability that the r-th success occurs on or before the k-th trial
# in a series of independent Bernoulli trials.
# This is a sum of PMF values.
static func negative_binomial_cdf(k_trials: int, r_successes: int, p_prob: float) -> float:
	assert(r_successes > 0, "Number of successes (r_successes) must be positive.")
	assert(p_prob > 0.0 and p_prob <= 1.0, "Success probability (p_prob) must be in (0,1].")
	
	if k_trials < r_successes: # Cannot have r successes in fewer than r trials.
		return 0.0
	
	var cumulative_prob: float = 0.0 # Renamed cum_prob
	for i in range(r_successes, k_trials + 1):
		cumulative_prob += StatMath.PmfPdfFunctions.negative_binomial_pmf(i, r_successes, p_prob)
	return cumulative_prob
