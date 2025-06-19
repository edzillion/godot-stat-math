# res://addons/godot-stat-math/core/cdf_functions.gd
class_name CdfFunctions extends RefCounted

## Cumulative Distribution Functions (CDF)
##
## This class provides static methods to calculate the cumulative distribution function 
## for various statistical distributions. The CDF, [code]F(x)[/code], gives the probability 
## that a random variable X will take a value less than or equal to x.
##
## Distribution Categories:
## • Continuous distributions (Normal, Exponential, Gamma, Beta, etc.)
## • Discrete distributions (Binomial, Poisson, Geometric, etc.)
## • Special distributions (Chi-Square, F-distribution, Student's t)
## • Heavy-tailed distributions (Pareto, Weibull)


# =============================================================================
# CONTINUOUS DISTRIBUTION CDFs
# =============================================================================

## Calculates the CDF of a uniform distribution: F(x; a, b).
##
## Returns the probability that a random variable from a uniform distribution 
## on the interval [code][a, b][/code] is less than or equal to x.
##
## Mathematical Note: [code]F(x) = (x-a)/(b-a)[/code] for [code]a ≤ x ≤ b[/code]
static func uniform_cdf(x: float, a: float, b: float) -> float:
	if not (a <= b):
		push_error("Parameter a must be less than or equal to b for Uniform CDF. Received a=%s, b=%s" % [a, b])
		return NAN
	if x < a:
		return 0.0
	if x >= b: # If x is b or greater, CDF is 1.0
		return 1.0
	return (x - a) / (b - a)


## Calculates the CDF of a normal distribution: F(x; μ, σ).
##
## Returns the probability that a random variable from a normal (Gaussian) 
## distribution with mean [code]μ[/code] and standard deviation [code]σ[/code] 
## is less than or equal to x. Uses the error function for computation.
##
## Mathematical Note: [code]F(x) = (1/2)[1 + erf((x-μ)/(σ√2))][/code]
static func normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
	if not (sigma > 0.0):
		push_error("Standard deviation (sigma) must be positive for Normal CDF. Received: %s" % sigma)
		return NAN
	var z: float = (x - mu) / sigma
	return 0.5 * (1.0 + StatMath.ErrorFunctions.erf(z / sqrt(2.0)))


## Calculates the CDF of an exponential distribution: F(x; λ).
##
## Returns the probability that a random variable from an exponential 
## distribution with rate parameter [code]λ[/code] is less than or equal to x.
##
## Mathematical Note: [code]F(x) = 1 - e^(-λx)[/code] for [code]x ≥ 0[/code]
static func exponential_cdf(x: float, lambda_param: float) -> float:
	if not (lambda_param > 0.0):
		push_error("Rate parameter (lambda_param) must be positive for Exponential CDF. Received: %s" % lambda_param)
		return NAN
	if x <= 0.0:
		return 0.0
	return 1.0 - exp(-lambda_param * x)


## Calculates the CDF of a beta distribution: F(x; α, β).
##
## Returns the probability that a random variable from a beta distribution 
## with shape parameters [code]α[/code] and [code]β[/code] is less than or equal to x.
## Uses the regularized incomplete beta function.
##
## Mathematical Note: [code]F(x) = I_x(α, β)[/code] where I is the incomplete beta function
static func beta_cdf(x: float, alpha: float, beta_param: float) -> float:
	if not (alpha > 0.0 and beta_param > 0.0):
		push_error("Shape parameters (alpha, beta_param) must be positive for Beta CDF. Received alpha=%s, beta_param=%s" % [alpha, beta_param])
		return NAN
	if x <= 0.0:
		return 0.0
	if x >= 1.0:
		return 1.0
	return StatMath.HelperFunctions.incomplete_beta(x, alpha, beta_param)


## Calculates the CDF of a gamma distribution: F(x; k, θ).
##
## Returns the probability that a random variable from a gamma distribution 
## with shape parameter [code]k[/code] and scale parameter [code]θ[/code] is less than or equal to x.
## Uses the regularized lower incomplete gamma function.
##
## Mathematical Note: [code]F(x) = P(k, x/θ)[/code] where P is the regularized incomplete gamma function
static func gamma_cdf(x: float, k_shape: float, theta_scale: float) -> float: # Renamed k, theta
	if not (k_shape > 0.0 and theta_scale > 0.0):
		push_error("Shape (k_shape) and scale (theta_scale) must be positive for Gamma CDF. Received k_shape=%s, theta_scale=%s" % [k_shape, theta_scale])
		return NAN
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


# =============================================================================
# SPECIAL DISTRIBUTION CDFs
# =============================================================================

## Calculates the CDF of a chi-square distribution: F(x; k).
##
## Returns the probability that a random variable from a chi-square distribution 
## with [code]k[/code] degrees of freedom is less than or equal to x.
## This is a special case of the gamma distribution.
##
## Mathematical Note: Chi-square with k df is [code]Gamma(k/2, 2)[/code]
static func chi_square_cdf(x: float, k_df: float) -> float:
	if not (k_df > 0.0):
		push_error("Degrees of freedom (k_df) must be positive for Chi-Square CDF. Received: %s" % k_df)
		return NAN
	if x <= 0.0: # Chi-square variable must be non-negative
		return 0.0
	# Chi-square with k_df degrees of freedom is Gamma(shape=k_df/2, scale=2)
	return gamma_cdf(x, k_df / 2.0, 2.0)


## Calculates the CDF of an F-distribution: F(x; d1, d2).
##
## Returns the probability that a random variable from an F-distribution 
## with [code]d1[/code] and [code]d2[/code] degrees of freedom is less than or equal to x.
## Uses the regularized incomplete beta function.
##
## Mathematical Note: [code]F(x) = I_z(d1/2, d2/2)[/code] where [code]z = d1x/(d1x + d2)[/code]
static func f_cdf(x: float, d1_df: float, d2_df: float) -> float: # Renamed d1, d2
	if not (d1_df > 0.0 and d2_df > 0.0):
		push_error("Degrees of freedom (d1_df, d2_df) must be positive for F-Distribution CDF. Received d1_df=%s, d2_df=%s" % [d1_df, d2_df])
		return NAN
	if x <= 0.0:
		return 0.0
	
	var val_z: float = (d1_df * x) / (d1_df * x + d2_df) # Renamed intermediate var from z_val
	return StatMath.HelperFunctions.incomplete_beta(val_z, d1_df / 2.0, d2_df / 2.0)


## Calculates the CDF of a Student's t-distribution: F(x; ν).
##
## Returns the probability that a random variable from a Student's t-distribution 
## with [code]ν[/code] (nu) degrees of freedom is less than or equal to x.
## Uses the regularized incomplete beta function.
##
## Mathematical Note: Uses transformation via incomplete beta function
static func t_cdf(x_val: float, df_nu: float) -> float: # Renamed x, df
	if not (df_nu > 0.0):
		push_error("Degrees of freedom (df_nu) must be positive for Student's t-Distribution CDF. Received: %s" % df_nu)
		return NAN
	
	var t_squared: float = x_val * x_val
	var z_val: float = df_nu / (df_nu + t_squared)
	
	if x_val < 0.0:
		return 0.5 * StatMath.HelperFunctions.incomplete_beta(z_val, df_nu / 2.0, 0.5)
	else:
		return 1.0 - 0.5 * StatMath.HelperFunctions.incomplete_beta(z_val, df_nu / 2.0, 0.5)


# =============================================================================
# DISCRETE DISTRIBUTION CDFs
# =============================================================================

## Calculates the CDF of a binomial distribution: F(k; n, p).
##
## Returns the probability of observing [code]k[/code] or fewer successes in [code]n[/code] 
## independent Bernoulli trials, each with success probability [code]p[/code].
## Computed as the sum of PMF values.
##
## Mathematical Note: [code]F(k) = Σᵢ₌₀ᵏ (n choose i) p^i (1-p)^(n-i)[/code]
static func binomial_cdf(k_successes: int, n_trials: int, p_prob: float) -> float:
	if not (n_trials >= 0):
		push_error("Number of trials (n_trials) must be non-negative. Received: %s" % n_trials)
		return NAN
	if not (p_prob >= 0.0 and p_prob <= 1.0):
		push_error("Probability (p_prob) must be between 0.0 and 1.0. Received: %s" % p_prob)
		return NAN
	
	if k_successes < 0:
		return 0.0
	if k_successes >= n_trials:
		return 1.0
	
	var cumulative_prob: float = 0.0 # Renamed cum_prob
	for i in range(k_successes + 1):
		cumulative_prob += StatMath.PmfPdfFunctions.binomial_pmf(i, n_trials, p_prob)
	return cumulative_prob


## Calculates the CDF of a Poisson distribution: F(k; λ).
##
## Returns the probability of observing [code]k[/code] or fewer events in a fixed interval, 
## given an average rate [code]λ[/code] of events. Computed as the sum of PMF values.
##
## Mathematical Note: [code]F(k) = Σᵢ₌₀ᵏ (λ^i e^(-λ))/i![/code]
static func poisson_cdf(k_events: int, lambda_param: float) -> float:
	if not (lambda_param >= 0.0):
		push_error("Rate parameter (lambda_param) must be non-negative for Poisson CDF. Received: %s" % lambda_param)
		return NAN
	
	if k_events < 0:
		return 0.0
	
	var cumulative_prob: float = 0.0 # Renamed cum_prob
	for i in range(k_events + 1):
		cumulative_prob += StatMath.PmfPdfFunctions.poisson_pmf(i, lambda_param)
	return cumulative_prob


## Calculates the CDF of a geometric distribution: F(k; p).
##
## Returns the probability that the first success in independent Bernoulli trials 
## occurs on or before the [code]k[/code]-th trial. Assumes [code]k ≥ 1[/code].
##
## Mathematical Note: [code]F(k) = 1 - (1-p)^k[/code]
static func geometric_cdf(k_trials: int, p_prob: float) -> float:
	if not (p_prob > 0.0 and p_prob <= 1.0):
		push_error("Success probability (p_prob) must be in (0,1]. Received: %s" % p_prob)
		return NAN
	
	if k_trials < 1: # First success cannot occur before the 1st trial.
		return 0.0
	return 1.0 - pow(1.0 - p_prob, float(k_trials)) # Ensure k_trials is float for pow if base is float


## Calculates the CDF of a negative binomial distribution: F(k; r, p).
##
## Returns the probability that the [code]r[/code]-th success occurs on or before 
## the [code]k[/code]-th trial in independent Bernoulli trials.
## Computed as the sum of PMF values.
##
## Mathematical Note: [code]F(k) = Σᵢ₌ᵣᵏ (i-1 choose r-1) p^r (1-p)^(i-r)[/code]
static func negative_binomial_cdf(k_trials: int, r_successes: int, p_prob: float) -> float:
	if not (r_successes > 0):
		push_error("Number of successes (r_successes) must be positive. Received: %s" % r_successes)
		return NAN
	if not (p_prob > 0.0 and p_prob <= 1.0):
		push_error("Success probability (p_prob) must be in (0,1]. Received: %s" % p_prob)
		return NAN
	
	if k_trials < r_successes: # Cannot have r successes in fewer than r trials.
		return 0.0
	
	var cumulative_prob: float = 0.0 # Renamed cum_prob
	for i in range(r_successes, k_trials + 1):
		cumulative_prob += StatMath.PmfPdfFunctions.negative_binomial_pmf(i, r_successes, p_prob)
	return cumulative_prob


# =============================================================================
# HEAVY-TAILED DISTRIBUTION CDFs
# =============================================================================

## Calculates the CDF of a Pareto distribution: F(x; scale, shape).
##
## Returns the probability that a random variable from a Pareto distribution 
## with scale parameter (minimum value) and shape parameter is less than or equal to x.
## Uses the closed-form solution.
##
## Mathematical Note: [code]F(x) = 1 - (scale/x)^shape[/code] for [code]x ≥ scale[/code]
static func pareto_cdf(x: float, scale_param: float, shape_param: float) -> float:
	if not (scale_param > 0.0):
		push_error("Scale parameter must be positive for Pareto CDF. Received: %s" % scale_param)
		return NAN
	if not (shape_param > 0.0):
		push_error("Shape parameter must be positive for Pareto CDF. Received: %s" % shape_param)
		return NAN
	
	if x < scale_param:
		return 0.0  # Pareto distribution has support [scale, +∞)
	
	# Closed-form solution: F(x) = 1 - (scale/x)^shape
	var ratio: float = scale_param / x
	var power_term: float = pow(ratio, shape_param)
	
	return 1.0 - power_term


## Calculates the CDF of a Weibull distribution: F(x; λ, k).
##
## Returns the probability that a random variable from a Weibull distribution 
## with scale parameter [code]λ[/code] and shape parameter [code]k[/code] is less than or equal to x.
## Uses the closed-form solution. Widely used for reliability analysis.
##
## Mathematical Note: [code]F(x) = 1 - exp(-(x/λ)^k)[/code] for [code]x ≥ 0[/code]
static func weibull_cdf(x: float, scale_param: float, shape_param: float) -> float:
	if not (scale_param > 0.0):
		push_error("Scale parameter must be positive for Weibull CDF. Received: %s" % scale_param)
		return NAN
	if not (shape_param > 0.0):
		push_error("Shape parameter must be positive for Weibull CDF. Received: %s" % shape_param)
		return NAN
	
	if x <= 0.0:
		return 0.0  # Weibull distribution has support [0, +∞)
	
	# Closed-form solution: F(x) = 1 - exp(-(x/λ)^k)
	var ratio: float = x / scale_param
	var power_term: float = pow(ratio, shape_param)
	var exp_term: float = exp(-power_term)
	
	return 1.0 - exp_term
