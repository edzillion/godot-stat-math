# res://addons/godot-stat-math/core/pmf_pdf_functions.gd
class_name PmfPdfFunctions extends RefCounted

## Probability Mass Functions (PMF) and Probability Density Functions (PDF)
##
## This class provides static methods to calculate probability mass functions for discrete 
## distributions and probability density functions for continuous distributions. The PMF/PDF 
## gives the probability (or density) of a random variable taking on a specific value.
##
## Distribution Categories:
## • PMF for discrete distributions (Binomial, Poisson, Negative Binomial)
## • PDF for continuous distributions (Normal, Exponential, Uniform, Gamma, Beta, Chi-squared, Student's t, F-distribution)
## • Uses logarithmic calculations for numerical stability


# =============================================================================
# DISCRETE DISTRIBUTION PMFs
# =============================================================================

## Calculates the PMF of a binomial distribution: P(X = k | n, p).
##
## Returns the probability of observing exactly [code]k[/code] successes in [code]n[/code] 
## independent Bernoulli trials, each with success probability [code]p[/code].
## Uses logarithmic calculations for numerical stability.
##
## Mathematical Note: [code]P(X = k) = (n choose k) p^k (1-p)^(n-k)[/code]
static func binomial_pmf(k_successes: int, n_trials: int, p_prob: float) -> float:
	if not (n_trials >= 0):
		push_error("Number of trials (n_trials) must be non-negative. Received: %s" % n_trials)
		return NAN
	if not (p_prob >= 0.0 and p_prob <= 1.0):
		push_error("Success probability (p_prob) must be between 0.0 and 1.0. Received: %s" % p_prob)
		return NAN
	
	if k_successes < 0 or k_successes > n_trials:
		return 0.0 # Not possible to have k < 0 or k > n successes.

	# Handle edge cases for p_prob to avoid log(0) or ensure correctness
	if p_prob == 0.0:
		return 1.0 if k_successes == 0 else 0.0
	if p_prob == 1.0:
		return 1.0 if k_successes == n_trials else 0.0
	
	# Formula: C(n, k) * p^k * (1-p)^(n-k)
	# Using logs: log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
	var log_binom_coeff: float = StatMath.HelperFunctions.log_binomial_coef(n_trials, k_successes)
	var term_p: float = float(k_successes) * log(p_prob)
	var term_one_minus_p: float = float(n_trials - k_successes) * log(1.0 - p_prob)
	
	var log_pmf_val: float = log_binom_coeff + term_p + term_one_minus_p
	return exp(log_pmf_val)


## Calculates the PMF of a Poisson distribution: P(X = k | λ).
##
## Returns the probability of observing exactly [code]k[/code] events in a fixed interval, 
## given an average rate [code]λ[/code] of events. Uses logarithmic calculations for 
## numerical stability.
##
## Mathematical Note: [code]P(X = k) = (λ^k e^(-λ)) / k![/code]
static func poisson_pmf(k_events: int, lambda_param: float) -> float:
	if not (lambda_param >= 0.0):
		push_error("Rate parameter (lambda_param) must be non-negative. Received: %s" % lambda_param)
		return NAN
	
	if k_events < 0:
		return 0.0 # Not possible to have a negative number of events.
	
	# If lambda is 0, PMF is 1 if k is 0, and 0 otherwise.
	if lambda_param == 0.0:
		return 1.0 if k_events == 0 else 0.0
	
	# Formula: (lambda^k * e^-lambda) / k!
	# Using logs: k*log(lambda) - lambda - log(k!)
	var term_lambda_k: float = float(k_events) * log(lambda_param)
	var log_factorial_k: float = StatMath.HelperFunctions.log_factorial(k_events)
	
	var log_pmf_val: float = term_lambda_k - lambda_param - log_factorial_k
	return exp(log_pmf_val)


## Calculates the PMF of a negative binomial distribution: P(X = k | r, p).
##
## Returns the probability that the [code]r[/code]-th success occurs on exactly the 
## [code]k[/code]-th trial in independent Bernoulli trials with success probability [code]p[/code].
## Uses logarithmic calculations for numerical stability.
##
## Mathematical Note: [code]P(X = k) = (k-1 choose r-1) p^r (1-p)^(k-r)[/code]
static func negative_binomial_pmf(k_trials: int, r_successes: int, p_prob: float) -> float:
	if not (r_successes > 0):
		push_error("Number of required successes (r_successes) must be positive. Received: %s" % r_successes)
		return NAN
	if not (p_prob > 0.0 and p_prob <= 1.0):
		push_error("Success probability (p_prob) must be in (0,1]. Received: %s" % p_prob)
		return NAN

	if k_trials < r_successes: # Need at least r_successes trials.
		return 0.0
	
	# Handle edge cases for p_prob
	# If p_prob is 1, r_successes must occur in exactly r_successes trials.
	if p_prob == 1.0:
		return 1.0 if k_trials == r_successes else 0.0
	# If p_prob is 0 (asserted against, but for defense), impossible to get r_successes > 0.

	# Formula: C(k-1, r-1) * p^r * (1-p)^(k-r)
	# Using logs: log(C(k-1,r-1)) + r*log(p) + (k-r)*log(1-p)
	var log_binom_coeff: float = StatMath.HelperFunctions.log_binomial_coef(k_trials - 1, r_successes - 1)
	var term_p_r: float = float(r_successes) * log(p_prob)
	var term_one_minus_p_k_minus_r: float = float(k_trials - r_successes) * log(1.0 - p_prob)
	
	var log_pmf_val: float = log_binom_coeff + term_p_r + term_one_minus_p_k_minus_r
	return exp(log_pmf_val)


# =============================================================================
# CONTINUOUS DISTRIBUTION PDFs
# =============================================================================

## Calculates the PDF of a normal distribution: f(x; μ, σ).
##
## Returns the probability density at [code]x[/code] for a normal (Gaussian) distribution 
## with mean [code]μ[/code] and standard deviation [code]σ[/code].
##
## Mathematical Note: [code]f(x) = (1/σ√(2π)) e^(-(x-μ)²/(2σ²))[/code]
static func normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
	if not (sigma > 0.0):
		push_error("Standard deviation (sigma) must be positive. Received: %s" % sigma)
		return NAN
	
	var variance: float = sigma * sigma
	var term1: float = 1.0 / (sigma * sqrt(2.0 * PI))
	var term2: float = exp(-pow(x - mu, 2.0) / (2.0 * variance))
	return term1 * term2


## Calculates the PDF of an exponential distribution: f(x; λ).
##
## Returns the probability density at [code]x[/code] for an exponential distribution 
## with rate parameter [code]λ[/code]. Used for modeling waiting times and decay processes.
##
## Mathematical Note: [code]f(x) = λe^(-λx)[/code] for [code]x ≥ 0[/code], [code]0[/code] otherwise
static func exponential_pdf(x: float, lambda_param: float) -> float:
	if not (lambda_param > 0.0):
		push_error("Rate parameter (lambda_param) must be positive. Received: %s" % lambda_param)
		return NAN
	
	if x < 0.0:
		return 0.0
	
	return lambda_param * exp(-lambda_param * x)


## Calculates the PDF of a uniform distribution: f(x; a, b).
##
## Returns the probability density at [code]x[/code] for a uniform distribution 
## on the interval [code][a, b][/code].
##
## Mathematical Note: [code]f(x) = 1/(b-a)[/code] for [code]a ≤ x ≤ b[/code], [code]0[/code] otherwise
static func uniform_pdf(x: float, a: float, b: float) -> float:
	if not (b > a):
		push_error("Parameter b must be greater than a. Received a=%s, b=%s" % [a, b])
		return NAN
	
	if x < a or x > b:
		return 0.0
	
	return 1.0 / (b - a)


## Calculates the PDF of a gamma distribution: f(x; k, θ).
##
## Returns the probability density at [code]x[/code] for a gamma distribution 
## with shape parameter [code]k[/code] and scale parameter [code]θ[/code].
##
## Mathematical Note: [code]f(x) = (1/(Γ(k)θ^k)) x^(k-1) e^(-x/θ)[/code] for [code]x ≥ 0[/code]
static func gamma_pdf(x: float, k_shape: float, theta_scale: float) -> float:
	if not (k_shape > 0.0):
		push_error("Shape parameter (k_shape) must be positive. Received: %s" % k_shape)
		return NAN
	if not (theta_scale > 0.0):
		push_error("Scale parameter (theta_scale) must be positive. Received: %s" % theta_scale)
		return NAN
	
	if x <= 0.0:
		return 0.0
	
	# Formula: (1/(Γ(k)θ^k)) * x^(k-1) * e^(-x/θ)
	# Using logs for numerical stability
	var log_gamma_k: float = StatMath.HelperFunctions.log_gamma(k_shape)
	var log_term1: float = -log_gamma_k - k_shape * log(theta_scale)
	var log_term2: float = (k_shape - 1.0) * log(x)
	var log_term3: float = -x / theta_scale
	
	var log_pdf_val: float = log_term1 + log_term2 + log_term3
	return exp(log_pdf_val)


## Calculates the PDF of a beta distribution: f(x; α, β).
##
## Returns the probability density at [code]x[/code] for a beta distribution 
## with shape parameters [code]α[/code] and [code]β[/code]. Defined on [0, 1].
##
## Mathematical Note: [code]f(x) = (Γ(α+β)/(Γ(α)Γ(β))) x^(α-1) (1-x)^(β-1)[/code]
static func beta_pdf(x: float, alpha: float, beta_param: float) -> float:
	if not (alpha > 0.0 and beta_param > 0.0):
		push_error("Shape parameters (alpha, beta_param) must be positive. Received alpha=%s, beta_param=%s" % [alpha, beta_param])
		return NAN
	
	if x <= 0.0 or x >= 1.0:
		return 0.0
	
	# Formula: (Γ(α+β)/(Γ(α)Γ(β))) * x^(α-1) * (1-x)^(β-1)
	# Using logs for numerical stability
	var log_beta_func: float = StatMath.HelperFunctions.log_gamma(alpha) + StatMath.HelperFunctions.log_gamma(beta_param) - StatMath.HelperFunctions.log_gamma(alpha + beta_param)
	var log_term1: float = -log_beta_func
	var log_term2: float = (alpha - 1.0) * log(x)
	var log_term3: float = (beta_param - 1.0) * log(1.0 - x)
	
	var log_pdf_val: float = log_term1 + log_term2 + log_term3
	return exp(log_pdf_val)


## Calculates the PDF of a Weibull distribution: f(x; λ, k).
##
## Returns the probability density at [code]x[/code] for a Weibull distribution 
## with scale parameter [code]λ[/code] and shape parameter [code]k[/code].
## Widely used in reliability analysis, survival analysis, and failure modeling.
##
## Mathematical Note: [code]f(x) = (k/λ)(x/λ)^(k-1) e^(-(x/λ)^k)[/code] for [code]x ≥ 0[/code]
static func weibull_pdf(x: float, scale_param: float, shape_param: float) -> float:
	if not (scale_param > 0.0):
		push_error("Scale parameter (scale_param) must be positive. Received: %s" % scale_param)
		return NAN
	if not (shape_param > 0.0):
		push_error("Shape parameter (shape_param) must be positive. Received: %s" % shape_param)
		return NAN
	
	if x < 0.0:
		return 0.0
	
	# Handle special case at x=0
	if x == 0.0:
		if shape_param < 1.0:
			return INF  # PDF approaches infinity for shape < 1
		elif shape_param == 1.0:
			return shape_param / scale_param  # Exponential case
		else:  # shape_param > 1.0
			return 0.0
	
	# Formula: (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)
	# Using logs for numerical stability when possible
	var x_over_lambda: float = x / scale_param
	var log_coefficient: float = log(shape_param) - log(scale_param)
	var log_power_term: float = (shape_param - 1.0) * log(x_over_lambda)
	var exponential_term: float = -pow(x_over_lambda, shape_param)
	
	var log_pdf_val: float = log_coefficient + log_power_term + exponential_term
	return exp(log_pdf_val)


## Calculates the PDF of a lognormal distribution: f(x; μ, σ).
##
## Returns the probability density at [code]x[/code] for a lognormal distribution 
## with location parameter [code]μ[/code] and scale parameter [code]σ[/code].
## If X ~ Lognormal(μ, σ), then ln(X) ~ Normal(μ, σ).
##
## Mathematical Note: [code]f(x) = (1/(xσ√(2π))) e^(-((ln(x)-μ)²)/(2σ²))[/code] for [code]x > 0[/code]
static func lognormal_pdf(x: float, mu: float, sigma: float) -> float:
	if not (sigma > 0.0):
		push_error("Standard deviation (sigma) must be positive. Received: %s" % sigma)
		return NAN
	
	if x <= 0.0:
		return 0.0  # Lognormal distribution has support (0, +∞)
	
	# Formula: (1/(x*σ*√(2π))) * exp(-((ln(x)-μ)²)/(2σ²))
	# Equivalent to: Normal PDF of ln(x) divided by x
	var ln_x: float = log(x)
	var normal_result: float = normal_pdf(ln_x, mu, sigma)
	return normal_result / x


## Calculates the PDF of a chi-squared distribution: f(x; k).
##
## Returns the probability density at [code]x[/code] for a chi-squared distribution 
## with [code]k[/code] degrees of freedom. This is a special case of the [StatMath.PmfPdfFunctions.gamma_pdf].
##
## Mathematical Note: [code]f(x) = (1/(2^(k/2)Γ(k/2))) x^(k/2-1) e^(-x/2)[/code] for [code]x ≥ 0[/code]
static func chi_squared_pdf(x: float, k_df: float) -> float:
	if not (k_df > 0.0):
		push_error("Degrees of freedom (k_df) must be positive. Received: %s" % k_df)
		return NAN
	
	# Chi-squared is Gamma(k/2, 2), so use gamma_pdf with appropriate parameters
	return gamma_pdf(x, k_df / 2.0, 2.0)


## Calculates the PDF of a Student's t-distribution: f(x; ν).
##
## Returns the probability density at [code]x[/code] for a Student's t-distribution 
## with [code]ν[/code] (nu) degrees of freedom.
##
## Mathematical Note: [code]f(x) = (Γ((ν+1)/2)/(√(νπ)Γ(ν/2))) (1 + x²/ν)^(-(ν+1)/2)[/code]
static func t_pdf(x: float, df_nu: float) -> float:
	if not (df_nu > 0.0):
		push_error("Degrees of freedom (df_nu) must be positive. Received: %s" % df_nu)
		return NAN
	
	# Formula: (Γ((ν+1)/2)/(√(νπ)Γ(ν/2))) * (1 + x²/ν)^(-(ν+1)/2)
	# Using logs for numerical stability
	var log_gamma_term: float = StatMath.HelperFunctions.log_gamma((df_nu + 1.0) / 2.0) - StatMath.HelperFunctions.log_gamma(df_nu / 2.0)
	var log_normalizer: float = log_gamma_term - 0.5 * log(df_nu * PI)
	var log_power_term: float = -(df_nu + 1.0) / 2.0 * log(1.0 + (x * x) / df_nu)
	
	var log_pdf_val: float = log_normalizer + log_power_term
	return exp(log_pdf_val)


## Calculates the PDF of an F-distribution: f(x; d1, d2).
##
## Returns the probability density at [code]x[/code] for an F-distribution 
## with numerator degrees of freedom [code]d1[/code] and denominator degrees of freedom [code]d2[/code].
##
## Mathematical Note: Uses [StatMath.HelperFunctions.beta_function] relationship for numerical stability
static func f_pdf(x: float, d1_df: float, d2_df: float) -> float:
	if not (d1_df > 0.0 and d2_df > 0.0):
		push_error("Degrees of freedom (d1_df, d2_df) must be positive. Received d1_df=%s, d2_df=%s" % [d1_df, d2_df])
		return NAN
	
	if x <= 0.0:
		return 0.0
	
	# Formula using beta function relationship
	# f(x) = (Γ((d1+d2)/2)/(Γ(d1/2)Γ(d2/2))) * (d1/d2)^(d1/2) * x^(d1/2-1) * (1 + (d1/d2)x)^(-(d1+d2)/2)
	var log_beta_term: float = StatMath.HelperFunctions.log_gamma((d1_df + d2_df) / 2.0) - StatMath.HelperFunctions.log_gamma(d1_df / 2.0) - StatMath.HelperFunctions.log_gamma(d2_df / 2.0)
	var log_ratio_term: float = (d1_df / 2.0) * log(d1_df / d2_df)
	var log_x_term: float = (d1_df / 2.0 - 1.0) * log(x)
	var log_denominator_term: float = -((d1_df + d2_df) / 2.0) * log(1.0 + (d1_df / d2_df) * x)
	
	var log_pdf_val: float = log_beta_term + log_ratio_term + log_x_term + log_denominator_term
	return exp(log_pdf_val)
