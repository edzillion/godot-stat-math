extends RefCounted

# Probability Mass Functions (PMF) and Probability Density Functions (PDF)
# This script provides static methods to calculate the PMF for discrete distributions
# and PDF for continuous distributions. The PMF/PDF gives the probability (or density)
# of a random variable taking on a specific value.

# Binomial Distribution PMF: P(X=k_successes | n_trials, p_prob)
# Calculates the probability of observing exactly k_successes in n_trials independent
# Bernoulli trials, each with a success probability p_prob.
# Uses logarithms for numerical stability with potentially large combinations or small probabilities.
static func binomial_pmf(k_successes: int, n_trials: int, p_prob: float) -> float:
	assert(n_trials >= 0, "Number of trials (n_trials) must be non-negative.")
	assert(p_prob >= 0.0 and p_prob <= 1.0, "Success probability (p_prob) must be between 0.0 and 1.0.")
	
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


# Poisson Distribution PMF: P(X=k_events | lambda_param)
# Calculates the probability of observing exactly k_events in a fixed interval,
# given an average rate lambda_param of events.
# Uses logarithms for numerical stability.
static func poisson_pmf(k_events: int, lambda_param: float) -> float:
	assert(lambda_param >= 0.0, "Rate parameter (lambda_param) must be non-negative.")
	
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


# Negative Binomial Distribution PMF: P(X=k_trials | r_successes, p_prob)
# Calculates the probability that the r_successes-th success occurs on exactly the k_trials-th trial
# in a series of independent Bernoulli trials with success probability p_prob.
# Uses logarithms for numerical stability.
static func negative_binomial_pmf(k_trials: int, r_successes: int, p_prob: float) -> float:
	assert(r_successes > 0, "Number of required successes (r_successes) must be positive.")
	assert(p_prob > 0.0 and p_prob <= 1.0, "Success probability (p_prob) must be in (0,1].")

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


# --- Probability Density Functions (PDF) ---
# (PDF functions will be added here later)
# Example:
# static func normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
#     assert(sigma > 0.0, "Standard deviation (sigma) must be positive for Normal PDF.")
#     var variance: float = sigma * sigma
#     var term1: float = 1.0 / (sigma * sqrt(2.0 * PI))
#     var term2: float = exp(-(pow(x - mu, 2.0)) / (2.0 * variance))
#     return term1 * term2
