extends RefCounted

# Inverse Cumulative Distribution Functions (ICDF), also known as Percentile Point Functions (PPF)
# or Quantile Functions. These functions return the value x such that CDF(x) = p.

# Constants MAX_ITERATIONS and EPSILON are now accessed via StatMath

# Probability Density Functions (PDF)

# Uniform Distribution PPF: uniform_ppf(p, a, b)
# Calculates the Percentile Point Function (inverse of the CDF) for the Uniform distribution.
# Returns the value x in [a, b] such that P(X <= x) = p.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   a: float - The lower bound of the distribution.
#   b: float - The upper bound of the distribution (must be >= a).
# Returns: float - The value x, or NAN if parameters are invalid.
static func uniform_ppf(p: float, a: float, b: float) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if b < a:
		push_error("Parameter b must be greater than or equal to a. Received a=%s, b=%s" % [a,b])
		return NAN
	
	return a + p * (b - a)

# Normal Distribution PPF: normal_ppf(p, mu, sigma)
# Calculates the PPF for the Normal (Gaussian) distribution.
# Returns the value x such that P(X <= x) = p for a N(mu, sigma) distribution.
# Uses Acklam's algorithm (an accurate approximation) for the standard normal N(0,1),
# then transforms it using x_transformed = mu + sigma * x_standard.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   mu: float - The mean of the distribution (default: 0.0).
#   sigma: float - The standard deviation of the distribution (must be > 0.0, default: 1.0).
# Returns: float - The value x. Returns -INF if p=0, INF if p=1, or NAN for invalid sigma.
static func normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if sigma <= 0.0:
		push_error("Standard deviation sigma must be positive. Received: %s" % sigma)
		return NAN
	
	# Handle edge cases
	if p == 0.0:
		return -INF
	if p == 1.0:
		return INF
	
	# Coefficients for the approximation (Acklam, 2010)
	# For p_low < p < p_high
	const A1 := -3.969683028665376e+01
	const A2 := 2.209460984245205e+02
	const A3 := -2.759285104469687e+02
	const A4 := 1.383577518672690e+02
	const A5 := -3.066479806614716e+01
	const A6 := 2.506628277459239e+00
	
	const B1 := -5.447609879822406e+01
	const B2 := 1.615858368580409e+02
	const B3 := -1.556989798598866e+02
	const B4 := 6.680131188771972e+01
	const B5 := -1.328068155288572e+01
	
	# For p <= p_low or p >= p_high
	const C1 := -7.784894002430293e-03
	const C2 := -3.223964580411365e-01
	const C3 := -2.400758277161838e+00
	const C4 := -2.549732539343734e+00
	const C5 := 4.374664141464968e+00
	const C6 := 2.938163982698783e+00
	
	const D1 := 7.784695709041462e-03
	const D2 := 3.224671290700398e-01
	const D3 := 2.445134137142996e+00
	const D4 := 3.754408661907416e+00
	
	const P_LOW := 0.02425
	const P_HIGH := 1.0 - P_LOW
	
	var x: float
	
	if p < P_LOW: # Left tail
		var q := sqrt(-2.0 * log(p))
		x = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) / \
			((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
	elif p <= P_HIGH: # Central region
		var q := p - 0.5
		var r := q * q
		x = (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q / \
			(((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0)
	else: # Right tail (p > P_HIGH)
		var q := sqrt(-2.0 * log(1.0 - p))
		x = -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) / \
			((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)

	# One refinement step for additional precision (optional, but good practice for Acklam's algorithm)
	# Using Normal CDF: 0.5 * (1 + erf(x / sqrt(2)))
	if abs(x) < 10: # Refinement is more effective for non-extreme values
		var cdf_val := 0.5 * (1.0 + StatMath.ErrorFunctions.error_function(x / sqrt(2.0)))
		var error_in_cdf := cdf_val - p
		var pdf_val := (1.0 / sqrt(2.0 * PI)) * exp(-0.5 * x * x)
		if pdf_val > 1e-10: # Avoid division by zero or very small numbers
			x -= error_in_cdf / pdf_val
	
	return mu + sigma * x

# Exponential Distribution PPF: exponential_ppf(p, lambda_param)
# Calculates the PPF for the Exponential distribution.
# Returns the value x such that P(X <= x) = p.
# Uses the closed-form solution: -log(1-p) / lambda_param.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   lambda_param: float - The rate parameter of the distribution (must be > 0.0).
# Returns: float - The value x. Returns 0.0 if p=0, INF if p=1, or NAN for invalid lambda_param.
static func exponential_ppf(p: float, lambda_param: float) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if lambda_param <= 0.0:
		push_error("Rate lambda_param must be positive. Received: %s" % lambda_param)
		return NAN
	
	if p == 1.0: # CDF approaches 1 as x -> INF
		return INF
	if p == 0.0: # CDF is 0 at x = 0
		return 0.0
	
	return -log(1.0 - p) / lambda_param

# Beta Distribution PPF: beta_ppf(p, alpha_shape, beta_shape)
# Calculates the PPF for the Beta distribution.
# Returns the value x in [0,1] such that P(X <= x) = p.
# Uses a numerical binary search (bisection) method, as a closed-form solution is not generally available.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   alpha_shape: float - The alpha shape parameter (must be > 0.0).
#   beta_shape: float - The beta shape parameter (must be > 0.0).
# Returns: float - The value x. Returns 0.0 if p=0, 1.0 if p=1, or NAN for invalid parameters or if search fails.
static func beta_ppf(p: float, alpha_shape: float, beta_shape: float) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if alpha_shape <= 0.0:
		push_error("Shape parameter alpha_shape must be positive. Received: %s" % alpha_shape)
		return NAN
	if beta_shape <= 0.0:
		push_error("Shape parameter beta_shape must be positive. Received: %s" % beta_shape)
		return NAN
	
	# Handle edge cases
	if p == 0.0:
		return 0.0
	if p == 1.0:
		return 1.0
	
	# Use binary search
	var lower_bound := 0.0
	var upper_bound := 1.0
	var x := 0.5 # Initial guess
	var iter := 0
	
	while iter < StatMath.MAX_ITERATIONS:
		# Calculate CDF at current x
		var cdf_val := StatMath.CdfFunctions.beta_cdf(x, alpha_shape, beta_shape)
		if is_nan(cdf_val):
			push_error("beta_cdf returned NaN during beta_ppf search.")
			return NAN # Propagate NaN

		# Check if we're close enough
		if abs(cdf_val - p) < StatMath.EPSILON:
			break
		
		# Update bounds
		if cdf_val < p:
			lower_bound = x
		else:
			upper_bound = x
		
		# Update x using bisection
		x = (lower_bound + upper_bound) / 2.0
		
		# Check for convergence if bounds are too close
		if abs(upper_bound - lower_bound) < StatMath.EPSILON * x : # Relative tolerance
			break
			
		iter += 1
	
	if iter == StatMath.MAX_ITERATIONS:
		push_warning("Beta PPF reached max iterations (%s) for p=%s, alpha=%s, beta=%s. Result might be an approximation." % [StatMath.MAX_ITERATIONS, p, alpha_shape, beta_shape])

	return x

# Gamma Distribution PPF: gamma_ppf(p, k_shape, theta_scale)
# Calculates the PPF for the Gamma distribution.
# Returns the value x such that P(X <= x) = p.
# Uses a numerical binary search (bisection) method with an initial guess often derived
# from the Wilson-Hilferty transformation (for k_shape > 1) or other approximations.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   k_shape: float - The shape parameter k (must be > 0.0).
#   theta_scale: float - The scale parameter theta (must be > 0.0).
# Returns: float - The value x. Returns 0.0 if p=0, INF if p=1, or NAN for invalid parameters or if search fails.
static func gamma_ppf(p: float, k_shape: float, theta_scale: float) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if k_shape <= 0.0:
		push_error("Shape parameter k_shape must be positive. Received: %s" % k_shape)
		return NAN
	if theta_scale <= 0.0:
		push_error("Scale parameter theta_scale must be positive. Received: %s" % theta_scale)
		return NAN
		
	# Handle edge cases
	if p == 0.0:
		return 0.0
	if p == 1.0:
		return INF # CDF approaches 1 as x -> INF
	
	# Initial guess
	var x: float
	if k_shape > 1.0 : # Wilson-Hilferty approximation is good for k_shape > 1
		var z_wh := normal_ppf(p) # Standard normal deviate for probability p. Renamed z to z_wh
		x = k_shape * pow(1.0 - 1.0/(9.0*k_shape) + z_wh / (3.0*sqrt(k_shape)), 3.0) * theta_scale
		if x <= 0: # Ensure guess is positive
			x = k_shape * theta_scale * p # Fallback to simpler guess if WH is non-positive
	else: # For small k_shape, use a simpler guess or median approximation if possible
		x = k_shape * theta_scale * pow(p, 1.0/k_shape) # Based on inverting x^k approx for small x
		if is_nan(x) or x <= 0:
			x = k_shape * theta_scale * p # Fallback
	
	x = max(x, StatMath.EPSILON * theta_scale) # Ensure x is positive for CDF calculation
	
	# Binary search refinement
	var lower_bound := 0.0
	var upper_bound := max(2.0 * x, k_shape * theta_scale + 10.0 * sqrt(k_shape) * theta_scale) 
	if p > 0.999:
		upper_bound = max(upper_bound, x * 10.0)
	upper_bound = max(upper_bound, theta_scale)
	
	var initial_cdf = StatMath.CdfFunctions.gamma_cdf(x, k_shape, theta_scale)
	if is_nan(initial_cdf):
		push_error("gamma_cdf returned NaN during gamma_ppf initial guess check.")
		return NAN

	if initial_cdf > p:
		upper_bound = x
	else: # If initial guess is too low, broaden the search space on the lower side too for safety
		lower_bound = x / 2.0
		if x < lower_bound : # If initial x was very small
			lower_bound = 0.0

	# Ensure x starts within the updated bounds if initial guess was adjusted
	x = clamp(x, lower_bound, upper_bound) 
	
	var iter := 0
	while iter < StatMath.MAX_ITERATIONS:
		var cdf_val := StatMath.CdfFunctions.gamma_cdf(x, k_shape, theta_scale)
		if is_nan(cdf_val):
			push_error("gamma_cdf returned NaN during gamma_ppf search.")
			return NAN

		if abs(cdf_val - p) < StatMath.EPSILON:
			break
		
		if cdf_val < p:
			lower_bound = x
		else:
			upper_bound = x
		
		var prev_x = x
		x = (lower_bound + upper_bound) / 2.0
		
		if abs(x - prev_x) < StatMath.EPSILON * x : # Relative change in x is small
			break
		
		iter += 1
	
	if iter == StatMath.MAX_ITERATIONS:
		push_warning("Gamma PPF reached max iterations (%s) for p=%s, k=%s, theta=%s. Result %s might be an approximation." % [StatMath.MAX_ITERATIONS, p, k_shape, theta_scale, x])
	
	return x

# Chi-Square Distribution PPF: chi_square_ppf(p, k_df)
# Calculates the PPF for the Chi-Square distribution.
# Returns the value x such that P(X <= x) = p.
# This is derived from the Gamma distribution PPF, as ChiSquare(k_df) is equivalent
# to Gamma(shape = k_df/2, scale = 2.0).
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   k_df: float - The degrees of freedom (must be > 0.0).
# Returns: float - The value x. Inherits return behavior (0.0 for p=0, INF for p=1, NAN for errors) from gamma_ppf.
static func chi_square_ppf(p: float, k_df: float) -> float:
	# Chi-square with k_df degrees of freedom is Gamma(shape=k_df/2, scale=2)
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if k_df <= 0.0:
		push_error("Degrees of freedom k_df must be positive. Received: %s" % k_df)
		return NAN
		
	return gamma_ppf(p, k_df / 2.0, 2.0)

# F Distribution PPF: f_ppf(p, d1, d2)
# Calculates the PPF for the F distribution.
# Returns the value x such that P(X <= x) = p.
# Uses a numerical binary search (bisection) method. An initial guess for x can be
# derived from the Beta distribution PPF due to their relationship.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   d1: float - The numerator degrees of freedom (must be > 0.0).
#   d2: float - The denominator degrees of freedom (must be > 0.0).
# Returns: float - The value x. Returns 0.0 if p=0, INF if p=1, or NAN for invalid parameters or if search fails.
static func f_ppf(p: float, d1: float, d2: float) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if d1 <= 0.0:
		push_error("Degrees of freedom d1 must be positive. Received: %s" % d1)
		return NAN
	if d2 <= 0.0:
		push_error("Degrees of freedom d2 must be positive. Received: %s" % d2)
		return NAN
		
	# Handle edge cases
	if p == 0.0:
		return 0.0
	if p == 1.0:
		return INF # CDF approaches 1 as x -> INF
	
	# Binary search method
	var lower_bound := 0.0
	# Initial guess for upper bound, can be quite large for F-dist.
	# Mean is approx d2/(d2-2) for d2>2. If d2 is small, can be large.
	var upper_bound := 1000.0 
	if d2 > 2.0: # A more informed upper starting point
		upper_bound = max(10.0, (d2 / (d2 - 2.0)) * 5.0) # 5x Mean as a rough guide
	if d1 < 2.0: # If d1 is small, variance can be very large or undefined
		upper_bound = max(upper_bound, 10000.0) # Use a larger default if d1 is small

	var x := 1.0  # Initial guess for x
	# Try to get a better initial guess based on relation to Beta distribution or approximations
	# F(p; d1, d2) relates to Beta PPF: if Y ~ Beta(d1/2, d2/2), then X = (d2*Y)/(d1*(1-Y)) ~ F(d1,d2)
	# So, Y_p = Beta.ppf(p, d1/2, d2/2). Then x_p = (d2*Y_p)/(d1*(1-Y_p))
	if d1 > 0 and d2 > 0:
		var beta_p_val = beta_ppf(p, d1/2.0, d2/2.0)
		if not is_nan(beta_p_val) and beta_p_val < 1.0 and beta_p_val > 0.0: # Check validity of beta_ppf output
			x = (d2 * beta_p_val) / (d1 * (1.0 - beta_p_val))
			if x <= 0 or is_nan(x): # Fallback if calculation failed
				x = 1.0
			else: # Use the calculated x as a better guess, refine bounds around it
				upper_bound = max(x * 2.0, upper_bound) # Ensure upper bound covers this guess
		else: # beta_ppf might return 0 or 1 for p=0 or p=1, or NaN for bad params
			x = 1.0 # Fallback to simple guess


	var iter := 0
	while iter < StatMath.MAX_ITERATIONS:
		var cdf_val := StatMath.CdfFunctions.f_cdf(x, d1, d2)
		if is_nan(cdf_val):
			push_error("f_cdf returned NaN during f_ppf search.")
			return NAN

		if abs(cdf_val - p) < StatMath.EPSILON or abs(upper_bound - lower_bound) < StatMath.EPSILON * x :
			break
		
		if cdf_val < p:
			lower_bound = x
		else:
			upper_bound = x
		
		x = (lower_bound + upper_bound) / 2.0
		iter += 1
	
	if iter == StatMath.MAX_ITERATIONS:
		push_warning("F PPF reached max iterations (%s) for p=%s, d1=%s, d2=%s. Result might be an approximation." % [StatMath.MAX_ITERATIONS, p, d1, d2])

	return x

# Student's t Distribution PPF: t_ppf(p, df)
# Calculates the PPF for the Student's t distribution.
# Returns the value x such that P(T <= x) = p.
# Uses a numerical binary search (bisection) method. For large degrees of freedom (df > 1000),
# it approximates using the Normal PPF. Utilizes symmetry for p > 0.5 (T_p(df) = -T_{1-p}(df)).
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   df: float - The degrees of freedom (must be > 0.0).
# Returns: float - The value x. Returns -INF if p=0, 0.0 if p=0.5, INF if p=1, or NAN for invalid df.
static func t_ppf(p: float, df: float) -> float:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return NAN
	if df <= 0.0:
		push_error("Degrees of freedom df must be positive. Received: %s" % df)
		return NAN
		
	# Handle edge cases
	if p == 0.0:
		return -INF
	if p == 0.5:
		return 0.0
	if p == 1.0:
		return INF
	
	# For large df, t-distribution approaches normal distribution
	if df > 1000: # Threshold for approximation
		return normal_ppf(p, 0.0, 1.0) # Standard normal
	
	# Use symmetry: T_p(df) = -T_{1-p}(df)
	if p > 0.5:
		return -t_ppf(1.0 - p, df) # Recurse on the other tail
	
	# Binary search method for p < 0.5 (so t is negative)
	var lower_bound := -100.0  # A reasonably small lower bound (e.g., for df=1, p=0.001, t ~ -300)
	if df == 1.0: # Cauchy distribution can have very extreme tails
		lower_bound = -max(1.0/p, 100.0) # Rough guide for Cauchy tail
	elif df < 5.0:
		lower_bound = -max(20.0/p, 100.0) # Wider range for small df

	var upper_bound := 0.0     # For p < 0.5, we know t is negative (or 0 if p=0.5, handled above)
	var x := -1.0              # Initial guess
	if p < 0.01 and df < 5: # Better initial guess for far tails with low df
		x = lower_bound * p # Very rough starting point

	var iter := 0
	while iter < StatMath.MAX_ITERATIONS:
		var cdf_val := StatMath.CdfFunctions.t_cdf(x, df)
		if is_nan(cdf_val):
			push_error("t_cdf returned NaN during t_ppf search.")
			return NAN

		if abs(cdf_val - p) < StatMath.EPSILON or abs(upper_bound - lower_bound) < StatMath.EPSILON * abs(x):
			break
		
		if cdf_val < p: # CDF is too small, need more negative x
			lower_bound = x
		else: # CDF is too large, need less negative x
			upper_bound = x
		
		x = (lower_bound + upper_bound) / 2.0
		iter += 1
	
	if iter == StatMath.MAX_ITERATIONS:
		push_warning("t-PPF reached max iterations (%s) for p=%s, df=%s. Result might be an approximation." % [StatMath.MAX_ITERATIONS, p, df])
	
	return x

# Binomial Distribution PPF: binomial_ppf(p, n, prob_success)
# Calculates the PPF for the Binomial distribution (a discrete distribution).
# Returns the smallest integer k (number of successes) such that CDF(k) >= p.
# Method: Linear search through possible values of k (0 to n), summing PMF values
# from StatMath.PdfPmfFunctions.binomial_pmf until the cumulative probability meets or exceeds p.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   n: int - The number of trials (must be non-negative).
#   prob_success: float - The probability of success on each trial (must be between 0.0 and 1.0).
# Returns: int - The smallest k. Returns 0 if p=0, n if p=1, or -1 for invalid parameters or if search fails.
static func binomial_ppf(p: float, n: int, prob_success: float) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return -1 # Error code for int function
	if n < 0:
		push_error("Number of trials n must be non-negative. Received: %s" % n)
		return -1
	if not (prob_success >= 0.0 and prob_success <= 1.0):
		push_error("Success probability pr must be between 0.0 and 1.0. Received: %s" % prob_success)
		return -1
	
	if n == 0: # If 0 trials, result is always 0 successes
		return 0

	# Handle edge cases for p
	if p == 0.0: # Smallest k such that CDF(k) >= 0. For Binomial, this is 0.
		return 0
	if p == 1.0: # Largest k such that CDF(k) >= 1. This is n.
		return n
	
	# Linear search for the correct quantile k (number of successes)
	var cumulative_prob := 0.0
	for k in range(n + 1): # k from 0 to n
		var pmf_val: float = StatMath.PmfPdfFunctions.binomial_pmf(k, n, prob_success)
		if is_nan(pmf_val):
			push_error("binomial_pmf returned NaN during binomial_ppf search.")
			return -1 # Propagate error
		cumulative_prob += pmf_val
		
		# Smallest k such that CDF(k) >= p
		if cumulative_prob >= p - StatMath.EPSILON: # Use EPSILON for float comparison
			return k
	
	return n  # Fallback: should be reached if p is very close to 1.0 and cum_prob sums up to slightly less due to precision.

# Poisson Distribution PPF: poisson_ppf(p, lambda_param)
# Calculates the PPF for the Poisson distribution (a discrete distribution).
# Returns the smallest integer k such that CDF(k) >= p.
# Method: Linear search by summing PMF values (from StatMath.PdfPmfFunctions.poisson_pmf)
# starting from k=0 up to a reasonable maximum (lambda + 10*sqrt(lambda) + 20).
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   lambda_param: float - The average rate of events (lambda, must be non-negative).
# Returns: int - The smallest k. Returns 0 if p=0 or lambda_param=0. Returns -1 for invalid parameters
# or if PMF calculation fails. May return a capped k if p is very close to 1.
static func poisson_ppf(p: float, lambda_param: float) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return -1 # Error code for int function
	if lambda_param < 0.0: # lambda = 0 is a valid degenerate case (always 0 events)
		push_error("Rate lambda_param must be non-negative. Received: %s" % lambda_param)
		return -1

	if lambda_param == 0.0: # If rate is 0, always 0 events
		return 0

	# Handle edge cases for p
	if p == 0.0: # Smallest k is 0
		return 0
	# For p=1.0, the PPF is theoretically infinity if lambda > 0.
	# This function returns a large integer (max_k) if the cumulative probability
	# doesn't reach p within the search limit.

	var cumulative_prob := 0.0
	var k := 0
	
	# Determine a reasonable upper bound for k to search.
	# Mean is lambda, StdDev is sqrt(lambda).
	# Search up to mean + several standard deviations, e.g., lambda + 10*sqrt(lambda).
	# Ensure a minimum search range, e.g., 20, if lambda is very small.
	var max_k := int(ceil(lambda_param + 10.0 * sqrt(lambda_param) + 20.0))
	if lambda_param == 0 : max_k = 1 # already handled, but for safety
	
	while k < max_k: # Iterate up to max_k
		var pmf_val: float = StatMath.PdfPmfFunctions.poisson_pmf(k, lambda_param)
		if is_nan(pmf_val):
			push_error("poisson_pmf returned NaN during poisson_ppf search.")
			return -1 # Propagate error
		cumulative_prob += pmf_val
		
		if cumulative_prob >= p - StatMath.EPSILON: # Smallest k such that CDF(k) >= p
			return k
		
		k += 1
	
	# If loop finishes, it means p is very close to 1.0 and max_k was reached,
	# or p=1.0 and the sum of PMFs up to max_k is still less than 1.0 (due to truncation).
	if p > cumulative_prob + StatMath.EPSILON : # If p is genuinely greater than achieved cum_prob
		push_warning("Poisson PPF search reached max_k (%s) for p=%s, lambda=%s. cum_prob=%s. Result might be capped." % [max_k, p, lambda_param, cumulative_prob])
	return k # Return the last k, which is max_k. This is the best guess within the search limit.

# Geometric Distribution PPF: geometric_ppf(p, prob_success)
# Calculates the PPF for the Geometric distribution (discrete distribution for k >= 1).
# This version defines k as the number of Bernoulli trials needed to get one success.
# Returns the smallest integer k (number of trials) such that CDF(k) >= p.
# Uses the closed-form formula: ceil(log(1-p) / log(1-prob_success)) for 0 < p < 1.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   prob_success: float - The probability of success on each trial (must be in (0.0, 1.0]).
# Returns: int - The smallest k. Returns 1 if p=0 or prob_success=1.
# Returns StatMath.INT_MAX_REPRESENTING_INF if p=1.0 and prob_success < 1.0.
# Returns -1 for invalid parameters.
static func geometric_ppf(p: float, prob_success: float) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return -1
	if not (prob_success > 0.0 and prob_success <= 1.0):
		push_error("Success probability pr must be in (0, 1]. Received: %s" % prob_success)
		return -1

	if prob_success == 1.0: # If success is certain on each trial
		return 1 # First trial is guaranteed to be a success. Applies for any p > 0.

	if p == 0.0: # Smallest k in geometric distribution (k>=1) is 1. CDF(0)=0.
		return 1 # So PPF(0) is 1. (Technically, value at lower bound of support for p=0)

	# If p is 1.0 and prob_success < 1.0, it theoretically takes infinite trials.
	if p == 1.0:
		push_warning("Geometric PPF for p=1.0 with pr < 1.0 is theoretically infinity. Returning StatMath.INT_MAX_REPRESENTING_INF.")
		return StatMath.INT_MAX_REPRESENTING_INF 

	# Closed-form PPF for Geometric (k>=1): k = ceil(log(1-p) / log(1-pr))
	# This works for 0 < p < 1 and 0 < pr < 1.
	var log_1_minus_p := log(1.0 - p)
	var log_1_minus_pr := log(1.0 - prob_success)
	
	if abs(log_1_minus_pr) < 1e-9: # Avoid division by zero if pr is extremely close to 1 (already handled by pr==1.0)
		push_error("Geometric PPF: log(1-pr) is too close to zero, pr might be too close to 1.0 (but not exactly 1.0). pr=%s" % prob_success)
		return -1 

	return int(ceil(log_1_minus_p / log_1_minus_pr))

# Negative Binomial Distribution PPF: negative_binomial_ppf(p, r_successes, prob_success)
# Calculates the PPF for the Negative Binomial distribution (discrete distribution).
# This version defines k as the number of trials to achieve r_successes successes, so k >= r_successes.
# Returns the smallest integer k (number of trials) such that CDF(k) >= p.
# Method: Linear search by summing PMF values (from StatMath.PdfPmfFunctions.negative_binomial_pmf)
# starting from k = r_successes up to a reasonable maximum.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   r_successes: int - The target number of successes (must be > 0).
#   prob_success: float - The probability of success on each trial (must be in (0.0, 1.0]).
# Returns: int - The smallest k. Returns r_successes if p=0 or prob_success=1.
# Returns StatMath.INT_MAX_REPRESENTING_INF if p=1.0 and prob_success < 1.0.
# Returns -1 for invalid parameters or if PMF calculation fails. May return a capped k.
static func negative_binomial_ppf(p: float, r_successes: int, prob_success: float) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return -1
	if r_successes <= 0:
		push_error("Number of successes r_successes must be a positive integer. Received: %s" % r_successes)
		return -1
	if not (prob_success > 0.0 and prob_success <= 1.0):
		push_error("Success probability pr must be in (0, 1]. Received: %s" % prob_success)
		return -1

	if prob_success == 1.0: # If success is certain on each trial
		return r_successes # Exactly r_successes trials are needed.

	# Handle p edge cases (now prob_success is in (0,1) )
	if p == 0.0: # Smallest k for Negative Binomial is r_successes. CDF(r-1)=0.
		return r_successes
	
	# If p=1.0 and prob_success < 1.0, theoretically infinite trials.
	if p == 1.0:
		push_warning("Negative Binomial PPF for p=1.0 with pr < 1.0 is theoretically infinity. Returning StatMath.INT_MAX_REPRESENTING_INF.")
		return StatMath.INT_MAX_REPRESENTING_INF 

	# Linear search for the correct quantile k (number of trials)
	var cumulative_prob := 0.0
	var k_trials := r_successes  # Start search from the minimum possible number of trials.
	
	# Determine a reasonable upper bound for k_trials.
	# Mean is r / pr. StdDev is sqrt(r*(1-pr)) / pr.
	# Search up to mean + several standard deviations.
	var mean_trials := float(r_successes) / prob_success
	var std_dev_trials := sqrt(float(r_successes) * (1.0 - prob_success)) / prob_success
	var max_k_trials := int(ceil(mean_trials + 10.0 * std_dev_trials + float(r_successes) + 20.0)) # Ensure it's well above r_successes and covers variability.
	max_k_trials = max(max_k_trials, r_successes + 100) # A fallback minimum search range.


	while k_trials < max_k_trials:
		var pmf_val: float = StatMath.PdfPmfFunctions.negative_binomial_pmf(k_trials, r_successes, prob_success)
		if is_nan(pmf_val):
			push_error("negative_binomial_pmf returned NaN during negative_binomial_ppf search (k=%s, r=%s, p=%s)." % [k_trials, r_successes, prob_success])
			return -1 
		cumulative_prob += pmf_val
		
		if cumulative_prob >= p - StatMath.EPSILON: # Smallest k_trials such that CDF(k_trials) >= p
			return k_trials
		
		k_trials += 1
	
	if p > cumulative_prob + StatMath.EPSILON:
		push_warning("Negative Binomial PPF search reached max_k_trials (%s) for p=%s, r=%s, pr=%s. cum_prob=%s. Result might be capped." % [max_k_trials, p, r_successes, prob_success, cumulative_prob])
	return k_trials # Return last k_trials (max_k_trials) as best guess.

# Bernoulli Distribution PPF: bernoulli_ppf(p, prob_success)
# Calculates the PPF for the Bernoulli distribution (a discrete distribution).
# This is a special case of the Binomial distribution where n (number of trials) is 1.
# Returns 0 (failure) or 1 (success) such that CDF(k) >= p.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   prob_success: float - The probability of success (must be between 0.0 and 1.0).
# Returns: int - 0 or 1. Returns -1 for invalid parameters.
static func bernoulli_ppf(p: float, prob_success: float) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return -1
	if not (prob_success >= 0.0 and prob_success <= 1.0):
		push_error("Success probability prob_success must be between 0.0 and 1.0. Received: %s" % prob_success)
		return -1

	# A Bernoulli trial is a Binomial trial with n=1.
	# binomial_ppf(p, 1, prob_success) will return 0 or 1.
	return binomial_ppf(p, 1, prob_success)

# Discrete Histogram PPF: discrete_histogram_ppf(p, values, probabilities)
# Calculates the PPF for a user-defined discrete distribution represented by a histogram.
# Returns the value from the 'values' array corresponding to the smallest cumulative probability >= p.
# Assumes 'values' and 'probabilities' are correctly ordered if a specific order is meaningful.
# Parameters:
#   p: float - The probability value (must be between 0.0 and 1.0).
#   values: Array - An array of outcome values (can be Variant types).
#   probabilities: Array[float] - An array of corresponding probabilities for each outcome.
# Returns: Variant - The outcome value from the 'values' array. Returns null for invalid parameters or errors.
static func discrete_histogram_ppf(p: float, values: Array, probabilities: Array[float]) -> Variant:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Probability p must be between 0.0 and 1.0 (inclusive). Received: %s" % p)
		return null
	if values.is_empty():
		push_error("Values array cannot be empty for discrete_histogram_ppf.")
		return null
	if probabilities.is_empty():
		push_error("Probabilities array cannot be empty for discrete_histogram_ppf.")
		return null
	if values.size() != probabilities.size():
		push_error("Values and probabilities arrays must have the same size. Received sizes %s and %s." % [values.size(), probabilities.size()])
		return null

	var sum_probs: float = 0.0
	for prob_val in probabilities:
		if prob_val < 0.0:
			push_error("All probabilities must be non-negative. Found: %s" % prob_val)
			return null
		sum_probs += prob_val
	
	if abs(sum_probs - 1.0) > StatMath.EPSILON:
		push_warning("Sum of probabilities (%s) is not close to 1.0 for discrete_histogram_ppf. Normalization may be needed." % sum_probs)
		# Proceeding, but this might indicate an issue with input data.

	var cumulative_prob: float = 0.0
	for i in range(values.size()):
		var current_value: Variant = values[i]
		var current_prob: float = probabilities[i]
		
		if current_prob < 0.0: # Should have been caught above, but as a safeguard during summation.
			push_error("Encountered negative probability during PPF calculation: %s" % current_prob)
			return null

		cumulative_prob += current_prob
		
		# Smallest value k such that CDF(k) >= p
		if cumulative_prob >= p - StatMath.EPSILON: # Use EPSILON for float comparison
			return current_value
	
	# If p is very close to 1.0 and the sum of probabilities is slightly less than 1 due to float precision,
	# or if p=1.0, return the last value.
	if values.size() > 0:
		return values[values.size() - 1]

	# Should not be reached if initial checks pass and arrays are not empty.
	push_error("discrete_histogram_ppf: Failed to find a value. This state should be unreachable.")
	return null
