# res://addons/godot-stat-math/core/distributions.gd
class_name Distributions extends RefCounted

## Random Variate Generation for Statistical Distributions
##
## This class provides methods to generate random numbers (variates) from various
## common statistical distributions. These functions are essential for simulations, 
## modeling, and statistical analysis in game development.
##
## Distribution Categories:
##
## * Discrete distributions (Bernoulli, Binomial, Geometric, Poisson)
##
## * Continuous distributions (Normal, Exponential, Gamma, Beta, etc.)
##
## * Specialized distributions (Triangular, Pareto, Weibull, Cauchy)
##
## * Custom distributions (Pseudo, Siege, Histogram)


# =============================================================================
# DISCRETE DISTRIBUTIONS
# =============================================================================

## Generates an integer from a Bernoulli distribution.
##
## Returns 1 (success) with probability [code]p[/code], and 0 (failure) with probability [code]1-p[/code].
## This is the fundamental building block for many other discrete distributions.
##
## Mathematical Note: [code]E[X] = p[/code], [code]Var(X) = p(1-p)[/code]
static func randi_bernoulli(p: float = 0.5) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Success probability (p) must be between 0.0 and 1.0. Received: %s" % p)
		return -1
	if StatMath.get_rng().randf() <= p:
		return 1
	else:
		return 0


## Generates an integer from a Binomial distribution.
##
## Returns the number of successes in [code]n[/code] independent Bernoulli trials, 
## each with success probability [code]p[/code]. Uses repeated geometric distribution sampling.
##
## Mathematical Note: [code]E[X] = np[/code], [code]Var(X) = np(1-p)[/code]
static func randi_binomial(p: float, n: int) -> int:
	if not (p >= 0.0 and p <= 1.0):
		push_error("Success probability (p) must be between 0.0 and 1.0. Received: %s" % p)
		return -1
	if not (n >= 0):
		push_error("Number of trials (n) must be non-negative. Received: %s" % n)
		return -1

	if is_equal_approx(p, 0.0): # If probability of success is 0
		return 0 # Then there are 0 successes

	var count: int = 0
	var current_n: int = n # Use a temporary variable for n to avoid modifying the input
	while(true):
		var curr: int = randi_geometric(p)
		if (curr > current_n):
			return count
		count += 1
		current_n -= curr

	# This line should ideally be unreachable if the logic is correct and p > 0.
	push_error("StatMath.Distributions.randi_binomial: Unreachable code hit after while(true) loop. Review logic for p=%s, n=%s." % [p, n])
	return -1 # Error/unexpected state


## Generates an integer from a Geometric distribution.
##
## Returns the number of Bernoulli trials needed to get one success (always ≥ 1).
## Uses inverse transform sampling for efficiency.
##
## Mathematical Note: [code]E[X] = 1/p[/code], [code]Var(X) = (1-p)/p²[/code]
static func randi_geometric(p: float) -> int:
	if not (p > 0.0 and p <= 1.0):
		push_error("Success probability (p) must be in (0,1]. Received: %s" % p)
		return -1
	
	if p == 1.0:
		return 1

	var under: float = log(1.0 - p) # This will be negative.

	# If p is extremely small, under is ~0. Division by ~0 can lead to INF or errors.
	# int(INF) results in int64.min, so explicitly return max int value.
	if is_equal_approx(under, 0.0):
		return StatMath.INT64_MAX_VAL

	# Inverse transform sampling: k = ceil(log(U) / log(1-p)), where U is randf() in (0,1).
	# Use StatMath.FLOAT_EPSILON to avoid log(0).
	var randf_val: float = StatMath.get_rng().randf_range(StatMath.FLOAT_EPSILON, 1.0) 
	var ra: float = log(randf_val) # ra will be < 0.

	var calc_value_float: float = ra / under # (negative / negative) = positive.

	# Handle potential overflow to INF from the division.
	if calc_value_float == INF:
		return StatMath.INT64_MAX_VAL
	
	var result_float: float = ceil(calc_value_float)
	var final_result: int = int(result_float)
	
	# Result must be >= 1. Handles cases where calc_value_float was ~0 or became negative.
	if final_result < 1:
		return 1
		
	return final_result


## Generates an integer from a Poisson distribution.
##
## Models the number of events occurring in a fixed interval when events occur 
## independently at a constant average rate [code]lambda_param[/code]. Uses Knuth's algorithm.
##
## Mathematical Note: [code]E[X] = Var(X) = λ[/code]
static func randi_poisson(lambda_param: float) -> int:
	if not (lambda_param > 0.0):
		push_error("Rate parameter (lambda_param) must be positive. Received: %s" % lambda_param)
		return -1
	var l_val: float = exp(-lambda_param)
	var k: int = 0
	var p_val: float = 1.0
	
	while(true):
		k += 1
		p_val *= StatMath.get_rng().randf()
		if p_val <= l_val:
			break
			
	return k - 1


## Generates an integer from a discrete uniform distribution.
##
## Returns a random integer uniformly distributed in [code][min_val, max_val][/code] (both inclusive).
## Wrapper around Godot's randi_range for consistency with other distribution functions.
static func randi_uniform(min_val: int, max_val: int) -> int:
	if not (min_val <= max_val):
		push_error("Minimum value must be less than or equal to maximum value for integer uniform distribution. Received min=%s, max=%s" % [min_val, max_val])
		return -1
	return StatMath.get_rng().randi_range(min_val, max_val)


# =============================================================================
# CUSTOM DISCRETE DISTRIBUTIONS
# =============================================================================

## Generates an integer from a custom pseudo-random process.
##
## Uses an iterative Bernoulli process with increasing success probability.
## Success probability starts at [code]c_param[/code] and increases by [code]c_param[/code] each trial.
static func randi_pseudo(c_param: float) -> int:
	if not (c_param > 0.0 and c_param <= 1.0):
		push_error("Probability increment (c_param) must be in (0.0, 1.0]. Received: %s" % c_param)
		return -1
	var current_c: float = c_param
	var trial: int = 0
	while current_c < 1.0:
		trial += 1
		if randi_bernoulli(current_c) == 1:
			break
		current_c += c_param
		# Safety break for very small c_param to prevent potential very long loops.
		if trial > 1000000:
			push_error("randi_pseudo: Exceeded 1,000,000 trials. Check c_param value: %s. This indicates c_param is too small and may cause infinite loops." % c_param)
			return trial 
		
	return trial


## Generates an integer from a custom siege scenario model.
##
## Simulates a scenario where capture probability changes based on win/loss outcomes.
## Attack probability is [code]w[/code], capture probability starts at [code]c_0[/code] and 
## changes by [code]c_win[/code] or [code]c_lose[/code] based on attack results.
static func randi_seige(w: float, c_0: float, c_win: float, c_lose: float) -> int:
	if not (w >= 0.0 and w <= 1.0):
		push_error("Parameter w (win probability) must be between 0.0 and 1.0. Received: %s" % w)
		return -1
	if not (c_0 >= 0.0 and c_0 <= 1.0):
		push_error("Parameter c_0 (initial capture probability) must be between 0.0 and 1.0. Received: %s" % c_0)
		return -1

	var c_val: float = c_0
	var trials: int = 0
	while(true):
		trials += 1
		if randi_bernoulli(w) == 1: # Attack wins
			c_val += c_win
		else: # Attack loses
			c_val += c_lose
		
		c_val = clamp(c_val, 0.0, 1.0) # Ensure c_val remains a valid probability.
			
		if randi_bernoulli(c_val) == 1: # Check for capture
			return trials
		
		# Safety break for parameters that might lead to extremely long or infinite loops.
		if trials > 1000000: 
			push_error("randi_seige: Exceeded 1,000,000 trials. Review parameters w=%s, c_0=%s, c_win=%s, c_lose=%s as they might prevent c_val from reaching a state where capture is likely." % [w, c_0, c_win, c_lose])
			return trials

	# This line should be unreachable given the loop structure and safety break.
	push_error("StatMath.Distributions.randi_seige: Unreachable code hit after while(true) loop. Review logic for w=%s, c_0=%s, c_win=%s, c_lose=%s." % [w, c_0, c_win, c_lose])
	return -1 # Error/unexpected state


# =============================================================================
# CONTINUOUS DISTRIBUTIONS
# =============================================================================

## Generates a float from a continuous uniform distribution.
##
## Returns a random float uniformly distributed in the interval [code][a, b)[/code].
## All values in the interval have equal probability density.
##
## Mathematical Note: [code]E[X] = (a+b)/2[/code], [code]Var(X) = (b-a)²/12[/code]
static func randf_uniform(a: float, b: float) -> float:
	if not (a <= b):
		push_error("Lower bound (a) must be less than or equal to upper bound (b) for Uniform distribution. Received a=%s, b=%s" % [a, b])
		return NAN
	if a == b:
		return a
	return StatMath.get_rng().randf() * (b - a) + a


## Generates a float from an Exponential distribution.
##
## Models the time between events in a Poisson process with rate [code]lambda_param[/code].
## Uses inverse transform sampling: [code]-log(1-U)/λ[/code] where U ~ Uniform(0,1).
##
## Mathematical Note: [code]E[X] = 1/λ[/code], [code]Var(X) = 1/λ²[/code]
static func randf_exponential(lambda_param: float) -> float:
	if not (lambda_param > 0.0):
		push_error("Rate parameter (lambda_param) must be positive for Exponential distribution. Received: %s" % lambda_param)
		return NAN
	# Ensure u is strictly (0,1) to avoid log(0) or log(1) from 1-u.
	var u: float = StatMath.get_rng().randf()
	while u == 0.0 or u == 1.0: 
		u = StatMath.get_rng().randf()
	return -log(1.0 - u) / lambda_param


## Generates a float from an Erlang distribution.
##
## Special case of [method Distributions.randf_gamma] distribution with integer shape parameter [code]k[/code].
## Represents the sum of [code]k[/code] independent Exponential([code]lambda_param[/code]) variables.
##
## Mathematical Note: [code]E[X] = k/λ[/code], [code]Var(X) = k/λ²[/code]
static func randf_erlang(k: int, lambda_param: float) -> float:
	if not (k > 0):
		push_error("Shape parameter (k) must be a positive integer for Erlang distribution. Received: %s" % k)
		return NAN
	if not (lambda_param > 0.0):
		push_error("Rate parameter (lambda_param) must be positive for Erlang distribution. Received: %s" % lambda_param)
		return NAN
	# Sum of k independent exponential variables, or product of k uniform variables method.
	var product: float = 1.0
	for _i in range(k):
		var u: float = StatMath.get_rng().randf()
		while u == 0.0: # Ensure product does not become 0 due to u being 0.
			u = StatMath.get_rng().randf()
		product *= u
		
	return -log(product) / lambda_param


## Generates a float from a Gamma distribution.
##
## Uses scale parameterization: [code]Gamma(α, θ)[/code] where mean = [code]αθ[/code] and variance = [code]αθ²[/code].
## Uses Marsaglia and Tsang's method for shape ≥ 1, rejection sampling for shape < 1.
##
## Mathematical Note: [code]E[X] = αθ[/code], [code]Var(X) = αθ²[/code]
static func randf_gamma(shape: float, scale: float = 1.0) -> float:
	if not (shape > 0.0):
		push_error("Shape parameter must be positive for Gamma distribution. Received: %s" % shape)
		return NAN
	if not (scale > 0.0):
		push_error("Scale parameter must be positive for Gamma distribution. Received: %s" % scale)
		return NAN
	
	var alpha: float = shape
	
	if alpha < 1.0:
		# For shape < 1, use Johnk's generator with rejection
		while true:
			var u: float = StatMath.get_rng().randf()
			var v: float = StatMath.get_rng().randf()
			var x: float = pow(u, 1.0 / alpha)
			var y: float = pow(v, 1.0 / (1.0 - alpha))
			if x + y <= 1.0:
				if x + y > 0.0:
					return scale * x * (-log(StatMath.get_rng().randf())) / (x + y)
	else:
		# For shape >= 1, use Marsaglia and Tsang's method
		var d: float = alpha - 1.0 / 3.0
		var c: float = 1.0 / sqrt(9.0 * d)
		
		while true:
			var x: float = randf_gaussian()
			var cube: float = (1.0 + c * x) * (1.0 + c * x) * (1.0 + c * x)
			var v: float = cube
			
			if v > 0.0:
				var u: float = StatMath.get_rng().randf()
				var x_squared: float = x * x
				
				if u < 1.0 - 0.0331 * x_squared * x_squared:
					return scale * d * v
				if log(u) < 0.5 * x_squared + d * (1.0 - v + log(v)):
					return scale * d * v
	
	# Should never reach here
	push_error("randf_gamma: Failed to generate value")
	return 0.0


## Generates a float from a Beta distribution.
##
## Uses the gamma-to-beta transformation: if [code]X ~ randf_gamma(α,1)[/code] and [code]Y ~ randf_gamma(β,1)[/code], 
## then [code]X/(X+Y) ~ Beta(α,β)[/code]. Values are always in [code][0,1][/code].
##
## Mathematical Note: [code]E[X] = α/(α+β)[/code], [code]Var(X) = αβ/[(α+β)²(α+β+1)][/code]
static func randf_beta(alpha: float, beta_param: float) -> float:
	if not (alpha > 0.0):
		push_error("Alpha parameter must be positive for Beta distribution. Received: %s" % alpha)
		return NAN
	if not (beta_param > 0.0):
		push_error("Beta parameter must be positive for Beta distribution. Received: %s" % beta_param)
		return NAN
	
	# Generate two independent gamma variates with scale=1
	var x: float = randf_gamma(alpha, 1.0)
	var y: float = randf_gamma(beta_param, 1.0)
	
	# Handle edge case where both values are very small
	if x + y <= 0.0:
		return 0.5  # Return midpoint as fallback
	
	return x / (x + y)


## Generates a float from a standard normal distribution N(0,1).
##
## Uses the Box-Muller transform to convert uniform random variables to normal.
## Returns one of the two generated variates (the other is discarded).
##
## Mathematical Note: [code]E[X] = 0[/code], [code]Var(X) = 1[/code]
static func randf_gaussian() -> float: 
	var u1: float = StatMath.get_rng().randf()
	while u1 == 0.0: # Avoid log(0) if randf() could return 0.
		u1 = StatMath.get_rng().randf()
	var u2: float = StatMath.get_rng().randf()
	
	var z0: float = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
	# var z1: float = sqrt(-2.0 * log(u1)) * sin(2.0 * PI * u2) # The second variable, if needed.
	return z0


## Generates a float from a normal distribution with specified mean and standard deviation.
##
## Transforms a standard normal variate: [code]Z*σ + μ[/code] where [code]Z ~ N(0,1)[/code].
## Defaults to [code]N(0,1)[/code] if parameters are not provided.
##
## Mathematical Note: [code]E[X] = μ[/code], [code]Var(X) = σ²[/code]
static func randf_normal(mu: float = 0.0, sigma: float = 1.0) -> float: 
	if not (sigma > 0.0):
		push_error("Standard deviation (sigma) must be positive. Received: %s" % sigma)
		return NAN
	return mu + sigma * randf_gaussian()


## Generates a float from a Lognormal distribution.
##
## Uses the fundamental relationship: if [code]X ~ Normal(μ, σ)[/code], then [code]exp(X) ~ Lognormal(μ, σ)[/code].
## The lognormal distribution models positive values and is commonly used for modeling 
## prices, incomes, and other quantities that cannot be negative.
##
## Mathematical Note: [code]E[X] = exp(μ + σ²/2)[/code], [code]Var(X) = [exp(σ²) - 1] × exp(2μ + σ²)[/code]
static func randf_lognormal(mu: float = 0.0, sigma: float = 1.0) -> float:
	if not (sigma > 0.0):
		push_error("Standard deviation (sigma) must be positive for Lognormal distribution. Received: %s" % sigma)
		return NAN
	# If X ~ Normal(μ, σ), then exp(X) ~ Lognormal(μ, σ)
	return exp(randf_normal(mu, sigma))


## Generates a float from a Cauchy (Lorentzian) distribution.
##
## Uses the ratio of two independent standard normal variates. The Cauchy distribution 
## has undefined mean and variance due to heavy tails, making it useful for modeling 
## extreme events and outliers.
##
## Mathematical Note: Mean and variance are undefined due to heavy tails
static func randf_cauchy(location: float = 0.0, scale: float = 1.0) -> float:
	if not (scale > 0.0):
		push_error("Scale parameter must be positive for Cauchy distribution. Received: %s" % scale)
		return NAN
	
	# Handle degenerate case where scale is zero (though assertion above prevents this)
	if scale == 0.0:
		return location
	
	var x: float = randf_gaussian()  # N(0,1)
	var y: float = randf_gaussian()  # N(0,1)
	
	# Robust handling of near-zero denominator
	# Use a threshold that balances numerical stability with preserving heavy tails
	while abs(y) < 1e-8:
		y = randf_gaussian()
	
	return location + scale * (x / y)


## Generates a float from a Triangular distribution.
##
## Creates values with a triangular probability density function, peaking at [code]mode_value[/code].
## Uses inverse transform sampling for efficiency. Commonly used in game development 
## when you know minimum, most likely, and maximum values.
##
## Mathematical Note: [code]E[X] = (a+b+c)/3[/code] where c is the mode
static func randf_triangular(min_value: float, max_value: float, mode_value: float) -> float:
	if not (max_value >= min_value):
		push_error("Maximum value must be greater than or equal to minimum value for Triangular distribution. Received min=%s, max=%s" % [min_value, max_value])
		return NAN
	if not (min_value <= mode_value):
		push_error("Mode value must be greater than or equal to minimum value for Triangular distribution. Received min=%s, mode=%s" % [min_value, mode_value])
		return NAN
	if not (mode_value <= max_value):
		push_error("Mode value must be less than or equal to maximum value for Triangular distribution. Received mode=%s, max=%s" % [mode_value, max_value])
		return NAN
	
	# Handle degenerate case where all values are the same
	if is_equal_approx(min_value, max_value):
		return min_value
	
	var uniform_random: float = StatMath.get_rng().randf()  # Random number in [0,1)
	var total_range: float = max_value - min_value           # Total width of distribution
	var left_range: float = mode_value - min_value          # Width from min to mode
	var right_range: float = max_value - mode_value         # Width from mode to max
	
	# Calculate the cumulative probability at the mode (where the triangle peaks)
	var mode_cumulative_probability: float = left_range / total_range
	
	# Use inverse transform method based on which side of the triangle we're sampling from
	if uniform_random < mode_cumulative_probability:
		# Left side of triangle: from min_value to mode_value
		# Formula: min + sqrt(U * total_range * left_range)
		var left_area_factor: float = uniform_random * total_range * left_range
		return min_value + sqrt(left_area_factor)
	else:
		# Right side of triangle: from mode_value to max_value  
		# Formula: max - sqrt((1-U) * total_range * right_range)
		var right_area_factor: float = (1.0 - uniform_random) * total_range * right_range
		return max_value - sqrt(right_area_factor)


## Generates a float from a Pareto distribution (power law).
##
## Models the famous "80/20 rule" and heavy-tailed distributions. Uses exponential 
## transformation method: if [code]Y ~ Exponential(shape)[/code], then 
## [code]X = scale * exp(Y) ~ Pareto(scale, shape)[/code].
##
## Mathematical Note: Mean exists only if [code]shape > 1[/code], variance exists only if [code]shape > 2[/code]
static func randf_pareto(scale_param: float, shape_param: float) -> float:
	if not (scale_param > 0.0):
		push_error("Scale parameter must be positive for Pareto distribution. Received: %s" % scale_param)
		return NAN
	if not (shape_param > 0.0):
		push_error("Shape parameter must be positive for Pareto distribution. Received: %s" % shape_param)
		return NAN
	
	# Efficient method using exponential transformation:
	# If Y ~ Exponential(shape), then X = scale * exp(Y) ~ Pareto(scale, shape)
	# This avoids expensive pow() operations and reuses existing exponential code
	var exponential_variate: float = randf_exponential(shape_param)
	
	return scale_param * exp(exponential_variate)


## Generates a float from a Weibull distribution.
##
## Widely used for reliability analysis, survival analysis, and weather modeling.
## Uses inverse transform sampling: [code]λ * (-ln(1-U))^(1/k)[/code] where U ~ Uniform(0,1).
##
## Mathematical Note: Mean = [code]λ * Γ(1 + 1/k)[/code] where Γ is the gamma function. See [method HelperFunctions.gamma_function] for the gamma function implementation.
static func randf_weibull(scale_param: float, shape_param: float) -> float:
	if not (scale_param > 0.0):
		push_error("Scale parameter must be positive for Weibull distribution. Received: %s" % scale_param)
		return NAN
	if not (shape_param > 0.0):
		push_error("Shape parameter must be positive for Weibull distribution. Received: %s" % shape_param)
		return NAN
	
	# Inverse transform sampling: F⁻¹(u) = λ * (-ln(1-u))^(1/k)
	# where u is uniform random variable in (0,1)
	var u: float = StatMath.get_rng().randf()
	
	# Ensure u is not exactly 0 or 1 to avoid log(0) or log(1-1)
	while u == 0.0 or u == 1.0:
		u = StatMath.get_rng().randf()
	
	# Apply inverse transform: scale * (-ln(1-u))^(1/shape)
	var log_term: float = -log(1.0 - u)
	var power_term: float = pow(log_term, 1.0 / shape_param)
	
	return scale_param * power_term


# =============================================================================
# HISTOGRAM DISTRIBUTION
# =============================================================================

## Generates a random value from a discrete histogram distribution.
##
## Samples from provided values using their associated probabilities. Probabilities 
## are automatically normalized, so they don't need to sum to 1. Uses cumulative 
## distribution function (CDF) for efficient sampling.
static func randv_histogram(values: Array, probabilities: Array) -> Variant:
	if values.is_empty():
		push_error("Values array cannot be empty.")
		return null
	if values.size() != probabilities.size():
		push_error("Values and probabilities arrays must have the same size. Received values size=%s, probabilities size=%s" % [values.size(), probabilities.size()])
		return null
	if probabilities.is_empty():
		push_error("Probabilities array cannot be empty.")
		return null

	var normalized_probs: Array[float] = []
	var sum_prob: float = 0.0

	for item in probabilities:
		var prob_val: float = 0.0
		if item is int or item is float:
			prob_val = float(item)
		else:
			push_error("Probabilities must be numbers (int or float). Received: %s" % str(item))
			return null
		
		if not (prob_val >= 0.0):
			push_error("Probabilities must be non-negative. Received: %s" % prob_val)
			return null
		normalized_probs.append(prob_val)
		sum_prob += prob_val
	
	if not (sum_prob > 0.0):
		push_error("Sum of probabilities must be positive for normalization. Received sum: %s" % sum_prob)
		return null

	# Normalize probabilities
	for i in range(normalized_probs.size()):
		normalized_probs[i] = normalized_probs[i] / sum_prob

	var rand_val: float = StatMath.get_rng().randf()
	var running_total: float = 0.0
	for i in range(normalized_probs.size()):
		running_total += normalized_probs[i]
		# Check if rand_val falls into the current probability bin.
		if rand_val < running_total or is_equal_approx(rand_val, running_total):
			# Ensure index is valid (should always be due to earlier assert on array sizes).
			return values[i]
				
	# Fallback for potential floating point inaccuracies or if rand_val is exactly 1.0 (though randf() is [0,1) ).
	# This ensures a value is always returned if probabilities sum correctly.
	if !values.is_empty():
		return values[values.size() - 1]
	
	# This state should ideally be unreachable if input validation is correct.
	push_error("randv_histogram: Failed to return a value. Check input arrays and logic.")
	return null
