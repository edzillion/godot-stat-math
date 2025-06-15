# addons/godot-stat-math/core/distributions.gd
class_name Distributions extends RefCounted

# Random Variate Generation for Statistical Distributions
# This script provides methods to generate random numbers (variates) from various
# common statistical distributions. These are essential for simulations, modeling,
# and various forms of statistical analysis.


# Bernoulli Distribution: randi_bernoulli(p)
# Generates an integer (0 or 1) from a Bernoulli distribution.
# Returns 1 (success) with probability p, and 0 (failure) with probability 1-p.
static func randi_bernoulli(p: float = 0.5) -> int:
	assert(p >= 0.0 and p <= 1.0, "Success probability (p) must be between 0.0 and 1.0.")
	if StatMath.get_rng().randf() <= p:
		return 1
	else:
		return 0


# Binomial Distribution: randi_binomial(p, n)
# Generates an integer representing the number of successes in n independent
# Bernoulli trials, each with success probability p.
# Algorithm uses repeated generation from a geometric distribution.
static func randi_binomial(p: float, n: int) -> int:
	assert(p >= 0.0 and p <= 1.0, "Success probability (p) must be between 0.0 and 1.0.")
	assert(n >= 0, "Number of trials (n) must be non-negative.")

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


# Geometric Distribution: randi_geometric(p)
# Returns the number of Bernoulli trials needed to get one success (always >= 1).
# Uses inverse transform sampling. For very small p, can return StatMath.INT64_MAX_VAL.
# Parameters:
#   p: float - Success probability per trial (0.0 < p <= 1.0).
# Returns: int - Number of trials.
static func randi_geometric(p: float) -> int:
	assert(p > 0.0 and p <= 1.0, "Success probability (p) must be in (0,1].")
	
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


# Poisson Distribution: randi_poisson(lambda_param)
# Generates an integer from a Poisson distribution with mean lambda_param (average rate of events).
# Uses Knuth's algorithm (multiplying uniform random numbers).
static func randi_poisson(lambda_param: float) -> int:
	assert(lambda_param > 0.0, "Rate parameter (lambda_param) must be positive.")
	var l_val: float = exp(-lambda_param)
	var k: int = 0
	var p_val: float = 1.0
	
	while(true):
		k += 1
		p_val *= StatMath.get_rng().randf()
		if p_val <= l_val:
			break
			
	return k - 1


# Pseudo Random Integer Generation (Custom/Specific use case)
# Generates an integer based on an iterative Bernoulli process with increasing success probability.
static func randi_pseudo(c_param: float) -> int:
	assert(c_param > 0.0 and c_param <= 1.0, "Probability increment (c_param) must be in (0.0, 1.0].")
	var current_c: float = c_param
	var trial: int = 0
	while current_c < 1.0:
		trial += 1
		if randi_bernoulli(current_c) == 1:
			break
		current_c += c_param
		# Safety break for very small c_param to prevent potential very long loops.
		if trial > 1000000:
			printerr("randi_pseudo: Exceeded 1,000,000 trials. Check c_param value (%f)." % c_param)
			return trial 
		
	return trial


# Siege Random Integer Generation (Custom/Specific use case)
# Simulates a scenario where capture probability changes based on win/loss outcomes.
static func randi_seige(w: float, c_0: float, c_win: float, c_lose: float) -> int:
	assert(w >= 0.0 and w <= 1.0, "Parameter w (win probability) must be between 0.0 and 1.0.")
	assert(c_0 >= 0.0 and c_0 <= 1.0, "Parameter c_0 (initial capture probability) must be between 0.0 and 1.0.")

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
			printerr("StatMath.Distributions.randi_seige: Exceeded 1,000,000 trials. Review parameters (w=%f, c_0=%f, c_win=%f, c_lose=%f) as they might prevent c_val from reaching a state where capture is likely." % [w, c_0, c_win, c_lose])
			return trials

	# This line should be unreachable given the loop structure and safety break.
	push_error("StatMath.Distributions.randi_seige: Unreachable code hit after while(true) loop. Review logic for w=%s, c_0=%s, c_win=%s, c_lose=%s." % [w, c_0, c_win, c_lose])
	return -1 # Error/unexpected state


# Uniform Distribution (Float): randf_uniform(a, b)
# Generates a random float uniformly distributed in the interval [a, b).
# If a = b, returns a.
static func randf_uniform(a: float, b: float) -> float:
	assert(a <= b, "Lower bound (a) must be less than or equal to upper bound (b) for Uniform distribution.")
	if a == b:
		return a
	return StatMath.get_rng().randf() * (b - a) + a


# Exponential Distribution (Float): randf_exponential(lambda_param)
# Generates a random float from an exponential distribution with rate parameter lambda_param.
# Uses inverse transform sampling method: -log(1-U)/lambda, where U is Uniform(0,1).
static func randf_exponential(lambda_param: float) -> float:
	assert(lambda_param > 0.0, "Rate parameter (lambda_param) must be positive for Exponential distribution.")
	# Ensure u is strictly (0,1) to avoid log(0) or log(1) from 1-u.
	var u: float = StatMath.get_rng().randf()
	while u == 0.0 or u == 1.0: 
		u = StatMath.get_rng().randf()
	return -log(1.0 - u) / lambda_param

	
# Erlang Distribution (Float): randf_erlang(k, lambda_param)
# Generates a random float from an Erlang distribution with shape k (positive integer)
# and rate lambda_param (positive float).
# An Erlang(k, lambda) variate is the sum of k independent Exponential(lambda) variates.
# This implementation uses the method based on product of k uniform variates.
static func randf_erlang(k: int, lambda_param: float) -> float:
	assert(k > 0, "Shape parameter (k) must be a positive integer for Erlang distribution.")
	assert(lambda_param > 0.0, "Rate parameter (lambda_param) must be positive for Erlang distribution.")
	# Sum of k independent exponential variables, or product of k uniform variables method.
	var product: float = 1.0
	for _i in range(k):
		var u: float = StatMath.get_rng().randf()
		while u == 0.0: # Ensure product does not become 0 due to u being 0.
			u = StatMath.get_rng().randf()
		product *= u
		
	return -log(product) / lambda_param


# Gamma Distribution (Float): randf_gamma(shape, scale)
# Generates a random float from a gamma distribution with shape and scale parameters.
# Uses Marsaglia and Tsang's method for shape >= 1, rejection sampling for shape < 1.
# Note: This uses scale parameterization (Gamma(α, θ)) where mean = α*θ and var = α*θ²
static func randf_gamma(shape: float, scale: float = 1.0) -> float:
	assert(shape > 0.0, "Shape parameter must be positive for Gamma distribution.")
	assert(scale > 0.0, "Scale parameter must be positive for Gamma distribution.")
	
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


# Beta Distribution (Float): randf_beta(alpha, beta)
# Generates a random float from a beta distribution using the gamma-to-beta transformation.
# Uses the relationship: if X~Gamma(α,1) and Y~Gamma(β,1), then X/(X+Y)~Beta(α,β)
# This avoids the need for complex special functions ("incomplete" beta, etc.)
static func randf_beta(alpha: float, beta_param: float) -> float:
	assert(alpha > 0.0, "Alpha parameter must be positive for Beta distribution.")
	assert(beta_param > 0.0, "Beta parameter must be positive for Beta distribution.")
	
	# Generate two independent gamma variates with scale=1
	var x: float = randf_gamma(alpha, 1.0)
	var y: float = randf_gamma(beta_param, 1.0)
	
	# Handle edge case where both values are very small
	if x + y <= 0.0:
		return 0.5  # Return midpoint as fallback
	
	return x / (x + y)


# Gaussian (Standard Normal) Distribution (Float): randf_gaussian()
# Generates a random float from a standard normal distribution N(0,1).
# Uses the Box-Muller transform, returning one of the two generated variates.
static func randf_gaussian() -> float: 
	var u1: float = StatMath.get_rng().randf()
	while u1 == 0.0: # Avoid log(0) if randf() could return 0.
		u1 = StatMath.get_rng().randf()
	var u2: float = StatMath.get_rng().randf()
	
	var z0: float = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
	# var z1: float = sqrt(-2.0 * log(u1)) * sin(2.0 * PI * u2) # The second variable, if needed.
	return z0


# Normal Distribution (Float): randf_normal(mu, sigma)
# Generates a random float from a normal (Gaussian) distribution with specified
# mean (mu) and standard deviation (sigma).
# Defaults to N(0,1) if mu and sigma are not provided.
# Transforms a standard normal variate: Z*sigma + mu.
static func randf_normal(mu: float = 0.0, sigma: float = 1.0) -> float: 
	assert(sigma >= 0.0, "Standard deviation (sigma) must be non-negative.")
	if sigma == 0.0:
		return mu # If sigma is 0, all values are the mean.
	return mu + sigma * randf_gaussian()


# Cauchy Distribution (Float): randf_cauchy(location, scale)
# Generates a random float from a Cauchy (Lorentzian) distribution with location and scale parameters.
# Uses the ratio of two independent standard normal variates: X/Y where X,Y ~ N(0,1).
# Note: Cauchy distribution has undefined mean and variance due to heavy tails.
# Useful for modeling extreme events, market fluctuations, and procedural generation with dramatic outliers.
# Parameters:
#   location: float - The location parameter (median of the distribution, default: 0.0).
#   scale: float - The scale parameter (controls spread, must be > 0.0, default: 1.0).
# Returns: float - A Cauchy-distributed random value.
static func randf_cauchy(location: float = 0.0, scale: float = 1.0) -> float:
	assert(scale > 0.0, "Scale parameter must be positive for Cauchy distribution.")
	
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


# Triangular Distribution (Float): randf_triangular(min_value, max_value, mode_value)
# Generates a random float from a triangular distribution with specified bounds and mode.
# The distribution forms a triangle shape with peak at mode_value between min_value and max_value.
# Uses inverse transform sampling method for efficiency and accuracy.
# Commonly used in game development for intuitive parameter generation where you know min/most_likely/max values.
# Parameters:
#   min_value: float - The minimum possible value (left bound of triangle).
#   max_value: float - The maximum possible value (right bound of triangle).
#   mode_value: float - The most likely value (peak of triangle), must satisfy min_value ≤ mode_value ≤ max_value.
# Returns: float - A triangular-distributed random value between min_value and max_value.
static func randf_triangular(min_value: float, max_value: float, mode_value: float) -> float:
	assert(max_value >= min_value, "Maximum value must be greater than or equal to minimum value for Triangular distribution.")
	assert(min_value <= mode_value, "Mode value must be greater than or equal to minimum value for Triangular distribution.")
	assert(mode_value <= max_value, "Mode value must be less than or equal to maximum value for Triangular distribution.")
	
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


# Pareto Distribution (Float): randf_pareto(scale_param, shape_param)
# Generates a random float from a Pareto distribution (also known as power law distribution).
# Models the famous "80/20 rule" and heavy-tailed distributions common in economics and nature.
# Uses inverse transform sampling method for efficiency and mathematical accuracy.
# Commonly used in game development for wealth distribution, loot rarity, city sizes, and resource allocation.
# Parameters:
#   scale_param: float - The scale parameter (minimum possible value, must be > 0.0).
#   shape_param: float - The shape parameter (controls heaviness of tail, must be > 0.0).
#                       Higher values = lighter tail, more concentration near minimum.
#                       Lower values = heavier tail, more extreme values possible.
# Returns: float - A Pareto-distributed random value ≥ scale_param.
# Note: Mean exists only if shape_param > 1, variance exists only if shape_param > 2.
static func randf_pareto(scale_param: float, shape_param: float) -> float:
	assert(scale_param > 0.0, "Scale parameter must be positive for Pareto distribution.")
	assert(shape_param > 0.0, "Shape parameter must be positive for Pareto distribution.")
	
	# Efficient method using exponential transformation:
	# If Y ~ Exponential(shape), then X = scale * exp(Y) ~ Pareto(scale, shape)
	# This avoids expensive pow() operations and reuses existing exponential code
	var exponential_variate: float = randf_exponential(shape_param)
	
	return scale_param * exp(exponential_variate)


# Histogram Distribution (Variant): randv_histogram(values, probabilities)
# Generates a random value from a discrete distribution based on provided values and probabilities.
# Uses cumulative distribution function (CDF) to determine which value to return.
# Parameters:
#   values: Array - Array of values to sample from.
#   probabilities: Array - Array of probabilities for each value.
# Returns: Variant - A random value from the distribution.
static func randv_histogram(values: Array, probabilities: Array) -> Variant:
	assert(!values.is_empty(), "Values array cannot be empty.")
	assert(values.size() == probabilities.size(), "Values and probabilities arrays must have the same size.")
	assert(!probabilities.is_empty(), "Probabilities array cannot be empty.")

	var normalized_probs: Array[float] = []
	var sum_prob: float = 0.0

	for item in probabilities:
		var prob_val: float = 0.0
		if item is int or item is float:
			prob_val = float(item)
		else:
			assert(false, "Probabilities must be numbers (int or float).")
		
		assert(prob_val >= 0.0, "Probabilities must be non-negative.")
		normalized_probs.append(prob_val)
		sum_prob += prob_val
	
	assert(sum_prob > 0.0, "Sum of probabilities must be positive for normalization.")

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
	assert(false, "randv_histogram: Failed to return a value. Check input arrays and logic.")
	return null
