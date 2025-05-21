extends RefCounted

# Core Mathematical Helper Functions
# This script provides a collection of static mathematical utility functions,
# including logarithms of factorials, binomial coefficients, direct calculation
# of binomial coefficients, and various special functions like Gamma, Beta,
# and their incomplete or regularized forms.
# These are fundamental for many statistical calculations.

# Constants are now defined in StatMath.gd

# --- Combinatorial Functions ---

# Binomial Coefficient: C(n, r) or "n choose r"
# Calculates the number of ways to choose r items from a set of n items
# without regard to the order of selection.
static func binomial_coefficient(n: int, r: int) -> float:
	assert(n >= 0, "Parameter n must be non-negative for binomial coefficient.")
	assert(r >= 0, "Parameter r must be non-negative for binomial coefficient.")

	if r < 0 or r > n:
		return 0.0 # By definition, C(n,r) is 0 if r is out of range [0, n]

	# Use symmetry C(n,r) = C(n, n-r). Choose smaller r for efficiency.
	var r_symmetric: int = r
	if r_symmetric > n / 2.0: # Corrected: Use / for division, ensure float context for comparison
		r_symmetric = n - r_symmetric

	if r_symmetric == 0: # C(n,0) = 1 and C(n,n) = 1
		return 1.0

	var coeff: float = 1.0
	# Iteratively calculate C(n, r_symmetric) = product_{i=1 to r_symmetric} (n - i + 1) / i
	for i in range(1, r_symmetric + 1):
		# Multiply by (n-i+1) then divide by i to maintain precision as much as possible
		# and reduce risk of intermediate numbers becoming too small before multiplication.
		coeff = coeff * float(n - i + 1) / float(i)
	
	return coeff


# Logarithm of Factorial: log(n!)
# Calculates the natural logarithm of n factorial.
# Useful for avoiding overflow with large factorials.
static func log_factorial(n: int) -> float:
	assert(n >= 0, "Factorial (and its log) is undefined for negative numbers.")
	if n <= 1: # log(0!) = log(1) = 0; log(1!) = log(1) = 0
		return 0.0
	
	var result: float = 0.0
	for i in range(2, n + 1):
		result += log(float(i)) # Ensure logarithm of float
	return result


# Logarithm of Binomial Coefficient: log(nCk) or log(n choose k)
# Calculates the natural logarithm of the binomial coefficient C(n, k).
static func log_binomial_coef(n: int, k: int) -> float:
	assert(n >= 0, "Parameter n must be non-negative for binomial coefficient.")
	assert(k >= 0, "Parameter k must be non-negative for binomial coefficient.")

	if k < 0 or k > n: # C(n,k) is 0 if k < 0 or k > n
		return -INF   # log(0) tends to -infinity
	
	if k == 0 or k == n: # C(n,0) = 1, C(n,n) = 1
		return 0.0       # log(1) = 0
	
	# Use symmetry C(n,k) = C(n, n-k) to use smaller k for efficiency.
	var actual_k: int = k
	if (n - k) < k:
		actual_k = n - k
		
	var result: float = 0.0
	# Formula: log(n! / (k! * (n-k)!)) = log(n!) - log(k!) - log((n-k)!)
	# More direct summation to avoid large intermediate factorials:
	# Sum_{i=1 to k} log(n-i+1) - Sum_{i=1 to k} log(i)
	for i in range(1, actual_k + 1):
		result += log(float(n - i + 1))
		result -= log(float(i))
	return result

# --- Gamma Function and Related --- 

# Gamma Function: Γ(z)
# Computes the Gamma function using the Lanczos approximation.
# Handles positive real numbers; uses reflection formula for z <= 0.
static func gamma_function(z: float) -> float:
	if z <= 0.0:
		# Reflection formula: Γ(z) * Γ(1-z) = π / sin(πz)
		# So, Γ(z) = π / (sin(πz) * Γ(1-z))
		# Check for poles at non-positive integers
		if is_equal_approx(z, floor(z)): # z is a non-positive integer
			return INF # Pole at 0, -1, -2, ...
		# Avoid issues with sin(PI*z) being zero if z is an integer (handled above)
		var sin_pi_z: float = sin(PI * z)
		if is_equal_approx(sin_pi_z, 0.0):
			return INF # Or NAN, effectively a pole or indeterminate form
		return PI / (sin_pi_z * gamma_function(1.0 - z))
	
	# Lanczos approximation for z > 0
	var x: float = z - 1.0 # Shifted variable for Lanczos coefficients
	var y_base: float = x + StatMath.LANCZOS_G # Base for power y^(x+0.5)
	
	var series_sum: float = StatMath.LANCZOS_P[0]
	for i in range(1, StatMath.LANCZOS_P.size()):
		series_sum += StatMath.LANCZOS_P[i] / (x + float(i))
	
	#sqrt(2π) * y^(x+0.5) * e^(-y) * sum
	return sqrt(2.0 * PI) * pow(y_base, x + 0.5) * exp(-y_base) * series_sum


# Logarithm of the Gamma Function: log(Γ(z))
# Computes the natural logarithm of the Gamma function using Lanczos approximation directly for log.
# More numerically stable than log(gamma_function(z)) for large z.
static func log_gamma(z: float) -> float:
	assert(z > 0.0, "Log Gamma function is typically defined for z > 0.")
	# Reflection formula for log_gamma can be complex due to sign changes of Gamma(z).
	# This implementation focuses on z > 0 where Gamma(z) is positive.

	var x: float = z - 1.0
	var y_base: float = x + StatMath.LANCZOS_G
	
	var series_sum_val: float = StatMath.LANCZOS_P[0]
	for i in range(1, StatMath.LANCZOS_P.size()):
		series_sum_val += StatMath.LANCZOS_P[i] / (x + float(i))
	
	# log(sqrt(2π)) + log(sum) + (x+0.5)*log(y) - y
	return log(sqrt(2.0 * PI)) + log(series_sum_val) + (x + 0.5) * log(y_base) - y_base

# --- Beta Function and Related --- 

# Beta Function: B(a, b)
# Defined as Γ(a)Γ(b) / Γ(a+b).
static func beta_function(a: float, b: float) -> float:
	assert(a > 0.0 and b > 0.0, "Parameters a and b must be positive for Beta function.")
	# Use logarithms for stability if intermediate Gamma values are very large/small.
	# log(B(a,b)) = logΓ(a) + logΓ(b) - logΓ(a+b)
	# B(a,b) = exp(logΓ(a) + logΓ(b) - logΓ(a+b))
	var log_gamma_a: float = log_gamma(a)
	var log_gamma_b: float = log_gamma(b)
	var log_gamma_a_plus_b: float = log_gamma(a + b)
	
	return exp(log_gamma_a + log_gamma_b - log_gamma_a_plus_b)


# Regularized Incomplete Beta Function: I_x(a, b)
# Calculates the regularized incomplete beta function, I_x(a,b) = B(x;a,b) / B(a,b).
# WARNING: THIS FUNCTION IS CURRENTLY A PLACEHOLDER AND NOT IMPLEMENTED.
# A robust implementation (e.g., using continued fractions like in Numerical Recipes betacf) is required.
static func incomplete_beta(x_val: float, a: float, b: float) -> float:
	assert(a > 0.0 and b > 0.0, "Shape parameters a and b must be positive.")
	assert(x_val >= 0.0 and x_val <= 1.0, "Parameter x_val must be between 0.0 and 1.0.")

	if x_val == 0.0:
		return 0.0
	if x_val == 1.0:
		return 1.0

	# TODO: Implement this using continued fractions like in Numerical Recipes betacf
	push_error("CRITICAL: StatMath.HelperFunctions.incomplete_beta(x=%s, a=%s, b=%s) is NOT IMPLEMENTED. It returns a placeholder. DO NOT USE FOR ACCURATE CALCULATIONS." % [x_val, a, b])
	return NAN # Return NaN to clearly indicate an uncomputed/error state.


# Direct Beta Function (avoid recomputing logs if gamma_function is directly available and stable)
# For use in incomplete_beta if the exp(log_gamma sum) is problematic or for direct calls.
static func log_beta_function_direct(a: float, b: float) -> float:
	assert(a > 0.0 and b > 0.0, "Parameters a and b must be positive for Beta function.")
	return log_gamma(a) + log_gamma(b) - log_gamma(a+b)


# Regularized Lower Incomplete Gamma Function: P(a,z) = γ(a,z) / Γ(a)
# WARNING: THIS FUNCTION IS CURRENTLY A PLACEHOLDER AND NOT IMPLEMENTED ACCURATELY FOR ALL CASES.
# The current implementation is a combination of series and continued fraction methods from Numerical Recipes
# but requires rigorous testing and verification for robustness across all input ranges.
# Using it may lead to inaccurate results, especially for certain parameter values.
static func lower_incomplete_gamma_regularized(a: float, z: float) -> float:
	assert(a > 0.0, "Shape parameter a must be positive for Incomplete Gamma function.")
	if z < 0.0:
		assert(false, "Parameter z must be non-negative for Lower Incomplete Gamma.")
		# Fallback for negative z, though assertions should catch typical use.
		push_error("StatMath.HelperFunctions.lower_incomplete_gamma_regularized called with negative z=%s. Returning NAN." % z)
		return NAN 

	if z == 0.0:
		return 0.0

	# TODO: Rigorously test and verify this implementation or replace with a known-good one.
	push_error("CRITICAL: StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a=%s, z=%s) is NOT FULLY VERIFIED and may be inaccurate. It returns a placeholder/unverified value. DO NOT USE FOR CRITICAL CALCULATIONS WITHOUT VALIDATION." % [a, z])
	
	# For now, to make it clear it's a placeholder despite the existing code block:
	# We will return NAN instead of the result of the current unverified algorithm.
	# To re-enable the experimental algorithm below for testing, comment out the next line.
	return NAN
