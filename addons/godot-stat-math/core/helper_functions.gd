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
	if not (n >= 0):
		push_error("Parameter n must be non-negative for binomial coefficient. Received: %s" % n)
		return NAN
	if not (r >= 0):
		push_error("Parameter r must be non-negative for binomial coefficient. Received: %s" % r)
		return NAN

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
	if not (n >= 0):
		push_error("Factorial (and its log) is undefined for negative numbers. Received: %s" % n)
		return NAN
	if n <= 1: # log(0!) = log(1) = 0; log(1!) = log(1) = 0
		return 0.0
	
	var result: float = 0.0
	for i in range(2, n + 1):
		result += log(float(i)) # Ensure logarithm of float
	return result


# Logarithm of Binomial Coefficient: log(nCk) or log(n choose k)
# Calculates the natural logarithm of the binomial coefficient C(n, k).
static func log_binomial_coef(n: int, k: int) -> float:
	if not (n >= 0):
		push_error("Parameter n must be non-negative for binomial coefficient. Received: %s" % n)
		return NAN
	if not (k >= 0):
		push_error("Parameter k must be non-negative for binomial coefficient. Received: %s" % k)
		return NAN

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
	if not (z > 0.0):
		push_error("Log Gamma function is typically defined for z > 0. Received: %s" % z)
		return NAN
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
	if not (a > 0.0 and b > 0.0):
		push_error("Parameters a and b must be positive for Beta function. Received a=%s, b=%s" % [a, b])
		return NAN
	# Use logarithms for stability if intermediate Gamma values are very large/small.
	# log(B(a,b)) = logΓ(a) + logΓ(b) - logΓ(a+b)
	# B(a,b) = exp(logΓ(a) + logΓ(b) - logΓ(a+b))
	var log_gamma_a: float = log_gamma(a)
	var log_gamma_b: float = log_gamma(b)
	var log_gamma_a_plus_b: float = log_gamma(a + b)
	
	return exp(log_gamma_a + log_gamma_b - log_gamma_a_plus_b)


# Regularized Incomplete Beta Function: I_x(a, b)
# Calculates the regularized incomplete beta function, I_x(a,b) = B(x;a,b) / B(a,b).
# IMPLEMENTATION NOTE: Uses simplified numerical integration method for basic functionality.
# For high-precision applications, consider implementing continued fractions method.
static func incomplete_beta(x_val: float, a: float, b: float) -> float:
	if not (a > 0.0 and b > 0.0):
		push_error("Shape parameters a and b must be positive. Received a=%s, b=%s" % [a, b])
		return NAN
	if not (x_val >= 0.0 and x_val <= 1.0):
		push_error("Parameter x_val must be between 0.0 and 1.0. Received: %s" % x_val)
		return NAN

	if x_val == 0.0:
		return 0.0
	if x_val == 1.0:
		return 1.0
	
	# Special case: Beta(2,2) has exact closed form
	if is_equal_approx(a, 2.0) and is_equal_approx(b, 2.0):
		return x_val * x_val * (3.0 - 2.0 * x_val)
	
	# General case: Use numerical integration (Simpson's rule)
	var n: int = 100  # Number of integration segments
	var h: float = x_val / float(n)
	var sum: float = 0.0
	
	for i in range(n + 1):
		var t: float = float(i) * h
		var weight: float = 1.0
		if i == 0 or i == n:
			weight = 1.0
		elif i % 2 == 1:
			weight = 4.0
		else:
			weight = 2.0
		
		if t > 0.0 and t < 1.0:
			sum += weight * pow(t, a - 1.0) * pow(1.0 - t, b - 1.0)
	
	var integral: float = (h / 3.0) * sum
	var beta_val: float = beta_function(a, b)
	
	if beta_val <= 0.0:
		push_warning("incomplete_beta: Beta function returned invalid value. Using simplified approximation.")
		return x_val  # Fallback approximation
	
	var result: float = integral / beta_val
	
	# Clamp result to valid range [0,1]
	result = clamp(result, 0.0, 1.0)
	
	if a >= 10.0 or b >= 10.0:
		push_warning("incomplete_beta: Using simplified numerical integration. For a=%s, b=%s, consider more advanced methods for higher precision." % [a, b])
	
	return result


# Direct Beta Function (avoid recomputing logs if gamma_function is directly available and stable)
# For use in incomplete_beta if the exp(log_gamma sum) is problematic or for direct calls.
static func log_beta_function_direct(a: float, b: float) -> float:
	if not (a > 0.0 and b > 0.0):
		push_error("Parameters a and b must be positive for Beta function. Received a=%s, b=%s" % [a, b])
		return NAN
	return log_gamma(a) + log_gamma(b) - log_gamma(a+b)


# Regularized Lower Incomplete Gamma Function: P(a,z) = γ(a,z) / Γ(a)
# IMPLEMENTATION NOTE: Uses simplified series expansion method for basic functionality.
# For high-precision applications, consider implementing continued fractions method.
static func lower_incomplete_gamma_regularized(a: float, z: float) -> float:
	if not (a > 0.0):
		push_error("Shape parameter a must be positive for Incomplete Gamma function. Received: %s" % a)
		return NAN
	if not (z >= 0.0):
		push_error("Parameter z must be non-negative for Lower Incomplete Gamma. Received: %s" % z)
		return NAN

	if z == 0.0:
		return 0.0
	
	# Special case for a = 1: P(1,z) = 1 - exp(-z)
	if is_equal_approx(a, 1.0):
		return 1.0 - exp(-z)
	
	# For very large z relative to a, P(a,z) approaches 1
	# Use a more conservative threshold to avoid premature convergence
	if z > a + 50.0:
		return 1.0
	
	# Use series expansion: P(a,z) = (z^a * e^(-z) / Γ(a)) * Σ(z^n / Γ(a+n+1))
	# Which simplifies to: P(a,z) = (z^a * e^(-z) / Γ(a)) * Σ(z^n / (a*(a+1)*...*(a+n)))
	var max_terms: int = 100
	var tolerance: float = 1e-12
	
	var series_sum: float = 1.0  # First term (n=0)
	var term: float = 1.0
	
	# Calculate the series sum
	for n in range(1, max_terms):
		term *= z / (a + float(n - 1))
		series_sum += term
		
		# Check convergence
		if abs(term) < tolerance:
			break
	
	# Calculate the final result: (z^a * e^(-z) / Γ(a)) * series_sum
	var log_prefix: float = a * log(z) - z - log_gamma(a)
	var result: float = exp(log_prefix) * series_sum
	
	# Clamp to valid range [0,1] but be careful not to clamp too aggressively
	# Only clamp if we're slightly outside bounds due to numerical errors
	if result > 1.0 and result < 1.001:
		result = 1.0
	elif result < 0.0 and result > -0.001:
		result = 0.0
	elif result < 0.0 or result > 1.0:
		push_warning("lower_incomplete_gamma_regularized: Result %s is outside [0,1] for a=%s, z=%s. Clamping." % [result, a, z])
		result = clamp(result, 0.0, 1.0)
	
	# Warning for edge cases
	if a < 0.5 or z > 50.0:
		push_warning("lower_incomplete_gamma_regularized: Using series expansion for a=%s, z=%s. Consider more advanced methods for extreme parameters." % [a, z])
	
	return result

# --- Data Preprocessing Functions ---

# Sanitize Numeric Array: Clean and sort numeric data
# Ingests an Array with elements of any type, sanitizes non-integers/floats, 
# and returns a sorted Array[float]. Non-numeric values are filtered out.
# This is useful for preprocessing data before statistical calculations.
static func sanitize_numeric_array(input_array: Array) -> Array[float]:
	var sanitized: Array[float] = []
	
	for element in input_array:
		if element is int or element is float:
			sanitized.append(float(element))
		# Non-numeric values (strings, nulls, objects, etc.) are silently skipped
	
	sanitized.sort()
	return sanitized
