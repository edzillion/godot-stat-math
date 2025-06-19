# res://addons/godot-stat-math/core/helper_functions.gd
class_name HelperFunctions extends RefCounted

## Core Mathematical Helper Functions
##
## This class provides essential mathematical utility functions including combinatorial 
## calculations, special functions (Gamma, Beta), and numerical utilities. These functions 
## serve as the mathematical foundation for statistical calculations throughout the StatMath library.
##
## Mathematical Categories:
## • Combinatorial functions (binomial coefficients, factorials)
## • Gamma and Beta functions with their incomplete variants
## • Array sanitization and preprocessing utilities
## • Logarithmic versions for numerical stability

# Constants are now defined in StatMath.gd

# =============================================================================
# COMBINATORIAL FUNCTIONS
# =============================================================================

## Calculates the binomial coefficient C(n, r) or "n choose r".
##
## Computes the number of ways to choose r items from a set of n items without 
## regard to the order of selection. Uses symmetry optimization and iterative 
## calculation to maintain numerical precision.
##
## Formula: [code]C(n,r) = n! / (r! × (n-r)!)[/code]
##
## Uses symmetry [code]C(n,r) = C(n, n-r)[/code] for efficiency.
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


## Calculates the natural logarithm of n factorial: log(n!).
##
## Computes [code]ln(n!)[/code] directly without calculating the factorial itself, 
## avoiding overflow issues with large factorials. Essential for statistical 
## calculations involving large numbers.
##
## Formula: [code]log(n!) = Σᵢ₌₂ⁿ log(i)[/code]
##
## Special Cases: [code]log(0!) = log(1!) = 0[/code]
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


## Calculates the natural logarithm of the binomial coefficient: log(C(n,k)).
##
## Computes [code]ln(C(n,k))[/code] using logarithmic arithmetic to avoid overflow
## issues with large binomial coefficients. More numerically stable than 
## [code]log(binomial_coefficient(n,k))[/code] for large values.
##
## Formula: [code]log(C(n,k)) = Σᵢ₌₁ᵏ [log(n-i+1) - log(i)][/code]
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


# =============================================================================
# GAMMA FUNCTION AND RELATED
# =============================================================================

## Computes the Gamma function Γ(z).
##
## The Gamma function is a generalization of the factorial function to real and complex numbers.
## For positive integers: [code]Γ(n) = (n-1)![/code]
## Uses the Lanczos approximation for positive values and the reflection formula for negative values.
##
## Mathematical Note: [code]Γ(z)Γ(1-z) = π/sin(πz)[/code] (reflection formula)
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


## Computes the natural logarithm of the Gamma function: log(Γ(z)).
##
## More numerically stable than [code]log(gamma_function(z))[/code] for large z.
## Uses Lanczos approximation directly in logarithmic form to avoid overflow.
##
## Mathematical Note: Only defined for [code]z > 0[/code] where [code]Γ(z) > 0[/code]
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


# =============================================================================
# BETA FUNCTION AND RELATED
# =============================================================================

## Computes the Beta function B(a, b).
##
## The Beta function is defined as [code]B(a,b) = Γ(a)Γ(b) / Γ(a+b)[/code].
## Uses logarithmic arithmetic for numerical stability with large parameter values.
##
## Mathematical Note: [code]B(a,b) = B(b,a)[/code] (symmetric property)
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


## Computes the natural logarithm of the Beta function: log(B(a,b)).
##
## More numerically stable than [code]log(beta_function(a,b))[/code] for large parameters.
## Formula: [code]log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))[/code]
static func log_beta_function_direct(a: float, b: float) -> float:
	if not (a > 0.0 and b > 0.0):
		push_error("Parameters a and b must be positive for Beta function. Received a=%s, b=%s" % [a, b])
		return NAN
	return log_gamma(a) + log_gamma(b) - log_gamma(a+b)


## Computes the regularized incomplete Beta function: I_x(a, b).
##
## Calculates [code]I_x(a,b) = B(x;a,b) / B(a,b)[/code] where [code]B(x;a,b)[/code] is 
## the incomplete Beta function. Uses numerical integration method for basic functionality.
##
## Mathematical Note: [code]I_0(a,b) = 0[/code], [code]I_1(a,b) = 1[/code]
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


## Computes the regularized lower incomplete Gamma function: P(a,z).
##
## Calculates [code]P(a,z) = γ(a,z) / Γ(a)[/code] where [code]γ(a,z)[/code] is the 
## lower incomplete Gamma function. Uses different numerical methods based on parameter 
## ranges for optimal stability.
##
## Mathematical Note: [code]P(a,0) = 0[/code], [code]P(a,∞) = 1[/code]
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
	if z > a + 50.0:
		return 1.0
	
	# IMPROVED ALGORITHM: Use different methods based on parameter ranges for better stability
	var result: float
	
	if z < a + 1.0:
		# Use series expansion for z < a + 1 (generally more stable in this range)
		result = _gamma_series_expansion(a, z)
	else:
		# Use continued fraction expansion for z >= a + 1 (more stable for larger z)
		result = 1.0 - _gamma_continued_fraction(a, z)
	
	# Final validation - should never be outside [0,1] with proper implementation
	if result < 0.0 or result > 1.0:
		push_warning("lower_incomplete_gamma_regularized: Numerical instability detected for a=%s, z=%s (result=%s). Using fallback." % [a, z, result])
		# Fallback to simple approximation for problematic cases
		if z < a:
			result = pow(z / (a + z), a) * 0.5  # Conservative lower bound
		else:
			result = 1.0 - exp(-z) * pow(z, a) / (gamma_function(a + 1.0))  # Upper bound approximation
		result = clamp(result, 0.0, 1.0)
	
	return result


## Helper function for series expansion method (used when z < a + 1).
##
## Implements the series expansion form of the incomplete Gamma function for better 
## numerical stability in the appropriate parameter range.
static func _gamma_series_expansion(a: float, z: float) -> float:
	var max_terms: int = 200  # Increased iterations for better convergence
	var tolerance: float = 1e-15  # Tighter tolerance
	
	var series_sum: float = 1.0
	var term: float = 1.0
	
	for n in range(1, max_terms):
		term *= z / (a + float(n - 1))
		series_sum += term
		
		# Check convergence with relative tolerance
		if abs(term / series_sum) < tolerance:
			break
	
	# More stable calculation using log space
	var log_result: float = a * log(z) - z - log_gamma(a) + log(series_sum)
	
	# Prevent overflow/underflow
	if log_result > 0.0:  # Result would be > 1.0
		return 1.0
	elif log_result < -50.0:  # Result would be essentially 0
		return 0.0
	
	return exp(log_result)


## Helper function for continued fraction method (used when z >= a + 1).
##
## Implements the continued fraction form of the incomplete Gamma function for better 
## numerical stability with larger z values relative to a.
static func _gamma_continued_fraction(a: float, z: float) -> float:
	var max_iterations: int = 200
	var tolerance: float = 1e-15
	
	# Continued fraction coefficients
	var b: float = z + 1.0 - a
	var c: float = 1e30  # Large number
	var d: float = 1.0 / b
	var h: float = d
	
	for i in range(1, max_iterations + 1):
		var an: float = -float(i) * (float(i) - a)
		b += 2.0
		
		d = an * d + b
		if abs(d) < 1e-30:
			d = 1e-30
		c = b + an / c
		if abs(c) < 1e-30:
			c = 1e-30
		
		d = 1.0 / d
		var del: float = d * c
		h *= del
		
		if abs(del - 1.0) < tolerance:
			break
	
	# Calculate final result
	var log_result: float = a * log(z) - z - log_gamma(a) + log(h)
	
	# Prevent overflow/underflow  
	if log_result > 0.0:
		return 1.0
	elif log_result < -50.0:
		return 0.0
	
	return exp(log_result)


# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

## Sanitizes and sorts a mixed-type array into a clean Array[float].
##
## Accepts an Array with elements of any type, filters out non-numeric values, 
## converts remaining elements to float, and returns a sorted array. Essential 
## for preprocessing data before statistical calculations.
##
## Non-numeric values (strings, nulls, objects, etc.) are silently skipped.
static func sanitize_numeric_array(input_array: Array) -> Array[float]:
	var sanitized: Array[float] = []
	
	for element in input_array:
		if element is int or element is float:
			sanitized.append(float(element))
		# Non-numeric values (strings, nulls, objects, etc.) are silently skipped
	
	sanitized.sort()
	return sanitized
