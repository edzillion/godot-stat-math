extends RefCounted

# Error Functions and their Inverses
# These functions are commonly used in probability, statistics, and partial differential equations.

# Constants are now defined in StatMath.gd

# Error Function erf(x_param)
# Computes the error function, an odd function (erf(-x) = -erf(x)).
# It is related to the cumulative distribution function (CDF) of the normal distribution.
# Approximation uses Abramowitz and Stegun formula 7.1.26, which has a maximum error of 1.5 * 10^-7.
static func error_function(x_param: float) -> float:
	# Save the sign of x_param
	var sign_val: float = 1.0
	if x_param < 0.0:
		sign_val = -1.0
	var abs_x: float = abs(x_param)
	
	var t: float = 1.0 / (1.0 + StatMath.P_ERR * abs_x)
	var y_err: float = 1.0 - (((((StatMath.A5_ERR * t + StatMath.A4_ERR) * t) + StatMath.A3_ERR) * t + StatMath.A2_ERR) * t + StatMath.A1_ERR) * t * exp(-abs_x * abs_x)
	
	return sign_val * y_err

# Complementary Error Function erfc(x_param)
# Defined as 1 - erf(x_param).
static func complementary_error_function(x_param: float) -> float:
	return 1.0 - error_function(x_param)

# Inverse Error Function erfinv(y)
# Returns x such that y = erf(x).
# The input y must be in the range [-1, 1].
# For y = -1 or y = 1, it returns -INF or INF respectively.
static func error_function_inverse(y: float) -> float:
	assert(y >= -1.0 and y <= 1.0, "Input y for erfinv must be in the range [-1, 1].")
	if y == -1.0:
		return -INF
	if y == 1.0:
		return INF
	var p_val_for_ppf: float = (1.0 + y) / 2.0
	return sqrt(2.0) * StatMath.PpfFunctions.normal_ppf(p_val_for_ppf, 0.0, 1.0)

# Inverse Complementary Error Function erfcinv(y)
# Returns x such that y = erfc(x).
# The input y must be in the range [0, 2].
# For y = 0 or y = 2, it returns INF or -INF respectively.
static func complementary_error_function_inverse(y: float) -> float:
	assert(y >= 0.0 and y <= 2.0, "Input y for erfcinv must be in the range [0, 2].")
	if y == 0.0:
		return INF
	if y == 2.0:
		return -INF
	return error_function_inverse(1.0 - y)
