# res://addons/godot-stat-math/tables/acklam_normal_ppf_coefficients.gd

## Coefficients for Peter John Acklam's approximation algorithm for the inverse 
## cumulative standard normal distribution function (normal PPF/quantile function).
##
## This algorithm provides a minimax approximation by rational functions with
## relative error whose absolute value is less than 1.15e-9.
##
## Source: Peter John Acklam's algorithm
## Reference: https://gist.github.com/roguetrainer/8188638
## Original Author: Peter John Acklam (pjacklam@online.no)
## Time-stamp: 2000-07-19 18:26:14
## WWW URL: http://home.online.no/~pjacklam

const VALUES: Dictionary = {
	"acklam_coefficients": {
		# Coefficients for central region approximation (P_LOW < p < P_HIGH)
		"central_numerator": [
			-3.969683028665376e+01,  # A1
			2.209460984245205e+02,   # A2
			-2.759285104469687e+02,  # A3
			1.383577518672690e+02,   # A4
			-3.066479806614716e+01,  # A5
			2.506628277459239e+00    # A6
		],
		"central_denominator": [
			-5.447609879822406e+01,  # B1
			1.615858368580409e+02,   # B2
			-1.556989798598866e+02,  # B3
			6.680131188771972e+01,   # B4
			-1.328068155288572e+01   # B5
		],
		# Coefficients for tail region approximation (p <= P_LOW or p >= P_HIGH)
		"tail_numerator": [
			-7.784894002430293e-03,  # C1
			-3.223964580411365e-01,  # C2
			-2.400758277161838e+00,  # C3
			-2.549732539343734e+00,  # C4
			4.374664141464968e+00,   # C5
			2.938163982698783e+00    # C6
		],
		"tail_denominator": [
			7.784695709041462e-03,   # D1
			3.224671290700398e-01,   # D2
			2.445134137142996e+00,   # D3
			3.754408661907416e+00    # D4
		],
		# Break-points for region selection
		"breakpoints": {
			"p_low": 0.02425,           # Lower breakpoint
			"p_high": 0.97575           # Upper breakpoint (1 - p_low)
		}
	}
} 