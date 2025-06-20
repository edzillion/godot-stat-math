# res://addons/godot-stat-math/core/error_functions.gd
class_name ErrorFunctions extends RefCounted

## Error Functions and Related Mathematical Functions
##
## These functions are commonly used in probability, statistics, and partial differential 
## equations. They are particularly important for working with normal distributions and 
## various statistical calculations.
##
## Mathematical Background:
## The error function is defined as: [code]erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt[/code]

## Computes the error function erf(x).
##
## The error function is an odd function ([code]erf(-x) = -erf(x)[/code]) and is related to 
## the cumulative distribution function (CDF) of the normal distribution. 
## Uses the Abramowitz and Stegun formula 7.1.26 approximation with a maximum error of 1.5 × 10⁻⁷.
## 
## Mathematical Note: [code]erf(0) = 0[/code], [code]erf(∞) = 1[/code], [code]erf(-∞) = -1[/code]
static func erf(x: float) -> float:
	if x == 0.0:
		return 0.0
	
	var abs_x: float = abs(x)
	var sign: float = 1.0 if x >= 0.0 else -1.0
	
	# Constants for Abramowitz and Stegun approximation
	var a1: float = 0.254829592
	var a2: float = -0.284496736
	var a3: float = 1.421413741
	var a4: float = -1.453152027
	var a5: float = 1.061405429
	var p: float = 0.3275911
	
	var t: float = 1.0 / (1.0 + p * abs_x)
	var y: float = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_x * abs_x)
	
	return sign * y


## Computes the complementary error function erfc(x).
##
## Defined as [code]erfc(x) = 1 - erf(x)[/code]. This function is useful for numerical 
## stability when computing values close to 1.
## 
## Mathematical Note: [code]erfc(0) = 1[/code], [code]erfc(∞) = 0[/code], [code]erfc(-∞) = 2[/code]
static func erfc(x: float) -> float:
	return 1.0 - erf(x)


## Computes the inverse error function erf⁻¹(x).
##
## Returns the value y such that [code]erf(y) = x[/code]. Uses an iterative approximation 
## method with Newton-Raphson refinement for improved accuracy.
## Valid input range: [code]-1 < x < 1[/code]
## 
## Mathematical Note: [code]erf_inv(0) = 0[/code], [code]erf_inv(1)[/code] approaches [code]∞[/code], 
## [code]erf_inv(-1)[/code] approaches [code]-∞[/code]
static func erf_inv(x: float) -> float:
	if x == 0.0:
		return 0.0
	if abs(x) > 1.0:
		push_error("Input y for erfinv must be in the range [-1, 1]. Received: %s" % x)
		return NAN
	if abs(x) == 1.0:
		return INF if x > 0 else -INF
	
	# Initial approximation using rational approximation
	var w: float = -log((1.0 - x) * (1.0 + x))
	var p: float
	
	if w < 6.25:
		w = w - 3.125
		p = -3.6444120640178196996e-21
		p = -1.685059138182016589e-19 + p * w
		p = 1.2858480715256400167e-18 + p * w
		p = 1.115787767802518096e-17 + p * w
		p = -1.333171662854620906e-16 + p * w
		p = 2.0972767875968561637e-17 + p * w
		p = 6.6376381343583238325e-15 + p * w
		p = -4.0545662729752068639e-14 + p * w
		p = -8.1519341976054721522e-14 + p * w
		p = 2.6335093153082322977e-12 + p * w
		p = -1.2975133253453532498e-11 + p * w
		p = -5.4154120542946279317e-11 + p * w
		p = 1.051212273321532285e-09 + p * w
		p = -4.1126339803469836976e-09 + p * w
		p = -2.9070369957882005086e-08 + p * w
		p = 4.2347877827932403518e-07 + p * w
		p = -1.3654692000834678645e-06 + p * w
		p = -1.3882523362786468719e-05 + p * w
		p = 0.0001867342080340571352 + p * w
		p = -0.00074070253416626697512 + p * w
		p = -0.0060336708714301490533 + p * w
		p = 0.24015818242558961693 + p * w
		p = 1.6536545626831027356 + p * w
	elif w < 16.0:
		w = sqrt(w) - 3.25
		p = 2.2137376921775787049e-09
		p = 9.0756561938885390979e-08 + p * w
		p = -2.7517406297064545428e-07 + p * w
		p = 1.8239629214389227755e-08 + p * w
		p = 1.5027403968909827627e-06 + p * w
		p = -4.013867526981545969e-06 + p * w
		p = 2.9234449089955446044e-06 + p * w
		p = 1.2475304481671778723e-05 + p * w
		p = -4.7318229009055733981e-05 + p * w
		p = 6.8284851459573175448e-05 + p * w
		p = 2.4031110387097893999e-05 + p * w
		p = -0.0003550375203628474796 + p * w
		p = 0.00095328937973738049703 + p * w
		p = -0.0016882755560235047313 + p * w
		p = 0.0024914420961078508066 + p * w
		p = -0.0037512085075692412107 + p * w
		p = 0.005370914553590063617 + p * w
		p = 1.0052589676941592334 + p * w
		p = 3.0838856104922207635 + p * w
	else:
		w = sqrt(w) - 5.0
		p = -2.7109920616438573243e-11
		p = -2.5556418169965252055e-10 + p * w
		p = 1.5076572693500548083e-09 + p * w
		p = -3.7894654401267369937e-09 + p * w
		p = 7.6157012080783393804e-09 + p * w
		p = -1.4960026627149240478e-08 + p * w
		p = 2.9147953450901080826e-08 + p * w
		p = -6.7711997758452339498e-08 + p * w
		p = 2.2900482228026654717e-07 + p * w
		p = -9.9298272942317002539e-07 + p * w
		p = 4.5260625972231537039e-06 + p * w
		p = -1.9681778105531670567e-05 + p * w
		p = 7.5995277030017761139e-05 + p * w
		p = -0.00021503011930044477347 + p * w
		p = -0.00013871931833623122026 + p * w
		p = 1.0103004648645343977 + p * w
		p = 4.8499064014085844221 + p * w
	
	var result: float = p * (1.0 if x > 0 else -1.0)
	
	# Newton-Raphson refinement
	for i in range(2):
		var err: float = erf(result) - x
		var derivative: float = (2.0 / sqrt(PI)) * exp(-result * result)
		result -= err / derivative
	
	return result


## Computes the inverse complementary error function erfc⁻¹(x).
##
## Returns the value y such that [code]erfc(y) = x[/code]. 
## Valid input range: [code]0 < x < 2[/code]
## 
## Mathematical Note: [code]erfc_inv(1) = 0[/code], [code]erfc_inv(0)[/code] approaches [code]∞[/code], 
## [code]erfc_inv(2)[/code] approaches [code]-∞[/code]
static func erfc_inv(x: float) -> float:
	if x < 0.0 or x > 2.0:
		push_error("Input y for erfcinv must be in the range [0, 2]. Received: %s" % x)
		return NAN
	if x == 0.0:
		return INF
	if x == 2.0:
		return -INF
	return erf_inv(1.0 - x)


## Natural logarithm of the absolute value of the gamma function.
##
## The gamma function [code]Γ(x)[/code] is a generalization of the factorial function to 
## real and complex numbers. For positive integers: [code]Γ(n) = (n-1)![/code]
## This function returns [code]ln|Γ(x)|[/code] to avoid overflow issues.
## 
## Mathematical Note: [code]Γ(1) = 1[/code], [code]Γ(n) = (n-1)![/code] for positive integers
static func log_gamma(x: float) -> float:
	if x <= 0.0:
		return NAN
	
	# Coefficients for Lanczos approximation
	var g: float = 7.0
	var coeff: Array[float] = [
		0.99999999999980993,
		676.5203681218851,
		-1259.1392167224028,
		771.32342877765313,
		-176.61502916214059,
		12.507343278686905,
		-0.13857109526572012,
		9.9843695780195716e-6,
		1.5056327351493116e-7
	]
	
	if x < 0.5:
		# Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
		return log(PI / sin(PI * x)) - log_gamma(1.0 - x)
	
	x -= 1.0
	var a: float = coeff[0]
	for i in range(1, coeff.size()):
		a += coeff[i] / (x + float(i))
	
	var t: float = x + g + 0.5
	return 0.5 * log(2.0 * PI) + (x + 0.5) * log(t) - t + log(a)


## The gamma function Γ(x).
##
## For positive integers: [code]Γ(n) = (n-1)![/code]
## Uses the exponential of [code]log_gamma[/code] to compute the actual gamma value.
## 
## Mathematical Note: [code]Γ(1) = 1[/code], [code]Γ(0.5) = √π[/code]
static func gamma(x: float) -> float:
	if x <= 0.0:
		return NAN
	
	return exp(log_gamma(x))
