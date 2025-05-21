extends Node

# --- Core Mathematical & Numerical Constants ---
# Represents a very large integer, for integer-returning functions where infinity is theoretical.
const INT_MAX_REPRESENTING_INF := 2147483647

# Represents the maximum value for a 64-bit signed integer.
const INT64_MAX_VAL: int = 9223372036854775807 # (1 << 63) - 1

# Smallest positive float x such that 1.0 + x != 1.0. Machine epsilon.
const FLOAT_EPSILON: float = 2.220446049250313e-16

# Constants for iterative approximations (from helper_functions.gd)
const MAX_ITERATIONS: int = 200  # Max iterations for series/continued fractions
const EPSILON: float = 1.0e-9 # Small epsilon for convergence checks & float comparisons

# Constants for Lanczos approximation of the Gamma function (from helper_functions.gd)
const LANCZOS_G: float = 7.5
const LANCZOS_P: Array[float] = [
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

# Constants for Abramowitz and Stegun approximation of erf(x) (from error_functions.gd)
const A1_ERR: float =  0.254829592
const A2_ERR: float = -0.284496736
const A3_ERR: float =  1.421413741
const A4_ERR: float = -1.453152027
const A5_ERR: float =  1.061405429
const P_ERR: float  =  0.3275911

# --- Preload Core Functionality Scripts ---
# These can be accessed via StatMath.ModuleName.function_name()
# e.g., StatMath.Distributions.randi_bernoulli(0.5)
# or constants via StatMath.CONSTANT_NAME, e.g. StatMath.EPSILON
const Distributions = preload("res://addons/godot-stat-math/core/distributions.gd")
const CdfFunctions = preload("res://addons/godot-stat-math/core/cdf_functions.gd")
const PmfPdfFunctions = preload("res://addons/godot-stat-math/core/pmf_pdf_functions.gd")
const PpfFunctions = preload("res://addons/godot-stat-math/core/ppf_functions.gd")
const ErrorFunctions = preload("res://addons/godot-stat-math/core/error_functions.gd")
const HelperFunctions = preload("res://addons/godot-stat-math/core/helper_functions.gd")

func _ready():
	print("StatMath addon loaded and ready. Access functions via StatMath.ModuleName.function_name() and constants via StatMath.CONSTANT_NAME.")

# No wrapper functions are needed here.
# All functions are accessed directly through the preloaded modules, e.g.:
# var x = StatMath.Distributions.randf_normal(0.0, 1.0)
# var p = StatMath.CdfFunctions.normal_cdf(x, 0.0, 1.0)
# var k_val = StatMath.HelperFunctions.binomial_coefficient(10, 3)
# var eps = StatMath.EPSILON
