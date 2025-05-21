extends Node

# --- Configuration for Random Number Generation ---
const MONTE_GODOT_SEED_VARIABLE_NAME: StringName = &"monte_godot_seed"
const _default_seed: int = 0 # Default seed if no global override is found
var _rng: RandomNumberGenerator = null

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
const SamplingGen = preload("res://addons/godot-stat-math/core/sampling_gen.gd")

func _ready() -> void:
	_initialize_rng()
	print("StatMath addon loaded and ready. RNG initialized. Access functions via StatMath.ModuleName.function_name() and constants via StatMath.CONSTANT_NAME.")

# No wrapper functions are needed here.
# All functions are accessed directly through the preloaded modules, e.g.:
# var x = StatMath.Distributions.randf_normal(0.0, 1.0)
# var p = StatMath.CdfFunctions.normal_cdf(x, 0.0, 1.0)
# var k_val = StatMath.HelperFunctions.binomial_coefficient(10, 3)
# var eps = StatMath.EPSILON

# --- RNG Management ---

# Helper to create and seed the RNG instance.
func _create_and_seed_rng(seed_val: int) -> void:
	_rng = RandomNumberGenerator.new()
	_rng.seed = seed_val # If seed_val is 0, Godot's RNG will pick a random seed.
	# The actual seed used (randomized if input was 0) can be read from _rng.seed after this.

func _initialize_rng() -> void:
	var global_seed_value: Variant = ProjectSettings.get_setting(MONTE_GODOT_SEED_VARIABLE_NAME, _default_seed)
	var seed_to_use: int
	
	if global_seed_value is int:
		print("StatMath: Found global seed 'monte_godot_seed' with value: %d. Using it." % global_seed_value)
		seed_to_use = global_seed_value
	else:
		# If the global var exists but is not an int, or doesn't exist (get_setting returns default)
		if ProjectSettings.has_setting(MONTE_GODOT_SEED_VARIABLE_NAME):
			printerr("StatMath: Global variable 'monte_godot_seed' is set but not an integer. Using default seed: %d." % _default_seed)
		else:
			print("StatMath: No global seed 'monte_godot_seed' found or it's not an integer. Using default seed (0 means random)." % str(_default_seed))
		seed_to_use = _default_seed
		
	_create_and_seed_rng(seed_to_use)
	print("StatMath: Initial RNG created and seeded. Effective seed: %d." % _rng.seed)
		
	# Ensure the project setting is actually created if it was defaulted, so user knows it's available.
	if not ProjectSettings.has_setting(MONTE_GODOT_SEED_VARIABLE_NAME):
		ProjectSettings.set_setting(MONTE_GODOT_SEED_VARIABLE_NAME, _default_seed)
		# ProjectSettings.save() # Not strictly necessary for it to be readable by get_setting in same session, but good for persistence if user wants to see it in project.godot

# Returns the addon's RandomNumberGenerator instance.
func get_rng() -> RandomNumberGenerator:
	# _initialize_rng should have been called in _ready, so _rng should not be null.
	# However, as a safeguard if StatMath is used before _ready (e.g. tool script or early access):
	if _rng == null:
		push_warning("StatMath.get_rng() called before _ready or RNG failed to initialize. Initializing RNG now.")
		_initialize_rng()
	return _rng

# Allows changing the seed of the addon's RandomNumberGenerator instance.
# This will create a new RNG instance.
func set_seed(new_seed: int) -> void:
	_create_and_seed_rng(new_seed)
	print("StatMath: RNG (re)created and seed explicitly set. Effective seed: %d" % _rng.seed)
