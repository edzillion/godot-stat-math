# res://addons/godot-stat-math/stat_math.gd
extends Node

## StatMath - Comprehensive Statistical Functions Library
##
## This is the main autoload singleton that provides access to all statistical functions
## and mathematical utilities. All core functionality is organized into specialized modules
## that can be accessed through this central interface.
##
## Usage Examples:
## [codeblock]
## # Generate random numbers from distributions
## var normal_sample = StatMath.Distributions.randf_normal(0.0, 1.0)
## var poisson_sample = StatMath.Distributions.randi_poisson(3.5)
## 
## # Calculate statistical measures
## var data = [1.0, 2.0, 3.0, 4.0, 5.0]
## var mean = StatMath.BasicStats.mean(data)
## var variance = StatMath.BasicStats.variance(data)
## 
## # Use distribution functions
## var cdf_value = StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)
## var quantile = StatMath.PpfFunctions.normal_ppf(0.975, 0.0, 1.0)
## 
## # Advanced sampling
## var samples = StatMath.SamplingGen.generate_samples(100, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
## [/codeblock]
##
## Module Organization:
## • [code]BasicStats[/code] - Descriptive statistics (mean, variance, quantiles, etc.)
## • [code]Distributions[/code] - Random number generation from various distributions
## • [code]CdfFunctions[/code] - Cumulative distribution functions
## • [code]PpfFunctions[/code] - Inverse CDF/quantile functions
## • [code]PmfPdfFunctions[/code] - Probability mass/density functions
## • [code]ErrorFunctions[/code] - Error functions and special mathematical functions
## • [code]HelperFunctions[/code] - Core mathematical utilities
## • [code]SamplingGen[/code] - Advanced sampling and quasi-random sequences


# =============================================================================
# CONFIGURATION AND RANDOM NUMBER GENERATION
# =============================================================================

## Project setting name for global random seed configuration.
const GODOT_STAT_MATH_SEED_VARIABLE_NAME: StringName = &"godot_stat_math_seed"

## Default seed value when no global seed is specified (0 means random).
const _default_seed: int = 0

## Internal RandomNumberGenerator instance for statistical functions.
var _rng: RandomNumberGenerator = null


# =============================================================================
# MATHEMATICAL AND NUMERICAL CONSTANTS
# =============================================================================

## First 100 prime numbers for use in statistical algorithms.
##
## Mathematical Note: Used primarily in Halton sequence generation and other quasi-random methods.
const PRIMES: Array[int] = [
	2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
	73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
	157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
	239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
	331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
	421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
	509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601
]

## Represents a very large integer for functions where infinity is theoretical.
##
## Used by discrete distribution functions when theoretical result would be infinite.
## Mathematical Note: [code]2^31 - 1[/code] (maximum 32-bit signed integer)
const INT_MAX_REPRESENTING_INF := 2147483647

## Maximum value for a 64-bit signed integer.
##
## Mathematical Note: [code]2^63 - 1 = 9,223,372,036,854,775,807[/code]
const INT64_MAX_VAL: int = 9223372036854775807

## Machine epsilon - smallest positive float x such that [code]1.0 + x ≠ 1.0[/code].
##
## Mathematical Note: Double precision floating point machine epsilon [code]≈ 2.22 × 10^-16[/code]
const FLOAT_EPSILON: float = 2.220446049250313e-16

## Maximum iterations for iterative approximations and series calculations.
##
## Used by convergence algorithms in special functions and numerical methods.
const MAX_ITERATIONS: int = 200

## Small epsilon for convergence checks and floating point comparisons.
##
## Mathematical Note: [code]10^-9[/code] provides good balance between precision and numerical stability
const EPSILON: float = 1.0e-9

## Standard floating point tolerance for test assertions and approximate comparisons.
##
## Mathematical Note: [code]10^-7[/code] provides robust floating point comparison for statistical calculations
const FLOAT_TOLERANCE: float = 1.0e-7

## High precision tolerance for algorithms requiring extra precision.
##
## Mathematical Note: [code]10^-9[/code] used for high-precision statistical calculations
const HIGH_PRECISION_TOLERANCE: float = 1.0e-9

## Error function approximation tolerance for iterative algorithms.
##
## Mathematical Note: [code]10^-5[/code] accounts for precision limitations in error function approximations
const ERF_APPROX_TOLERANCE: float = 1.0e-5

## Numerical differentiation step size for derivative approximations.
##
## Mathematical Note: [code]10^-6[/code] provides optimal balance between accuracy and numerical stability
const NUMERICAL_DIFFERENTIATION_H: float = 1.0e-6

## Tolerance for statistical sample mean calculations.
##
## Mathematical Note: [code]10^-6[/code] appropriate for probability density integrations
const SAMPLE_MEAN_TOLERANCE: float = 1.0e-6

## Deterministic sampling tolerance for quasi-random sequences.
##
## Mathematical Note: [code]10^-7[/code] ensures reproducibility in deterministic sampling methods
const SAMPLING_DETERMINISM_TOLERANCE: float = 1.0e-7

## Inverse function (PPF) calculation tolerance for quantile computations.
##
## Mathematical Note: [code]2×10^-6[/code] appropriate for Newton-Raphson and binary search PPF algorithms  
const INVERSE_FUNCTION_TOLERANCE: float = 2.0e-6

## Numerical algorithm tolerance for iterative approximations.
##
## Mathematical Note: [code]10^-5[/code] balances accuracy with convergence speed
const NUMERICAL_TOLERANCE: float = 1.0e-5

## CDF-PPF consistency tolerance for round-trip validation.
##
## Mathematical Note: `10^-5` ensures inverse function consistency
const CDF_PPF_CONSISTENCY_TOLERANCE: float = 1.0e-5

## Derivative tolerance for CDF-PDF relationship validation.
##
## Mathematical Note: `10^-3` accounts for numerical differentiation errors
const DERIVATIVE_TOLERANCE: float = 1.0e-3

## Probability calculation tolerance for statistical tests.
##
## Mathematical Note: [code]10^-6[/code] suitable for probability mass/density calculations
const PROBABILITY_TOLERANCE: float = 1.0e-6

## Special mathematical values tolerance (ln(2), π, z-scores, etc.).
##
## Mathematical Note: [code]2×10^-6[/code] appropriate for well-known mathematical constants and statistical tables
const SPECIAL_VALUES_TOLERANCE: float = 2.0e-6

## Boundary condition tolerance for extreme probability values.
##
## Mathematical Note: [code]10^-10[/code] for testing probability bounds (0, 1) and distribution limits
const BOUNDARY_TOLERANCE: float = 1.0e-10

## Tolerance for asymptotic approximations and convergence tests.
##
## Mathematical Note: [code]10^-2[/code] for large-parameter approximations (e.g., t-distribution → normal)
const ASYMPTOTIC_TOLERANCE: float = 1.0e-2

## Interpolation tolerance for percentile calculations.
##
## Mathematical Note: [code]10^-4[/code] appropriate for linear interpolation in statistical functions
const INTERPOLATION_TOLERANCE: float = 1.0e-4



## Numerical integration tolerance for mathematical computations.
##
## Mathematical Note: [code]5×10^-3[/code] accounts for truncation errors in infinite-tail distributions
const NUMERICAL_INTEGRATION_TOLERANCE: float = 5.0e-3

## Symmetry validation tolerance for testing mathematical properties.
##
## Mathematical Note: [code]10^-4[/code] for symmetry tests around mathematical points
const SYMMETRY_TOLERANCE: float = 1.0e-4

## Error function inverse approximation tolerance.
##
## Mathematical Note: [code]2×10^-2[/code] accounts for Newton-Raphson convergence limitations
const ERF_INV_TOLERANCE: float = 2.0e-2

## Statistical test standard deviation multiplier for confidence intervals.
##
## Mathematical Note: [code]4.0[/code] represents ~4σ confidence level for statistical validation
const STATISTICAL_TEST_STD_DEV_MULTIPLIER: float = 4.0

## Default tolerance factor for adaptive tolerance calculations.
##
## Mathematical Note: [code]1.0[/code] baseline multiplier for dynamic tolerance adjustment
const DEFAULT_TOLERANCE_FACTOR: float = 1.0

## Sampling tolerance for statistical distributions and random number generation.
##
## Mathematical Note: [code]10^-6[/code] for sampling validation and distribution testing
const SAMPLING_TOLERANCE: float = 1.0e-6

## Inverse consistency tolerance for PPF-CDF round-trip validation.
##
## Mathematical Note: [code]10^-5[/code] ensures inverse function accuracy
const INVERSE_CONSISTENCY_TOLERANCE: float = 1.0e-5

## Stability tolerance for numerical algorithm convergence.
##
## Mathematical Note: [code]10^-6[/code] for algorithm stability testing
const STABILITY_TOLERANCE: float = 1.0e-6

## Stress test boundary value for extreme parameter testing.
##
## Mathematical Note: [code]10^-17[/code] for testing numerical stability at extreme scales
const STRESS_TEST_BOUNDARY: float = 1.0e-17

## Stress test small value for boundary condition testing.
##
## Mathematical Note: [code]10^-3[/code] for testing small parameter behavior
const STRESS_TEST_SMALL_VALUE: float = 1.0e-3

## Interface tolerance for API consistency testing.
##
## Mathematical Note: [code]10^-7[/code] for testing interface contracts and return value consistency
const INTERFACE_TOLERANCE: float = 1.0e-7

## Determinism tolerance for reproducible random number generation.
##
## Mathematical Note: [code]10^-7[/code] ensures exact reproducibility in deterministic contexts
const DETERMINISM_TOLERANCE: float = 1.0e-7

## Distribution-specific tolerance for hypergeometric distribution tests.
##
## Mathematical Note: [code]0.15[/code] accounts for discrete distribution sampling variance
const HYPERGEOMETRIC_TOLERANCE: float = 0.15

## Distribution-specific tolerance for negative binomial distribution tests.
##
## Mathematical Note: [code]0.2[/code] accounts for higher variance in negative binomial sampling
const NEGATIVE_BINOMIAL_TOLERANCE: float = 0.2

## Distribution-specific high tolerance for challenging distributions.
##
## Mathematical Note: [code]0.5[/code] for distributions with high variability or convergence challenges
const HIGH_DISTRIBUTION_TOLERANCE: float = 0.5

## Distribution-specific tolerance for beta distribution tests.
##
## Mathematical Note: [code]0.1[/code] for bounded distributions with moderate variance
const BETA_TOLERANCE: float = 0.1

## Lanczos approximation parameter for Gamma function calculations.
##
## Mathematical Note: [code]g = 7.5[/code] provides optimal accuracy for the Lanczos method
const LANCZOS_G: float = 7.5

## Lanczos coefficients for high-precision Gamma function approximation.
##
## Mathematical Note: Coefficients optimized for [code]g = 7.5[/code] providing ~15 decimal digits accuracy
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

## Abramowitz and Stegun approximation coefficients for error function.
##
## Mathematical Note: Coefficients for maximum error [code]< 1.5 × 10^-7[/code] in [code]erf(x)[/code] approximation
const A1_ERR: float =  0.254829592
const A2_ERR: float = -0.284496736
const A3_ERR: float =  1.421413741
const A4_ERR: float = -1.453152027
const A5_ERR: float =  1.061405429
const P_ERR: float  =  0.3275911


# =============================================================================
# DISTRIBUTION ENUMS
# =============================================================================

## Enumeration of all statistical distributions supported by the library.
##
## This enum provides type-safe distribution identification across all modules,
## replacing string literals for better maintainability and IDE support.
##
## Usage Examples:
## [codeblock]
## # Instead of strings:
## var sample = StatMath.Distributions.randf_from_distribution(StatMath.SupportedDistributions.NORMAL, [0.0, 1.0])
## 
## # Use enum values:
## var sample = StatMath.Distributions.randf_from_distribution(StatMath.SupportedDistributions.NORMAL, [0.0, 1.0])
## [/codeblock]
enum SupportedDistributions {
	## Standard normal distribution and variants with arbitrary mean and variance.
	## Parameters: [mean, standard_deviation]
	NORMAL,
	
	## Exponential distribution for modeling time between events.
	## Parameters: [rate] (lambda parameter)
	EXPONENTIAL,
	
	## Uniform distribution over a continuous interval.
	## Parameters: [min_value, max_value]
	UNIFORM,
	
	## Gamma distribution with shape and scale parameters.
	## Parameters: [shape, scale] (k and theta parameters)
	GAMMA,
	
	## Beta distribution bounded between 0 and 1.
	## Parameters: [alpha, beta] (shape parameters)
	BETA,
	
	## Chi-square distribution (special case of gamma).
	## Parameters: [degrees_of_freedom]
	CHI_SQUARE,
	
	## Student's t-distribution for small sample statistics.
	## Parameters: [degrees_of_freedom]
	T_DISTRIBUTION,
	
	## F-distribution for variance ratio testing.
	## Parameters: [degrees_of_freedom_1, degrees_of_freedom_2]
	F_DISTRIBUTION,
	
	## Weibull distribution for reliability and survival analysis.
	## Parameters: [scale, shape] (lambda and k parameters)
	WEIBULL,
	
	## Pareto distribution for power-law phenomena.
	## Parameters: [scale, shape] (minimum value and alpha parameters)
	PARETO,
	
	## Binomial distribution for fixed number of trials.
	## Parameters: [num_trials, success_probability]
	BINOMIAL,
	
	## Poisson distribution for counting rare events.
	## Parameters: [rate] (lambda parameter)
	POISSON,
	
	## Geometric distribution for number of trials until first success.
	## Parameters: [success_probability]
	GEOMETRIC,
	
	## Negative binomial distribution for number of failures before r successes.
	## Parameters: [num_successes, success_probability]
	NEGATIVE_BINOMIAL
}


# =============================================================================
# CORE FUNCTIONALITY MODULES
# =============================================================================

## Core statistical functions module - descriptive statistics and data analysis.
const BasicStats = preload("res://addons/godot-stat-math/core/basic_stats.gd")

## Random number generation module - samples from various statistical distributions.
const Distributions = preload("res://addons/godot-stat-math/core/distributions.gd")

## Cumulative distribution functions module - probability calculations.
const CdfFunctions = preload("res://addons/godot-stat-math/core/cdf_functions.gd")

## Probability mass and density functions module - discrete and continuous distributions.
const PmfPdfFunctions = preload("res://addons/godot-stat-math/core/pmf_pdf_functions.gd")

## Inverse CDF/quantile functions module - percentile calculations.
const PpfFunctions = preload("res://addons/godot-stat-math/core/ppf_functions.gd")

## Error functions and special mathematical functions module.
const ErrorFunctions = preload("res://addons/godot-stat-math/core/error_functions.gd")

## Mathematical helper utilities module - combinatorics, special functions, preprocessing.
const HelperFunctions = preload("res://addons/godot-stat-math/core/helper_functions.gd")

## Advanced sampling and quasi-random sequences module - Monte Carlo methods.
const SamplingGen = preload("res://addons/godot-stat-math/core/sampling_gen.gd")


# =============================================================================
# INITIALIZATION AND LIFECYCLE
# =============================================================================

## Initializes the StatMath addon and sets up the random number generator.
##
## Automatically called when the addon is loaded as an autoload singleton.
## Reads the global seed configuration and initializes the RNG system.
func _ready() -> void:
	_initialize_rng()
	print("StatMath addon loaded and ready. RNG initialized. Access functions via StatMath.ModuleName.function_name() and constants via StatMath.CONSTANT_NAME.")


# =============================================================================
# RANDOM NUMBER GENERATOR MANAGEMENT
# =============================================================================

## Creates and seeds the internal RandomNumberGenerator instance.
##
## If [code]seed_val[/code] is 0, Godot will automatically choose a random seed.
## The actual seed used can be read from [code]_rng.seed[/code] after initialization.
func _create_and_seed_rng(seed_val: int) -> void:
	_rng = RandomNumberGenerator.new()
	_rng.seed = seed_val
	# The actual seed used (randomized if input was 0) can be read from _rng.seed after this.


## Initializes the RNG system using global project settings.
##
## Checks for the project setting [code]godot_stat_math_seed[/code] to determine
## the seed value. Creates the setting with default value if it doesn't exist.
func _initialize_rng() -> void:
	var global_seed_value: Variant = ProjectSettings.get_setting(GODOT_STAT_MATH_SEED_VARIABLE_NAME, _default_seed)
	var seed_to_use: int
	
	if global_seed_value is int:
		print("StatMath: Found global seed 'godot_stat_math_seed' with value: %d. Using it." % global_seed_value)
		seed_to_use = global_seed_value
	else:
		# If the global var exists but is not an int, or doesn't exist (get_setting returns default)
		if ProjectSettings.has_setting(GODOT_STAT_MATH_SEED_VARIABLE_NAME):
			push_error("Global variable 'godot_stat_math_seed' is set but not an integer. Received type: %s. Using default seed: %d." % [typeof(global_seed_value), _default_seed])
		else:
			print("StatMath: No global seed 'godot_stat_math_seed' found or it's not an integer. Using default seed (0 means random)." % str(_default_seed))
		seed_to_use = _default_seed
		
	_create_and_seed_rng(seed_to_use)
	print("StatMath: Initial RNG created and seeded. Effective seed: %d." % _rng.seed)
		
	# Ensure the project setting is actually created if it was defaulted, so user knows it's available.
	if not ProjectSettings.has_setting(GODOT_STAT_MATH_SEED_VARIABLE_NAME):
		ProjectSettings.set_setting(GODOT_STAT_MATH_SEED_VARIABLE_NAME, _default_seed)


## Returns the addon's RandomNumberGenerator instance.
##
## This RNG is used by all statistical functions that require randomness.
## Provides consistent seeding and reproducible results across the addon.
##
## Mathematical Note: All statistical sampling uses this centralized RNG for reproducibility
func get_rng() -> RandomNumberGenerator:
	# _initialize_rng should have been called in _ready, so _rng should not be null.
	# However, as a safeguard if StatMath is used before _ready (e.g. tool script or early access):
	if _rng == null:
		push_warning("StatMath.get_rng() called before _ready or RNG failed to initialize. Initializing RNG now.")
		_initialize_rng()
	return _rng


## Changes the global seed and recreates the RandomNumberGenerator.
##
## This affects all subsequent random number generation throughout the addon.
## Useful for creating reproducible statistical simulations and tests.
##
## Mathematical Note: Changing seed allows reproducible statistical experiments
func set_global_seed(new_seed: int) -> void:
	_create_and_seed_rng(new_seed)
	print("StatMath: RNG (re)created and seed explicitly set. Effective seed: %d" % _rng.seed)
