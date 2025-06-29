StatMath
========

StatMath - Comprehensive Statistical Functions Library
This is the main autoload singleton that provides access to all statistical functions
and mathematical utilities.

All core functionality is organized into specialized modules
that can be accessed through this central interface.
Usage Examples:
[codeblock]
# Generate random numbers from distributions
var normal_sample = StatMath.Distributions.randf_normal(0.0, 1.0)
var poisson_sample = StatMath.Distributions.randi_poisson(3.5)
# Calculate statistical measures
var data = [1.0, 2.0, 3.0, 4.0, 5.0]
var mean = StatMath.BasicStats.mean(data)
var variance = StatMath.BasicStats.variance(data)

Usage
-----

.. code-block:: gdscript

   # StatMath is the main singleton - access modules through it
   var result = StatMath.ModuleName.function_name(parameters)
   
   # Or access constants directly
   var epsilon = StatMath.EPSILON

Constants
---------

.. data:: GODOT_STAT_MATH_SEED_VARIABLE_NAME

   Value: ``StringName = &"godot_stat_math_seed"``

.. data:: _default_seed

   Value: ``int = 0``

.. data:: PRIMES

   Value: ``Array[int] = [``

.. data:: INT_MAX_REPRESENTING_INF

   Value: ``= 2147483647``

.. data:: INT64_MAX_VAL

   Value: ``int = 9223372036854775807``

.. data:: FLOAT_EPSILON

   Value: ``float = 2.220446049250313e-16``

.. data:: MAX_ITERATIONS

   Value: ``int = 200``

.. data:: EPSILON

   Value: ``float = 1.0e-9``

.. data:: FLOAT_TOLERANCE

   Value: ``float = 1.0e-7``

.. data:: HIGH_PRECISION_TOLERANCE

   Value: ``float = 1.0e-9``

.. data:: ERF_APPROX_TOLERANCE

   Value: ``float = 1.0e-5``

.. data:: NUMERICAL_DIFFERENTIATION_H

   Value: ``float = 1.0e-6``

.. data:: SAMPLE_MEAN_TOLERANCE

   Value: ``float = 1.0e-6``

.. data:: SAMPLING_DETERMINISM_TOLERANCE

   Value: ``float = 1.0e-7``

.. data:: INVERSE_FUNCTION_TOLERANCE

   Value: ``float = 2.0e-6``

.. data:: NUMERICAL_TOLERANCE

   Value: ``float = 1.0e-5``

.. data:: CDF_PPF_CONSISTENCY_TOLERANCE

   Value: ``float = 1.0e-5``

.. data:: DERIVATIVE_TOLERANCE

   Value: ``float = 1.0e-3``

.. data:: PROBABILITY_TOLERANCE

   Value: ``float = 1.0e-6``

.. data:: SPECIAL_VALUES_TOLERANCE

   Value: ``float = 2.0e-6``

.. data:: BOUNDARY_TOLERANCE

   Value: ``float = 1.0e-10``

.. data:: ASYMPTOTIC_TOLERANCE

   Value: ``float = 1.0e-2``

.. data:: INTERPOLATION_TOLERANCE

   Value: ``float = 1.0e-4``

.. data:: NUMERICAL_INTEGRATION_TOLERANCE

   Value: ``float = 5.0e-3``

.. data:: SYMMETRY_TOLERANCE

   Value: ``float = 1.0e-4``

.. data:: ERF_INV_TOLERANCE

   Value: ``float = 2.0e-2``

.. data:: STATISTICAL_TEST_STD_DEV_MULTIPLIER

   Value: ``float = 4.0``

.. data:: DEFAULT_TOLERANCE_FACTOR

   Value: ``float = 1.0``

.. data:: SAMPLING_TOLERANCE

   Value: ``float = 1.0e-6``

.. data:: INVERSE_CONSISTENCY_TOLERANCE

   Value: ``float = 1.0e-5``

.. data:: STABILITY_TOLERANCE

   Value: ``float = 1.0e-6``

.. data:: STRESS_TEST_BOUNDARY

   Value: ``float = 1.0e-17``

.. data:: STRESS_TEST_SMALL_VALUE

   Value: ``float = 1.0e-3``

.. data:: INTERFACE_TOLERANCE

   Value: ``float = 1.0e-7``

.. data:: DETERMINISM_TOLERANCE

   Value: ``float = 1.0e-7``

.. data:: HYPERGEOMETRIC_TOLERANCE

   Value: ``float = 0.15``

.. data:: NEGATIVE_BINOMIAL_TOLERANCE

   Value: ``float = 0.2``

.. data:: HIGH_DISTRIBUTION_TOLERANCE

   Value: ``float = 0.5``

.. data:: BETA_TOLERANCE

   Value: ``float = 0.1``

.. data:: LANCZOS_G

   Value: ``float = 7.5``

.. data:: LANCZOS_P

   Value: ``Array[float] = [``

.. data:: A1_ERR

   Value: ``float =  0.254829592``

.. data:: A2_ERR

   Value: ``float = -0.284496736``

.. data:: A3_ERR

   Value: ``float =  1.421413741``

.. data:: A4_ERR

   Value: ``float = -1.453152027``

.. data:: A5_ERR

   Value: ``float =  1.061405429``

.. data:: P_ERR

   Value: ``float  =  0.3275911``

.. data:: BasicStats

   Value: ``preload("res://addons/godot-stat-math/core/basic_stats.gd")``

.. data:: Distributions

   Value: ``preload("res://addons/godot-stat-math/core/distributions.gd")``

.. data:: CdfFunctions

   Value: ``preload("res://addons/godot-stat-math/core/cdf_functions.gd")``

.. data:: PmfPdfFunctions

   Value: ``preload("res://addons/godot-stat-math/core/pmf_pdf_functions.gd")``

.. data:: PpfFunctions

   Value: ``preload("res://addons/godot-stat-math/core/ppf_functions.gd")``

.. data:: ErrorFunctions

   Value: ``preload("res://addons/godot-stat-math/core/error_functions.gd")``

.. data:: HelperFunctions

   Value: ``preload("res://addons/godot-stat-math/core/helper_functions.gd")``

.. data:: SamplingGen

   Value: ``preload("res://addons/godot-stat-math/core/sampling_gen.gd")``

Functions
---------

.. function:: _ready() -> void:

   Initializes the StatMath addon and sets up the random number generator.

   Automatically called when the addon is loaded as an autoload singleton.
   Reads the global seed configuration and initializes the RNG system.

.. function:: _create_and_seed_rng(seed_val: int) -> void:

   Creates and seeds the internal RandomNumberGenerator instance.

   If ``seed_val`` is 0, Godot will automatically choose a random seed.
   The actual seed used can be read from ``_rng.seed`` after initialization.

.. function:: _initialize_rng() -> void:

   Initializes the RNG system using global project settings.

   Checks for the project setting ``godot_stat_math_seed`` to determine
   the seed value. Creates the setting with default value if it doesn't exist.

.. function:: get_rng() -> RandomNumberGenerator:

   Returns the addon's RandomNumberGenerator instance.

   This RNG is used by all statistical functions that require randomness.
   Provides consistent seeding and reproducible results across the addon.

   Mathematical Note: All statistical sampling uses this centralized RNG for reproducibility

.. function:: set_global_seed(new_seed: int) -> void:

   Changes the global seed and recreates the RandomNumberGenerator.

   This affects all subsequent random number generation throughout the addon.
   Useful for creating reproducible statistical simulations and tests.

   Mathematical Note: Changing seed allows reproducible statistical experiments

