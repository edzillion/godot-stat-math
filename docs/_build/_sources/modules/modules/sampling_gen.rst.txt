StatMath.SamplingGen
====================

Advanced Sampling and Quasi-Random Number Generation
This class provides sophisticated sampling methods including quasi-random sequences,
Latin hypercube sampling, and coordinated shuffling for statistical simulations.

Designed for high-performance multi-dimensional sampling with threading support.
Features:

* Quasi-random sequences (Sobol, Halton) for low-discrepancy sampling

* Latin Hypercube sampling for space-filling designs

* Coordinated shuffling with statistical guarantees

* Memory pooling and threading for performance optimization

* Multiple selection strategies (with/without replacement)
Defines the available sampling methods for generating random sequences.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.SamplingGen.function_name(parameters)

Constants
---------

.. data:: _SOBOL_DATA

   Value: ``preload("res://addons/godot-stat-math/tables/sobol_data.gd")``

.. data:: _PRIME_DATA

   Value: ``preload("res://addons/godot-stat-math/tables/prime_numbers_data.gd")``

.. data:: _SOBOL_BITS

   Value: ``int = 30``

.. data:: _SOBOL_MAX_VAL_FLOAT

   Value: ``float = float(1 << _SOBOL_BITS)``

.. data:: MAX_POOLED_DECKS_PER_SIZE

   Value: ``int = 16``

Functions
---------

.. function:: _init() -> void:

.. function:: _get_pooled_deck(deck_size: int) -> Array[int]:

   Gets a deck array from the pool or creates a new one

.. function:: _return_pooled_deck(deck: Array[int], deck_size: int) -> void:

   Returns a deck array to the pool for reuse

.. function:: _ensure_sobol_vectors_initialized(max_dimension: int) -> void:

   Ensures Sobol direction vectors are initialized up to the specified dimension.
   This method is idempotent and safe to call multiple times.
   SINGLE-THREADED: Should only be called from main thread before spawning workers.

.. function:: _generate_direction_vectors_for_dimension(dimension: int) -> void:

   Generates direction vectors for a specific dimension using authoritative Joe-Kuo direction numbers.

.. function:: _init(dim: int, draws: int, start_idx: int, sampling_method: SamplingMethod):

.. function:: _generate_dimension_samples_worker(task: SobolDimensionTask) -> void:

   Thread worker function for generating a single dimension's samples

.. function:: _generate_samples_nd(

   Threaded version of generate_samples_nd for high-dimensional cases

.. function:: _init(size: int, n_shuffles: int, samples: Array):

.. function:: _coordinated_batch_shuffles_threaded(

   Optimized two-phase batch shuffle generation
   Phase 1: Multi-threaded bulk sample generation
   Phase 2: Multi-threaded shuffling with pre-generated samples

.. function:: _batch_shuffle_worker(task: BatchShuffleTask) -> void:

   Simplified worker function that only does shuffling with pre-generated samples
   No sample generation needed - samples are pre-generated in Phase 1

.. function:: _create_unshuffled_deck(deck_size: int) -> Array[int]:

   Helper function to create an unshuffled deck for error cases

.. function:: _coordinated_shuffle_with_samples(deck_size: int, sobol_point: Array) -> Array[int]:

   Optimized shuffle using pre-generated samples - avoids redundant sample generation

.. function:: _coordinated_shuffle_worker(task: BatchShuffleTask) -> void:

   Worker function for individual shuffle generation

.. function:: generate_samples(

   Unified interface for generating samples in 1, 2, or N dimensions.

   Returns different types based on dimensions: ``Array[float]`` for 1D,
   ``Array[Vector2]`` for 2D, ``Array[Array[float]]`` for N-D.
   Supports all sampling methods including quasi-random sequences.

   Mathematical Note: For quasi-random methods, low-discrepancy sequences provide better coverage than pseudo-random

.. function:: generate_samples_nd(

   Generates N-dimensional samples using the specified method.

   Returns an array of samples where each sample is an array of ``dimensions`` values.
   Uses threading for dimensions ≥ 3 for optimal performance. Supports all sampling methods.

   Mathematical Note: Quasi-random sequences maintain uniformity across all dimensions simultaneously

.. function:: coordinated_shuffle(

   Performs a complete coordinated shuffle using multi-dimensional sampling.

   Uses a single multi-dimensional point to drive the Fisher-Yates shuffle algorithm,
   ensuring statistical guarantees across the entire shuffle operation.
   This is the core method for coordinated shuffling.

   Mathematical Note: Uses ``(deck_size-1)`` dimensional point for Fisher-Yates coordination

.. function:: coordinated_batch_shuffles(

   Generates multiple coordinated shuffles efficiently.

   Creates multiple shuffles using sequential points from the specified sampling sequence.
   Uses threading for ``n_shuffles ≥ 2`` to maximize performance with batch operations.

   Mathematical Note: Each shuffle uses consecutive points from the quasi-random sequence for coordination

.. function:: sample_indices(

   Samples indices from a finite population using advanced selection strategies.

   Combines sampling methods (how to generate random numbers) with selection strategies
   (how to use those numbers for population sampling). Supports both replacement and
   non-replacement sampling with various optimization strategies.

   Mathematical Note: Selection strategies optimize for different use cases - bootstrap (with replacement), surveys (without replacement)

.. function:: _with_replacement_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Sampling with replacement - allows duplicates.

.. function:: _fisher_yates_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Fisher-Yates shuffle with custom sampling method for randomness.

.. function:: _reservoir_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Reservoir sampling with custom sampling method.

.. function:: _selection_tracking_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Selection tracking with custom sampling method.

.. function:: _get_nth_prime(n: int) -> int:

   Returns the nth prime number using the PrimeNumbersData table.
   Used for Halton sequence base selection.

.. function:: _get_sobol_1d_integers(ndraws: int, dimension_index: int, starting_index: int = 0) -> Array[int]:

   Generates Sobol sequence integers for a specific dimension.
   ASSUMES: Direction vectors for dimension_index are already initialized.

.. function:: _generate_sobol_1d(ndraws: int, dimension_index: int, starting_index: int = 0) -> Array[float]:

   Generates 1D Sobol samples for a specific dimension.

.. function:: _generate_halton_1d(ndraws: int, base: int, starting_index: int = 0) -> Array[float]:

.. function:: _generate_latin_hypercube_1d(ndraws: int, rng: RandomNumberGenerator) -> Array[float]:

.. function:: _fast_random_batch_shuffles(deck_size: int, n_shuffles: int, sample_seed: int) -> Array:

.. function:: _fast_random_shuffle(deck_size: int, sample_seed: int) -> Array[int]:

