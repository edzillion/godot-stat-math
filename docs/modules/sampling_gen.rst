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

   Retrieves a deck array from the memory pool or creates a new one.

   Provides significant performance improvements for shuffle operations by reusing
   pre-allocated arrays. Thread-safe implementation with mutex protection.
   Automatically resets deck to default state [0, 1, 2, ..., deck_size-1].

.. function:: _return_pooled_deck(deck: Array[int], deck_size: int) -> void:

   Returns a deck array to the memory pool for future reuse.

   Enables efficient memory recycling for shuffle operations. Only stores a limited
   number of decks per size to prevent unlimited memory growth. Thread-safe with
   mutex protection for concurrent access.

.. function:: _ensure_sobol_vectors_initialized(max_dimension: int) -> void:

   Ensures Sobol direction vectors are initialized up to the specified dimension.

   This method is idempotent and safe to call multiple times. Direction vectors are
   cached for performance, using authoritative Joe-Kuo data for optimal low-discrepancy
   properties. Should only be called from main thread before spawning workers.

   Mathematical Note: Uses Joe-Kuo direction numbers for superior uniformity compared to primitive polynomials

.. function:: _generate_direction_vectors_for_dimension(dimension: int) -> void:

   Generates direction vectors for a specific dimension using authoritative Joe-Kuo direction numbers.

   Converts Joe-Kuo direction numbers (m_i values) to direction vectors using the relationship
   V_i = m_i / 2^(i+1). For dimensions without Joe-Kuo data, uses fallback patterns
   to ensure all dimensions remain functional.

   Mathematical Note: Direction vectors V_i determine the binary digit distribution in Sobol sequences

.. function:: _init(dim: int, draws: int, start_idx: int, sampling_method: SamplingMethod):

.. function:: _generate_dimension_samples_worker(task: SobolDimensionTask) -> void:

   Thread worker function for generating a single dimension's samples.

   Processes different sampling methods in parallel threads for optimal performance.
   Each thread operates independently on its assigned dimension, with deterministic
   seeding for reproducible results. Includes fallback mechanisms for sequence limits.

   Mathematical Note: Each dimension uses independent sequences to maintain low-discrepancy properties

.. function:: _generate_samples_nd(

   Threaded version of generate_samples_nd for high-dimensional cases.

   Optimizes multi-dimensional sample generation by parallelizing across dimensions.
   Pre-initializes all required Sobol direction vectors before spawning threads to
   ensure true parallelization. Uses worker thread pool for scalable performance.

   Mathematical Note: Maintains uniformity guarantees across all dimensions simultaneously

.. function:: _init(size: int, n_shuffles: int, samples: Array):

.. function:: _coordinated_batch_shuffles_threaded(

   Optimized two-phase batch shuffle generation.

   Phase 1: Multi-threaded bulk sample generation for all shuffles using unified N-dimensional sampling.
   Phase 2: Multi-threaded shuffling with pre-generated samples to avoid redundant computations.
   This approach maximizes parallelization while maintaining coordinated shuffle properties.

   Mathematical Note: Preserves Fisher-Yates statistical guarantees across all generated shuffles

.. function:: _batch_shuffle_worker(task: BatchShuffleTask) -> void:

   Simplified worker function that performs shuffling with pre-generated samples.

   No sample generation needed as samples are pre-generated in Phase 1 of the
   batch shuffle process. Each worker processes a chunk of shuffles using
   their assigned pre-generated multi-dimensional points.

.. function:: _create_unshuffled_deck(deck_size: int) -> Array[int]:

   Helper function to create an unshuffled deck for error cases.

   Returns a deck in default order [0, 1, 2, ..., deck_size-1] when shuffle
   operations encounter errors. Uses memory pooling for performance consistency.

.. function:: _coordinated_shuffle_with_samples(deck_size: int, sobol_point: Array) -> Array[int]:

   Optimized shuffle using pre-generated samples.

   Performs coordinated Fisher-Yates shuffle using a pre-generated N-dimensional point
   to avoid redundant sample generation. Each dimension of the point drives one step
   of the Fisher-Yates algorithm for perfect coordination.

   Mathematical Note: Uses (deck_size-1) dimensional point for Fisher-Yates coordination

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

   Sampling with replacement allowing duplicates.

   Generates random indices where the same index can appear multiple times.
   Essential for bootstrap sampling and simulation techniques. Uses the specified
   sampling method to ensure proper statistical properties.

   Mathematical Note: Each draw is independent with uniform probability 1/population_size

.. function:: _fisher_yates_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Fisher-Yates shuffle with custom sampling method for randomness.

   Performs partial Fisher-Yates shuffle to select exactly ``draw_count`` items
   without replacement. More efficient than full shuffle when drawing small samples
   from large populations.

   Mathematical Note: Maintains uniform selection probability for each remaining item at each step

.. function:: _reservoir_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Reservoir sampling with custom sampling method.

   Implements Vitter's reservoir algorithm for memory-efficient sampling from large
   populations. Maintains exactly ``draw_count`` items with uniform selection
   probability without requiring full population in memory.

   Mathematical Note: Each item has probability draw_count/population_size of selection

.. function:: _selection_tracking_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:

   Selection tracking with custom sampling method.

   Uses hash table to track selected indices, avoiding duplicates through rejection sampling.
   More memory-efficient than Fisher-Yates for small samples but can be slower due to
   potential rejection cycles.

   Mathematical Note: Expected attempts = draw_count × population_size / (population_size - selected_count + 1)

.. function:: _get_nth_prime(n: int) -> int:

   Returns the nth prime number using the PrimeNumbersData table.

   Used for Halton sequence base selection to ensure different dimensions use
   coprime bases, which is essential for maintaining low-discrepancy properties
   across multiple dimensions.

   Mathematical Note: Coprime bases ensure dimensional independence in Halton sequences

.. function:: _get_sobol_1d_integers(ndraws: int, dimension_index: int, starting_index: int = 0) -> Array[int]:

   Generates Sobol sequence integers for a specific dimension.

   Produces the raw integer representation of Sobol sequence points before normalization.
   Direction vectors for the specified dimension must be pre-initialized. Returns -1
   to signal errors when sequence limits are exceeded.

   Mathematical Note: Uses XOR operations with direction vectors to generate low-discrepancy points

.. function:: _generate_sobol_1d(ndraws: int, dimension_index: int, starting_index: int = 0) -> Array[float]:

   Generates 1D Sobol samples for a specific dimension.

   Converts Sobol integers to floating-point values in [0,1) range. Uses authoritative
   Joe-Kuo direction numbers for optimal uniformity. Returns -1.0 to signal errors
   when sequence generation fails.

   Mathematical Note: Normalization factor is 2^30 for 30-bit Sobol precision

.. function:: _generate_halton_1d(ndraws: int, base: int, starting_index: int = 0) -> Array[float]:

   Generates 1D Halton sequence samples using the specified base.

   Implements the radical inverse function to create Halton sequences. Each dimension
   should use a different prime base to maintain low-discrepancy properties across
   multiple dimensions.

   Mathematical Note: Radical inverse in base b: Φ_b(n) = Σ(a_i × b^(-i-1)) where n = Σ(a_i × b^i)

.. function:: _generate_latin_hypercube_1d(ndraws: int, rng: RandomNumberGenerator) -> Array[float]:

   Generates 1D Latin Hypercube samples with Fisher-Yates shuffling.

   Creates stratified samples by dividing [0,1) into equal intervals and sampling
   once from each interval. Fisher-Yates shuffle ensures uniform permutation
   for optimal space-filling properties.

   Mathematical Note: Each interval [i/n, (i+1)/n) contains exactly one sample point

.. function:: _fast_random_batch_shuffles(deck_size: int, n_shuffles: int, sample_seed: int) -> Array:


   Optimized path for random shuffles that bypasses complex multi-dimensional
   sampling infrastructure. Uses simple Fisher-Yates with standard RNG for
   maximum performance when coordination is not required.

.. function:: _fast_random_shuffle(deck_size: int, sample_seed: int) -> Array[int]:

   Fast single shuffle generation for RANDOM method.

   Simple Fisher-Yates shuffle implementation for cases where quasi-random
   coordination is not needed. Provides optimal performance for standard
   random shuffling operations.

   Mathematical Note: Standard Fisher-Yates ensures each permutation has probability 1/n!

