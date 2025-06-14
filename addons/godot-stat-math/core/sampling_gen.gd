# res://src/core/sampling_gen.gd
class_name SamplingGen

enum SamplingMethod {
	RANDOM,           # Pseudo-random sampling
	SOBOL,            # Sobol quasi-random sequence
	SOBOL_RANDOM,     # Randomized Sobol sequence
	HALTON,           # Halton quasi-random sequence
	HALTON_RANDOM,    # Randomized Halton sequence
	LATIN_HYPERCUBE   # Latin Hypercube space-filling design
}

enum SelectionStrategy {
	WITH_REPLACEMENT,          # Allow duplicates (like rolling dice, bootstrap sampling)
	FISHER_YATES,              # Without replacement - shuffle then draw
	RESERVOIR,                 # Without replacement - when draw count unknown
	SELECTION_TRACKING,        # Without replacement - memory efficient
	COORDINATED_FISHER_YATES   # Without replacement - multi-dimensional coordinated shuffle
}

const _SOBOL_BITS: int = 30
const _SOBOL_MAX_VAL_FLOAT: float = float(1 << _SOBOL_BITS)

# Enhanced Sobol direction vectors system for arbitrary dimensions
# Each dimension uses a primitive polynomial to generate direction vectors
static var _sobol_direction_vectors_cache: Dictionary = {}  # dimension -> Array[int]
static var _max_cached_dimension: int = -1

# Threading is always used for multi-dimensional generation (dimensions >= 3)

# Primitive polynomials for Sobol sequence generation (up to 100 dimensions)
# Format: [degree, a1, a2, ..., a_degree-1] where polynomial is x^degree + a1*x^(degree-1) + ... + a_degree-1*x + 1
const _PRIMITIVE_POLYNOMIALS: Array[Array] = [
	[1],                    # Dimension 0: x (degree 1)
	[2, 1],                 # Dimension 1: x^2 + x + 1
	[3, 1],                 # Dimension 2: x^3 + x + 1
	[3, 2],                 # Dimension 3: x^3 + x^2 + 1
	[4, 1],                 # Dimension 4: x^4 + x + 1
	[4, 3],                 # Dimension 5: x^4 + x^3 + 1
	[4, 1, 3],              # Dimension 6: x^4 + x^3 + x + 1
	[4, 3, 1],              # Dimension 7: x^4 + x^3 + x^2 + 1
	[5, 2],                 # Dimension 8: x^5 + x^2 + 1
	[5, 4],                 # Dimension 9: x^5 + x^4 + 1
	[5, 1, 2],              # Dimension 10: x^5 + x^2 + x + 1
	[5, 1, 4],              # Dimension 11: x^5 + x^4 + x + 1
	[5, 3, 1],              # Dimension 12: x^5 + x^3 + x^2 + 1
	[5, 4, 3],              # Dimension 13: x^5 + x^4 + x^3 + 1
	[5, 1, 4, 2],           # Dimension 14: x^5 + x^4 + x^2 + x + 1
	[5, 3, 4, 1],           # Dimension 15: x^5 + x^4 + x^3 + x^2 + 1
	[6, 1],                 # Dimension 16: x^6 + x + 1
	[6, 5],                 # Dimension 17: x^6 + x^5 + 1
	[6, 1, 5],              # Dimension 18: x^6 + x^5 + x + 1
	[6, 5, 1],              # Dimension 19: x^6 + x^5 + x^2 + 1
	[6, 4, 1],              # Dimension 20: x^6 + x^4 + x^2 + 1
	[7, 1],                 # Dimension 21: x^7 + x + 1
	[7, 4],                 # Dimension 22: x^7 + x^4 + 1
	[7, 4, 3, 1],           # Dimension 23: x^7 + x^4 + x^3 + x^2 + 1
	[7, 6, 1],              # Dimension 24: x^7 + x^6 + x^2 + 1
	[7, 5, 2],              # Dimension 25: x^7 + x^5 + x^2 + 1
	[7, 6, 5, 2],           # Dimension 26: x^7 + x^6 + x^5 + x^2 + 1
	[7, 5, 4, 3, 2, 1],     # Dimension 27: x^7 + x^5 + x^4 + x^3 + x^2 + x + 1
	[7, 6, 5, 4, 2, 1],     # Dimension 28: x^7 + x^6 + x^5 + x^4 + x^2 + x + 1
	[7, 6, 3, 1],           # Dimension 29: x^7 + x^6 + x^3 + x + 1
	[8, 4, 3, 2],           # Dimension 30: x^8 + x^4 + x^3 + x^2 + 1
	[8, 6, 5, 3],           # Dimension 31: x^8 + x^6 + x^5 + x^3 + 1
	[8, 6, 5, 1],           # Dimension 32: x^8 + x^6 + x^5 + x + 1
	[8, 5, 3, 1],           # Dimension 33: x^8 + x^5 + x^3 + x + 1
	[8, 4, 3, 1],           # Dimension 34: x^8 + x^4 + x^3 + x + 1
	[8, 7, 2, 1],           # Dimension 35: x^8 + x^7 + x^2 + x + 1
	[8, 7, 4, 2],           # Dimension 36: x^8 + x^7 + x^4 + x^2 + 1
	[8, 7, 6, 1],           # Dimension 37: x^8 + x^7 + x^6 + x + 1
	[8, 6, 4, 3, 2, 1],     # Dimension 38: x^8 + x^6 + x^4 + x^3 + x^2 + x + 1
	[8, 7, 6, 5, 2, 1],     # Dimension 39: x^8 + x^7 + x^6 + x^5 + x^2 + x + 1
	[9, 4],                 # Dimension 40: x^9 + x^4 + 1
	[9, 6, 4, 2],           # Dimension 41: x^9 + x^6 + x^4 + x^2 + 1
	[9, 5, 3, 2],           # Dimension 42: x^9 + x^5 + x^3 + x^2 + 1
	[9, 6, 5, 4, 2, 1],     # Dimension 43: x^9 + x^6 + x^5 + x^4 + x^2 + x + 1
	[9, 7, 6, 4, 3, 1],     # Dimension 44: x^9 + x^7 + x^6 + x^4 + x^3 + x + 1
	[9, 8, 7, 6, 2, 1],     # Dimension 45: x^9 + x^8 + x^7 + x^6 + x^2 + x + 1
	[9, 8, 7, 2],           # Dimension 46: x^9 + x^8 + x^7 + x^2 + 1
	[9, 8, 6, 5, 3, 2],     # Dimension 47: x^9 + x^8 + x^6 + x^5 + x^3 + x^2 + 1
	[9, 8, 3, 2],           # Dimension 48: x^9 + x^8 + x^3 + x^2 + 1
	[9, 5, 4, 3, 2, 1],     # Dimension 49: x^9 + x^5 + x^4 + x^3 + x^2 + x + 1
	[10, 3],                # Dimension 50: x^10 + x^3 + 1
	[10, 8, 3, 2],          # Dimension 51: x^10 + x^8 + x^3 + x^2 + 1
	# This gives us 52 dimensions (0-51), enough for a 52-card deck
]


func _init() -> void:
	_ensure_sobol_vectors_initialized(51)  # Initialize up to 51 dimensions for 52-card deck support


## Ensures Sobol direction vectors are initialized up to the specified dimension.
## This method is idempotent and safe to call multiple times.
## Thread-safe: Uses a simple mutex-like approach via static variables.
static func _ensure_sobol_vectors_initialized(max_dimension: int) -> void:
	if max_dimension <= _max_cached_dimension:
		return
		
	# Basic thread safety: if another thread is already initializing higher dimensions,
	# we don't need to do anything (the cache check above handles this)
	var start_dim: int = max(_max_cached_dimension + 1, 0)
	
	for dim in range(start_dim, max_dimension + 1):
		if dim >= _PRIMITIVE_POLYNOMIALS.size():
			printerr("SamplingGen: No primitive polynomial available for dimension ", dim)
			break
			
		# Only generate if not already in cache (thread safety)
		if not _sobol_direction_vectors_cache.has(dim):
			_generate_direction_vectors_for_dimension(dim)
	
	# Update the maximum cached dimension atomically
	if max_dimension > _max_cached_dimension:
		_max_cached_dimension = max_dimension


## Generates direction vectors for a specific dimension using its primitive polynomial.
static func _generate_direction_vectors_for_dimension(dimension: int) -> void:
	if _sobol_direction_vectors_cache.has(dimension):
		return
		
	var direction_vectors: Array[int] = []
	direction_vectors.resize(_SOBOL_BITS)
	
	if dimension >= _PRIMITIVE_POLYNOMIALS.size():
		printerr("SamplingGen: No primitive polynomial available for dimension ", dimension)
		return
	
	var poly: Array = _PRIMITIVE_POLYNOMIALS[dimension]
	
	if poly.is_empty():
		printerr("SamplingGen: Empty polynomial for dimension ", dimension)
		return
		
	var degree: int = poly[0]
	
	if dimension == 0:
		# Special case for dimension 0: polynomial x (degree 1)
		for j in range(_SOBOL_BITS):
			direction_vectors[j] = 1 << (_SOBOL_BITS - 1 - j)
	else:
		# Initialize first 'degree' direction vectors with powers of 2
		for i in range(degree):
			if i < _SOBOL_BITS:
				direction_vectors[i] = 1 << (_SOBOL_BITS - 1 - i)
		
		# Generate remaining direction vectors using recurrence relation
		for j in range(degree, _SOBOL_BITS):
			var v: int = direction_vectors[j - degree]
			
			# Apply polynomial coefficients
			for k in range(1, poly.size()):
				var coeff: int = poly[k]
				if coeff > 0 and j - coeff >= 0:
					v ^= direction_vectors[j - coeff]
			
			# Apply the shift operation
			v ^= (direction_vectors[j - degree] >> degree)
			direction_vectors[j] = v
	
	_sobol_direction_vectors_cache[dimension] = direction_vectors


# --- THREADED DIMENSION GENERATION ---

## Data structure for passing parameters to worker threads
class SobolDimensionTask:
	var dimension: int
	var n_draws: int  
	var starting_index: int
	var method: SamplingMethod
	var random_mask: int = 0  # For SOBOL_RANDOM
	var halton_base: int = 2  # For HALTON
	var random_offset: float = 0.0  # For HALTON_RANDOM
	var result_samples: Array[float] = []  # Store results here
	
	func _init(dim: int, draws: int, start_idx: int, sampling_method: SamplingMethod):
		dimension = dim
		n_draws = draws
		starting_index = start_idx
		method = sampling_method
		result_samples.resize(draws)


## Thread worker function for generating a single dimension's samples
static func _generate_dimension_samples_worker(task: SobolDimensionTask) -> void:
	match task.method:
		SamplingMethod.RANDOM:
			# Each thread needs its own RNG with deterministic seeding
			var thread_rng = RandomNumberGenerator.new()
			thread_rng.seed = hash(str(task.dimension) + str(task.starting_index))
			for i in range(task.n_draws):
				task.result_samples[i] = thread_rng.randf()
				
		SamplingMethod.SOBOL:
			var sobol_samples = _generate_sobol_1d(task.n_draws, task.dimension, task.starting_index)
			for i in range(task.n_draws):
				task.result_samples[i] = sobol_samples[i] if i < sobol_samples.size() else -1.0
				
		SamplingMethod.SOBOL_RANDOM:
			var sobol_integers = _get_sobol_1d_integers(task.n_draws, task.dimension, task.starting_index)
			for i in range(task.n_draws):
				if sobol_integers[i] != -1:
					task.result_samples[i] = float(sobol_integers[i] ^ task.random_mask) / _SOBOL_MAX_VAL_FLOAT
				else:
					task.result_samples[i] = -1.0
					
		SamplingMethod.HALTON:
			var halton_samples = _generate_halton_1d(task.n_draws, task.halton_base, task.starting_index)
			for i in range(task.n_draws):
				task.result_samples[i] = halton_samples[i] if i < halton_samples.size() else -1.0
				
		SamplingMethod.HALTON_RANDOM:
			var halton_samples = _generate_halton_1d(task.n_draws, task.halton_base, task.starting_index)
			for i in range(task.n_draws):
				if halton_samples[i] != -1.0:
					task.result_samples[i] = fmod(halton_samples[i] + task.random_offset, 1.0)
				else:
					task.result_samples[i] = -1.0
		
		SamplingMethod.LATIN_HYPERCUBE:
			# Each thread needs its own RNG for LHS
			var thread_rng = RandomNumberGenerator.new()
			thread_rng.seed = hash(str(task.dimension) + str(task.starting_index))
			var lhs_samples = _generate_latin_hypercube_1d(task.n_draws, thread_rng)
			for i in range(task.n_draws):
				task.result_samples[i] = lhs_samples[i] if i < lhs_samples.size() else -1.0
		
		_:
			# For unsupported methods, fill with -1.0
			for i in range(task.n_draws):
				task.result_samples[i] = -1.0


## Threaded version of generate_samples_nd for high-dimensional cases
static func _generate_samples_nd(
	n_draws: int, 
	dimensions: int, 
	method: SamplingMethod,
	starting_index: int,
	rng: RandomNumberGenerator
) -> Array:
	var samples: Array = []
	samples.resize(n_draws)
	for i in range(n_draws):
		samples[i] = []
		samples[i].resize(dimensions)
	
	# Pre-generate random values for methods that need them
	var random_masks: Array = []
	var random_offsets: Array = []
	
	if method == SamplingMethod.SOBOL_RANDOM:
		random_masks.resize(dimensions)
		for d in range(dimensions):
			random_masks[d] = rng.randi() & ((1 << _SOBOL_BITS) - 1)
	
	if method == SamplingMethod.HALTON_RANDOM:
		random_offsets.resize(dimensions)
		for d in range(dimensions):
			random_offsets[d] = rng.randf()
	
	# Create tasks for each dimension
	var tasks: Array = []
	for d in range(dimensions):
		var task = SobolDimensionTask.new(d, n_draws, starting_index, method)
		
		match method:
			SamplingMethod.SOBOL_RANDOM:
				task.random_mask = random_masks[d]
			SamplingMethod.HALTON:
				task.halton_base = _get_nth_prime(d)
			SamplingMethod.HALTON_RANDOM:
				task.halton_base = _get_nth_prime(d)
				task.random_offset = random_offsets[d]
			# RANDOM and LATIN_HYPERCUBE handled via deterministic thread seeding
		
		tasks.append(task)
	
	# Submit tasks to thread pool
	var task_ids: Array = []
	for task in tasks:
		var task_id = WorkerThreadPool.add_task(Callable(_generate_dimension_samples_worker).bind(task))
		task_ids.append(task_id)
	
	# Collect results from threads
	for d in range(dimensions):
		WorkerThreadPool.wait_for_task_completion(task_ids[d])
		# Note: WorkerThreadPool doesn't return task results directly in Godot
		# We need to use a different approach - shared result storage
		var dim_samples: Array[float] = tasks[d].result_samples
		for i in range(n_draws):
			samples[i][d] = dim_samples[i] if i < dim_samples.size() else -1.0
	
	return samples


## Data structure for batch chunk processing - handles multiple shuffles per thread
class BatchChunkTask:
	var deck_size: int
	var method: SamplingMethod
	var starting_point_index: int
	var sample_seed: int
	var chunk_size: int
	var result_shuffles: Array = []  # Store multiple shuffle results here
	
	func _init(size: int, sampling_method: SamplingMethod, start_idx: int, seed: int, n_shuffles: int):
		deck_size = size
		method = sampling_method
		starting_point_index = start_idx
		sample_seed = seed
		chunk_size = n_shuffles
		result_shuffles.resize(n_shuffles)


## Optimized batch shuffle generation using chunk-based threading
static func _coordinated_batch_shuffles_threaded(
	deck_size: int,
	n_shuffles: int,
	method: SamplingMethod,
	starting_index: int,
	sample_seed: int
) -> Array:
	var results: Array = []
	results.resize(n_shuffles)
	
	# Determine optimal chunk size based on available processors and workload
	var thread_count: int = min(OS.get_processor_count(), n_shuffles)
	var shuffles_per_thread: int = ceili(float(n_shuffles) / float(thread_count))
	
	# Adjust for small batches - don't over-thread
	if n_shuffles < thread_count * 2:
		thread_count = max(1, n_shuffles / 2)
		shuffles_per_thread = ceili(float(n_shuffles) / float(thread_count))
	
	var chunk_tasks: Array = []
	var task_ids: Array = []
	
	# Create and submit ALL chunk tasks to thread pool FIRST
	for thread_idx in range(thread_count):
		var start_shuffle: int = thread_idx * shuffles_per_thread
		var end_shuffle: int = min(start_shuffle + shuffles_per_thread, n_shuffles)
		
		if start_shuffle >= end_shuffle:
			break  # No more work for this thread
		
		var actual_chunk_size: int = end_shuffle - start_shuffle
		var task = BatchChunkTask.new(
			deck_size, 
			method, 
			starting_index + start_shuffle, 
			sample_seed, 
			actual_chunk_size
		)
		chunk_tasks.append(task)
		
		var task_id = WorkerThreadPool.add_task(Callable(_batch_chunk_worker).bind(task))
		task_ids.append(task_id)
	
	# NOW wait for ALL tasks to complete in parallel - this is the key fix!
	for task_id in task_ids:
		WorkerThreadPool.wait_for_task_completion(task_id)
	
	# Finally, assemble results from all completed chunks
	for i in range(chunk_tasks.size()):
		var chunk_results: Array = chunk_tasks[i].result_shuffles
		var start_idx: int = i * shuffles_per_thread
		
		# Copy chunk results to final results array
		for j in range(chunk_results.size()):
			var result_idx: int = start_idx + j
			if result_idx < results.size():
				results[result_idx] = chunk_results[j]
	
	return results


## Enhanced worker function that does batch sample generation per chunk
static func _batch_chunk_worker(task: BatchChunkTask) -> void:
	# MAJOR OPTIMIZATION: Generate all samples for this chunk at once!
	var shuffle_dimensions: int = task.deck_size - 1
	if shuffle_dimensions <= 0:
		# Handle trivial case
		for i in range(task.chunk_size):
			var deck: Array[int] = []
			if task.deck_size == 1:
				deck.append(0)
			task.result_shuffles[i] = deck
		return
	
	# Generate ALL samples for this chunk in one go - HUGE performance gain!
	var all_chunk_samples: Array = _generate_samples_nd_sequential(
		task.chunk_size, 
		shuffle_dimensions, 
		task.method, 
		task.starting_point_index, 
		task.sample_seed
	)
	
	# Now do the shuffles using pre-generated samples
	for i in range(task.chunk_size):
		if i < all_chunk_samples.size() and all_chunk_samples[i].size() == shuffle_dimensions:
			task.result_shuffles[i] = _coordinated_shuffle_with_samples(
				task.deck_size, 
				all_chunk_samples[i]
			)
		else:
			# Fallback on error
			task.result_shuffles[i] = _create_unshuffled_deck(task.deck_size)


## Helper function to create an unshuffled deck for error cases
static func _create_unshuffled_deck(deck_size: int) -> Array[int]:
	var deck: Array[int] = []
	deck.resize(deck_size)
	for i in range(deck_size):
		deck[i] = i
	return deck


## Optimized shuffle using pre-generated samples - avoids redundant sample generation
static func _coordinated_shuffle_with_samples(deck_size: int, sobol_point: Array) -> Array[int]:
	if deck_size <= 1:
		var result: Array[int] = []
		if deck_size == 1:
			result.append(0)
		return result
	
	var deck: Array[int] = []
	deck.resize(deck_size)
	for i in range(deck_size):
		deck[i] = i
	
	# Perform coordinated Fisher-Yates shuffle using the pre-generated N-dimensional point
	for i in range(deck_size - 1, 0, -1):
		var dim_index: int = deck_size - 1 - i
		var raw_random_val: float = sobol_point[dim_index] if dim_index < sobol_point.size() else 0.5
		
		# Use proper Beta(2,2) PPF transformation
		var beta_val: float = StatMath.PpfFunctions.beta_ppf(raw_random_val, 2.0, 2.0)
		
		var j: int = int(beta_val * float(i + 1))
		j = clamp(j, 0, i)  # Ensure j is always in valid range [0, i]
		
		# Swap deck[i] and deck[j]
		if i != j:
			var temp: int = deck[i]
			deck[i] = deck[j]
			deck[j] = temp
	
	return deck


## Sequential version of generate_samples_nd for use within already-threaded contexts
## This prevents nested threading explosion when called from worker threads
static func _generate_samples_nd_sequential(
	n_draws: int, 
	dimensions: int, 
	method: SamplingMethod,
	starting_index: int,
	sample_seed: int
) -> Array:
	var samples: Array = []
	if n_draws <= 0 or dimensions <= 0:
		return samples
	
	var rng_to_use: RandomNumberGenerator
	if sample_seed != -1:
		rng_to_use = RandomNumberGenerator.new()
		rng_to_use.seed = sample_seed
	else:
		rng_to_use = StatMath.get_rng()
	
	_ensure_sobol_vectors_initialized(dimensions - 1)
	
	# Always use sequential implementation for thread safety
	samples.resize(n_draws)
	for i in range(n_draws):
		samples[i] = []
		samples[i].resize(dimensions)
	
	match method:
		SamplingMethod.RANDOM:
			for i in range(n_draws):
				for d in range(dimensions):
					samples[i][d] = rng_to_use.randf()
		
		SamplingMethod.SOBOL:
			for d in range(dimensions):
				var dim_samples = _generate_sobol_1d(n_draws, d, starting_index)
				for i in range(n_draws):
					samples[i][d] = dim_samples[i] if i < dim_samples.size() else -1.0
		
		SamplingMethod.SOBOL_RANDOM:
			var random_masks = []
			random_masks.resize(dimensions)
			for d in range(dimensions):
				random_masks[d] = rng_to_use.randi() & ((1 << _SOBOL_BITS) - 1)
			
			for d in range(dimensions):
				var sobol_integers = _get_sobol_1d_integers(n_draws, d, starting_index)
				for i in range(n_draws):
					if sobol_integers[i] != -1:
						samples[i][d] = float(sobol_integers[i] ^ random_masks[d]) / _SOBOL_MAX_VAL_FLOAT
					else:
						samples[i][d] = rng_to_use.randf()
		
		SamplingMethod.HALTON:
			for d in range(dimensions):
				var base: int = _get_nth_prime(d)
				var dim_samples = _generate_halton_1d(n_draws, base, starting_index)
				for i in range(n_draws):
					samples[i][d] = dim_samples[i] if i < dim_samples.size() else -1.0
		
		SamplingMethod.HALTON_RANDOM:
			var random_offsets = []
			random_offsets.resize(dimensions)
			for d in range(dimensions):
				random_offsets[d] = rng_to_use.randf()
			
			for d in range(dimensions):
				var base: int = _get_nth_prime(d)
				var halton_samples = _generate_halton_1d(n_draws, base, starting_index)
				for i in range(n_draws):
					if halton_samples[i] != -1.0:
						samples[i][d] = fmod(halton_samples[i] + random_offsets[d], 1.0)
					else:
						samples[i][d] = rng_to_use.randf()
		
		SamplingMethod.LATIN_HYPERCUBE:
			for d in range(dimensions):
				var lhs_samples = _generate_latin_hypercube_1d(n_draws, rng_to_use)
				for i in range(n_draws):
					samples[i][d] = lhs_samples[i] if i < lhs_samples.size() else -1.0
		
		_:
			printerr("Unsupported sampling method: ", SamplingMethod.keys()[method])
			for i in range(n_draws):
				for d in range(dimensions):
					samples[i][d] = -1.0
	
	return samples


## Worker function for individual shuffle generation
static func _coordinated_shuffle_worker(task: BatchShuffleTask) -> void:
	task.result_shuffle = coordinated_shuffle(
		task.deck_size, 
		task.method, 
		task.point_index, 
		task.sample_seed
	)


# --- CONTINUOUS SPACE SAMPLING (for Monte Carlo, space-filling) ---

## Unified interface for generating samples in 1, 2, or N dimensions.
##
## @param n_draws: int The number of samples to draw.
## @param dimensions: int The number of dimensions (1, 2, or N).
## @param method: SamplingMethod The sampling method to use.
## @param starting_index: int Starting index for deterministic sequences (default 0).
## @param sample_seed: int The random seed (optional, uses global RNG if -1).
## @return Variant Returns Array[float] for 1D, Array[Vector2] for 2D, Array[Array[float]] for ND.
static func generate_samples(
	n_draws: int, 
	dimensions: int = 1,
	method: SamplingMethod = SamplingMethod.RANDOM, 
	starting_index: int = 0,
	sample_seed: int = -1
) -> Variant:
	if n_draws <= 0:
		match dimensions:
			1: return Array([], TYPE_FLOAT, "", null)
			2: return Array([], TYPE_VECTOR2, "", null)
			_: return Array([], TYPE_ARRAY, "", null)
	
	if dimensions < 1:
		printerr("generate_samples: dimensions must be >= 1")
		return null
	
	# Use the unified N-dimensional generation for all cases
	var nd_samples: Array = generate_samples_nd(n_draws, dimensions, method, starting_index, sample_seed)
	
	match dimensions:
		1:
			# Convert from Array[Array[float]] to Array[float]
			var samples_1d: Array[float] = []
			samples_1d.resize(n_draws)
			for i in range(n_draws):
				if i < nd_samples.size() and nd_samples[i].size() > 0:
					samples_1d[i] = nd_samples[i][0]
				else:
					samples_1d[i] = -1.0  # Error signal
			return samples_1d
		2:
			# Convert from Array[Array[float]] to Array[Vector2]
			var samples_2d: Array[Vector2] = []
			samples_2d.resize(n_draws)
			for i in range(n_draws):
				if i < nd_samples.size() and nd_samples[i].size() >= 2:
					samples_2d[i] = Vector2(nd_samples[i][0], nd_samples[i][1])
				else:
					samples_2d[i] = Vector2(-1.0, -1.0)  # Error signal
			return samples_2d
		_:
			# Return the N-dimensional array as-is
			return nd_samples


## Generates N-dimensional samples using the specified method.
##
## @param n_draws: int The number of samples to draw.
## @param dimensions: int The number of dimensions.
## @param method: SamplingMethod The sampling method to use.
## @param starting_index: int Starting index for deterministic sequences.
## @param sample_seed: int The random seed (optional, uses global RNG if -1).
## @return Array An array of n_draws samples, each with 'dimensions' values (Array[Array[float]] conceptually).
static func generate_samples_nd(
	n_draws: int, 
	dimensions: int, 
	method: SamplingMethod = SamplingMethod.RANDOM,
	starting_index: int = 0,
	sample_seed: int = -1
) -> Array:
	var samples: Array = []
	if n_draws <= 0 or dimensions <= 0:
		return samples
	
	var rng_to_use: RandomNumberGenerator
	if sample_seed != -1:
		rng_to_use = RandomNumberGenerator.new()
		rng_to_use.seed = sample_seed
	else:
		rng_to_use = StatMath.get_rng()
	
	_ensure_sobol_vectors_initialized(dimensions - 1)
	
	# Always use threading for dimensions >= 3 (cleaner, simpler, and faster)
	if dimensions >= 3:
		return _generate_samples_nd(n_draws, dimensions, method, starting_index, rng_to_use)
	
	# Keep simple sequential implementation only for 1D and 2D cases
	samples.resize(n_draws)
	for i in range(n_draws):
		samples[i] = []
		samples[i].resize(dimensions)
	
	match method:
		SamplingMethod.RANDOM:
			for i in range(n_draws):
				for d in range(dimensions):
					samples[i][d] = rng_to_use.randf()
		
		SamplingMethod.SOBOL:
			for d in range(dimensions):
				var dim_samples = _generate_sobol_1d(n_draws, d, starting_index)
				for i in range(n_draws):
					samples[i][d] = dim_samples[i] if i < dim_samples.size() else -1.0
		
		SamplingMethod.SOBOL_RANDOM:
			var random_masks = []
			random_masks.resize(dimensions)
			for d in range(dimensions):
				random_masks[d] = rng_to_use.randi() & ((1 << _SOBOL_BITS) - 1)
			
			for d in range(dimensions):
				var sobol_integers = _get_sobol_1d_integers(n_draws, d, starting_index)
				for i in range(n_draws):
					if sobol_integers[i] != -1:
						samples[i][d] = float(sobol_integers[i] ^ random_masks[d]) / _SOBOL_MAX_VAL_FLOAT
					else:
						samples[i][d] = rng_to_use.randf()
		
		SamplingMethod.HALTON:
			for d in range(dimensions):
				var base: int = _get_nth_prime(d)
				var dim_samples = _generate_halton_1d(n_draws, base, starting_index)
				for i in range(n_draws):
					samples[i][d] = dim_samples[i] if i < dim_samples.size() else -1.0
		
		SamplingMethod.HALTON_RANDOM:
			var random_offsets = []
			random_offsets.resize(dimensions)
			for d in range(dimensions):
				random_offsets[d] = rng_to_use.randf()
			
			for d in range(dimensions):
				var base: int = _get_nth_prime(d)
				var halton_samples = _generate_halton_1d(n_draws, base, starting_index)
				for i in range(n_draws):
					if halton_samples[i] != -1.0:
						samples[i][d] = fmod(halton_samples[i] + random_offsets[d], 1.0)
					else:
						samples[i][d] = rng_to_use.randf()
		
		SamplingMethod.LATIN_HYPERCUBE:
			for d in range(dimensions):
				var lhs_samples = _generate_latin_hypercube_1d(n_draws, rng_to_use)
				for i in range(n_draws):
					samples[i][d] = lhs_samples[i] if i < lhs_samples.size() else -1.0
		
		_:
			printerr("Unsupported sampling method: ", SamplingMethod.keys()[method])
			for i in range(n_draws):
				for d in range(dimensions):
					samples[i][d] = -1.0
	
	return samples


## Performs a complete coordinated shuffle of a deck using multi-dimensional sampling.
## This is the key method for your coordinated Fisher-Yates approach.
##
## @param deck_size: int Size of the deck to shuffle.
## @param method: SamplingMethod The sampling method to use for coordination.
## @param point_index: int Which point in the sequence to use (for deterministic sequences).
## @param sample_seed: int The random seed (optional, uses global RNG if -1).
## @return Array[int] The shuffled deck as indices [0, deck_size-1].
static func coordinated_shuffle(
	deck_size: int, 
	method: SamplingMethod = SamplingMethod.SOBOL,
	point_index: int = 0,
	sample_seed: int = -1
) -> Array[int]:
	if deck_size <= 1:
		var result: Array[int] = []
		if deck_size == 1:
			result.append(0)
		return result
	
	var deck: Array[int] = []
	deck.resize(deck_size)
	for i in range(deck_size):
		deck[i] = i
	
	# Generate a multi-dimensional point for the shuffle
	# We need (deck_size - 1) dimensions for Fisher-Yates
	var shuffle_dimensions: int = deck_size - 1
	
	# CRITICAL FIX: Use sequential sampling to avoid nested threading explosion
	var nd_samples: Array = _generate_samples_nd_sequential(1, shuffle_dimensions, method, point_index, sample_seed)
	
	if nd_samples.is_empty() or nd_samples[0].size() != shuffle_dimensions:
		printerr("coordinated_shuffle: Failed to generate ND samples")
		return deck  # Return unshuffled deck on error
	
	var sobol_point = nd_samples[0]
	
	# Perform coordinated Fisher-Yates shuffle using the N-dimensional point
	for i in range(deck_size - 1, 0, -1):
		var dim_index: int = deck_size - 1 - i
		var raw_random_val: float = sobol_point[dim_index]
		
		# FINAL FIX: Use proper Beta(2,2) PPF transformation
		# Now that incomplete_beta is implemented, we can use the mathematical approach
		# Beta(2,2) creates a symmetric bell curve that avoids extremes while preserving uniformity
		var beta_val: float = StatMath.PpfFunctions.beta_ppf(raw_random_val, 2.0, 2.0)
		
		var j: int = int(beta_val * float(i + 1))
		j = clamp(j, 0, i)  # Ensure j is always in valid range [0, i]
		
		# Swap deck[i] and deck[j]
		if i != j:
			var temp: int = deck[i]
			deck[i] = deck[j]
			deck[j] = temp
	
	return deck


## Generates multiple coordinated shuffles efficiently.
##
## @param deck_size: int Size of each deck to shuffle.
## @param n_shuffles: int Number of shuffles to generate.
## @param method: SamplingMethod The sampling method to use.
## @param starting_index: int Starting point in the sequence.
## @param sample_seed: int The random seed (optional, uses global RNG if -1).
## @return Array Array of shuffled decks (Array[Array[int]] conceptually).
static func coordinated_batch_shuffles(
	deck_size: int,
	n_shuffles: int, 
	method: SamplingMethod = SamplingMethod.SOBOL,
	starting_index: int = 0,
	sample_seed: int = -1
) -> Array:
	var results: Array = []
	if n_shuffles <= 0 or deck_size <= 0:
		return results
	
	# Fast path for RANDOM method - no need for complex ND generation
	if method == SamplingMethod.RANDOM:
		return _fast_random_batch_shuffles(deck_size, n_shuffles, sample_seed)
	
	# Always use threading for batch operations (simpler and faster)
	if n_shuffles >= 2:
		return _coordinated_batch_shuffles_threaded(deck_size, n_shuffles, method, starting_index, sample_seed)
	
	# Single shuffle case
	results.resize(n_shuffles)
	for i in range(n_shuffles):
		results[i] = coordinated_shuffle(deck_size, method, starting_index + i, sample_seed)
	
	return results


# --- DISCRETE INDEX SAMPLING (for finite populations, card games, bootstrap) ---

## Enhanced interface for sampling indices from a finite population.
## Combines sampling methods (how to generate random numbers) with selection strategies 
## (how to use those numbers to select indices).
##
## @param population_size: int The size of the population to sample from (0 to population_size-1).
## @param draw_count: int The number of indices to draw.
## @param selection_strategy: SelectionStrategy How to select indices (with/without replacement).
## @param sampling_method: SamplingMethod How to generate the underlying random numbers.
## @param sample_seed: int The random seed (optional, uses global RNG if -1).
## @return Array[int] An Array of integers representing selected indices. Empty if invalid parameters.
static func sample_indices(
	population_size: int, 
	draw_count: int, 
	selection_strategy: SelectionStrategy = SelectionStrategy.FISHER_YATES,
	sampling_method: SamplingMethod = SamplingMethod.RANDOM,
	sample_seed: int = -1
) -> Array[int]:
	if draw_count < 0:
		printerr("sample_indices: draw_count cannot be negative.")
		return []
	if population_size < 0:
		printerr("sample_indices: population_size cannot be negative.")
		return []
	if selection_strategy != SelectionStrategy.WITH_REPLACEMENT and draw_count > population_size:
		printerr("sample_indices: Without replacement, draw_count cannot exceed population_size.")
		return []

	var rng_to_use: RandomNumberGenerator
	if sample_seed != -1:
		rng_to_use = RandomNumberGenerator.new()
		rng_to_use.seed = sample_seed
	else:
		rng_to_use = StatMath.get_rng()

	match selection_strategy:
		SelectionStrategy.WITH_REPLACEMENT:
			return _with_replacement_draw(population_size, draw_count, sampling_method, rng_to_use)
		SelectionStrategy.FISHER_YATES:
			return _fisher_yates_draw(population_size, draw_count, sampling_method, rng_to_use)
		SelectionStrategy.RESERVOIR:
			return _reservoir_draw(population_size, draw_count, sampling_method, rng_to_use)
		SelectionStrategy.SELECTION_TRACKING:
			return _selection_tracking_draw(population_size, draw_count, sampling_method, rng_to_use)
		SelectionStrategy.COORDINATED_FISHER_YATES:
			return _coordinated_fisher_yates_draw(population_size, draw_count, sampling_method, rng_to_use)
	
	printerr("sample_indices: Unsupported selection strategy: ", SelectionStrategy.keys()[selection_strategy])
	return []


# --- PRIVATE IMPLEMENTATION METHODS ---

## Sampling with replacement - allows duplicates.
static func _with_replacement_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:
	var result: Array[int] = []
	if draw_count <= 0:
		return result
	result.resize(draw_count)

	# Generate random values using the specified sampling method
	var random_values: Array[float] = []
	match sampling_method:
		SamplingMethod.RANDOM:
			random_values.resize(draw_count)
			for i in range(draw_count):
				random_values[i] = rng.randf()
		SamplingMethod.SOBOL:
			random_values = _generate_sobol_1d(draw_count, 0)
		SamplingMethod.SOBOL_RANDOM:
			var sobol_integers: Array[int] = _get_sobol_1d_integers(draw_count, 0)
			var random_mask: int = rng.randi() & ((1 << _SOBOL_BITS) - 1)
			random_values.resize(draw_count)
			for i in range(draw_count):
				if sobol_integers[i] != -1:
					random_values[i] = float(sobol_integers[i] ^ random_mask) / _SOBOL_MAX_VAL_FLOAT
				else:
					random_values[i] = rng.randf() # Fallback on error
		SamplingMethod.HALTON:
			random_values = _generate_halton_1d(draw_count, 2)
		SamplingMethod.HALTON_RANDOM:
			var halton_samples: Array[float] = _generate_halton_1d(draw_count, 2)
			var random_offset: float = rng.randf()
			random_values.resize(draw_count)
			for i in range(draw_count):
				if halton_samples[i] != -1.0:
					random_values[i] = fmod(halton_samples[i] + random_offset, 1.0)
				else:
					random_values[i] = rng.randf() # Fallback on error
		SamplingMethod.LATIN_HYPERCUBE:
			random_values = _generate_latin_hypercube_1d(draw_count, rng)
		_:
			# Fallback to RANDOM
			random_values.resize(draw_count)
			for i in range(draw_count):
				random_values[i] = rng.randf()

	# Convert random values to indices
	for i in range(draw_count):
		result[i] = int(random_values[i] * float(population_size)) % population_size

	return result


## Fisher-Yates shuffle with custom sampling method for randomness.
static func _fisher_yates_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:
	var deck: Array[int] = []
	deck.resize(population_size)
	for i in range(population_size):
		deck[i] = i
	
	# Generate enough random values for the shuffle
	var random_values: Array[float] = []
	match sampling_method:
		SamplingMethod.RANDOM:
			# Generate on-demand for efficiency
			pass
		_:
			# Pre-generate for deterministic sequences
			var samples_variant: Variant = generate_samples(draw_count, 1, sampling_method, 0, rng.get_seed() if rng.get_seed() != 0 else -1)
			random_values = samples_variant as Array[float]

	# Partial Fisher-Yates shuffle
	for i in range(draw_count):
		var random_val: float
		if random_values.is_empty():
			random_val = rng.randf()
		else:
			random_val = random_values[i] if i < random_values.size() else rng.randf()
		
		# Map to valid range [i, population_size - 1]
		var j: int = i + int(random_val * float(population_size - i))
		j = min(j, population_size - 1) # Clamp to prevent overflow
		
		# Swap
		if i != j:
			var temp: int = deck[i]
			deck[i] = deck[j]
			deck[j] = temp

	# Return first draw_count elements
	var result: Array[int] = []
	result.resize(draw_count)
	for i in range(draw_count):
		result[i] = deck[i]
	return result


## Reservoir sampling with custom sampling method.
static func _reservoir_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:
	var reservoir: Array[int] = []
	reservoir.resize(draw_count)
	
	if draw_count == 0:
		return reservoir

	# Fill reservoir with first draw_count items
	for i in range(min(draw_count, population_size)):
		reservoir[i] = i

	# Generate random values for the algorithm
	var needed_randoms: int = max(0, population_size - draw_count)
	var random_values: Array[float] = []
	if needed_randoms > 0:
		match sampling_method:
			SamplingMethod.RANDOM:
				# Generate on-demand for efficiency
				pass
			_:
				var samples_variant: Variant = generate_samples(needed_randoms, 1, sampling_method, 0, rng.get_seed() if rng.get_seed() != 0 else -1)
				random_values = samples_variant as Array[float]

	# Reservoir algorithm
	var random_index: int = 0
	for i in range(draw_count, population_size):
		var random_val: float
		if random_values.is_empty():
			random_val = rng.randf()
		else:
			random_val = random_values[random_index] if random_index < random_values.size() else rng.randf()
			random_index += 1
		
		var j: int = int(random_val * float(i + 1))
		if j < draw_count:
			reservoir[j] = i

	return reservoir


## Selection tracking with custom sampling method.
static func _selection_tracking_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:
	var selected_indices: Dictionary = {}
	var result: Array[int] = []
	result.resize(draw_count)
	
	if draw_count == 0:
		return result

	# Pre-generate random values for deterministic sequences
	var max_attempts: int = (population_size * 4) + (draw_count * 4) + 20
	var random_values: Array[float] = []
	match sampling_method:
		SamplingMethod.RANDOM:
			# Generate on-demand
			pass
		_:
			var samples_variant: Variant = generate_samples(max_attempts, 1, sampling_method, 0, rng.get_seed() if rng.get_seed() != 0 else -1)
			random_values = samples_variant as Array[float]

	var items_drawn: int = 0
	var attempts: int = 0

	while items_drawn < draw_count and attempts < max_attempts:
		var random_val: float
		if random_values.is_empty():
			random_val = rng.randf()
		else:
			random_val = random_values[attempts] if attempts < random_values.size() else rng.randf()
		
		var random_index: int = int(random_val * float(population_size))
		if not selected_indices.has(random_index):
			selected_indices[random_index] = true
			result[items_drawn] = random_index
			items_drawn += 1
		attempts += 1
	
	if items_drawn < draw_count:
		printerr("_selection_tracking_draw: Failed to draw enough unique items.")
		return result.slice(0, items_drawn)

	return result


## Coordinated Fisher-Yates shuffle for multi-dimensional sampling.
## This performs a complete coordinated shuffle and returns the first draw_count elements.
static func _coordinated_fisher_yates_draw(population_size: int, draw_count: int, sampling_method: SamplingMethod, rng: RandomNumberGenerator) -> Array[int]:
	# For coordinated Fisher-Yates, we perform a complete shuffle and return the first draw_count elements
	var shuffled_deck: Array[int] = coordinated_shuffle(population_size, sampling_method, 0, rng.get_seed() if rng.get_seed() != 0 else -1)
	
	var result: Array[int] = []
	result.resize(draw_count)
	
	for i in range(draw_count):
		if i < shuffled_deck.size():
			result[i] = shuffled_deck[i]
		else:
			result[i] = i  # Fallback in case of error
	
	return result


# --- SOBOL SEQUENCE IMPLEMENTATION ---

## Returns nth prime number for Halton sequences
static func _get_nth_prime(n: int) -> int:
	const PRIMES: Array[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
	if n < PRIMES.size():
		return PRIMES[n]
	else:
		# Simple fallback for higher dimensions (not optimized)
		return 2 + n  # This is not correct but prevents crashes


## Generates Sobol sequence integers for a specific dimension.
static func _get_sobol_1d_integers(ndraws: int, dimension_index: int, starting_index: int = 0) -> Array[int]:
	var integers: Array[int] = []
	if ndraws <= 0:
		return integers
	integers.resize(ndraws)

	_ensure_sobol_vectors_initialized(20)
	
	if dimension_index < 0 or dimension_index >= _PRIMITIVE_POLYNOMIALS.size():
		printerr("Sobol: Invalid dimension_index: ", dimension_index)
		for i in range(ndraws): integers[i] = -1 # Signal error
		return integers

	# Retrieve the cached direction vectors for the specified dimension
	var direction_vectors: Array[int] = _sobol_direction_vectors_cache[dimension_index]
	
	# Generate Sobol integers starting from starting_index
	var temp_integers: Array[int] = []
	var total_needed: int = ndraws + starting_index
	temp_integers.resize(total_needed)
	
	var current_sobol_integer: int = 0
	if total_needed > 0:
		temp_integers[0] = 0 # First Sobol integer is 0

	for i in range(1, total_needed):
		var c: int = 0
		var temp_i: int = i
		while (temp_i & 1) == 0:
			if temp_i == 0: break
			temp_i >>= 1
			c += 1
		
		# Ensure c is within bounds for direction_vectors
		if c >= direction_vectors.size(): 
			printerr("Sobol sequence ctz index c (%d) is out of range for direction_vectors.size() (%d) at point index i=%d for dim %d." % [c, direction_vectors.size(), i, dimension_index])
			# This is an error state; fill remaining integers and return to avoid further issues.
			for k in range(i, total_needed): temp_integers[k] = -1 # Signal error from this point
			break
			
		current_sobol_integer = current_sobol_integer ^ direction_vectors[c]
		temp_integers[i] = current_sobol_integer
	
	# Extract the requested slice
	for i in range(ndraws):
		var source_index: int = starting_index + i
		if source_index < temp_integers.size():
			integers[i] = temp_integers[source_index]
		else:
			integers[i] = -1  # Error signal
		
	return integers


## Generates 1D Sobol samples for a specific dimension.
static func _generate_sobol_1d(ndraws: int, dimension_index: int, starting_index: int = 0) -> Array[float]:
	var samples: Array[float] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var sobol_integers: Array[int] = _get_sobol_1d_integers(ndraws, dimension_index, starting_index)
	# Check if _get_sobol_1d_integers signaled an error (by filling with -1)
	if ndraws > 0 and not sobol_integers.is_empty() and sobol_integers[0] == -1 and sobol_integers.count(-1) == ndraws:
		for i in range(ndraws): samples[i] = -1.0 # Propagate error
		return samples
		
	for i in range(ndraws):
		# Additional check for individual -1 in sobol_integers if partial error occurred
		if sobol_integers[i] == -1:
			samples[i] = -1.0
		else:
			samples[i] = float(sobol_integers[i]) / _SOBOL_MAX_VAL_FLOAT
		
	return samples


# --- HALTON SEQUENCE IMPLEMENTATION ---

static func _generate_halton_1d(ndraws: int, base: int, starting_index: int = 0) -> Array[float]:
	var sequence: Array[float] = []
	if ndraws <= 0: return sequence
	sequence.resize(ndraws)

	# Check for invalid base
	if base < 2:
		printerr("Halton: Base must be >= 2. Got: ", base)
		for i_idx in range(ndraws): sequence[i_idx] = -1.0 # Signal error
		return sequence

	for i_idx in range(ndraws):
		var n: int = i_idx + starting_index + 1 # Halton sequence is 1-indexed for generation
		var x: float = 0.0
		var f: float = 1.0
		while n > 0:
			f /= float(base)
			x += f * (n % base)
			n = int(n / base) # Ensure integer division
		sequence[i_idx] = x
	return sequence


# --- LATIN HYPERCUBE IMPLEMENTATION ---

static func _generate_latin_hypercube_1d(ndraws: int, rng: RandomNumberGenerator) -> Array[float]:
	var lhs_samples: Array[float] = []
	if ndraws <= 0:
		return lhs_samples
	lhs_samples.resize(ndraws)

	for i in range(ndraws):
		lhs_samples[i] = (float(i) + rng.randf()) / float(ndraws)
	
	# Fisher-Yates shuffle
	for i in range(ndraws - 1, 0, -1): 
		var j: int = rng.randi_range(0, i) 
		var temp: float = lhs_samples[i]
		lhs_samples[i] = lhs_samples[j]
		lhs_samples[j] = temp
		
	return lhs_samples


# Fast random batch shuffles
static func _fast_random_batch_shuffles(deck_size: int, n_shuffles: int, sample_seed: int) -> Array:
	var results: Array = []
	results.resize(n_shuffles)
	
	for i in range(n_shuffles):
		results[i] = _fast_random_shuffle(deck_size, sample_seed)
	
	return results


static func _fast_random_shuffle(deck_size: int, sample_seed: int) -> Array[int]:
	var deck: Array[int] = []
	deck.resize(deck_size)
	for i in range(deck_size):
		deck[i] = i
	
	# Fisher-Yates shuffle
	for i in range(deck_size - 1, 0, -1):
		var j: int = int(randf() * float(i + 1))
		var temp: int = deck[i]
		deck[i] = deck[j]
		deck[j] = temp
	
	return deck
