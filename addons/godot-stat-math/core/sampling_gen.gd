# res://src/core/sampling_gen.gd
class_name SamplingGen

enum SamplingMethod {
	RANDOM,
	SOBOL,
	SOBOL_RANDOM,
	HALTON,
	HALTON_RANDOM,
	LATIN_HYPERCUBE
}

# Dependencies:
# res://src/core/enums.gd (implicitly used via SamplingMethod)

const _SOBOL_BITS: int = 30
const _SOBOL_MAX_VAL_FLOAT: float = float(1 << _SOBOL_BITS)
# Stores direction vectors for multiple dimensions.
# _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[dim_idx][bit_idx]
# This must be Array[Array] due to GDScript limitations on nested typed arrays.
# The inner arrays will contain integers.
static var _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC: Array[Array] = []


func _init() -> void: # Called if SamplingGen is an Autoload/instantiated
	_ensure_sobol_vectors_initialized()


## Ensures Sobol direction vectors are initialized for up to 2 dimensions.
## This method is idempotent and safe to call multiple times.
static func _ensure_sobol_vectors_initialized() -> void:
	if not _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC.is_empty():
		return

	_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC.resize(2) # Max 2 dimensions for now

	# Dimension 0 (equivalent to old 1D Sobol, polynomial 'x')
	# Initialize as generic Array, elements will be int
	_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[0] = [] 
	_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[0].resize(_SOBOL_BITS)
	for j in range(_SOBOL_BITS):
		_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[0][j] = 1 << (_SOBOL_BITS - 1 - j)

	# Dimension 1 (based on primitive polynomial x^2 + x + 1; degree s=2, coefficient a1=1)
	# Initialize as generic Array, elements will be int
	_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1] = [] 
	_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1].resize(_SOBOL_BITS)
	if _SOBOL_BITS > 0:
		# Initial values for v_0, v_1 (m_0=1, m_1=1 for this poly)
		_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1][0] = 1 << (_SOBOL_BITS - 1) 
	if _SOBOL_BITS > 1:
		_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1][1] = 1 << (_SOBOL_BITS - 1 - 1) 
	
	# Recurrence for j >= 2 (degree s=2, a1=1)
	# v_j = a1*v_{j-1} ^ v_{j-2} ^ (v_{j-2} >> s)  becomes v_j = v_{j-1} ^ v_{j-2} ^ (v_{j-2} >> 2)
	# We access elements of a generic Array, and they are known to be ints from initialization.
	for j in range(2, _SOBOL_BITS):
		var v_jm1: int = _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1][j-1] # Implicitly int
		var v_jm2: int = _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1][j-2] # Implicitly int
		_SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[1][j] = v_jm1 ^ v_jm2 ^ (v_jm2 >> 2)


## Generates an array of 1D sample values based on the specified method.
##
## @param ndraws: int The number of samples to draw.
## @param method: SamplingMethod The sampling method to use.
## @param seed: int The random seed (optional, uses global RNG if -1).
## @return Array[float] An Array of floats. Empty if ndraws <= 0.
static func generate_samples_1d(ndraws: int, method: SamplingMethod, seed: int = -1) -> Array[float]:
	var samples: Array[float] = []
	if ndraws <= 0:
		return samples
		
	samples.resize(ndraws)

	var rng_to_use: RandomNumberGenerator
	if seed != -1:
		rng_to_use = RandomNumberGenerator.new()
		rng_to_use.seed = seed
	else:
		rng_to_use = StatMath.get_rng()

	_ensure_sobol_vectors_initialized()

	match method:
		SamplingMethod.RANDOM:
			for i in range(ndraws):
				samples[i] = rng_to_use.randf()
		
		SamplingMethod.SOBOL:
			samples = _generate_sobol_1d(ndraws, 0)
		
		SamplingMethod.SOBOL_RANDOM:
			var sobol_integers: Array[int] = _get_sobol_1d_integers(ndraws, 0)
			var random_mask: int = rng_to_use.randi() & ((1 << _SOBOL_BITS) - 1)
			# Check if _get_sobol_1d_integers signaled an error
			if ndraws > 0 and not sobol_integers.is_empty() and sobol_integers[0] == -1 and sobol_integers.count(-1) == ndraws:
				for i in range(ndraws): samples[i] = -1.0 # Propagate error
			else:
				for i in range(ndraws):
					samples[i] = float(sobol_integers[i] ^ random_mask) / _SOBOL_MAX_VAL_FLOAT

		SamplingMethod.HALTON:
			samples = _generate_halton_1d(ndraws, 2) # Base 2 for 1D
		
		SamplingMethod.HALTON_RANDOM:
			var halton_samples_1d: Array[float] = _generate_halton_1d(ndraws, 2)
			var random_offset_1d: float = rng_to_use.randf()
			# Error check based on _generate_halton_1d potential error signal
			if ndraws > 0 and not halton_samples_1d.is_empty() and halton_samples_1d[0] == -1.0 and halton_samples_1d.count(-1.0) == ndraws:
				for i in range(ndraws): samples[i] = -1.0 # Propagate error
			else:
				for i in range(ndraws):
					samples[i] = fmod(halton_samples_1d[i] + random_offset_1d, 1.0)

		SamplingMethod.LATIN_HYPERCUBE:
			samples = _generate_latin_hypercube_1d(ndraws, rng_to_use)
		
		_:
			printerr("Unsupported sampling method: ", SamplingMethod.keys()[method])
			for i in range(ndraws): samples[i] = -1.0 

	return samples


## Generates an array of 2D sample values based on the specified method.
##
## @param ndraws: int The number of samples to draw.
## @param method: SamplingMethod The sampling method to use.
## @param seed: int The random seed (optional, uses global RNG if -1).
## @return Array[Vector2] An Array of Vector2. Empty if ndraws <= 0.
static func generate_samples_2d(ndraws: int, method: SamplingMethod, seed: int = -1) -> Array[Vector2]:
	var samples: Array[Vector2] = []
	if ndraws <= 0:
		return samples
		
	samples.resize(ndraws)

	var rng_to_use: RandomNumberGenerator
	if seed != -1:
		rng_to_use = RandomNumberGenerator.new()
		rng_to_use.seed = seed
	else:
		rng_to_use = StatMath.get_rng()

	_ensure_sobol_vectors_initialized()

	match method:
		SamplingMethod.RANDOM:
			for i in range(ndraws):
				samples[i] = Vector2(rng_to_use.randf(), rng_to_use.randf())
		
		SamplingMethod.SOBOL:
			samples = _generate_sobol_2d(ndraws)
		
		SamplingMethod.SOBOL_RANDOM:
			samples = _generate_sobol_random_2d(ndraws, rng_to_use)

		SamplingMethod.HALTON:
			samples = _generate_halton_2d(ndraws)
		
		SamplingMethod.HALTON_RANDOM:
			samples = _generate_halton_random_2d(ndraws, rng_to_use)

		SamplingMethod.LATIN_HYPERCUBE:
			samples = _generate_latin_hypercube_2d(ndraws, rng_to_use)
		
		_:
			printerr("Unsupported sampling method: ", SamplingMethod.keys()[method])
			for i in range(ndraws): samples[i] = Vector2(-1.0, -1.0)

	return samples


# --- Private Static Helper Functions ---

## Generates Sobol sequence integers for a specific dimension.
static func _get_sobol_1d_integers(ndraws: int, dimension_index: int) -> Array[int]:
	var integers: Array[int] = []
	if ndraws <= 0:
		return integers
	integers.resize(ndraws)

	_ensure_sobol_vectors_initialized()
	
	if dimension_index < 0 or dimension_index >= _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC.size():
		printerr("Sobol: Invalid dimension_index: ", dimension_index)
		for i in range(ndraws): integers[i] = -1 # Signal error
		return integers

	# Retrieve the generic inner array for the specified dimension.
	var generic_inner_array: Array = _SOBOL_DIRECTION_VECTORS_DIM_SPECIFIC[dimension_index]
	
	# Manually construct the typed Array[int] for direction_vectors_for_dim.
	var direction_vectors_for_dim: Array[int] = []
	# Check if the generic array is valid and has the expected size (_SOBOL_BITS)
	if generic_inner_array.is_empty() or generic_inner_array.size() != _SOBOL_BITS:
		printerr("Sobol: Direction vectors for dim ", dimension_index, " are empty or have incorrect size. Expected: ", _SOBOL_BITS, " Got: ", generic_inner_array.size())
		for i in range(ndraws): integers[i] = -1 # Signal error
		return integers
	
	direction_vectors_for_dim.resize(generic_inner_array.size())
	for i in range(generic_inner_array.size()):
		var val = generic_inner_array[i]
		if val is int:
			direction_vectors_for_dim[i] = val
		else:
			printerr("Sobol: Non-integer found in direction vectors for dim ", dimension_index, " at index ", i, ". Value: ", val)
			for k in range(ndraws): integers[k] = -1 # Signal error
			return integers

	var current_sobol_integer: int = 0
	if ndraws > 0:
		integers[0] = 0 # First Sobol integer is 0

	for i in range(1, ndraws):
		var c: int = 0
		var temp_i: int = i
		while (temp_i & 1) == 0:
			if temp_i == 0: break
			temp_i >>= 1
			c += 1
		
		# Ensure c is within bounds for direction_vectors_for_dim
		if c >= direction_vectors_for_dim.size(): 
			printerr("Sobol sequence ctz index c (%d) is out of range for direction_vectors_for_dim.size() (%d) at point index i=%d for dim %d." % [c, direction_vectors_for_dim.size(), i, dimension_index])
			# This is an error state; fill remaining integers and return to avoid further issues.
			for k in range(i, ndraws): integers[k] = -1 # Signal error from this point
			return integers
			
		current_sobol_integer = current_sobol_integer ^ direction_vectors_for_dim[c]
		integers[i] = current_sobol_integer
		
	return integers


## Generates 1D Sobol samples for a specific dimension.
static func _generate_sobol_1d(ndraws: int, dimension_index: int) -> Array[float]:
	var samples: Array[float] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var sobol_integers: Array[int] = _get_sobol_1d_integers(ndraws, dimension_index)
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


## Generates 2D Sobol samples.
static func _generate_sobol_2d(ndraws: int) -> Array[Vector2]:
	var samples: Array[Vector2] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var sobol_integers_x: Array[int] = _get_sobol_1d_integers(ndraws, 0)
	var sobol_integers_y: Array[int] = _get_sobol_1d_integers(ndraws, 1)

	# Check if _get_sobol_1d_integers signaled an error for X
	var error_in_x: bool = false
	if ndraws > 0 and not sobol_integers_x.is_empty():
		if sobol_integers_x[0] == -1 and sobol_integers_x.count(-1) == ndraws:
			error_in_x = true
	
	# Check if _get_sobol_1d_integers signaled an error for Y
	var error_in_y: bool = false
	if ndraws > 0 and not sobol_integers_y.is_empty():
		if sobol_integers_y[0] == -1 and sobol_integers_y.count(-1) == ndraws:
			error_in_y = true

	if error_in_x or error_in_y:
		for i in range(ndraws): samples[i] = Vector2(-1.0, -1.0) # Propagate error
		return samples

	for i in range(ndraws):
		# Additional check for individual -1 in sobol_integers if partial error occurred
		var x_val: float = -1.0
		var y_val: float = -1.0
		if sobol_integers_x[i] != -1:
			x_val = float(sobol_integers_x[i]) / _SOBOL_MAX_VAL_FLOAT
		if sobol_integers_y[i] != -1:
			y_val = float(sobol_integers_y[i]) / _SOBOL_MAX_VAL_FLOAT
		samples[i] = Vector2(x_val, y_val)
		
	return samples


## Generates 2D randomized Sobol samples.
static func _generate_sobol_random_2d(ndraws: int, rng: RandomNumberGenerator) -> Array[Vector2]:
	var samples: Array[Vector2] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var sobol_integers_x: Array[int] = _get_sobol_1d_integers(ndraws, 0)
	var sobol_integers_y: Array[int] = _get_sobol_1d_integers(ndraws, 1)
	
	var error_in_x: bool = false
	if ndraws > 0 and not sobol_integers_x.is_empty():
		if sobol_integers_x[0] == -1 and sobol_integers_x.count(-1) == ndraws:
			error_in_x = true
	
	var error_in_y: bool = false
	if ndraws > 0 and not sobol_integers_y.is_empty():
		if sobol_integers_y[0] == -1 and sobol_integers_y.count(-1) == ndraws:
			error_in_y = true

	if error_in_x or error_in_y:
		for i in range(ndraws): samples[i] = Vector2(-1.0, -1.0) # Propagate error
		return samples

	var random_mask_x: int = rng.randi() & ((1 << _SOBOL_BITS) - 1)
	var random_mask_y: int = rng.randi() & ((1 << _SOBOL_BITS) - 1)


	for i in range(ndraws):
		var x_val: float = -1.0
		var y_val: float = -1.0
		if sobol_integers_x[i] != -1:
			x_val = float(sobol_integers_x[i] ^ random_mask_x) / _SOBOL_MAX_VAL_FLOAT
		if sobol_integers_y[i] != -1:
			y_val = float(sobol_integers_y[i] ^ random_mask_y) / _SOBOL_MAX_VAL_FLOAT
		samples[i] = Vector2(x_val, y_val)
		
	return samples


static func _generate_halton_1d(ndraws: int, base: int) -> Array[float]:
	var sequence: Array[float] = []
	if ndraws <= 0: return sequence
	sequence.resize(ndraws)

	# Check for invalid base
	if base < 2:
		printerr("Halton: Base must be >= 2. Got: ", base)
		for i_idx in range(ndraws): sequence[i_idx] = -1.0 # Signal error
		return sequence

	for i_idx in range(ndraws):
		var n: int = i_idx + 1 # Halton sequence is 1-indexed for generation
		var x: float = 0.0
		var f: float = 1.0
		while n > 0:
			f /= float(base)
			x += f * (n % base)
			n = int(n / base) # Ensure integer division
		sequence[i_idx] = x
	return sequence


## Generates 2D Halton samples using bases 2 and 3.
static func _generate_halton_2d(ndraws: int) -> Array[Vector2]:
	var samples: Array[Vector2] = []
	if ndraws <= 0: return samples
	samples.resize(ndraws)

	var halton_x: Array[float] = _generate_halton_1d(ndraws, 2) # Base 2 for X
	var halton_y: Array[float] = _generate_halton_1d(ndraws, 3) # Base 3 for Y

	# Check for errors from _generate_halton_1d
	var error_in_x: bool = ndraws > 0 and not halton_x.is_empty() and halton_x[0] == -1.0 and halton_x.count(-1.0) == ndraws
	var error_in_y: bool = ndraws > 0 and not halton_y.is_empty() and halton_y[0] == -1.0 and halton_y.count(-1.0) == ndraws

	if error_in_x or error_in_y:
		for i in range(ndraws): samples[i] = Vector2(-1.0, -1.0)
		return samples

	for i in range(ndraws):
		samples[i] = Vector2(halton_x[i], halton_y[i])
	return samples


## Generates 2D randomized Halton samples.
static func _generate_halton_random_2d(ndraws: int, rng: RandomNumberGenerator) -> Array[Vector2]:
	var samples: Array[Vector2] = []
	if ndraws <= 0: return samples
	samples.resize(ndraws)
	
	var halton_points_2d: Array[Vector2] = _generate_halton_2d(ndraws)
	# Check for errors from _generate_halton_2d
	if ndraws > 0 and not halton_points_2d.is_empty() and halton_points_2d[0] == Vector2(-1.0, -1.0) and halton_points_2d.count(Vector2(-1.0,-1.0)) == ndraws:
		for i in range(ndraws): samples[i] = Vector2(-1.0, -1.0) # Propagate error
		return samples

	var random_offset_x: float = rng.randf()
	var random_offset_y: float = rng.randf()

	for i in range(ndraws):
		samples[i] = Vector2(
			fmod(halton_points_2d[i].x + random_offset_x, 1.0),
			fmod(halton_points_2d[i].y + random_offset_y, 1.0)
		)
	return samples


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


## Generates 2D Latin Hypercube samples.
static func _generate_latin_hypercube_2d(ndraws: int, rng: RandomNumberGenerator) -> Array[Vector2]:
	var samples: Array[Vector2] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var lhs_x: Array[float] = _generate_latin_hypercube_1d(ndraws, rng)
	var lhs_y: Array[float] = _generate_latin_hypercube_1d(ndraws, rng)

	# _generate_latin_hypercube_1d already returns empty on error or ndraws <=0
	if lhs_x.size() != ndraws or lhs_y.size() != ndraws: 
		for i in range(ndraws): samples[i] = Vector2(-1.0,-1.0)
		return samples

	for i in range(ndraws):
		samples[i] = Vector2(lhs_x[i], lhs_y[i])
		
	return samples
