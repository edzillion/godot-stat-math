# res://src/core/sampling.gd
class_name Sampling

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
static var _SOBOL_DIRECTION_VECTORS: Array[int] = [] # Static-like variable, initialized by _ensure_sobol_vectors_initialized or _init


func _init() -> void: # Called if MCSampling is an Autoload/instantiated
	_ensure_sobol_vectors_initialized()


## Ensures Sobol direction vectors are initialized.
## This method is idempotent and safe to call multiple times.
static func _ensure_sobol_vectors_initialized() -> void:
	if _SOBOL_DIRECTION_VECTORS.is_empty():
		# This initialization should ideally happen once.
		# If called from multiple threads simultaneously before initialization (not typical in GDScript main thread),
		# it could theoretically lead to redundant work or a race, but practically safe for most GDScript use cases.
		_SOBOL_DIRECTION_VECTORS.resize(_SOBOL_BITS)
		for j in range(_SOBOL_BITS):
			_SOBOL_DIRECTION_VECTORS[j] = 1 << (_SOBOL_BITS - 1 - j)


## Generates an array of sample values based on the specified method.
##
## @param ndraws: int The number of samples to draw.
## @param method: SamplingMethod The sampling method to use.
## @param seed: int The random seed (optional, uses system time if not provided or 0).
## @return Array[float] An array of floats between 0.0 and 1.0.
static func generate_samples(ndraws: int, method: SamplingMethod, seed: int = 0) -> Array[float]:
	var samples: Array[float] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	if seed != 0:
		rng.seed = seed
	else:
		rng.randomize()

	_ensure_sobol_vectors_initialized() # Ensure vectors are ready for any SOBOL methods

	match method:
		SamplingMethod.RANDOM:
			for i in range(ndraws):
				samples[i] = rng.randf()
		SamplingMethod.SOBOL:
			samples = _generate_sobol_1d(ndraws)
		SamplingMethod.SOBOL_RANDOM:
			var sobol_integers: Array[int] = _get_sobol_1d_integers(ndraws)
			var random_mask: int = rng.randi() & ((1 << _SOBOL_BITS) - 1) 
			for i in range(ndraws):
				samples[i] = float(sobol_integers[i] ^ random_mask) / _SOBOL_MAX_VAL_FLOAT
		SamplingMethod.HALTON:
			samples = _generate_halton_1d(ndraws, 2) # Using base 2 for 1D Halton
		SamplingMethod.HALTON_RANDOM:
			var halton_samples: Array[float] = _generate_halton_1d(ndraws, 2)
			var random_offset: float = rng.randf()
			for i in range(ndraws):
				samples[i] = fmod(halton_samples[i] + random_offset, 1.0)
		SamplingMethod.LATIN_HYPERCUBE:
			samples = _generate_latin_hypercube_1d(ndraws, rng)
		_:
			printerr("Unsupported sampling method: ", SamplingMethod.keys()[method])
			for i in range(ndraws): samples[i] = -1.0 # Fill with an error indicator

	return samples


# --- Private Static Helper Functions ---

static func _get_sobol_1d_integers(ndraws: int) -> Array[int]:
	var integers: Array[int] = []
	if ndraws <= 0:
		return integers
	integers.resize(ndraws)

	_ensure_sobol_vectors_initialized() # Ensure vectors are ready

	var current_sobol_integer: int = 0
	if ndraws > 0: 
		integers[0] = 0 # First Sobol integer is 0

	for i in range(1, ndraws): # i is the 1-based index for ctz calculation
		var c: int = 0
		var temp_i: int = i
		while (temp_i & 1) == 0: # Count trailing zeros (ctz)
			if temp_i == 0: break # Should not be reached if i > 0
			temp_i >>= 1
			c += 1
		# c is now the 0-indexed position of the rightmost 1-bit of i
		
		if c >= _SOBOL_BITS:
			printerr("Sobol sequence ctz index c (%d) is out of range for _SOBOL_BITS (%d) at point index i=%d. Reduce ndraws or increase _SOBOL_BITS." % [c, _SOBOL_BITS, i])
			c = _SOBOL_BITS - 1 # Cap c to avoid crash, though sequence quality degrades.
			
		current_sobol_integer = current_sobol_integer ^ _SOBOL_DIRECTION_VECTORS[c]
		integers[i] = current_sobol_integer
		
	return integers


static func _generate_sobol_1d(ndraws: int) -> Array[float]:
	var samples: Array[float] = []
	if ndraws <= 0:
		return samples
	samples.resize(ndraws)

	var sobol_integers: Array[int] = _get_sobol_1d_integers(ndraws)
	for i in range(ndraws):
		samples[i] = float(sobol_integers[i]) / _SOBOL_MAX_VAL_FLOAT
		
	return samples


static func _generate_halton_1d(ndraws: int, base: int) -> Array[float]:
	var sequence: Array[float] = []
	if ndraws <= 0: return sequence
	sequence.resize(ndraws)

	for i in range(ndraws):
		var n: int = i + 1 # Halton sequence conventionally starts from index 1 for calculation
		var x: float = 0.0
		var f: float = 1.0
		while n > 0:
			f /= float(base)
			x += f * (n % base)
			n = n / base # Integer division
		sequence[i] = x
	return sequence


static func _generate_latin_hypercube_1d(ndraws: int, rng: RandomNumberGenerator) -> Array[float]:
	var lhs_samples: Array[float] = []
	if ndraws <= 0:
		return lhs_samples
	lhs_samples.resize(ndraws)

	# Generate stratified samples
	for i in range(ndraws):
		lhs_samples[i] = (float(i) + rng.randf()) / float(ndraws)
	
	# Shuffle the samples (Fisher-Yates shuffle)
	for i in range(ndraws - 1, 0, -1): # Iterate from ndraws-1 down to 1
		var j: int = rng.randi_range(0, i) # Pick an index from 0 to i (inclusive)
		# Swap elements
		var temp: float = lhs_samples[i]
		lhs_samples[i] = lhs_samples[j]
		lhs_samples[j] = temp
		
	return lhs_samples
