# addons/godot-stat-math/tests/core/sampling_test.gd
class_name SamplingTest extends GdUnitTestSuite

# Member variables here if needed, e.g. for complex setups or shared resources.


# Called before each test.
func before_test() -> void:
	pass


# Called after each test.
func after_test() -> void:
	pass


# --- Test Cases for generate_samples ---

func test_generate_samples_random_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(ndraws) # "Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "Sample value should be <= 1.0"


func test_generate_samples_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(0) # "Should return an empty array for ndraws = 0"


func test_generate_samples_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(0) # "Should return an empty array for ndraws < 0"


func test_generate_samples_sobol_basic() -> void:
	var ndraws: int = 5
	# Ensure Sobol vectors are initialized for the test if not autoloaded
	# StatMath.Sampling._ensure_sobol_vectors_initialized() # This is static and called internally
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL)

	assert_int(samples.size()).is_equal(ndraws) # "SOBOL: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "SOBOL: Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "SOBOL: Sample value should be <= 1.0"

	# Check specific Sobol sequence values for the first few points (1D)
	# Values depend on _SOBOL_BITS and the direction vector initialization.
	# With _SOBOL_BITS = 30 and standard 1D direction vectors (powers of 2):
	# s0 = 0/N = 0
	# s1 = (0 ^ V0)/N = (0 ^ 2^29) / 2^30 = 0.5
	# s2 = (s1_int ^ V1)/N = (2^29 ^ 2^28) / 2^30 = (0.5 + 0.25) = 0.75
	# s3 = (s2_int ^ V0)/N = ((2^29 ^ 2^28) ^ 2^29) / 2^30 = 2^28 / 2^30 = 0.25
	# s4 = (s3_int ^ V2)/N = (2^28 ^ 2^27) / 2^30 = (0.25 + 0.125) = 0.375
	var expected_sobol_samples: Array[float] = [0.0, 0.5, 0.75, 0.25, 0.375]
	for i in range(ndraws):
		assert_float(samples[i]).is_equal_approx(expected_sobol_samples[i], 0.00001)
			# "SOBOL: Sample %d should match expected value" % i


func test_generate_samples_sobol_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL)
	assert_int(samples.size()).is_equal(0) # "SOBOL: Should return an empty array for ndraws = 0"


func test_generate_samples_sobol_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL)
	assert_int(samples.size()).is_equal(0) # "SOBOL: Should return an empty array for ndraws < 0"


func test_generate_samples_sobol_random_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL_RANDOM)

	assert_int(samples.size()).is_equal(ndraws) # "SOBOL_RANDOM: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "SOBOL_RANDOM: Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "SOBOL_RANDOM: Sample value should be <= 1.0"


func test_generate_samples_sobol_random_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 12345
	var samples1: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL_RANDOM, seed)
	var samples2: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL_RANDOM, seed)

	assert_int(samples1.size()).is_equal(ndraws) # "SOBOL_RANDOM (seed): Correct number of samples for first set"
	assert_int(samples2.size()).is_equal(ndraws) # "SOBOL_RANDOM (seed): Correct number of samples for second set"
	for i in range(ndraws):
		assert_float(samples1[i]).is_greater_equal(0.0) # "SOBOL_RANDOM (seed): Sample1[%d] should be >= 0.0" % i
		assert_float(samples1[i]).is_less_equal(1.0) # "SOBOL_RANDOM (seed): Sample1[%d] should be <= 1.0" % i
		assert_float(samples2[i]).is_greater_equal(0.0) # "SOBOL_RANDOM (seed): Sample2[%d] should be >= 0.0" % i
		assert_float(samples2[i]).is_less_equal(1.0) # "SOBOL_RANDOM (seed): Sample2[%d] should be <= 1.0" % i
		assert_float(samples1[i]).is_equal_approx(samples2[i], 0.0000001) # "SOBOL_RANDOM (seed): Samples should be reproducible with the same seed"


func test_generate_samples_sobol_random_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL_RANDOM)
	assert_int(samples.size()).is_equal(0) # "SOBOL_RANDOM: Should return an empty array for ndraws = 0"


func test_generate_samples_sobol_random_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.SOBOL_RANDOM)
	assert_int(samples.size()).is_equal(0) # "SOBOL_RANDOM: Should return an empty array for ndraws < 0"


func test_generate_samples_halton_basic() -> void:
	var ndraws: int = 5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON)

	assert_int(samples.size()).is_equal(ndraws) # "HALTON: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "HALTON: Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "HALTON: Sample value should be < 1.0 (strictly for base > 1)"

	# Expected Halton sequence for base 2 (first 5 values):
	# H(1,2) = 1/2 = 0.5
	# H(2,2) = 1/4 = 0.25
	# H(3,2) = 3/4 = 0.75
	# H(4,2) = 1/8 = 0.125
	# H(5,2) = 5/8 = 0.625
	# Note: The implementation _generate_halton_1d uses i from 0 to ndraws-1, and calculates for n = i + 1.
	# So for ndraws=5, it calculates for n=1,2,3,4,5.
	var expected_halton_samples: Array[float] = [0.5, 0.25, 0.75, 0.125, 0.625]
	for i in range(ndraws):
		assert_float(samples[i]).is_equal_approx(expected_halton_samples[i], 0.00001)
			# "HALTON: Sample %d should match expected value" % i


func test_generate_samples_halton_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON)
	assert_int(samples.size()).is_equal(0) # "HALTON: Should return an empty array for ndraws = 0"


func test_generate_samples_halton_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON)
	assert_int(samples.size()).is_equal(0) # "HALTON: Should return an empty array for ndraws < 0"


func test_generate_samples_halton_random_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON_RANDOM)

	assert_int(samples.size()).is_equal(ndraws) # "HALTON_RANDOM: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "HALTON_RANDOM: Sample value should be >= 0.0"
		assert_float(sample_val).is_less(1.0) # "HALTON_RANDOM: Sample value should be < 1.0 (due to fmod)"


func test_generate_samples_halton_random_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 54321
	var samples1: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON_RANDOM, seed)
	var samples2: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON_RANDOM, seed)

	assert_int(samples1.size()).is_equal(ndraws) # "HALTON_RANDOM (seed): Correct number of samples for first set"
	assert_int(samples2.size()).is_equal(ndraws) # "HALTON_RANDOM (seed): Correct number of samples for second set"
	for i in range(ndraws):
		assert_float(samples1[i]).is_greater_equal(0.0) # "HALTON_RANDOM (seed): Sample1[%d] should be >= 0.0" % i
		assert_float(samples1[i]).is_less(1.0) # "HALTON_RANDOM (seed): Sample1[%d] should be < 1.0" % i
		assert_float(samples2[i]).is_greater_equal(0.0) # "HALTON_RANDOM (seed): Sample2[%d] should be >= 0.0" % i
		assert_float(samples2[i]).is_less(1.0) # "HALTON_RANDOM (seed): Sample2[%d] should be < 1.0" % i
		assert_float(samples1[i]).is_equal_approx(samples2[i], 0.0000001) # "HALTON_RANDOM (seed): Samples should be reproducible with the same seed"


func test_generate_samples_halton_random_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON_RANDOM)
	assert_int(samples.size()).is_equal(0) # "HALTON_RANDOM: Should return an empty array for ndraws = 0"


func test_generate_samples_halton_random_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.HALTON_RANDOM)
	assert_int(samples.size()).is_equal(0) # "HALTON_RANDOM: Should return an empty array for ndraws < 0"


func test_generate_samples_latin_hypercube_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.LATIN_HYPERCUBE)

	assert_int(samples.size()).is_equal(ndraws) # "LHS: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "LHS: Sample value should be >= 0.0"
		assert_float(sample_val).is_less(1.0) # "LHS: Sample value should be < 1.0"


func test_generate_samples_latin_hypercube_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 67890
	var samples1: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.LATIN_HYPERCUBE, seed)
	var samples2: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.LATIN_HYPERCUBE, seed)

	assert_int(samples1.size()).is_equal(ndraws) # "LHS (seed): Correct number of samples for first set"
	assert_int(samples2.size()).is_equal(ndraws) # "LHS (seed): Correct number of samples for second set"
	for i in range(ndraws):
		assert_float(samples1[i]).is_greater_equal(0.0) # "LHS (seed): Sample1[%d] should be >= 0.0" % i
		assert_float(samples1[i]).is_less(1.0) # "LHS (seed): Sample1[%d] should be < 1.0" % i
		assert_float(samples2[i]).is_greater_equal(0.0) # "LHS (seed): Sample2[%d] should be >= 0.0" % i
		assert_float(samples2[i]).is_less(1.0) # "LHS (seed): Sample2[%d] should be < 1.0" % i
		assert_float(samples1[i]).is_equal_approx(samples2[i], 0.0000001) # "LHS (seed): Samples should be reproducible with the same seed"


func test_generate_samples_latin_hypercube_stratification() -> void:
	var ndraws: int = 20 # Use a reasonable number for checking stratification
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.LATIN_HYPERCUBE, 123)
	
	assert_int(samples.size()).is_equal(ndraws) # "LHS (strat): Correct number of samples"
	
	# To check stratification, sort the samples and verify one per stratum
	var sorted_samples: Array[float] = samples.duplicate()
	sorted_samples.sort()
	
	for i in range(ndraws):
		var lower_bound: float = float(i) / float(ndraws)
		var upper_bound: float = float(i + 1) / float(ndraws)
		assert_float(sorted_samples[i]).is_greater_equal(lower_bound)
			# "LHS (strat): Sample %d (%.4f) should be >= %.4f" % [i, sorted_samples[i], lower_bound]
		# For the upper bound, allow it to be equal if it's the last sample and exactly 1.0, otherwise strictly less.
		if i == ndraws -1 and sorted_samples[i] > (upper_bound - 0.000001) and sorted_samples[i] < (upper_bound + 0.000001): # approx upper_bound
			assert_float(sorted_samples[i]).is_less_equal(upper_bound)
				# "LHS (strat): Sample %d (%.4f) should be <= %.4f (last stratum)" % [i, sorted_samples[i], upper_bound]
		else:
			assert_float(sorted_samples[i]).is_less(upper_bound)
				# "LHS (strat): Sample %d (%.4f) should be < %.4f" % [i, sorted_samples[i], upper_bound]


func test_generate_samples_latin_hypercube_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.LATIN_HYPERCUBE)
	assert_int(samples.size()).is_equal(0) # "LHS: Should return an empty array for ndraws = 0"


func test_generate_samples_latin_hypercube_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.Sampling.generate_samples(ndraws, StatMath.Sampling.SamplingMethod.LATIN_HYPERCUBE)
	assert_int(samples.size()).is_equal(0) # "LHS: Should return an empty array for ndraws < 0" 
