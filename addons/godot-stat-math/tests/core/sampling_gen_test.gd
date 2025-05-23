# addons/godot-stat-math/tests/core/sampling_gen_test.gd
class_name SamplingGenTest extends GdUnitTestSuite

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
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(ndraws) # "Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "Sample value should be <= 1.0"


func test_generate_samples_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(0) # "Should return an empty array for ndraws = 0"


func test_generate_samples_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(0) # "Should return an empty array for ndraws < 0"


func test_generate_samples_sobol_basic() -> void:
	var ndraws: int = 5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL)

	assert_int(samples.size()).is_equal(ndraws) # "SOBOL: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "SOBOL: Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "SOBOL: Sample value should be <= 1.0"

	var expected_sobol_samples: Array[float] = [0.0, 0.5, 0.75, 0.25, 0.375]
	for i in range(ndraws):
		assert_float(samples[i]).is_equal_approx(expected_sobol_samples[i], 0.00001)
			# "SOBOL: Sample %d should match expected value" % i


func test_generate_samples_sobol_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL)
	assert_int(samples.size()).is_equal(0) # "SOBOL: Should return an empty array for ndraws = 0"


func test_generate_samples_sobol_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL)
	assert_int(samples.size()).is_equal(0) # "SOBOL: Should return an empty array for ndraws < 0"


func test_generate_samples_sobol_random_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM)

	assert_int(samples.size()).is_equal(ndraws) # "SOBOL_RANDOM: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "SOBOL_RANDOM: Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "SOBOL_RANDOM: Sample value should be <= 1.0"


func test_generate_samples_sobol_random_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 12345
	var samples1: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, seed)
	var samples2: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, seed)

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
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM)
	assert_int(samples.size()).is_equal(0) # "SOBOL_RANDOM: Should return an empty array for ndraws = 0"


func test_generate_samples_sobol_random_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM)
	assert_int(samples.size()).is_equal(0) # "SOBOL_RANDOM: Should return an empty array for ndraws < 0"


func test_generate_samples_halton_basic() -> void:
	var ndraws: int = 5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON)

	assert_int(samples.size()).is_equal(ndraws) # "HALTON: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "HALTON: Sample value should be >= 0.0"
		assert_float(sample_val).is_less_equal(1.0) # "HALTON: Sample value should be < 1.0 (strictly for base > 1)"

	var expected_halton_samples: Array[float] = [0.5, 0.25, 0.75, 0.125, 0.625]
	for i in range(ndraws):
		assert_float(samples[i]).is_equal_approx(expected_halton_samples[i], 0.00001)
			# "HALTON: Sample %d should match expected value" % i


func test_generate_samples_halton_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON)
	assert_int(samples.size()).is_equal(0) # "HALTON: Should return an empty array for ndraws = 0"


func test_generate_samples_halton_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON)
	assert_int(samples.size()).is_equal(0) # "HALTON: Should return an empty array for ndraws < 0"


func test_generate_samples_halton_random_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM)

	assert_int(samples.size()).is_equal(ndraws) # "HALTON_RANDOM: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "HALTON_RANDOM: Sample value should be >= 0.0"
		assert_float(sample_val).is_less(1.0) # "HALTON_RANDOM: Sample value should be < 1.0 (due to fmod)"


func test_generate_samples_halton_random_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 54321
	var samples1: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM, seed)
	var samples2: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM, seed)

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
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM)
	assert_int(samples.size()).is_equal(0) # "HALTON_RANDOM: Should return an empty array for ndraws = 0"


func test_generate_samples_halton_random_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM)
	assert_int(samples.size()).is_equal(0) # "HALTON_RANDOM: Should return an empty array for ndraws < 0"


func test_generate_samples_latin_hypercube_basic() -> void:
	var ndraws: int = 10
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE)

	assert_int(samples.size()).is_equal(ndraws) # "LHS: Should return the correct number of samples"
	for sample_val in samples:
		assert_float(sample_val).is_greater_equal(0.0) # "LHS: Sample value should be >= 0.0"
		assert_float(sample_val).is_less(1.0) # "LHS: Sample value should be < 1.0"


func test_generate_samples_latin_hypercube_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 67890
	var samples1: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)
	var samples2: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)

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
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 123)
	
	assert_int(samples.size()).is_equal(ndraws) # "LHS (strat): Correct number of samples"
	
	var sorted_samples: Array[float] = samples.duplicate()
	sorted_samples.sort()
	
	for i in range(ndraws):
		var lower_bound: float = float(i) / float(ndraws)
		var upper_bound: float = float(i + 1) / float(ndraws)
		assert_float(sorted_samples[i]).is_greater_equal(lower_bound)
		if i == ndraws -1 and sorted_samples[i] > (upper_bound - 0.000001) and sorted_samples[i] < (upper_bound + 0.000001): 
			assert_float(sorted_samples[i]).is_less_equal(upper_bound)
		else:
			assert_float(sorted_samples[i]).is_less(upper_bound)


func test_generate_samples_latin_hypercube_ndraws_zero() -> void:
	var ndraws: int = 0
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE)
	assert_int(samples.size()).is_equal(0) # "LHS: Should return an empty array for ndraws = 0"


func test_generate_samples_latin_hypercube_ndraws_negative() -> void:
	var ndraws: int = -5
	var samples: Array[float] = StatMath.SamplingGen.generate_samples_1d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE)
	assert_int(samples.size()).is_equal(0) # "LHS: Should return an empty array for ndraws < 0"


func test_rng_determinism_with_set_seed() -> void:
	const TEST_SEED: int = 888
	const NDRAWS: int = 5
	
	var results_run1_random: Array[float]
	var results_run1_lhs: Array[float]
	var results_run2_random: Array[float]
	var results_run2_lhs: Array[float]

	StatMath.set_seed(TEST_SEED)
	results_run1_random = StatMath.SamplingGen.generate_samples_1d(NDRAWS, StatMath.SamplingGen.SamplingMethod.RANDOM)
	results_run1_lhs = StatMath.SamplingGen.generate_samples_1d(NDRAWS, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE)

	StatMath.set_seed(TEST_SEED) 
	results_run2_random = StatMath.SamplingGen.generate_samples_1d(NDRAWS, StatMath.SamplingGen.SamplingMethod.RANDOM)
	results_run2_lhs = StatMath.SamplingGen.generate_samples_1d(NDRAWS, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE)

	assert_int(results_run1_random.size()).is_equal(NDRAWS) 
	assert_int(results_run2_random.size()).is_equal(NDRAWS) 
	for i in range(NDRAWS):
		assert_float(results_run1_random[i]).is_equal(results_run2_random[i])

	assert_int(results_run1_lhs.size()).is_equal(NDRAWS) 
	assert_int(results_run2_lhs.size()).is_equal(NDRAWS) 
	for i in range(NDRAWS):
		assert_float(results_run1_lhs[i]).is_equal(results_run2_lhs[i])


# --- Test Cases for 2D Sample Generation ---

# test_generate_samples_invalid_dimensions removed as it's no longer applicable.

func test_generate_samples_ndraws_zero_2d() -> void:
	var ndraws: int = 0
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	assert_int(samples.size()).is_equal(0)


# --- RANDOM 2D Tests ---
func test_generate_samples_random_2d_basic() -> void:
	var ndraws: int = 10
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM)
	
	assert_int(samples.size()).is_equal(ndraws)
	for sample_vec in samples:
		assert_float(sample_vec.x).is_greater_equal(0.0)
		assert_float(sample_vec.x).is_less_equal(1.0)
		assert_float(sample_vec.y).is_greater_equal(0.0)
		assert_float(sample_vec.y).is_less_equal(1.0)

func test_generate_samples_random_2d_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 12321
	var sensitivity: Vector2 = Vector2(0.00001, 0.00001)
	var samples1: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM, seed)
	var samples2: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.RANDOM, seed)

	assert_int(samples1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(samples1[i]).is_equal_approx(samples2[i], sensitivity)


# --- SOBOL 2D Tests ---
func test_generate_samples_sobol_2d_basic() -> void:
	var ndraws: int = 5
	var sensitivity: Vector2 = Vector2(0.00001, 0.00001)
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL)

	assert_int(samples.size()).is_equal(ndraws)
	for sample_vec in samples:
		assert_float(sample_vec.x).is_greater_equal(0.0)
		assert_float(sample_vec.x).is_less_equal(1.0)
		assert_float(sample_vec.y).is_greater_equal(0.0)
		assert_float(sample_vec.y).is_less_equal(1.0)

	var expected_sobol_2d_samples: Array[Vector2] = [
		Vector2(0.0, 0.0),
		Vector2(0.5, 0.5),
		Vector2(0.75, 0.75),
		Vector2(0.25, 0.25),
		Vector2(0.375, 0.625) 
	]
	assert_int(samples.size()).is_equal(expected_sobol_2d_samples.size())
	for i in range(min(ndraws, expected_sobol_2d_samples.size())):
		assert_vector(samples[i]).is_equal_approx(expected_sobol_2d_samples[i], sensitivity)

func test_generate_samples_sobol_random_2d_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 56789
	var sensitivity: Vector2 = Vector2(0.00001, 0.00001)
	var samples1: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, seed)
	var samples2: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM, seed)

	assert_int(samples1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(samples1[i]).is_equal_approx(samples2[i], sensitivity)
		assert_float(samples1[i].x).is_between(0.0, 1.0)
		assert_float(samples1[i].y).is_between(0.0, 1.0)


# --- HALTON 2D Tests ---
func test_generate_samples_halton_2d_basic() -> void:
	var ndraws: int = 5
	var sensitivity: Vector2 = Vector2(0.00001, 0.00001)
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON)

	assert_int(samples.size()).is_equal(ndraws)
	for sample_vec in samples:
		assert_float(sample_vec.x).is_greater_equal(0.0)
		assert_float(sample_vec.x).is_less(1.0)
		assert_float(sample_vec.y).is_greater_equal(0.0)
		assert_float(sample_vec.y).is_less(1.0)
	
	var expected_halton_2d_samples: Array[Vector2] = [
		Vector2(0.5, 1.0/3.0),
		Vector2(0.25, 2.0/3.0),
		Vector2(0.75, 1.0/9.0),
		Vector2(0.125, 4.0/9.0),
		Vector2(0.625, 7.0/9.0)
	]
	assert_int(samples.size()).is_equal(expected_halton_2d_samples.size())
	for i in range(min(ndraws, expected_halton_2d_samples.size())):
		assert_vector(samples[i]).is_equal_approx(expected_halton_2d_samples[i], sensitivity)

func test_generate_samples_halton_random_2d_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 98765
	var sensitivity: Vector2 = Vector2(0.00001, 0.00001)
	var samples1: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM, seed)
	var samples2: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.HALTON_RANDOM, seed)

	assert_int(samples1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(samples1[i]).is_equal_approx(samples2[i], sensitivity)
		assert_float(samples1[i].x).is_greater_equal(0.0)
		assert_float(samples1[i].x).is_less(1.0)
		assert_float(samples1[i].y).is_greater_equal(0.0)
		assert_float(samples1[i].y).is_less(1.0)


# --- LATIN HYPERCUBE 2D Tests ---
func test_generate_samples_latin_hypercube_2d_basic() -> void:
	var ndraws: int = 10
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE)
	
	assert_int(samples.size()).is_equal(ndraws)
	for sample_vec in samples:
		assert_float(sample_vec.x).is_greater_equal(0.0)
		assert_float(sample_vec.x).is_less(1.0)
		assert_float(sample_vec.y).is_greater_equal(0.0)
		assert_float(sample_vec.y).is_less(1.0)

func test_generate_samples_latin_hypercube_2d_with_seed() -> void:
	var ndraws: int = 5
	var seed: int = 24680
	var sensitivity: Vector2 = Vector2(0.00001, 0.00001)
	var samples1: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)
	var samples2: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)

	assert_int(samples1.size()).is_equal(ndraws)
	for i in range(ndraws):
		assert_vector(samples1[i]).is_equal_approx(samples2[i], sensitivity)

func test_generate_samples_latin_hypercube_2d_stratification() -> void:
	var ndraws: int = 20
	var seed: int = 13579
	var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples_2d(ndraws, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, seed)
	
	assert_int(samples.size()).is_equal(ndraws)
	
	var samples_x: Array[float] = []
	samples_x.resize(ndraws)
	for i in range(ndraws): samples_x[i] = samples[i].x
	samples_x.sort()
	for i in range(ndraws):
		var lower_bound: float = float(i) / float(ndraws)
		var upper_bound: float = float(i + 1) / float(ndraws)
		assert_float(samples_x[i]).is_greater_equal(lower_bound)
		if i == ndraws -1 and abs(samples_x[i] - upper_bound) < 0.000001:
			assert_float(samples_x[i]).is_less_equal(upper_bound)
		else:
			assert_float(samples_x[i]).is_less(upper_bound)

	var samples_y: Array[float] = []
	samples_y.resize(ndraws)
	for i in range(ndraws): samples_y[i] = samples[i].y
	samples_y.sort()
	for i in range(ndraws):
		var lower_bound: float = float(i) / float(ndraws)
		var upper_bound: float = float(i + 1) / float(ndraws)
		assert_float(samples_y[i]).is_greater_equal(lower_bound)
		if i == ndraws -1 and abs(samples_y[i] - upper_bound) < 0.000001:
			assert_float(samples_y[i]).is_less_equal(upper_bound)
		else:
			assert_float(samples_y[i]).is_less(upper_bound)


# --- Test Cases for draw_without_replacement ---

# Helper function to check common properties of drawn samples
func _assert_drawn_samples_valid(samples: Array[int], deck_size: int, draw_count: int, method_name: String) -> void:
	# Check 1: Correct number of samples
	assert_int(samples.size()).is_equal(draw_count) # "%s: Should return %d samples" % [method_name, draw_count]

	if draw_count == 0: # No further checks needed for empty result
		return

	# Check 2: All samples are unique
	var unique_samples: Dictionary = {}
	for sample_val in samples:
		assert_bool(unique_samples.has(sample_val)).is_false() # "%s: Sample %d should be unique" % [method_name, sample_val]
		unique_samples[sample_val] = true
		# Check 3: All samples are within the valid range [0, deck_size - 1]
		assert_int(sample_val).is_greater_equal(0) # "%s: Sample %d should be >= 0" % [method_name, sample_val]
		assert_int(sample_val).is_less(deck_size) # "%s: Sample %d should be < deck_size (%d)" % [method_name, sample_val, deck_size]


# --- FISHER_YATES Tests ---
func test_draw_without_replacement_fisher_yates_basic() -> void:
	var deck_size: int = 52
	var draw_count: int = 5
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "FISHER_YATES_BASIC")


func test_draw_without_replacement_fisher_yates_draw_zero() -> void:
	var deck_size: int = 52
	var draw_count: int = 0
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "FISHER_YATES_DRAW_ZERO")
	assert_int(samples.size()).is_equal(0) # "FISHER_YATES_DRAW_ZERO: Should be an empty array"


func test_draw_without_replacement_fisher_yates_draw_all() -> void:
	var deck_size: int = 10
	var draw_count: int = 10
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "FISHER_YATES_DRAW_ALL")


func test_draw_without_replacement_fisher_yates_deck_zero_draw_zero() -> void:
	var deck_size: int = 0
	var draw_count: int = 0
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "FISHER_YATES_DECK_ZERO_DRAW_ZERO")
	assert_int(samples.size()).is_equal(0) # "FISHER_YATES_DECK_ZERO_DRAW_ZERO: Should be an empty array"


# --- RESERVOIR Tests ---
func test_draw_without_replacement_reservoir_basic() -> void:
	var deck_size: int = 52
	var draw_count: int = 5
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.RESERVOIR)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "RESERVOIR_BASIC")


func test_draw_without_replacement_reservoir_draw_zero() -> void:
	var deck_size: int = 52
	var draw_count: int = 0
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.RESERVOIR)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "RESERVOIR_DRAW_ZERO")
	assert_int(samples.size()).is_equal(0) # "RESERVOIR_DRAW_ZERO: Should be an empty array"


func test_draw_without_replacement_reservoir_draw_all() -> void:
	var deck_size: int = 10
	var draw_count: int = 10
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.RESERVOIR)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "RESERVOIR_DRAW_ALL")


func test_draw_without_replacement_reservoir_deck_zero_draw_zero() -> void:
	var deck_size: int = 0
	var draw_count: int = 0
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.RESERVOIR)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "RESERVOIR_DECK_ZERO_DRAW_ZERO")
	assert_int(samples.size()).is_equal(0) # "RESERVOIR_DECK_ZERO_DRAW_ZERO: Should be an empty array"


# --- SELECTION_TRACKING Tests ---
func test_draw_without_replacement_selection_tracking_basic() -> void:
	var deck_size: int = 52
	var draw_count: int = 5
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.SELECTION_TRACKING)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "SELECTION_TRACKING_BASIC")


func test_draw_without_replacement_selection_tracking_draw_zero() -> void:
	var deck_size: int = 52
	var draw_count: int = 0
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.SELECTION_TRACKING)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "SELECTION_TRACKING_DRAW_ZERO")
	assert_int(samples.size()).is_equal(0) # "SELECTION_TRACKING_DRAW_ZERO: Should be an empty array"


func test_draw_without_replacement_selection_tracking_draw_all() -> void:
	var deck_size: int = 10
	var draw_count: int = 10
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.SELECTION_TRACKING)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "SELECTION_TRACKING_DRAW_ALL")


func test_draw_without_replacement_selection_tracking_deck_zero_draw_zero() -> void:
	var deck_size: int = 0
	var draw_count: int = 0
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.SELECTION_TRACKING)
	_assert_drawn_samples_valid(samples, deck_size, draw_count, "SELECTION_TRACKING_DECK_ZERO_DRAW_ZERO")
	assert_int(samples.size()).is_equal(0) # "SELECTION_TRACKING_DECK_ZERO_DRAW_ZERO: Should be an empty array"


# --- Invalid Input Tests for draw_without_replacement ---
func test_draw_without_replacement_invalid_draw_count_negative() -> void:
	var deck_size: int = 10
	var draw_count: int = -1
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	assert_int(samples.size()).is_equal(0) # "INVALID_DRAW_NEGATIVE: Should return empty array for negative draw_count"


func test_draw_without_replacement_invalid_deck_size_negative() -> void:
	var deck_size: int = -10
	var draw_count: int = 5
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	assert_int(samples.size()).is_equal(0) # "INVALID_DECK_NEGATIVE: Should return empty array for negative deck_size"


func test_draw_without_replacement_invalid_draw_count_greater_than_deck_size() -> void:
	var deck_size: int = 5
	var draw_count: int = 10
	var samples: Array[int] = StatMath.SamplingGen.draw_without_replacement(deck_size, draw_count, StatMath.SamplingGen.SamplingMethod.FISHER_YATES)
	assert_int(samples.size()).is_equal(0) # "INVALID_DRAW_GREATER: Should return empty array if draw_count > deck_size"
