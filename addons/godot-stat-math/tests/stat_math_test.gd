# addons/godot-stat-math/tests/stat_math_test.gd
class_name StatMathTest extends GdUnitTestSuite

# Test project settings key name
const TEST_SEED_KEY: StringName = &"godot_stat_math_seed"

func before_each() -> void:
	# Clear any existing project setting before each test
	if ProjectSettings.has_setting(TEST_SEED_KEY):
		ProjectSettings.set_setting(TEST_SEED_KEY, null)

func after_each() -> void:
	# Clean up project settings after each test
	if ProjectSettings.has_setting(TEST_SEED_KEY):
		ProjectSettings.set_setting(TEST_SEED_KEY, null)

# --- Constants Tests ---
func test_constants_accessibility() -> void:
	# Test that all major constants are accessible
	assert_int(StatMath.INT_MAX_REPRESENTING_INF).is_equal(2147483647)
	assert_int(StatMath.INT64_MAX_VAL).is_equal(9223372036854775807)
	assert_float(StatMath.FLOAT_EPSILON).is_equal_approx(2.220446049250313e-16, 1e-20)
	assert_int(StatMath.MAX_ITERATIONS).is_equal(200)
	assert_float(StatMath.EPSILON).is_equal_approx(1.0e-9, 1e-12)

func test_lanczos_constants() -> void:
	# Test Lanczos approximation constants
	assert_float(StatMath.LANCZOS_G).is_equal_approx(7.5, 1e-7)
	assert_int(StatMath.LANCZOS_P.size()).is_equal(9)
	assert_float(StatMath.LANCZOS_P[0]).is_equal_approx(0.99999999999980993, 1e-12)

func test_error_function_constants() -> void:
	# Test error function approximation constants
	assert_float(StatMath.A1_ERR).is_equal_approx(0.254829592, 1e-7)
	assert_float(StatMath.A2_ERR).is_equal_approx(-0.284496736, 1e-7)
	assert_float(StatMath.P_ERR).is_equal_approx(0.3275911, 1e-7)

# --- Module Preloading Tests ---
func test_modules_preloaded() -> void:
	# Test that all modules are accessible
	assert_object(StatMath.Distributions).is_not_null()
	assert_object(StatMath.CdfFunctions).is_not_null()
	assert_object(StatMath.PmfPdfFunctions).is_not_null()
	assert_object(StatMath.PpfFunctions).is_not_null()
	assert_object(StatMath.ErrorFunctions).is_not_null()
	assert_object(StatMath.HelperFunctions).is_not_null()
	assert_object(StatMath.SamplingGen).is_not_null()
	assert_object(StatMath.BasicStats).is_not_null()

func test_module_function_access() -> void:
	# Test that we can access functions from modules
	# Using simple functions that don't require complex setup
	var bernoulli_result: int = StatMath.Distributions.randi_bernoulli(0.5)
	assert_bool(bernoulli_result >= 0 and bernoulli_result <= 1).is_true()
	
	var mean_result: float = StatMath.BasicStats.mean([1.0, 2.0, 3.0])
	assert_float(mean_result).is_equal_approx(2.0, 1e-7)

# --- RNG Management Tests ---
func test_get_rng_returns_valid_instance() -> void:
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_object(rng).is_not_null()
	assert_object(rng).is_instanceof(RandomNumberGenerator)

func test_get_rng_returns_same_instance() -> void:
	var rng1: RandomNumberGenerator = StatMath.get_rng()
	var rng2: RandomNumberGenerator = StatMath.get_rng()
	assert_object(rng1).is_same(rng2)

func test_set_global_seed_changes_rng() -> void:
	var original_rng: RandomNumberGenerator = StatMath.get_rng()
	var original_seed: int = original_rng.seed
	
	StatMath.set_global_seed(12345)
	
	var new_rng: RandomNumberGenerator = StatMath.get_rng()
	assert_int(new_rng.seed).is_equal(12345)
	assert_object(new_rng).is_not_same(original_rng)

func test_set_global_seed_zero_valid() -> void:
	# Setting seed to 0 is valid (0 is a legitimate seed value)
	StatMath.set_global_seed(0)
	var rng: RandomNumberGenerator = StatMath.get_rng()
	# The seed should be 0 when explicitly set to 0
	assert_int(rng.seed).is_equal(0)

func test_rng_produces_reproducible_results() -> void:
	# Test that the same seed produces the same sequence
	StatMath.set_global_seed(42)
	var rng1: RandomNumberGenerator = StatMath.get_rng()
	var value1: float = rng1.randf()
	
	StatMath.set_global_seed(42)
	var rng2: RandomNumberGenerator = StatMath.get_rng()
	var value2: float = rng2.randf()
	
	assert_float(value1).is_equal_approx(value2, 1e-10)

# --- Project Settings Integration Tests ---
func test_default_seed_behavior() -> void:
	# Simulate no project setting (should use default of 0)
	# Clear any existing setting
	if ProjectSettings.has_setting(TEST_SEED_KEY):
		ProjectSettings.set_setting(TEST_SEED_KEY, null)
	
	# Force re-initialization by creating a new StatMath instance scenario
	# Since we can't easily reinitialize the singleton, we test the logic indirectly
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_object(rng).is_not_null()

func test_project_setting_integer_seed() -> void:
	# Set a specific integer seed in project settings
	ProjectSettings.set_setting(TEST_SEED_KEY, 54321)
	
	# The setting is read during initialization, but since StatMath is already initialized,
	# we test by calling set_global_seed which should work with the same value
	StatMath.set_global_seed(54321)
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_int(rng.seed).is_equal(54321)

func test_project_setting_invalid_type() -> void:
	# Set an invalid type in project settings (string instead of int)
	ProjectSettings.set_setting(TEST_SEED_KEY, "not_an_integer")
	
	# This would trigger an error during initialization, but we can't easily test 
	# the initialization process. Instead, we verify the fallback behavior
	# by ensuring the RNG still works
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_object(rng).is_not_null()

# --- Integration Tests ---
func test_stat_math_integration_with_distributions() -> void:
	# Test that StatMath integrates properly with the Distributions module
	StatMath.set_global_seed(999)
	
	# Test normal distribution
	var normal_val: float = StatMath.Distributions.randf_normal(0.0, 1.0)
	assert_float(normal_val).is_not_equal(0.0) # Should generate some value
	
	# Test uniform distribution
	var uniform_val: int = StatMath.Distributions.randi_uniform(1, 100)
	assert_bool(uniform_val >= 1 and uniform_val <= 100).is_true()

func test_stat_math_integration_with_basic_stats() -> void:
	# Test integration with BasicStats module
	var test_data: Array[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
	
	var mean_val: float = StatMath.BasicStats.mean(test_data)
	assert_float(mean_val).is_equal_approx(3.0, 1e-7)
	
	var variance_val: float = StatMath.BasicStats.variance(test_data)
	assert_float(variance_val).is_equal_approx(2.0, 1e-7)

func test_stat_math_integration_with_helper_functions() -> void:
	# Test integration with HelperFunctions module
	var binomial_coeff: int = StatMath.HelperFunctions.binomial_coefficient(10, 3)
	assert_int(binomial_coeff).is_equal(120) # C(10,3) = 120

func test_stat_math_integration_with_error_functions() -> void:
	# Test integration with ErrorFunctions module
	var erf_val: float = StatMath.ErrorFunctions.error_function(1.0)
	assert_float(erf_val).is_equal_approx(0.84270079, 1e-6)

# --- Edge Cases and Error Handling ---
func test_rng_initialization_before_ready() -> void:
	# Test that get_rng() works even if called before _ready
	# This is already covered by other tests, but explicitly test the warning case
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_object(rng).is_not_null()

func test_large_seed_values() -> void:
	# Test with large seed values
	var large_seed: int = 2147483647 # MAX_INT
	StatMath.set_global_seed(large_seed)
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_int(rng.seed).is_equal(large_seed)

func test_negative_seed_values() -> void:
	# Test with negative seed values
	var negative_seed: int = -12345
	StatMath.set_global_seed(negative_seed)
	var rng: RandomNumberGenerator = StatMath.get_rng()
	assert_int(rng.seed).is_equal(negative_seed)

# --- Performance and Consistency Tests ---
func test_multiple_rng_calls_consistency() -> void:
	# Test that multiple calls to get_rng() are consistent
	StatMath.set_global_seed(777)
	
	var rng1: RandomNumberGenerator = StatMath.get_rng()
	var rng2: RandomNumberGenerator = StatMath.get_rng()
	var rng3: RandomNumberGenerator = StatMath.get_rng()
	
	assert_object(rng1).is_same(rng2)
	assert_object(rng2).is_same(rng3)
	assert_int(rng1.seed).is_equal(rng2.seed)
	assert_int(rng2.seed).is_equal(rng3.seed)

func test_seed_changes_affect_all_modules() -> void:
	# Test that changing the global seed affects all modules that use RNG
	StatMath.set_global_seed(1111)
	
	# Generate some values
	var dist_val1: float = StatMath.Distributions.randf_normal(0.0, 1.0)
	var uniform_val1: int = StatMath.Distributions.randi_uniform(1, 1000)
	
	# Reset to same seed
	StatMath.set_global_seed(1111)
	
	# Generate same values again
	var dist_val2: float = StatMath.Distributions.randf_normal(0.0, 1.0)
	var uniform_val2: int = StatMath.Distributions.randi_uniform(1, 1000)
	
	# Should be identical
	assert_float(dist_val1).is_equal_approx(dist_val2, 1e-10)
	assert_int(uniform_val1).is_equal(uniform_val2)

# --- Configuration Tests ---
func test_seed_variable_name_constant() -> void:
	# Test that the seed variable name constant is correct
	assert_str(str(StatMath.GODOT_STAT_MATH_SEED_VARIABLE_NAME)).is_equal("godot_stat_math_seed")

func test_default_seed_constant() -> void:
	# Test that setting seed to 0 explicitly works (seed 0 is valid)
	StatMath.set_global_seed(0)
	var rng: RandomNumberGenerator = StatMath.get_rng()
	# The seed should be 0 when explicitly set to 0
	assert_int(rng.seed).is_equal(0) 
