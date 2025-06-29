# res://addons/godot-stat-math/tests/core/sampling_gen/sampling_gen_scipy_validation_tests.gd
class_name SamplingGenScipyValidationTests extends GdUnitTestSuite

## =============================================================================
## SCIPY VALIDATION TESTS - DATA-DRIVEN
## =============================================================================
##
## This file contains actual scipy validation tests that validate our sampling
## methods against scipy.stats.qmc quasi-Monte Carlo functions.

const SAMPLING_TEST_DATA = preload("res://addons/godot-stat-math/tables/sampling_test_data.gd")

## Tests Sobol sequence generation against scipy.stats.qmc.Sobol
func test_generate_samples_sobol_scipy_validation() -> void:
	# Test 1D Sobol sequence against known scipy values
	var n_draws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	
	# These values match the expected Sobol sequence from scipy.stats.qmc.Sobol
	var expected_sobol: Array[float] = SAMPLING_TEST_DATA.EXPECTED_SOBOL_1D
	
	assert_int(typed_samples.size()).is_equal(n_draws)
	for i in range(n_draws):
		assert_float(typed_samples[i]).is_equal_approx(expected_sobol[i], StatMath.FLOAT_TOLERANCE)

## Tests Halton sequence generation against scipy.stats.qmc.Halton  
func test_generate_samples_halton_scipy_validation() -> void:
	# Test 1D Halton sequence against known scipy values
	var n_draws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.HALTON)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	
	# These values match the expected Halton sequence from scipy.stats.qmc.Halton with base 2
	var expected_halton: Array[float] = SAMPLING_TEST_DATA.EXPECTED_HALTON_1D
	
	assert_int(typed_samples.size()).is_equal(n_draws)
	for i in range(n_draws):
		assert_float(typed_samples[i]).is_equal_approx(expected_halton[i], StatMath.FLOAT_TOLERANCE)

## Tests 2D Sobol sequence generation against scipy.stats.qmc.Sobol
func test_generate_samples_sobol_2d_scipy_validation() -> void:
	# Test 2D Sobol sequence against known scipy values
	var n_draws: int = 5
	var samples: Variant = StatMath.SamplingGen.generate_samples(n_draws, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
	var typed_samples: Array[Vector2] = samples as Array[Vector2]
	
	# These values match the expected 2D Sobol sequence from scipy.stats.qmc.Sobol
	var expected_sobol_2d: Array[Vector2] = SAMPLING_TEST_DATA.EXPECTED_SOBOL_2D
	
	assert_int(typed_samples.size()).is_equal(n_draws)
	for i in range(n_draws):
		assert_float(typed_samples[i].x).is_equal_approx(expected_sobol_2d[i].x, StatMath.FLOAT_TOLERANCE)
		assert_float(typed_samples[i].y).is_equal_approx(expected_sobol_2d[i].y, StatMath.FLOAT_TOLERANCE)

## Note: Latin Hypercube sampling validation is more complex as it involves randomization
## This test validates that the stratification property is maintained
func test_generate_samples_latin_hypercube_stratification_property() -> void:
	# Latin Hypercube should maintain stratification property
	var n_draws: int = 10
	var samples: Variant = StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE, 0, 42)
	var typed_samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(samples)
	
	assert_int(typed_samples.size()).is_equal(n_draws)
	
	# Check that each stratum [i/n, (i+1)/n) contains exactly one sample
	var strata_filled: Array[bool] = []
	strata_filled.resize(n_draws)
	strata_filled.fill(false)
	
	for sample_val in typed_samples:
		assert_float(sample_val).is_between(0.0, 1.0)
		var stratum: int = int(sample_val * float(n_draws))
		stratum = min(stratum, n_draws - 1)  # Handle edge case where sample_val == 1.0
		
		# Each stratum should be used exactly once
		assert_bool(strata_filled[stratum]).is_false()
		strata_filled[stratum] = true
	
	# All strata should be filled
	for filled in strata_filled:
		assert_bool(filled).is_true() 