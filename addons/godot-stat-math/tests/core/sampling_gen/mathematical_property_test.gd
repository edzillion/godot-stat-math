# res://addons/godot-stat-math/tests/core/sampling_gen/mathematical_property_test.gd
class_name SamplingGenMathematicalPropertyTest extends GdUnitTestSuite

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

## Tests mathematical properties of sampling methods
func test_sampling_uniformity_property() -> void:
	# Test that sampling methods produce uniform distributions
	var n_draws: int = 1000
	var samples: Array[float] = StatMath.HelperFunctions.convert_to_float_array(
		StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	)
	
	# Check that samples are uniformly distributed in [0,1]
	var min_val: float = StatMath.BasicStats.minimum(samples)
	var max_val: float = StatMath.BasicStats.maximum(samples)
	
	assert_float(min_val).is_greater_equal(0.0)
	assert_float(max_val).is_less_equal(1.0)
	
	# Check that samples are well-distributed (not clustered)
	var mean_val: float = StatMath.BasicStats.mean(samples)
	assert_float(mean_val).is_equal_approx(0.5, 0.1)  # Should be close to 0.5 for uniform distribution

## Tests deterministic properties of quasi-random sequences
func test_quasi_random_determinism() -> void:
	# Test that quasi-random sequences are deterministic
	var n_draws: int = 10
	var samples1: Array[float] = StatMath.HelperFunctions.convert_to_float_array(
		StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	)
	var samples2: Array[float] = StatMath.HelperFunctions.convert_to_float_array(
		StatMath.SamplingGen.generate_samples(n_draws, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
	)
	
	# Should be identical
	for i in range(n_draws):
		assert_float(samples1[i]).is_equal_approx(samples2[i], StatMath.DETERMINISM_TOLERANCE)


# --- HELPER FUNCTIONS ---

 
