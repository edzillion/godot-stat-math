# res://addons/godot-stat-math/tables/sampling_test_data.gd

# =============================================================================
# SAMPLING TEST CONSTANTS AND PARAMETERS
# =============================================================================

## Test dimensions and sample counts for sampling algorithms
const TEST_DIMENSIONS: Dictionary = {
	"basic_1d": 1,
	"basic_2d": 2,
	"basic_5d": 5,
	"basic_nd": 4,
	"high_dimensions": 20,
	"sobol_halton_test": 3
}

## Sample counts for different test scenarios
const SAMPLE_COUNTS: Dictionary = {
	"small_test": 3,
	"medium_test": 5,
	"standard_test": 10,
	"large_test": 100,
	"royal_flush_trials": 1000
}

## Deck and shuffle parameters
const DECK_PARAMETERS: Dictionary = {
	"small_deck": 5,
	"medium_deck": 10,
	"standard_deck": 52
}

## Seed values for deterministic testing
const TEST_SEEDS: Dictionary = {
	"standard_seed": 42,
	"alternate_seed": 12345,
	"sobol_point_42": 42,
	"sobol_point_43": 43
}

## Threshold for unique hand validation
const MINIMUM_UNIQUE_HAND_RATIO: float = 0.8

# =============================================================================
# EXPECTED DETERMINISTIC SEQUENCES
# =============================================================================

## Expected Sobol sequence values for 1D generation (first 5 points)
const EXPECTED_SOBOL_1D: Array[float] = [0.0, 0.5, 0.75, 0.25, 0.375]

## Expected Halton sequence values for 1D generation with base 2 (first 5 points)  
const EXPECTED_HALTON_1D: Array[float] = [0.5, 0.25, 0.75, 0.125, 0.625]

## Expected Sobol sequence values for 2D generation (first 5 points)
const EXPECTED_SOBOL_2D: Array[Vector2] = [
	Vector2(0.0, 0.0),
	Vector2(0.5, 0.5), 
	Vector2(0.75, 0.5),
	Vector2(0.25, 0.0),
	Vector2(0.375, 0.125)
]

# =============================================================================
# DISTRIBUTION TEST PARAMETERS
# =============================================================================

## Parameters for distribution-specific sampling tests
const DISTRIBUTION_PARAMS: Dictionary = {
	"normal_test": {"mean": 0.0, "std_dev": 1.0},
	"exponential_test": {"lambda": 1.0},
	"gamma_test": {"shape": 2.0, "scale": 1.0}
}

## Statistical validation parameters  
const STATISTICAL_VALIDATION: Dictionary = {
	"confidence_multiplier": 4.0,  # 4σ confidence interval
	"minimum_sample_size": 100,
	"maximum_sample_size": 10000
} 