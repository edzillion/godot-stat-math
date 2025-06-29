# res://addons/godot-stat-math/tables/distributions_test_data.gd
class_name DistributionsTestData extends RefCounted

## Test data constants to eliminate magic numbers from distributions tests

# =============================================================================
# BASIC TEST PARAMETERS
# =============================================================================

## Sample sizes for statistical validation tests
const SAMPLE_SIZES := {
	"small": 100,
	"medium": 1000,
	"large": 2000,
	"xl": 5000
}

## Statistical test parameters
const STATISTICAL_PARAMS := {
	"binomial_n_trials": 20,
	"binomial_p": 0.4,
	"geometric_p_half": 0.5,
	"poisson_lambda": 3.0,
	"poisson_small_lambda": 0.1,
	"uniform_range_small": {"min": 1, "max": 10},
	"uniform_range_negative": {"min": -10, "max": -1},
	"uniform_range_mixed": {"min": -5, "max": 5},
	"uniform_range_single": {"min": 42, "max": 42},
	"uniform_range_large": {"min": 1, "max": 100},
	"exponential_rate": 2.0,
	"chi_squared_df": 3.0
}

## Test seeds for deterministic behavior
const TEST_SEEDS := {
	"primary": 42,
	"secondary": 123,
	"tertiary": 456
}

## Boundary test values
const BOUNDARY_VALUES := {
	"probability_zero": 0.0,
	"probability_half": 0.5,
	"probability_one": 1.0,
	"trials_zero": 0,
	"trials_five": 5,
	"single_trial": 1,
	"small_iterations": 3,
	"medium_iterations": 10,
	"large_positive": 1000,
	"sentinel_invalid": -1
}

## Invalid parameter test cases
const INVALID_PARAMS := {
	"probability_negative": -0.1,
	"probability_above_one": 1.1,
	"negative_trials": -1,
	"negative_lambda": -1.0,
	"invalid_range_reversed": {"min": 10, "max": 5},
	"invalid_range_float_reversed": {"a": 5.0, "b": 2.0}
}

# =============================================================================
# PSEUDO DISTRIBUTION TEST CASES
# =============================================================================

## randi_pseudo test parameters
const PSEUDO_PARAMS := {
	"c_param_one": 1.0,
	"c_param_third": 0.3,
	"c_param_half": 0.5,
	"expected_trials_for_third": {"min": 1, "max": 3},
	"expected_trials_for_half": 1
}

# =============================================================================
# SIEGE DISTRIBUTION TEST CASES  
# =============================================================================

## randi_seige test parameters
const SIEGE_PARAMS := {
	"initial_capture_guaranteed": {
		"w": 0.5, "c_0": 1.0, "c_win": 0.1, "c_lose": -0.1,
		"expected_max_trials": 2
	},
	"guaranteed_win_and_capture": {
		"w": 1.0, "c_0": 0.0, "c_win": 1.0, "c_lose": 0.0,
		"expected_trials": 1
	},
	"typical_case": {
		"w": 0.5, "c_0": 0.1, "c_win": 0.2, "c_lose": -0.05
	},
	"no_change_case": {
		"w": 0.5, "c_0": 0.1, "c_win": 0.0, "c_lose": 0.0
	}
}

# =============================================================================
# FLOAT UNIFORM DISTRIBUTION TEST CASES
# =============================================================================

## randf_uniform test parameters  
const FLOAT_UNIFORM_PARAMS := {
	"equal_bounds": {"a": 5.0, "b": 5.0},
	"typical_range": {"a": 2.0, "b": 5.0},
	"negative_range": {"a": -5.0, "b": -2.0},
	"mixed_sign_range": {"a": -3.0, "b": 3.0}
}

# =============================================================================
# NORMAL DISTRIBUTION TEST CASES
# =============================================================================

## randf_normal test parameters
const NORMAL_PARAMS := {
	"standard": {"mu": 0.0, "sigma": 1.0},
	"custom": {"mu": 5.0, "sigma": 2.0},
	"small_sigma": {"mu": 0.0, "sigma": 0.1},
	"large_sigma": {"mu": 0.0, "sigma": 10.0},
	"zero_sigma_invalid": {"mu": 0.0, "sigma": 0.0},
	"negative_sigma_invalid": {"mu": 0.0, "sigma": -1.0}
}

## Expected ranges for normal distribution validation (approximately ±3σ)
const NORMAL_EXPECTED_RANGES := {
	"standard_3_sigma": {"min": -3.0, "max": 3.0},
	"custom_3_sigma": {"min": -1.0, "max": 11.0},  # 5 ± 3*2
	"tolerance_range_factor": 3.0
}

# =============================================================================
# CHI-SQUARED DISTRIBUTION TEST CASES
# =============================================================================

## randf_chi_squared test parameters
const CHI_SQUARED_PARAMS := {
	"typical_df": 3.0,
	"single_df": 1.0,
	"large_df": 10.0,
	"zero_df_invalid": 0.0,
	"negative_df_invalid": -1.0
}

# =============================================================================
# STATISTICAL VALIDATION PARAMETERS
# =============================================================================

## Parameters for validating statistical properties
const VALIDATION_PARAMS := {
	"confidence_multiplier": 4.0,  # ~4σ confidence for statistical tests
	"min_iterations_for_convergence": 1000,
	"max_expected_deviation_percent": 0.05,  # 5% tolerance for sample statistics
	"determinism_test_iterations": 10
} 