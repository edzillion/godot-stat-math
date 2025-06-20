# res://addons/godot-stat-math/tables/cdf_pdf_integration_test_data.gd

# =============================================================================
# CDF-PDF INTEGRATION TEST DATA
# =============================================================================

## Test points for derivative relationship testing
const DERIVATIVE_TEST_POINTS: Dictionary = {
	StatMath.SupportedDistributions.NORMAL: [-2.0, -1.0, 0.0, 1.0, 2.0],
	StatMath.SupportedDistributions.EXPONENTIAL: [0.1, 0.5, 1.0, 2.0, 5.0],  # Avoid x=0 for stability
	StatMath.SupportedDistributions.UNIFORM: [1.5, 2.0, 2.5, 3.0, 3.5],  # Points strictly inside [a,b]
	StatMath.SupportedDistributions.BETA: [0.1, 0.3, 0.5, 0.7, 0.9],  # Points strictly inside (0,1)
	StatMath.SupportedDistributions.WEIBULL: [0.5, 1.0, 1.5, 2.0, 3.0]  # Points > 0
}

## Distribution parameters for testing
const DISTRIBUTION_PARAMETERS: Dictionary = {
	"normal_standard": {"mu": 0.0, "sigma": 1.0},
	"exponential_rate_2": {"lambda_param": 2.0},
	"uniform_1_to_4": {"a": 1.0, "b": 4.0},
	"beta_2_3": {"alpha": 2.0, "beta_param": 3.0},
	"gamma_equivalence": {"shape": 1.0, "scale": 2.0, "lambda_equiv": 0.5},
	"weibull_2_2": {"scale_param": 2.0, "shape_param": 2.0}
}

## Monotonicity testing data
const MONOTONICITY_TEST_DATA: Array[Dictionary] = [
	{
		"name": StatMath.SupportedDistributions.NORMAL, 
		"params": [0.0, 1.0], 
		"points": [-3.0, -1.0, 0.0, 1.0, 3.0]
	},
	{
		"name": StatMath.SupportedDistributions.EXPONENTIAL, 
		"params": [1.0], 
		"points": [0.1, 0.5, 1.0, 2.0, 5.0]
	},
	{
		"name": StatMath.SupportedDistributions.UNIFORM, 
		"params": [1.0, 4.0], 
		"points": [1.0, 1.5, 2.5, 3.5, 4.0]
	},
	{
		"name": StatMath.SupportedDistributions.BETA, 
		"params": [2.0, 3.0], 
		"points": [0.0, 0.25, 0.5, 0.75, 1.0]
	},
	{
		"name": StatMath.SupportedDistributions.GAMMA, 
		"params": [2.0, 1.5], 
		"points": [0.1, 1.0, 2.0, 4.0, 6.0]
	},
	{
		"name": StatMath.SupportedDistributions.WEIBULL, 
		"params": [2.0, 2.0], 
		"points": [0.1, 1.0, 2.0, 3.0, 4.0]
	}
]

## Cross-function consistency test data
const CROSS_FUNCTION_TEST_DATA: Array[Dictionary] = [
	{
		"name": StatMath.SupportedDistributions.NORMAL, 
		"cdf_params": [1.5, 0.0, 1.0], 
		"pdf_params": [1.5, 0.0, 1.0], 
		"ppf_params": [0.0, 1.0]
	},
	{
		"name": StatMath.SupportedDistributions.EXPONENTIAL, 
		"cdf_params": [2.0, 1.0], 
		"pdf_params": [2.0, 1.0], 
		"ppf_params": [1.0]
	},
	{
		"name": StatMath.SupportedDistributions.UNIFORM, 
		"cdf_params": [2.5, 1.0, 4.0], 
		"pdf_params": [2.5, 1.0, 4.0], 
		"ppf_params": [1.0, 4.0]
	}
]

## End-to-end workflow test parameters
const WORKFLOW_TEST_PARAMETERS: Dictionary = {
	"sample_size": 1000,
	"normal_params": {"mu": 5.0, "sigma": 2.0},
	"test_seed": 12345,
	"percentile_value": 95.0
}

## Extreme value testing parameters
const EXTREME_VALUE_PARAMETERS: Dictionary = {
	"small_sigma": 1e-6,
	"large_sigma": 1e6,
	"tiny_lambda": 1e-8,
	"huge_lambda": 1e8
} 
