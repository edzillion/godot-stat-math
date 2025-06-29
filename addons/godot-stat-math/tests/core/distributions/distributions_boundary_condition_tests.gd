# res://addons/godot-stat-math/tests/core/distributions/distributions_boundary_condition_tests.gd
class_name DistributionsBoundaryConditionTests extends GdUnitTestSuite

# =============================================================================
# BOUNDARY CONDITION TESTS - DETERMINISTIC EDGE CASES
# =============================================================================

func test_randi_bernoulli_deterministic_cases() -> void:
	# Test deterministic boundary cases with fixed seeds
	StatMath.set_global_seed(DistributionsTestData.TEST_SEEDS.primary)
	
	# p = 0.0 should always return 0
	var result_zero: int = StatMath.Distributions.randi_bernoulli(DistributionsTestData.BOUNDARY_VALUES.probability_zero)
	assert_int(result_zero).is_equal(0)
	
	# p = 1.0 should always return 1  
	var result_one: int = StatMath.Distributions.randi_bernoulli(DistributionsTestData.BOUNDARY_VALUES.probability_one)
	assert_int(result_one).is_equal(1)

func test_randi_bernoulli_p_zero() -> void:
	var result: int = StatMath.Distributions.randi_bernoulli(0.0)
	assert_int(result).is_equal(0)


func test_randi_bernoulli_p_one() -> void:
	var result: int = StatMath.Distributions.randi_bernoulli(1.0)
	assert_int(result).is_equal(1)


func test_randi_bernoulli_p_half() -> void:
	var result: int = StatMath.Distributions.randi_bernoulli(0.5)
	assert_bool(result == 0 or result == 1).is_true() # Result should be 0 or 1 for p=0.5 


# Tests for randi_binomial
func test_randi_binomial_p_zero() -> void:
	var result: int = StatMath.Distributions.randi_binomial(0.0, 10)
	assert_int(result).is_equal(0)


func test_randi_binomial_p_one() -> void:
	var n_trials: int = 5
	var result: int = StatMath.Distributions.randi_binomial(1.0, n_trials)
	assert_int(result).is_equal(n_trials)


func test_randi_binomial_n_zero() -> void:
	var result: int = StatMath.Distributions.randi_binomial(0.5, 0)
	assert_int(result).is_equal(0)


func test_randi_binomial_deterministic_cases() -> void:
	# Test deterministic boundary cases with fixed seeds
	StatMath.set_global_seed(DistributionsTestData.TEST_SEEDS.primary)
	
	# p = 0.0 should always return 0
	var result_zero: int = StatMath.Distributions.randi_binomial(
		DistributionsTestData.BOUNDARY_VALUES.probability_zero, 
		DistributionsTestData.BOUNDARY_VALUES.trials_five
	)
	assert_int(result_zero).is_equal(0)
	
	# p = 1.0 should return n_trials
	var n_trials: int = DistributionsTestData.BOUNDARY_VALUES.trials_five
	var result_one: int = StatMath.Distributions.randi_binomial(
		DistributionsTestData.BOUNDARY_VALUES.probability_one, 
		n_trials
	)
	assert_int(result_one).is_equal(n_trials)
	
	# n = 0 should always return 0
	var result_zero_trials: int = StatMath.Distributions.randi_binomial(
		DistributionsTestData.BOUNDARY_VALUES.probability_half, 
		DistributionsTestData.BOUNDARY_VALUES.trials_zero
	)
	assert_int(result_zero_trials).is_equal(0)


# Tests for randi_geometric
func test_randi_geometric_deterministic_cases() -> void:
	# Test deterministic boundary cases with fixed seeds
	StatMath.set_global_seed(DistributionsTestData.TEST_SEEDS.primary)
	
	# p = 1.0 should always return 1 (first trial succeeds)
	var result_one: int = StatMath.Distributions.randi_geometric(DistributionsTestData.BOUNDARY_VALUES.probability_one)
	assert_int(result_one).is_equal(1)
	
	# p = 0.5 with fixed seed should produce deterministic result >= 1
	var result_half: int = StatMath.Distributions.randi_geometric(DistributionsTestData.BOUNDARY_VALUES.probability_half)
	assert_bool(result_half >= 1).is_true()
	
	# Very small p should produce large result or int64.max
	var p_very_small: float = StatMath.STRESS_TEST_BOUNDARY
	var result_small: int = StatMath.Distributions.randi_geometric(p_very_small)
	assert_bool(result_small > DistributionsTestData.BOUNDARY_VALUES.large_positive or result_small == StatMath.INT64_MAX_VAL).is_true()


# Tests for randi_poisson
func test_randi_poisson_deterministic_cases() -> void:
	# Test deterministic behavior with fixed seeds
	StatMath.set_global_seed(DistributionsTestData.TEST_SEEDS.primary)
	
	# Typical lambda case - result should be non-negative
	var result_typical: int = StatMath.Distributions.randi_poisson(DistributionsTestData.STATISTICAL_PARAMS.poisson_lambda)
	assert_bool(result_typical >= 0).is_true()
	
	# Small lambda case - result should be non-negative
	var result_small: int = StatMath.Distributions.randi_poisson(DistributionsTestData.STATISTICAL_PARAMS.poisson_small_lambda)
	assert_bool(result_small >= 0).is_true()


# Tests for randi_pseudo
func test_randi_pseudo_deterministic_cases() -> void:
	# Test deterministic boundary cases with fixed seeds
	StatMath.set_global_seed(DistributionsTestData.TEST_SEEDS.primary)
	
	# c_param = 1.0 should return 0 (loop condition false initially)
	var result_one: int = StatMath.Distributions.randi_pseudo(DistributionsTestData.PSEUDO_PARAMS.c_param_one)
	assert_int(result_one).is_equal(0)
	
	# c_param = 0.5 should always return 1
	var result_half: int = StatMath.Distributions.randi_pseudo(DistributionsTestData.PSEUDO_PARAMS.c_param_half)
	assert_int(result_half).is_equal(DistributionsTestData.PSEUDO_PARAMS.expected_trials_for_half)
	
	# c_param = 0.3 should return between 1 and 3
	var result_third: int = StatMath.Distributions.randi_pseudo(DistributionsTestData.PSEUDO_PARAMS.c_param_third)
	assert_bool(result_third >= DistributionsTestData.PSEUDO_PARAMS.expected_trials_for_third.min and 
				result_third <= DistributionsTestData.PSEUDO_PARAMS.expected_trials_for_third.max).is_true()


# Tests for randi_seige
func test_randi_seige_deterministic_cases() -> void:
	# Test deterministic cases with fixed seeds
	StatMath.set_global_seed(DistributionsTestData.TEST_SEEDS.primary)
	
	# Initial capture guaranteed case
	var params_guaranteed := DistributionsTestData.SIEGE_PARAMS.initial_capture_guaranteed
	var result_guaranteed: int = StatMath.Distributions.randi_seige(
		params_guaranteed.w, params_guaranteed.c_0, params_guaranteed.c_win, params_guaranteed.c_lose
	)
	assert_bool(result_guaranteed >= 1 and result_guaranteed <= params_guaranteed.expected_max_trials).is_true()
	
	# Guaranteed win and capture case
	var params_win := DistributionsTestData.SIEGE_PARAMS.guaranteed_win_and_capture
	var result_win: int = StatMath.Distributions.randi_seige(
		params_win.w, params_win.c_0, params_win.c_win, params_win.c_lose
	)
	assert_int(result_win).is_equal(params_win.expected_trials)
	
	# Typical case - should be at least 1
	var params_typical := DistributionsTestData.SIEGE_PARAMS.typical_case
	var result_typical: int = StatMath.Distributions.randi_seige(
		params_typical.w, params_typical.c_0, params_typical.c_win, params_typical.c_lose
	)
	assert_bool(result_typical >= 1).is_true()
	
	# No change case - should eventually capture
	var params_no_change := DistributionsTestData.SIEGE_PARAMS.no_change_case
	var result_no_change: int = StatMath.Distributions.randi_seige(
		params_no_change.w, params_no_change.c_0, params_no_change.c_win, params_no_change.c_lose
	)
	assert_bool(result_no_change >= 1).is_true()


 
