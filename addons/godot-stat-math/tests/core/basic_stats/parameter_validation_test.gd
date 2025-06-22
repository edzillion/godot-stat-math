# res://addons/godot-stat-math/tests/core/basic_stats/parameter_validation_test.gd
class_name BasicStatsParameterValidationTest extends GdUnitTestSuite


# Import test data for data-driven tests
const BASIC_STATS_TEST_DATA = preload("res://addons/godot-stat-math/tables/basic_stats_test_data.gd")

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

func test_mean_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.mean([])
	await assert_error(test_call).is_push_error("Cannot calculate mean of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.mean([])
	assert_bool(is_nan(result)).is_true()

func test_median_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median([])
	await assert_error(test_call).is_push_error("Cannot calculate median of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.median([])
	assert_bool(is_nan(result)).is_true()

func test_variance_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.variance([])
	await assert_error(test_call).is_push_error("Cannot calculate variance of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.variance([])
	assert_bool(is_nan(result)).is_true()

func test_standard_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.standard_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate standard deviation of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.standard_deviation([])
	assert_bool(is_nan(result)).is_true()

func test_sample_variance_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var test_call: Callable = func():
		StatMath.BasicStats.sample_variance(data)
	await assert_error(test_call).is_push_error("Cannot calculate sample variance with fewer than 2 data points. Received size: 1")

func test_sample_standard_deviation_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var test_call: Callable = func():
		StatMath.BasicStats.sample_standard_deviation(data)
	await assert_error(test_call).is_push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: 1")

func test_range_spread_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.range_spread([])
	await assert_error(test_call).is_push_error("Cannot calculate range of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.range_spread([])
	assert_bool(is_nan(result)).is_true()

func test_minimum_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.minimum([])
	await assert_error(test_call).is_push_error("Cannot find minimum of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.minimum([])
	assert_bool(is_nan(result)).is_true()

func test_maximum_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.maximum([])
	await assert_error(test_call).is_push_error("Cannot find maximum of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.maximum([])
	assert_bool(is_nan(result)).is_true()

func test_median_absolute_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median_absolute_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate median absolute deviation of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.median_absolute_deviation([])
	assert_bool(is_nan(result)).is_true() 