# addons/godot-stat-math/tests/core/basic_stats_test.gd
class_name BasicStatsTest extends GdUnitTestSuite

const FLOAT_TOLERANCE: float = 1e-7

# Test data sets
var simple_data: Array[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
var decimal_data: Array[float] = [1.5, 2.3, 1.8, 2.1, 1.9, 2.4, 1.7]
var unsorted_decimal_data: Array[float] = [1.5, 2.3, 1.8, 2.1, 1.9, 2.4, 1.7]
var single_value: Array[float] = [42.0]
var two_values: Array[float] = [10.0, 20.0]

func _ready() -> void:
	# Sorting here to satisfy median() precondition
	decimal_data.sort()

# --- Mean Tests ---
func test_mean_simple_data() -> void:
	var result: float = StatMath.BasicStats.mean(simple_data)
	assert_float(result).is_equal_approx(3.0, FLOAT_TOLERANCE)

func test_mean_decimal_data() -> void:
	var result: float = StatMath.BasicStats.mean(decimal_data)
	var expected: float = 13.7 / 7.0  # Sum is 13.7, count is 7
	assert_float(result).is_equal_approx(expected, FLOAT_TOLERANCE)

func test_mean_single_value() -> void:
	var result: float = StatMath.BasicStats.mean(single_value)
	assert_float(result).is_equal_approx(42.0, FLOAT_TOLERANCE)

func test_mean_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.mean([])
	await assert_error(test_call).is_push_error("Cannot calculate mean of empty array.")

# --- Median Tests ---
func test_median_odd_count() -> void:
	var result: float = StatMath.BasicStats.median(simple_data)
	assert_float(result).is_equal_approx(3.0, FLOAT_TOLERANCE)

func test_median_even_count() -> void:
	var even_data: Array[float] = [1.0, 2.0, 3.0, 4.0]
	var result: float = StatMath.BasicStats.median(even_data)
	assert_float(result).is_equal_approx(2.5, FLOAT_TOLERANCE)

func test_median_unsorted_data() -> void:
	# The median function requires pre-sorted data.
	var data: Array[float] = unsorted_decimal_data.duplicate()
	data.sort()
	# Sorted: [1.5, 1.7, 1.8, 1.9, 2.1, 2.3, 2.4], median is 1.9
	var result: float = StatMath.BasicStats.median(data)
	assert_float(result).is_equal_approx(1.9, FLOAT_TOLERANCE)

func test_median_single_value() -> void:
	var result: float = StatMath.BasicStats.median(single_value)
	assert_float(result).is_equal_approx(42.0, FLOAT_TOLERANCE)

func test_median_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median([])
	await assert_error(test_call).is_push_error("Cannot calculate median of empty array.")

# --- Variance Tests ---
func test_variance_simple_data() -> void:
	var result: float = StatMath.BasicStats.variance(simple_data)
	# Variance of [1,2,3,4,5] with mean 3.0 is ((1-3)²+(2-3)²+(3-3)²+(4-3)²+(5-3)²)/5 = (4+1+0+1+4)/5 = 2.0
	assert_float(result).is_equal_approx(2.0, FLOAT_TOLERANCE)

func test_variance_single_value() -> void:
	var result: float = StatMath.BasicStats.variance(single_value)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_variance_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.variance([])
	await assert_error(test_call).is_push_error("Cannot calculate variance of empty array.")

# --- Standard Deviation Tests ---
func test_standard_deviation_simple_data() -> void:
	var result: float = StatMath.BasicStats.standard_deviation(simple_data)
	assert_float(result).is_equal_approx(sqrt(2.0), FLOAT_TOLERANCE)

func test_standard_deviation_single_value() -> void:
	var result: float = StatMath.BasicStats.standard_deviation(single_value)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_standard_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.standard_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate standard deviation of empty array.")

# --- Sample Variance Tests ---
func test_sample_variance_simple_data() -> void:
	var result: float = StatMath.BasicStats.sample_variance(simple_data)
	# Sample variance uses N-1 denominator: 10/4 = 2.5
	assert_float(result).is_equal_approx(2.5, FLOAT_TOLERANCE)

func test_sample_variance_two_values() -> void:
	var result: float = StatMath.BasicStats.sample_variance(two_values)
	# Sample variance of [10, 20] with mean 15.0 is ((10-15)²+(20-15)²)/(2-1) = (25+25)/1 = 50.0
	assert_float(result).is_equal_approx(50.0, FLOAT_TOLERANCE)

func test_sample_variance_single_value() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.sample_variance(single_value)
	await assert_error(test_call).is_push_error("Cannot calculate sample variance with fewer than 2 data points. Received size: 1")

# --- Sample Standard Deviation Tests ---
func test_sample_standard_deviation_simple_data() -> void:
	var result: float = StatMath.BasicStats.sample_standard_deviation(simple_data)
	assert_float(result).is_equal_approx(sqrt(2.5), FLOAT_TOLERANCE)

func test_sample_standard_deviation_single_value() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.sample_standard_deviation(single_value)
	await assert_error(test_call).is_push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: 1")

# --- Median Absolute Deviation Tests ---
func test_median_absolute_deviation_simple_data() -> void:
	var result: float = StatMath.BasicStats.median_absolute_deviation(simple_data)
	# Median is 3.0, deviations are [2,1,0,1,2], median of deviations is 1.0
	assert_float(result).is_equal_approx(1.0, FLOAT_TOLERANCE)

func test_median_absolute_deviation_single_value() -> void:
	var result: float = StatMath.BasicStats.median_absolute_deviation(single_value)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_median_absolute_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median_absolute_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate MAD of empty array.")

# --- Range Tests ---
func test_range_spread_simple_data() -> void:
	var result: float = StatMath.BasicStats.range_spread(simple_data)
	assert_float(result).is_equal_approx(4.0, FLOAT_TOLERANCE)  # 5.0 - 1.0

func test_range_spread_single_value() -> void:
	var result: float = StatMath.BasicStats.range_spread(single_value)
	assert_float(result).is_equal_approx(0.0, FLOAT_TOLERANCE)

func test_range_spread_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.range_spread([])
	await assert_error(test_call).is_push_error("Cannot calculate range of empty array.")

# --- Minimum Tests ---
func test_minimum_simple_data() -> void:
	var result: float = StatMath.BasicStats.minimum(simple_data)
	assert_float(result).is_equal_approx(1.0, FLOAT_TOLERANCE)

func test_minimum_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.minimum([])
	await assert_error(test_call).is_push_error("Cannot find minimum of empty array.")

# --- Maximum Tests ---
func test_maximum_simple_data() -> void:
	var result: float = StatMath.BasicStats.maximum(simple_data)
	assert_float(result).is_equal_approx(5.0, FLOAT_TOLERANCE)

func test_maximum_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.maximum([])
	await assert_error(test_call).is_push_error("Cannot find maximum of empty array.")

# --- Summary Statistics Tests ---
func test_summary_statistics_structure() -> void:
	var result: Dictionary = StatMath.BasicStats.summary_statistics(simple_data)
	
	# Test that the dictionary contains all expected keys
	assert_that(result.has("mean")).is_true()
	assert_that(result.has("median")).is_true()
	assert_that(result.has("variance")).is_true()
	assert_that(result.has("standard_deviation")).is_true()
	assert_that(result.has("sample_variance")).is_true()
	assert_that(result.has("sample_standard_deviation")).is_true()
	assert_that(result.has("median_absolute_deviation")).is_true()
	assert_that(result.has("range")).is_true()
	assert_that(result.has("minimum")).is_true()
	assert_that(result.has("maximum")).is_true()
	assert_that(result.has("count")).is_true()
	
	# Test that count is always an integer
	assert_int(result["count"]).is_equal(5)

func test_summary_statistics_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.summary_statistics([])
	await assert_error(test_call).is_push_error("Cannot calculate summary statistics of empty array.")

# --- Percentile Tests ---
func test_percentile_simple_data() -> void:
	var data: Array[float] = [10.0, 20.0, 30.0, 40.0, 50.0]
	assert_float(StatMath.BasicStats.percentile(data, 0.0)).is_equal_approx(10.0, FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(data, 25.0)).is_equal_approx(20.0, FLOAT_TOLERANCE) # (5-1)*0.25=1 -> index 1
	assert_float(StatMath.BasicStats.percentile(data, 50.0)).is_equal_approx(30.0, FLOAT_TOLERANCE) # (5-1)*0.5=2 -> index 2
	assert_float(StatMath.BasicStats.percentile(data, 75.0)).is_equal_approx(40.0, FLOAT_TOLERANCE) # (5-1)*0.75=3 -> index 3
	assert_float(StatMath.BasicStats.percentile(data, 100.0)).is_equal_approx(50.0, FLOAT_TOLERANCE)

func test_percentile_interpolation() -> void:
	var data: Array[float] = [10.0, 20.0, 30.0, 40.0]
	# Position = (4-1) * (33.333/100) = 0.99999
	# Lower index = 0, weight = 0.99999
	# 10 * (1-0.99999) + 20 * 0.99999 ~= 20
	assert_float(StatMath.BasicStats.percentile(data, 33.333)).is_equal_approx(20.0, 1e-4)
	
	# Position = (4-1) * (50/100) = 1.5
	# Lower index = 1, weight = 0.5
	# 20 * 0.5 + 30 * 0.5 = 25
	assert_float(StatMath.BasicStats.percentile(data, 50.0)).is_equal_approx(25.0, FLOAT_TOLERANCE)

func test_percentile_single_value() -> void:
	var result: float = StatMath.BasicStats.percentile(single_value, 50.0)
	assert_float(result).is_equal_approx(42.0, FLOAT_TOLERANCE)

func test_percentile_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.percentile([], 50.0)
	await assert_error(test_call).is_push_error("Cannot calculate percentile of empty array.")

func test_percentile_invalid_percentile() -> void:
	var test_call_low: Callable = func():
		StatMath.BasicStats.percentile(simple_data, -10.0)
	await assert_error(test_call_low).is_push_error("Percentile value must be between 0.0 and 100.0. Received: -10.000000")
	
	var test_call_high: Callable = func():
		StatMath.BasicStats.percentile(simple_data, 110.0)
	await assert_error(test_call_high).is_push_error("Percentile value must be between 0.0 and 100.0. Received: 110.000000") 
