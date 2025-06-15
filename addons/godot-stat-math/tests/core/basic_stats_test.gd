# addons/godot-stat-math/tests/core/basic_stats_test.gd
class_name BasicStatsTest extends GdUnitTestSuite

# Test data sets
var simple_data: Array[float] = [1.0, 2.0, 3.0, 4.0, 5.0]
var decimal_data: Array[float] = [1.5, 2.3, 1.8, 2.1, 1.9, 2.4, 1.7]
var single_value: Array[float] = [42.0]
var two_values: Array[float] = [10.0, 20.0]

# --- Mean Tests ---
func test_mean_simple_data() -> void:
	var result: float = StatMath.BasicStats.mean(simple_data)
	assert_float(result).is_equal_approx(3.0, 1e-7)

func test_mean_decimal_data() -> void:
	var result: float = StatMath.BasicStats.mean(decimal_data)
	var expected: float = 13.7 / 7.0  # Sum is 13.7, count is 7
	assert_float(result).is_equal_approx(expected, 1e-7)

func test_mean_single_value() -> void:
	var result: float = StatMath.BasicStats.mean(single_value)
	assert_float(result).is_equal_approx(42.0, 1e-7)

func test_mean_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.mean([])
	await assert_error(test_call).is_push_error("Cannot calculate mean of empty array.")

# --- Median Tests ---
func test_median_odd_count() -> void:
	var result: float = StatMath.BasicStats.median(simple_data)
	assert_float(result).is_equal_approx(3.0, 1e-7)

func test_median_even_count() -> void:
	var even_data: Array[float] = [1.0, 2.0, 3.0, 4.0]
	var result: float = StatMath.BasicStats.median(even_data)
	assert_float(result).is_equal_approx(2.5, 1e-7)

func test_median_single_value() -> void:
	var result: float = StatMath.BasicStats.median(single_value)
	assert_float(result).is_equal_approx(42.0, 1e-7)

func test_median_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median([])
	await assert_error(test_call).is_push_error("Cannot calculate median of empty array.")

# --- Variance Tests ---
func test_variance_simple_data() -> void:
	var result: float = StatMath.BasicStats.variance(simple_data)
	# Variance of [1,2,3,4,5] with mean 3.0 is ((1-3)²+(2-3)²+(3-3)²+(4-3)²+(5-3)²)/5 = (4+1+0+1+4)/5 = 2.0
	assert_float(result).is_equal_approx(2.0, 1e-7)

func test_variance_single_value() -> void:
	var result: float = StatMath.BasicStats.variance(single_value)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_variance_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.variance([])
	await assert_error(test_call).is_push_error("Cannot calculate variance of empty array.")

# --- Standard Deviation Tests ---
func test_standard_deviation_simple_data() -> void:
	var result: float = StatMath.BasicStats.standard_deviation(simple_data)
	assert_float(result).is_equal_approx(sqrt(2.0), 1e-7)

func test_standard_deviation_single_value() -> void:
	var result: float = StatMath.BasicStats.standard_deviation(single_value)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_standard_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.standard_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate standard deviation of empty array.")

# --- Sample Variance Tests ---
func test_sample_variance_simple_data() -> void:
	var result: float = StatMath.BasicStats.sample_variance(simple_data)
	# Sample variance uses N-1 denominator: 10/4 = 2.5
	assert_float(result).is_equal_approx(2.5, 1e-7)

func test_sample_variance_two_values() -> void:
	var result: float = StatMath.BasicStats.sample_variance(two_values)
	# Sample variance of [10, 20] with mean 15.0 is ((10-15)²+(20-15)²)/(2-1) = (25+25)/1 = 50.0
	assert_float(result).is_equal_approx(50.0, 1e-7)

func test_sample_variance_single_value() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.sample_variance(single_value)
	await assert_error(test_call).is_push_error("Cannot calculate sample variance with fewer than 2 data points.")

# --- Sample Standard Deviation Tests ---
func test_sample_standard_deviation_simple_data() -> void:
	var result: float = StatMath.BasicStats.sample_standard_deviation(simple_data)
	assert_float(result).is_equal_approx(sqrt(2.5), 1e-7)

func test_sample_standard_deviation_single_value() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.sample_standard_deviation(single_value)
	await assert_error(test_call).is_push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: 1")

# --- Median Absolute Deviation Tests ---
func test_median_absolute_deviation_simple_data() -> void:
	var result: float = StatMath.BasicStats.median_absolute_deviation(simple_data)
	# Median is 3.0, deviations are [2,1,0,1,2], median of deviations is 1.0
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_median_absolute_deviation_single_value() -> void:
	var result: float = StatMath.BasicStats.median_absolute_deviation(single_value)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_median_absolute_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median_absolute_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate MAD of empty array.")

# --- Range Tests ---
func test_range_spread_simple_data() -> void:
	var result: float = StatMath.BasicStats.range_spread(simple_data)
	assert_float(result).is_equal_approx(4.0, 1e-7)  # 5.0 - 1.0

func test_range_spread_single_value() -> void:
	var result: float = StatMath.BasicStats.range_spread(single_value)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_range_spread_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.range_spread([])
	await assert_error(test_call).is_push_error("Cannot calculate range of empty array.")

# --- Minimum Tests ---
func test_minimum_simple_data() -> void:
	var result: float = StatMath.BasicStats.minimum(simple_data)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_minimum_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.minimum([])
	await assert_error(test_call).is_push_error("Cannot find minimum of empty array.")

# --- Maximum Tests ---
func test_maximum_simple_data() -> void:
	var result: float = StatMath.BasicStats.maximum(simple_data)
	assert_float(result).is_equal_approx(5.0, 1e-7)

func test_maximum_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.maximum([])
	await assert_error(test_call).is_push_error("Cannot find maximum of empty array.")

# --- Summary Statistics Tests ---
func test_summary_statistics_simple_data() -> void:
	var result: Dictionary = StatMath.BasicStats.summary_statistics(simple_data)
	
	assert_float(result["mean"]).is_equal_approx(3.0, 1e-7)
	assert_float(result["median"]).is_equal_approx(3.0, 1e-7)
	assert_float(result["variance"]).is_equal_approx(2.0, 1e-7)
	assert_float(result["standard_deviation"]).is_equal_approx(sqrt(2.0), 1e-7)
	assert_float(result["sample_variance"]).is_equal_approx(2.5, 1e-7)
	assert_float(result["sample_standard_deviation"]).is_equal_approx(sqrt(2.5), 1e-7)
	assert_float(result["median_absolute_deviation"]).is_equal_approx(1.0, 1e-7)
	assert_float(result["range"]).is_equal_approx(4.0, 1e-7)
	assert_float(result["minimum"]).is_equal_approx(1.0, 1e-7)
	assert_float(result["maximum"]).is_equal_approx(5.0, 1e-7)
	assert_int(result["count"]).is_equal(5)

func test_summary_statistics_single_value() -> void:
	var result: Dictionary = StatMath.BasicStats.summary_statistics(single_value)
	
	assert_float(result["mean"]).is_equal_approx(42.0, 1e-7)
	assert_float(result["median"]).is_equal_approx(42.0, 1e-7)
	assert_float(result["variance"]).is_equal_approx(0.0, 1e-7)
	assert_float(result["standard_deviation"]).is_equal_approx(0.0, 1e-7)
	assert_that(is_nan(result["sample_variance"])).is_true()
	assert_that(is_nan(result["sample_standard_deviation"])).is_true()
	assert_float(result["median_absolute_deviation"]).is_equal_approx(0.0, 1e-7)
	assert_float(result["range"]).is_equal_approx(0.0, 1e-7)
	assert_float(result["minimum"]).is_equal_approx(42.0, 1e-7)
	assert_float(result["maximum"]).is_equal_approx(42.0, 1e-7)
	assert_int(result["count"]).is_equal(1)

func test_summary_statistics_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.summary_statistics([])
	await assert_error(test_call).is_push_error("Cannot calculate summary statistics of empty array.") 
