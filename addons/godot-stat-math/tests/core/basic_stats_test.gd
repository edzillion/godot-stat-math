# res://addons/godot-stat-math/tests/core/basic_stats_test.gd
class_name BasicStatsTest extends GdUnitTestSuite


# Import test data for data-driven tests
const BASIC_STATS_TEST_DATA = preload("res://addons/godot-stat-math/tables/basic_stats_test_data.gd")

# Test data sets eliminated - now using scipy-generated data from BASIC_STATS_TEST_DATA

# --- Mean Tests ---
func test_mean_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.mean(data)
	assert_float(result).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)

func test_mean_integer_like_floats() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["integer_like_floats"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.mean(data)
	assert_float(result).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)

func test_mean_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.mean(data)
	assert_float(result).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)

func test_mean_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.mean([])
	await assert_error(test_call).is_push_error("Cannot calculate mean of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.mean([])
	assert_bool(is_nan(result)).is_true()

# --- Median Tests ---
func test_median_odd_count() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]  # 10 values, even count for this case
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_median_even_count() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["power_law_data"]  # 8 values, even count
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_median_unsorted_data() -> void:
	# Using right_skewed_data for this test - has sorted and unsorted versions
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["right_skewed_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = data.duplicate()
	sorted_data.sort()
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_median_with_unsorted_array_violation() -> void:
	# Test that median function behavior with unsorted data follows crash early philosophy
	# We expect this to produce an incorrect result since the function assumes sorted data
	var unsorted_data: Array[float] = [5.0, 1.0, 3.0, 2.0, 4.0]
	# Without sorting, this takes the middle element by index, not value
	var result: float = StatMath.BasicStats.median(unsorted_data) 
	assert_float(result).is_equal_approx(3.0, StatMath.FLOAT_TOLERANCE)  # Middle element by index, not median by value
	
	# The correct median should be 3.0 after sorting
	var sorted_data: Array[float] = unsorted_data.duplicate()
	sorted_data.sort()
	var correct_result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(correct_result).is_equal_approx(3.0, StatMath.FLOAT_TOLERANCE)
	
	# In this specific case they're the same, but let's test a case where they differ
	var unsorted_data2: Array[float] = [5.0, 1.0, 10.0, 2.0, 4.0]
	var unsorted_result: float = StatMath.BasicStats.median(unsorted_data2) 
	assert_float(unsorted_result).is_equal_approx(10.0, StatMath.FLOAT_TOLERANCE)  # Middle element by index
	
	var sorted_data2: Array[float] = unsorted_data2.duplicate()
	sorted_data2.sort()
	var correct_sorted_result: float = StatMath.BasicStats.median(sorted_data2)
	assert_float(correct_sorted_result).is_equal_approx(4.0, StatMath.FLOAT_TOLERANCE)  # True median

func test_median_enhanced_decimal_precision() -> void:
	# Test with high-precision decimal data
	var high_precision_data: Array[float] = [
		1.123456789, 2.987654321, 1.555555555, 
		2.444444444, 1.777777777, 2.666666666, 1.999999999
	]
	high_precision_data.sort()
	# Sorted: [1.123456789, 1.555555555, 1.777777777, 1.999999999, 2.444444444, 2.666666666, 2.987654321]
	# Median should be 1.999999999 (middle element)
	var result: float = StatMath.BasicStats.median(high_precision_data)
	assert_float(result).is_equal_approx(1.999999999, StatMath.HIGH_PRECISION_TOLERANCE)

func test_median_with_repeated_decimal_values() -> void:
	# Test median with repeated decimal values
	var repeated_decimals: Array[float] = [1.5, 1.5, 2.1, 2.1, 2.1, 3.7, 3.7]
	var result: float = StatMath.BasicStats.median(repeated_decimals)
	assert_float(result).is_equal_approx(2.1, StatMath.FLOAT_TOLERANCE)  # Middle element of 7

func test_median_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_median_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median([])
	await assert_error(test_call).is_push_error("Cannot calculate median of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.median([])
	assert_bool(is_nan(result)).is_true()

# --- Variance Tests ---
func test_variance_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.variance(data)
	assert_float(result).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)

func test_variance_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.variance(data)
	assert_float(result).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)

func test_variance_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.variance([])
	await assert_error(test_call).is_push_error("Cannot calculate variance of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.variance([])
	assert_bool(is_nan(result)).is_true()

# --- Standard Deviation Tests ---
func test_standard_deviation_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.standard_deviation(data)
	assert_float(result).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)

func test_standard_deviation_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.standard_deviation(data)
	assert_float(result).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)

func test_standard_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.standard_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate standard deviation of empty array.")
	
	# Test sentinel return value
	var result: float = StatMath.BasicStats.standard_deviation([])
	assert_bool(is_nan(result)).is_true()

# --- Sample Variance Tests ---
func test_sample_variance_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.sample_variance(data)
	assert_float(result).is_equal_approx(test_data["expected_sample_variance"], StatMath.FLOAT_TOLERANCE)

func test_sample_variance_very_large_numbers() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["very_large_numbers"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.sample_variance(data)
	assert_float(result).is_equal_approx(test_data["expected_sample_variance"], StatMath.FLOAT_TOLERANCE)

func test_sample_variance_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var test_call: Callable = func():
		StatMath.BasicStats.sample_variance(data)
	await assert_error(test_call).is_push_error("Cannot calculate sample variance with fewer than 2 data points. Received size: 1")

# --- Sample Standard Deviation Tests ---
func test_sample_standard_deviation_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.sample_standard_deviation(data)
	assert_float(result).is_equal_approx(test_data["expected_sample_std"], StatMath.FLOAT_TOLERANCE)

func test_sample_standard_deviation_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var test_call: Callable = func():
		StatMath.BasicStats.sample_standard_deviation(data)
	await assert_error(test_call).is_push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: 1")

# --- Median Absolute Deviation Tests ---
func test_median_absolute_deviation_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median_absolute_deviation(sorted_data)
	# Calculated MAD for bimodal data [1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0]:
	# Median = 5.0, Absolute deviations = [4.0, 3.5, 3.0, 2.5, 2.0, 2.0, 2.5, 3.0, 3.5, 4.0]
	# Sorted abs devs = [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.0]
	# MAD = median([2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.0]) = (3.0 + 3.0) / 2 = 3.0
	var expected_mad: float = 3.0
	assert_float(result).is_equal_approx(expected_mad, StatMath.FLOAT_TOLERANCE)

func test_median_absolute_deviation_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median_absolute_deviation(sorted_data)
	assert_float(result).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_median_absolute_deviation_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.median_absolute_deviation([])
	await assert_error(test_call).is_push_error("Cannot calculate MAD of empty array.")

# --- Range Tests ---
func test_range_spread_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.range_spread(data)
	assert_float(result).is_equal_approx(test_data["expected_range"], StatMath.FLOAT_TOLERANCE)

# --- Minimum Tests ---
func test_minimum_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.minimum(data)
	assert_float(result).is_equal_approx(test_data["expected_min"], StatMath.FLOAT_TOLERANCE)

# --- Maximum Tests ---
func test_maximum_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.maximum(data)
	assert_float(result).is_equal_approx(test_data["expected_max"], StatMath.FLOAT_TOLERANCE)

# --- Summary Statistics Tests ---
func test_summary_statistics_structure() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["integer_like_floats"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: Dictionary = StatMath.BasicStats.summary_statistics(data)
	
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
	assert_int(result["count"]).is_equal(5)  # integer_like_floats has 5 values

func test_summary_statistics_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.summary_statistics([])
	await assert_error(test_call).is_push_error("Cannot calculate summary statistics of empty array.")

func test_summary_statistics_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: Dictionary = StatMath.BasicStats.summary_statistics(data)
	
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
	assert_int(result["count"]).is_equal(10)

# --- Percentile Tests ---
func test_percentile_bimodal_data() -> void:
	var data: Array[float] = [10.0, 20.0, 30.0, 40.0, 50.0]
	assert_float(StatMath.BasicStats.percentile(data, 0.0)).is_equal_approx(10.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(data, 25.0)).is_equal_approx(20.0, StatMath.FLOAT_TOLERANCE) # (5-1)*0.25=1 -> index 1
	assert_float(StatMath.BasicStats.percentile(data, 50.0)).is_equal_approx(30.0, StatMath.FLOAT_TOLERANCE) # (5-1)*0.5=2 -> index 2
	assert_float(StatMath.BasicStats.percentile(data, 75.0)).is_equal_approx(40.0, StatMath.FLOAT_TOLERANCE) # (5-1)*0.75=3 -> index 3
	assert_float(StatMath.BasicStats.percentile(data, 100.0)).is_equal_approx(50.0, StatMath.FLOAT_TOLERANCE)

func test_percentile_interpolation() -> void:
	var data: Array[float] = [10.0, 20.0, 30.0, 40.0]
	# Position = (4-1) * (33.333/100) = 0.99999
	# Lower index = 0, weight = 0.99999
	# 10 * (1-0.99999) + 20 * 0.99999 ~= 20
	assert_float(StatMath.BasicStats.percentile(data, 33.333)).is_equal_approx(20.0, StatMath.INTERPOLATION_TOLERANCE)
	
	# Position = (4-1) * (50/100) = 1.5
	# Lower index = 1, weight = 0.5
	# 20 * 0.5 + 30 * 0.5 = 25
	assert_float(StatMath.BasicStats.percentile(data, 50.0)).is_equal_approx(25.0, StatMath.FLOAT_TOLERANCE)

func test_percentile_enhanced_decimal_precision() -> void:
	# Test percentiles with high-precision decimal data
	var high_precision_data: Array[float] = [
		1.123456789, 2.987654321, 1.555555555, 
		2.444444444, 1.777777777, 2.666666666, 1.999999999
	]
	high_precision_data.sort()
	
	# Test various percentiles with high precision
	assert_float(StatMath.BasicStats.percentile(high_precision_data, 0.0)).is_equal_approx(1.123456789, StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(high_precision_data, 100.0)).is_equal_approx(2.987654321, StatMath.HIGH_PRECISION_TOLERANCE)
	
	# Test 50th percentile (median)
	assert_float(StatMath.BasicStats.percentile(high_precision_data, 50.0)).is_equal_approx(1.999999999, StatMath.HIGH_PRECISION_TOLERANCE)

func test_percentile_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.percentile(data, 50.0)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_percentile_empty_array() -> void:
	var test_call: Callable = func():
		StatMath.BasicStats.percentile([], 50.0)
	await assert_error(test_call).is_push_error("Cannot calculate percentile of empty array.")

func test_percentile_invalid_percentile_negative() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var test_call: Callable = func():
		StatMath.BasicStats.percentile(sorted_data, -10.0)
	await assert_error(test_call).is_push_error("Percentile value must be between 0.0 and 100.0. Received: -10.000000")

func test_percentile_invalid_percentile_over_100() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var test_call: Callable = func():
		StatMath.BasicStats.percentile(sorted_data, 110.0)
	await assert_error(test_call).is_push_error("Percentile value must be between 0.0 and 100.0. Received: 110.000000")

# --- Enhanced Decimal Data Tests ---
func test_mean_enhanced_decimal_precision() -> void:
	# Test mean with high-precision decimal data
	var high_precision_data: Array[float] = [
		1.123456789, 2.987654321, 1.555555555, 
		2.444444444, 1.777777777
	]
	var expected_mean: float = (1.123456789 + 2.987654321 + 1.555555555 + 2.444444444 + 1.777777777) / 5.0
	var result: float = StatMath.BasicStats.mean(high_precision_data)
	assert_float(result).is_equal_approx(expected_mean, StatMath.HIGH_PRECISION_TOLERANCE)

func test_variance_enhanced_decimal_precision() -> void:
	# Test variance with high-precision decimal data
	var high_precision_data: Array[float] = [1.111111111, 2.222222222, 3.333333333]
	var result: float = StatMath.BasicStats.variance(high_precision_data)
	
	# Manual calculation for verification
	var mean_val: float = StatMath.BasicStats.mean(high_precision_data)
	var expected_variance: float = 0.0
	for value in high_precision_data:
		expected_variance += (value - mean_val) * (value - mean_val)
	expected_variance /= high_precision_data.size()
	
	assert_float(result).is_equal_approx(expected_variance, StatMath.HIGH_PRECISION_TOLERANCE)

func test_standard_deviation_enhanced_decimal_precision() -> void:
	# Test standard deviation with repeating decimal pattern
	var repeating_decimals: Array[float] = [1.666666667, 1.833333333, 2.166666667, 2.333333333]
	var result: float = StatMath.BasicStats.standard_deviation(repeating_decimals)
	var variance_result: float = StatMath.BasicStats.variance(repeating_decimals)
	var expected_std: float = sqrt(variance_result)
	
	assert_float(result).is_equal_approx(expected_std, StatMath.HIGH_PRECISION_TOLERANCE) 

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- Phase 3 Task 1: Non-Normal Distribution Tests ---

func test_right_skewed_data_behavior() -> void:
	# Test that our functions handle right-skewed data correctly (common in game analytics)
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["right_skewed_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# Test all basic statistics with scipy-validated expected values
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.sample_variance(data)).is_equal_approx(test_data["expected_sample_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.sample_standard_deviation(data)).is_equal_approx(test_data["expected_sample_std"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.range_spread(data)).is_equal_approx(test_data["expected_range"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.minimum(data)).is_equal_approx(test_data["expected_min"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.maximum(data)).is_equal_approx(test_data["expected_max"], StatMath.FLOAT_TOLERANCE)
	
	# For skewed data, median should be more robust than mean
	# In right-skewed data, mean > median (pulled by outliers)
	assert_that(StatMath.BasicStats.mean(data) > StatMath.BasicStats.median(sorted_data)).is_true()

func test_left_skewed_data_behavior() -> void:
	# Test left-skewed data (rare high scores concentrated at upper end)
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["left_skewed_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.sample_variance(data)).is_equal_approx(test_data["expected_sample_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.sample_standard_deviation(data)).is_equal_approx(test_data["expected_sample_std"], StatMath.FLOAT_TOLERANCE)
	
	# For left-skewed data, mean < median (pulled down by outliers)
	assert_that(StatMath.BasicStats.mean(data) < StatMath.BasicStats.median(sorted_data)).is_true()

func test_heavy_tailed_data_robustness() -> void:
	# Test heavy-tailed data (damage spikes, network latency) - emphasizes MAD vs StdDev
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["heavy_tailed_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# MAD should be more robust to outliers than standard deviation
	var mad: float = StatMath.BasicStats.median_absolute_deviation(sorted_data)
	var std_dev: float = StatMath.BasicStats.standard_deviation(data)
	
	# With heavy tails, MAD should be significantly smaller than std dev
	assert_that(mad < std_dev).is_true()
	
	# Extreme values should not cause overflow
	assert_that(not is_inf(std_dev)).is_true()
	assert_that(not is_nan(std_dev)).is_true()

func test_bimodal_data_characteristics() -> void:
	# Test bimodal data (two distinct player skill groups)
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# For this specific bimodal distribution, mean and median should be equal (symmetric)
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(StatMath.BasicStats.median(sorted_data), StatMath.FLOAT_TOLERANCE)

func test_power_law_data_handling() -> void:
	# Test power-law distributed data (common in gaming analytics)
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["power_law_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# Power-law data: mean > median due to heavy right tail
	assert_that(StatMath.BasicStats.mean(data) > StatMath.BasicStats.median(sorted_data)).is_true()

# --- Phase 3 Task 2: Numerical Stability Tests ---

func test_very_large_numbers_stability() -> void:
	# Test numerical stability with very large numbers
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["very_large_numbers"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# All calculations should remain stable, no overflow/underflow
	var mean_result: float = StatMath.BasicStats.mean(data)
	var variance_result: float = StatMath.BasicStats.variance(data)
	var std_result: float = StatMath.BasicStats.standard_deviation(data)
	
	assert_float(mean_result).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(variance_result).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(std_result).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# Verify no overflow conditions
	assert_that(not is_inf(mean_result)).is_true()
	assert_that(not is_inf(variance_result)).is_true()
	assert_that(not is_inf(std_result)).is_true()
	assert_that(not is_nan(variance_result)).is_true()

func test_very_small_numbers_precision() -> void:
	# Test precision preservation with very small numbers
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["very_small_numbers"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.HIGH_PRECISION_TOLERANCE)
	
	# Verify no underflow to zero when it shouldn't
	assert_that(StatMath.BasicStats.variance(data) > 0.0).is_true()

func test_mixed_magnitude_data_robustness() -> void:
	# Test with data spanning many orders of magnitude
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["mixed_magnitude_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# Should handle extreme range without precision loss
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# Extreme range should not cause computational issues
	var range_result: float = StatMath.BasicStats.range_spread(data)
	assert_that(not is_inf(range_result)).is_true()
	assert_that(not is_nan(range_result)).is_true()

func test_close_numbers_precision_stability() -> void:
	# Test precision with very close numbers
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["close_numbers"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# Should preserve precision even with tiny differences
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.HIGH_PRECISION_TOLERANCE)
	
	# Variance should be positive (not rounded to zero) for distinct values
	assert_that(StatMath.BasicStats.variance(data) > 0.0).is_true()

func test_identical_values_numerical_stability() -> void:
	# Test numerical stability with identical values
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["identical_values"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# All variance measures should be exactly zero
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.sample_variance(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.sample_standard_deviation(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.range_spread(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# Mean should equal the identical value
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

# --- Phase 3 Task 3: Integer vs Float Input Consistency Tests ---

func test_integer_like_float_precision() -> void:
	# Test that integer-like floats (game scores) maintain precision
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["integer_like_floats"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# Should produce exact results for integer-like inputs
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# No precision artifacts should appear in calculations
	var mean_result: float = StatMath.BasicStats.mean(data)
	assert_that(mean_result == 2000.0).is_true()  # Exact equality for this case

func test_mixed_integer_float_consistency() -> void:
	# Test consistency between pure integer-like data and mixed precision
	var integer_like: Array[float] = [100.0, 200.0, 300.0, 400.0, 500.0]
	var mixed_precision: Array[float] = [100.0, 200.5, 300.0, 400.5, 500.0]
	
	# Integer-like data should produce clean results
	var int_mean: float = StatMath.BasicStats.mean(integer_like)
	var int_variance: float = StatMath.BasicStats.variance(integer_like)
	
	assert_float(int_mean).is_equal_approx(300.0, StatMath.FLOAT_TOLERANCE)
	assert_that(int_variance == 20000.0).is_true()  # Should be exact
	
	# Mixed precision should handle gracefully
	var mixed_mean: float = StatMath.BasicStats.mean(mixed_precision)
	var mixed_variance: float = StatMath.BasicStats.variance(mixed_precision)
	
	assert_that(not is_nan(mixed_mean)).is_true()
	assert_that(not is_nan(mixed_variance)).is_true()
	assert_that(mixed_mean > int_mean).is_true()  # Slightly higher due to .5 values

# --- Phase 3 Task 4: Enhanced Single-Element Edge Case Tests ---

func test_single_element_comprehensive_zero() -> void:
	# Comprehensive test for single zero element
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_zero"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# All basic statistics should handle single zero correctly
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.range_spread(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.minimum(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.maximum(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# All percentiles should return the single value
	assert_float(StatMath.BasicStats.percentile(sorted_data, 0.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(sorted_data, 25.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(sorted_data, 50.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(sorted_data, 75.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.percentile(sorted_data, 100.0)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)

func test_single_element_comprehensive_negative() -> void:
	# Comprehensive test for single negative element
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(-42.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(-42.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.range_spread(data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.minimum(data)).is_equal_approx(-42.0, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.maximum(data)).is_equal_approx(-42.0, StatMath.FLOAT_TOLERANCE)
	
	# Percentiles should all return the single negative value
	assert_float(StatMath.BasicStats.percentile(sorted_data, 50.0)).is_equal_approx(-42.0, StatMath.FLOAT_TOLERANCE)

func test_single_element_extreme_values() -> void:
	# Test single element with extreme values
	var large_test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_large"]
	var large_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(large_test_data["data"])
	var large_sorted: Array[float] = StatMath.HelperFunctions.convert_to_float_array(large_test_data["sorted_data"])
	
	var small_test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_small"]
	var small_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(small_test_data["data"])
	var small_sorted: Array[float] = StatMath.HelperFunctions.convert_to_float_array(small_test_data["sorted_data"])
	
	# Large single element
	assert_float(StatMath.BasicStats.mean(large_data)).is_equal_approx(1e10, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(large_sorted)).is_equal_approx(1e10, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(large_data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# Small single element  
	assert_float(StatMath.BasicStats.mean(small_data)).is_equal_approx(1e-10, StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.median(small_sorted)).is_equal_approx(1e-10, StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(small_data)).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	
	# No computational issues with extreme single values
	assert_that(not is_inf(StatMath.BasicStats.mean(large_data))).is_true()
	assert_that(not is_nan(StatMath.BasicStats.mean(large_data))).is_true()
	assert_that(not is_nan(StatMath.BasicStats.mean(small_data))).is_true()

func test_single_element_summary_statistics_consistency() -> void:
	# Test that summary statistics work correctly for single elements
	var test_data: Array[float] = [123.456]
	var summary: Dictionary = StatMath.BasicStats.summary_statistics(test_data)
	
	# All statistics should be consistent for single element
	assert_float(summary["mean"]).is_equal_approx(123.456, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["median"]).is_equal_approx(123.456, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["variance"]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["standard_deviation"]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["median_absolute_deviation"]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["range"]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["minimum"]).is_equal_approx(123.456, StatMath.FLOAT_TOLERANCE)
	assert_float(summary["maximum"]).is_equal_approx(123.456, StatMath.FLOAT_TOLERANCE)
	assert_int(summary["count"]).is_equal(1)
	
	# Sample variance and std dev should error appropriately for single element
	var test_call_sample_var: Callable = func():
		StatMath.BasicStats.sample_variance(test_data)
	await assert_error(test_call_sample_var).is_push_error("Cannot calculate sample variance with fewer than 2 data points. Received size: 1")
	
	var test_call_sample_std: Callable = func():
		StatMath.BasicStats.sample_standard_deviation(test_data)
	await assert_error(test_call_sample_std).is_push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: 1") 