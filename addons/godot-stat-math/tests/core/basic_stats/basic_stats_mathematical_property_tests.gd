# res://addons/godot-stat-math/tests/core/basic_stats/basic_stats_mathematical_property_tests.gd
class_name BasicStatsMathematicalPropertyTests extends GdUnitTestSuite


# Import test data for data-driven tests
const BASIC_STATS_TEST_DATA = preload("res://addons/godot-stat-math/tables/basic_stats_test_data.gd")

# =============================================================================  
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

# --- Precision and Special Cases ---
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
	var unsorted_data: Array[float] = [5.0, 1.0, 3.0, 2.0, 4.0]
	var result: float = StatMath.BasicStats.median(unsorted_data) 
	assert_float(result).is_equal_approx(3.0, StatMath.FLOAT_TOLERANCE)  # Middle element by index, not median by value
	
	# The correct median should be 3.0 after sorting
	var sorted_data: Array[float] = unsorted_data.duplicate()
	sorted_data.sort()
	var correct_result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(correct_result).is_equal_approx(3.0, StatMath.FLOAT_TOLERANCE)
	
	# Test a case where they differ
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
	var result: float = StatMath.BasicStats.median(high_precision_data)
	assert_float(result).is_equal_approx(1.999999999, StatMath.HIGH_PRECISION_TOLERANCE)

func test_median_with_repeated_decimal_values() -> void:
	# Test median with repeated decimal values
	var repeated_decimals: Array[float] = [1.5, 1.5, 2.1, 2.1, 2.1, 3.7, 3.7]
	var result: float = StatMath.BasicStats.median(repeated_decimals)
	assert_float(result).is_equal_approx(2.1, StatMath.FLOAT_TOLERANCE)

# --- Non-Normal Distribution Tests ---
func test_right_skewed_data_behavior() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["right_skewed_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# For skewed data, median should be more robust than mean
	# In right-skewed data, mean > median (pulled by outliers)
	assert_that(StatMath.BasicStats.mean(data) > StatMath.BasicStats.median(sorted_data)).is_true()

func test_left_skewed_data_behavior() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["left_skewed_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# For left-skewed data, mean < median (pulled down by outliers)
	assert_that(StatMath.BasicStats.mean(data) < StatMath.BasicStats.median(sorted_data)).is_true()

func test_heavy_tailed_data_robustness() -> void:
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
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["power_law_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# Power-law data: mean > median due to heavy right tail
	assert_that(StatMath.BasicStats.mean(data) > StatMath.BasicStats.median(sorted_data)).is_true()

# --- Numerical Stability Tests ---
func test_very_large_numbers_stability() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["very_large_numbers"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	
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
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["mixed_magnitude_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# Extreme range should not cause computational issues
	var range_result: float = StatMath.BasicStats.range_spread(data)
	assert_that(not is_inf(range_result)).is_true()
	assert_that(not is_nan(range_result)).is_true()

func test_close_numbers_precision_stability() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["close_numbers"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.HIGH_PRECISION_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.HIGH_PRECISION_TOLERANCE)
	
	# Variance should be positive (not rounded to zero) for distinct values
	assert_that(StatMath.BasicStats.variance(data) > 0.0).is_true()

func test_identical_values_numerical_stability() -> void:
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

func test_integer_like_float_precision() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["integer_like_floats"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	assert_float(StatMath.BasicStats.mean(data)).is_equal_approx(test_data["expected_mean"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.median(sorted_data)).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.variance(data)).is_equal_approx(test_data["expected_variance"], StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.BasicStats.standard_deviation(data)).is_equal_approx(test_data["expected_std"], StatMath.FLOAT_TOLERANCE)
	
	# No precision artifacts should appear in calculations
	var mean_result: float = StatMath.BasicStats.mean(data)
	assert_that(mean_result == 2000.0).is_true()  # Exact equality for this case

# --- Variance and Standard Deviation Relationship Tests ---
func test_variance_standard_deviation_relationship() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	
	var variance_result: float = StatMath.BasicStats.variance(data)
	var std_result: float = StatMath.BasicStats.standard_deviation(data)
	var expected_std: float = sqrt(variance_result)
	
	assert_float(std_result).is_equal_approx(expected_std, StatMath.HIGH_PRECISION_TOLERANCE)

# --- Median Absolute Deviation Property Tests ---
func test_median_absolute_deviation_basic_properties() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	var mad: float = StatMath.BasicStats.median_absolute_deviation(sorted_data)
	
	# MAD should be non-negative
	assert_that(mad >= 0.0).is_true()
	
	# MAD should be finite
	assert_that(not is_inf(mad)).is_true()
	assert_that(not is_nan(mad)).is_true()

# --- Percentile Boundary Property Tests ---
func test_percentile_boundary_properties() -> void:
	# Test that percentile(data, 0) == minimum(data) and percentile(data, 100) == maximum(data)
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	# 0th percentile should equal minimum
	var percentile_0: float = StatMath.BasicStats.percentile(sorted_data, 0.0)
	var minimum_value: float = StatMath.BasicStats.minimum(sorted_data)
	assert_float(percentile_0).is_equal_approx(minimum_value, StatMath.BOUNDARY_TOLERANCE)
	
	# 100th percentile should equal maximum
	var percentile_100: float = StatMath.BasicStats.percentile(sorted_data, 100.0)
	var maximum_value: float = StatMath.BasicStats.maximum(sorted_data)
	assert_float(percentile_100).is_equal_approx(maximum_value, StatMath.BOUNDARY_TOLERANCE)

func test_percentile_median_property() -> void:
	# Test that percentile(data, 50) == median(data)
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	var percentile_50: float = StatMath.BasicStats.percentile(sorted_data, 50.0)
	var median_value: float = StatMath.BasicStats.median(sorted_data)
	assert_float(percentile_50).is_equal_approx(median_value, StatMath.BOUNDARY_TOLERANCE)

# --- Summary Statistics Consistency Tests ---
func test_summary_statistics_consistency() -> void:
	# Test that the values in the returned dictionary match the results from calling individual functions
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	
	var summary: Dictionary = StatMath.BasicStats.summary_statistics(data)
	
	# Test that each value in summary matches individual function calls
	assert_float(summary["mean"]).is_equal_approx(StatMath.BasicStats.mean(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["median"]).is_equal_approx(StatMath.BasicStats.median(sorted_data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["variance"]).is_equal_approx(StatMath.BasicStats.variance(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["standard_deviation"]).is_equal_approx(StatMath.BasicStats.standard_deviation(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["sample_variance"]).is_equal_approx(StatMath.BasicStats.sample_variance(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["sample_standard_deviation"]).is_equal_approx(StatMath.BasicStats.sample_standard_deviation(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["median_absolute_deviation"]).is_equal_approx(StatMath.BasicStats.median_absolute_deviation(sorted_data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["range"]).is_equal_approx(StatMath.BasicStats.range_spread(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["minimum"]).is_equal_approx(StatMath.BasicStats.minimum(data), StatMath.FLOAT_TOLERANCE)
	assert_float(summary["maximum"]).is_equal_approx(StatMath.BasicStats.maximum(data), StatMath.FLOAT_TOLERANCE)
	assert_int(summary["count"]).is_equal(data.size()) 