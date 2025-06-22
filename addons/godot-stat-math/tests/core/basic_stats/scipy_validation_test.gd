# res://addons/godot-stat-math/tests/core/basic_stats/scipy_validation_test.gd
class_name BasicStatsScipyValidationTest extends GdUnitTestSuite


# Import test data for data-driven tests
const BASIC_STATS_TEST_DATA = preload("res://addons/godot-stat-math/tables/basic_stats_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

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

# --- Median Tests ---
func test_median_odd_count() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_median_even_count() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["power_law_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

func test_median_single_value() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["single_negative"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_median"], StatMath.FLOAT_TOLERANCE)

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

# --- Sample Standard Deviation Tests ---
func test_sample_standard_deviation_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.sample_standard_deviation(data)
	assert_float(result).is_equal_approx(test_data["expected_sample_std"], StatMath.FLOAT_TOLERANCE)

# --- Median Absolute Deviation Tests ---
func test_median_absolute_deviation_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var sorted_data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["sorted_data"])
	var result: float = StatMath.BasicStats.median_absolute_deviation(sorted_data)
	assert_float(result).is_equal_approx(test_data["expected_mad"], StatMath.FLOAT_TOLERANCE)

# --- Range and Min/Max Tests ---
func test_range_spread_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.range_spread(data)
	assert_float(result).is_equal_approx(test_data["expected_range"], StatMath.FLOAT_TOLERANCE)

func test_minimum_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.minimum(data)
	assert_float(result).is_equal_approx(test_data["expected_min"], StatMath.FLOAT_TOLERANCE)

func test_maximum_bimodal_data() -> void:
	var test_data: Dictionary = BASIC_STATS_TEST_DATA.VALUES["bimodal_data"]
	var data: Array[float] = StatMath.HelperFunctions.convert_to_float_array(test_data["data"])
	var result: float = StatMath.BasicStats.maximum(data)
	assert_float(result).is_equal_approx(test_data["expected_max"], StatMath.FLOAT_TOLERANCE) 