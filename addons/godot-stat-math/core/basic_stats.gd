# res://addons/godot-stat-math/core/basic_stats.gd
class_name BasicStats extends RefCounted

## Basic Statistical Functions
##
## This class provides static methods to calculate common descriptive statistics
## from data arrays. These functions are designed for practical game development use
## where you need to analyze player data, game metrics, performance statistics, etc.
##
## Note: All functions expect [code]Array[float][/code] input. Use 
## [StatMath.HelperFunctions.sanitize_numeric_array] to preprocess 
## mixed-type arrays before calling these functions.


# =============================================================================
# MEASURES OF CENTRAL TENDENCY
# =============================================================================

## Calculates the arithmetic mean (average) of a dataset.
##
## The arithmetic mean is the sum of all values divided by the count of values.
## Formula: [code]μ = (Σx) / n[/code]
static func mean(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate mean of empty array.")
		return NAN
	
	var sum_val: float = 0.0
	for value in data:
		sum_val += value
	
	# # TEMPORARY: Intentional delay to test performance failure collection
	# OS.delay_msec(100)
	
	return sum_val / float(data.size())


## Calculates the middle value of a sorted dataset.
##
## For even-sized arrays, returns the average of the two middle values.
## This function assumes the input array is already sorted.
## Use [StatMath.HelperFunctions.sanitize_numeric_array] which sorts automatically.
static func median(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate median of empty array.")
		return NAN
	
	var size: int = data.size()
	
	if size % 2 != 0:
		# Odd number of elements - return middle element
		return data[size / 2]
	else:
		# Even number of elements - return average of two middle elements
		var mid_lower: int = (size - 1) / 2
		var mid_upper: int = size / 2
		return (data[mid_lower] + data[mid_upper]) * 0.5


# =============================================================================
# MEASURES OF VARIABILITY & SPREAD
# =============================================================================

## Calculates the population variance of a dataset.
##
## Population variance measures how spread out the data points are from the mean.
## Formula: [code]σ² = Σ(x - μ)² / N[/code]
static func variance(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate variance of empty array.")
		return NAN
	
	var mean_val: float = mean(data)
	var variance_sum: float = 0.0
	
	for value in data:
		var deviation: float = value - mean_val
		variance_sum += deviation * deviation
	
	return variance_sum / float(data.size())


## Calculates the population standard deviation of a dataset.
##
## Returns the square root of the variance, providing a measure of spread
## in the same units as the original data.
static func standard_deviation(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate standard deviation of empty array.")
		return NAN
	
	return sqrt(variance(data))


## Calculates the sample variance of a dataset with Bessel's correction.
##
## Uses sample variance formula: `Σ(x - x̄)² / (N-1)`. Use this when your 
## data represents a sample from a larger population, as it provides an unbiased 
## estimate of the population variance.
static func sample_variance(data: Array[float]) -> float:
	if not (data.size() > 1):
		push_error("Cannot calculate sample variance with fewer than 2 data points. Received size: %s" % data.size())
		return NAN
	
	var mean_val: float = mean(data)
	var variance_sum: float = 0.0
	
	for value in data:
		var deviation: float = value - mean_val
		variance_sum += deviation * deviation
	
	return variance_sum / float(data.size() - 1)


## Calculates the sample standard deviation of a dataset.
##
## Returns the square root of the sample variance. Use this when your data 
## represents a sample from a larger population.
static func sample_standard_deviation(data: Array[float]) -> float:
	if not (data.size() > 1):
		push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: %s" % data.size())
		return NAN
	
	return sqrt(sample_variance(data))


## Calculates the median absolute deviation (MAD) of a dataset.
##
## A robust measure of variability that is less sensitive to outliers than standard 
## deviation. Formula: `median(|x - median(x)|)`
static func median_absolute_deviation(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate median absolute deviation of empty array.")
		return NAN
	
	var median_val: float = median(data)
	var deviations: Array[float] = []
	
	for value in data:
		deviations.append(abs(value - median_val))
	
	deviations.sort()
	return median(deviations)


## Calculates the range (spread) of a dataset.
##
## Returns the difference between the maximum and minimum values, providing 
## a simple measure of the data's spread.
static func range_spread(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate range of empty array.")
		return NAN
	
	return data.max() - data.min()


# =============================================================================
# EXTREME VALUES
# =============================================================================

## Returns the smallest value in the dataset.
static func minimum(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot find minimum of empty array.")
		return NAN
	
	return data.min()


## Returns the largest value in the dataset.
static func maximum(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot find maximum of empty array.")
		return NAN
	
	return data.max()


# =============================================================================
# QUANTILES & PERCENTILES
# =============================================================================

## Calculates the value below which a given percentage of data falls.
##
## Uses linear interpolation for percentiles that fall between data points 
## using the "R-6" quantile method (commonly used).
## This function assumes the input array is already sorted.
static func percentile(data: Array[float], percentile_value: float) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate percentile of empty array.")
		return NAN
	
	if percentile_value < 0.0 or percentile_value > 100.0:
		push_error("Percentile value must be between 0.0 and 100.0. Received: %f" % percentile_value)
		return NAN
	
	var size: int = data.size()
	
	# Handle edge cases
	if percentile_value == 0.0:
		return data[0]
	if percentile_value == 100.0:
		return data[size - 1]
	
	# Calculate position using the "R-6" quantile method (commonly used)
	var position: float = (percentile_value / 100.0) * (size - 1)
	var lower_index: int = int(position)
	var upper_index: int = lower_index + 1
	
	# If position is exactly on an index, return that value
	if position == float(lower_index):
		return data[lower_index]
	
	# If upper_index would be out of bounds, return the last element
	if upper_index >= size:
		return data[size - 1]
	
	# Linear interpolation between the two surrounding values
	var weight: float = position - float(lower_index)
	return data[lower_index] * (1.0 - weight) + data[upper_index] * weight


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

## Calculates all basic statistics and returns them in a Dictionary.
##
## Provides a comprehensive statistical summary of the dataset including all
## measures of central tendency, variability, and extreme values.
## Dictionary contains: `mean`, `median`, `variance`, `standard_deviation`,
## `sample_variance`, `sample_standard_deviation`, `median_absolute_deviation`,
## `range`, `minimum`, `maximum`, `count`.
static func summary_statistics(data: Array[float]) -> Dictionary:
	if not (data.size() > 0):
		push_error("Cannot calculate summary statistics of empty array.")
		return {}
	
	return {
		"mean": mean(data),
		"median": median(data),
		"variance": variance(data),
		"standard_deviation": standard_deviation(data),
		"sample_variance": sample_variance(data) if data.size() > 1 else NAN,
		"sample_standard_deviation": sample_standard_deviation(data) if data.size() > 1 else NAN,
		"median_absolute_deviation": median_absolute_deviation(data),
		"range": range_spread(data),
		"minimum": minimum(data),
		"maximum": maximum(data),
		"count": data.size()
	} 
