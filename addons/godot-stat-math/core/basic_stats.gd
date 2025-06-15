# res://addons/godot-stat-math/core/basic_stats.gd
extends RefCounted

# Basic Statistical Functions
# This script provides static methods to calculate common descriptive statistics
# from data arrays. These functions are designed for practical game development use
# where you need to analyze player data, game metrics, performance statistics, etc.
#
# All functions expect Array[float] input. Use StatMath.HelperFunctions.sanitize_numeric_array()
# to preprocess mixed-type arrays before calling these functions.

# Mean (Average): Calculates the arithmetic mean of a dataset
# Returns the sum of all values divided by the number of values.
static func mean(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate mean of empty array.")
		return NAN
	
	var sum_val: float = 0.0
	for value in data:
		sum_val += value
	
	return sum_val / float(data.size())


# Median: Calculates the middle value of a sorted dataset
# For even-sized arrays, returns the average of the two middle values.
# Note: This function assumes the input array is already sorted.
# Use StatMath.HelperFunctions.sanitize_numeric_array() which sorts automatically.
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


# Variance: Calculates the population variance of a dataset
# Measures how spread out the data points are from the mean.
# Uses population variance formula: Σ(x - μ)² / N
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


# Standard Deviation: Calculates the population standard deviation of a dataset  
# Returns the square root of the variance, providing a measure of spread
# in the same units as the original data.
static func standard_deviation(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate standard deviation of empty array.")
		return NAN
	
	return sqrt(variance(data))


# Sample Variance: Calculates the sample variance of a dataset
# Uses sample variance formula with Bessel's correction: Σ(x - x̄)² / (N-1)
# Use this when your data represents a sample from a larger population.
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


# Sample Standard Deviation: Calculates the sample standard deviation of a dataset
# Returns the square root of the sample variance.
# Use this when your data represents a sample from a larger population.
static func sample_standard_deviation(data: Array[float]) -> float:
	if not (data.size() > 1):
		push_error("Cannot calculate sample standard deviation with fewer than 2 data points. Received size: %s" % data.size())
		return NAN
	
	return sqrt(sample_variance(data))


# Median Absolute Deviation (MAD): Calculates the median absolute deviation
# A robust measure of variability that is less sensitive to outliers than standard deviation.
# Formula: median(|x - median(x)|)
static func median_absolute_deviation(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate MAD of empty array.")
		return NAN
	
	var median_val: float = median(data)
	var deviations: Array[float] = []
	
	for value in data:
		deviations.append(abs(value - median_val))
	
	deviations.sort()
	return median(deviations)


# Range (Spread): Calculates the range of a dataset
# Returns the difference between the maximum and minimum values.
static func range_spread(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot calculate range of empty array.")
		return NAN
	
	return data.max() - data.min()


# Minimum: Returns the smallest value in the dataset
static func minimum(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot find minimum of empty array.")
		return NAN
	
	return data.min()


# Maximum: Returns the largest value in the dataset  
static func maximum(data: Array[float]) -> float:
	if not (data.size() > 0):
		push_error("Cannot find maximum of empty array.")
		return NAN
	
	return data.max()


# Summary Statistics: Calculates all basic statistics and returns them in a Dictionary
# Provides a comprehensive statistical summary of the dataset.
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
