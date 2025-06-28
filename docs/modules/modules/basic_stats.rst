StatMath.BasicStats
===================

Basic Statistical Functions
This class provides static methods to calculate common descriptive statistics
from data arrays.

These functions are designed for practical game development use
where you need to analyze player data, game metrics, performance statistics, etc.
Note: All functions expect ``Array[float]`` input. Use
`sanitize_numeric_array() <helper_functions.html#sanitize_numeric_array>`_ to preprocess
mixed-type arrays before calling these functions.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.BasicStats.function_name(parameters)

Functions
---------

.. function:: mean(data: Array[float]) -> float:

   Calculates the arithmetic mean (average) of a dataset.

   The arithmetic mean is the sum of all values divided by the count of values.
   Formula: ``μ = (Σx) / n``

.. function:: median(data: Array[float]) -> float:

   Calculates the middle value of a sorted dataset.

   For even-sized arrays, returns the average of the two middle values.
   This function assumes the input array is already sorted.
   Use `sanitize_numeric_array() <helper_functions.html#sanitize_numeric_array>`_ which sorts automatically.

.. function:: variance(data: Array[float]) -> float:

   Calculates the population variance of a dataset.

   Population variance measures how spread out the data points are from the mean.
   Formula: ``σ² = Σ(x - μ)² / N``

.. function:: standard_deviation(data: Array[float]) -> float:

   Calculates the population standard deviation of a dataset.

   Returns the square root of the variance, providing a measure of spread
   in the same units as the original data.

.. function:: sample_variance(data: Array[float]) -> float:

   Calculates the sample variance of a dataset with Bessel's correction.

   Uses sample variance formula: `Σ(x - x̄)² / (N-1)`. Use this when your
   data represents a sample from a larger population, as it provides an unbiased
   estimate of the population variance.

.. function:: sample_standard_deviation(data: Array[float]) -> float:

   Calculates the sample standard deviation of a dataset.

   Returns the square root of the sample variance. Use this when your data
   represents a sample from a larger population.

.. function:: median_absolute_deviation(data: Array[float]) -> float:

   Calculates the median absolute deviation (MAD) of a dataset.

   A robust measure of variability that is less sensitive to outliers than standard
   deviation. Formula: `median(|x - median(x)|)`

.. function:: range_spread(data: Array[float]) -> float:

   Calculates the range (spread) of a dataset.

   Returns the difference between the maximum and minimum values, providing
   a simple measure of the data's spread.

.. function:: minimum(data: Array[float]) -> float:

   Returns the smallest value in the dataset.

.. function:: maximum(data: Array[float]) -> float:

   Returns the largest value in the dataset.

.. function:: percentile(data: Array[float], percentile_value: float) -> float:

   Calculates the value below which a given percentage of data falls.

   Uses linear interpolation for percentiles that fall between data points
   using the "R-6" quantile method (commonly used).
   This function assumes the input array is already sorted.
   Use `sanitize_numeric_array() <helper_functions.html#sanitize_numeric_array>`_ which sorts automatically.

.. function:: summary_statistics(data: Array[float]) -> Dictionary:

   Calculates all basic statistics and returns them in a Dictionary.

   Provides a comprehensive statistical summary of the dataset including all
   measures of central tendency, variability, and extreme values.
   Dictionary contains: `mean`, `median`, `variance`, `standard_deviation`,
   `sample_variance`, `sample_standard_deviation`, `median_absolute_deviation`,
   `range`, `minimum`, `maximum`, `count`.

