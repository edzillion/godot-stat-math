StatMath.PpfFunctionsMathematicalPropertyTest
=============================================

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PpfFunctionsMathematicalPropertyTest.function_name(parameters)

Functions
---------

.. function:: test_normal_cdf_ppf_round_trip_consistency() -> void:

   Tests that CDF and PPF are inverse functions - Normal Distribution

.. function:: test_exponential_cdf_ppf_round_trip_consistency() -> void:

   Tests that CDF and PPF are inverse functions - Exponential Distribution

.. function:: test_uniform_cdf_ppf_round_trip_consistency() -> void:

   Tests that CDF and PPF are inverse functions - Uniform Distribution

.. function:: test_normal_ppf_cdf_round_trip_consistency() -> void:

   Tests that PPF and CDF are inverse functions - Normal Distribution

.. function:: test_exponential_ppf_cdf_round_trip_consistency() -> void:

   Tests that PPF and CDF are inverse functions - Exponential Distribution

.. function:: test_normal_ppf_monotonicity() -> void:

   Tests that PPF functions are monotonically increasing - Normal Distribution

.. function:: test_exponential_ppf_monotonicity() -> void:

   Tests that PPF functions are monotonically increasing - Exponential Distribution

.. function:: test_normal_ppf_boundary_conditions() -> void:

   Tests PPF behavior at probability boundaries - Normal Distribution

.. function:: test_uniform_ppf_boundary_conditions() -> void:

   Tests PPF behavior at probability boundaries - Uniform Distribution

.. function:: test_exponential_weibull_equivalence() -> void:

   Tests special mathematical relationships - Exponential Weibull Equivalence

.. function:: test_exponential_median_special_value() -> void:

   Tests special values - Exponential Median

