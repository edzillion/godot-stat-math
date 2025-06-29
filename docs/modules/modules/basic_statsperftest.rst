StatMath.BasicStatsPerfTest
===========================

Performance Test Suite for StatMath.

BasicStats Module

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.BasicStatsPerfTest.function_name(parameters)

Functions
---------

.. function:: get_module_name() -> String:

   Performance Test Suite for StatMath.BasicStats Module

   Tests statistical analysis functions on various dataset sizes
   to catch performance regressions during development.

.. function:: test_mean_variance_performance() -> void:

.. function:: test_median_performance() -> void:

.. function:: test_median_absolute_deviation_performance() -> void:

.. function:: test_sample_statistics_performance() -> void:

.. function:: test_min_max_range_performance() -> void:

