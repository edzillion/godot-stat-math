StatMath.HelperFunctionsPerfTest
================================

Performance Test Suite for StatMath.

HelperFunctions Module

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.HelperFunctionsPerfTest.function_name(parameters)

Functions
---------

.. function:: get_module_name() -> String:

   Performance Test Suite for StatMath.HelperFunctions Module

   Tests computationally intensive mathematical helper functions
   to catch performance regressions during development.

.. function:: test_gamma_function_performance() -> void:

.. function:: test_beta_function_performance() -> void:

.. function:: test_incomplete_beta_function_performance() -> void:

.. function:: test_binomial_coefficient_performance() -> void:

.. function:: test_log_gamma_function_performance() -> void:

.. function:: test_log_factorial_performance() -> void:

.. function:: test_log_binomial_coefficient_performance() -> void:

.. function:: test_sanitize_numeric_array_performance() -> void:

