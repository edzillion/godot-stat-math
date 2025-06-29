StatMath.CdfFunctionsPerfTest
=============================

Performance Test Suite for StatMath.

CdfFunctions Module
Tests cumulative distribution function calculations that use
complex mathematical operations like error functions and incomplete functions.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.CdfFunctionsPerfTest.function_name(parameters)

Constants
---------

.. data:: TEST_VALUES

   Value: ``Array[float] = [-2.0, 0.0, 1.0, 2.0]``

Functions
---------

.. function:: get_module_name() -> String:

.. function:: test_normal_cdf_performance() -> void:

.. function:: test_gamma_cdf_performance() -> void:

.. function:: test_beta_cdf_performance() -> void:

.. function:: test_weibull_cdf_performance() -> void:

.. function:: test_exponential_cdf_performance() -> void:

.. function:: test_chi_square_cdf_performance() -> void:

.. function:: test_f_cdf_performance() -> void:

.. function:: test_t_cdf_performance() -> void:

.. function:: test_binomial_cdf_performance() -> void:

.. function:: test_poisson_cdf_performance() -> void:

.. function:: test_geometric_cdf_performance() -> void:

.. function:: test_negative_binomial_cdf_performance() -> void:

.. function:: test_pareto_cdf_performance() -> void:

