StatMath.DistributionsPerfTest
==============================

Performance Test Suite for StatMath.

Distributions Module

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.DistributionsPerfTest.function_name(parameters)

Functions
---------

.. function:: get_module_name() -> String:

   Performance Test Suite for StatMath.Distributions Module

   Tests the most computationally intensive random variate generation functions
   to catch performance regressions during development.

.. function:: test_normal_distribution_performance() -> void:

.. function:: test_gamma_distribution_performance() -> void:

.. function:: test_beta_distribution_performance() -> void:

.. function:: test_binomial_distribution_performance() -> void:

.. function:: test_poisson_distribution_performance() -> void:

.. function:: test_weibull_distribution_performance() -> void:

.. function:: test_uniform_int_distribution_performance() -> void:

.. function:: test_uniform_float_distribution_performance() -> void:

.. function:: test_exponential_distribution_performance() -> void:

.. function:: test_pareto_distribution_performance() -> void:

.. function:: test_cauchy_distribution_performance() -> void:

.. function:: test_triangular_distribution_performance() -> void:

