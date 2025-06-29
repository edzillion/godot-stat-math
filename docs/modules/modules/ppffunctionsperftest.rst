StatMath.PpfFunctionsPerfTest
=============================

Performance Test Suite for StatMath.

PpfFunctions Module
Tests computationally intensive percent point functions (quantiles)
which often use iterative methods like Newton-Raphson.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PpfFunctionsPerfTest.function_name(parameters)

Constants
---------

.. data:: PROBABILITY_VALUES

   Value: ``Array[float] = [0.1, 0.5, 0.9]``

Functions
---------

.. function:: get_module_name() -> String:

.. function:: test_normal_ppf_performance() -> void:

.. function:: test_gamma_ppf_performance() -> void:

.. function:: test_beta_ppf_performance() -> void:

.. function:: test_weibull_ppf_performance() -> void:

