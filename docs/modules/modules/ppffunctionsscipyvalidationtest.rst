StatMath.PpfFunctionsScipyValidationTest
========================================

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PpfFunctionsScipyValidationTest.function_name(parameters)

Constants
---------

.. data:: PPF_TEST_DATA

   Value: ``preload("res://addons/godot-stat-math/tables/ppf_test_data.gd")``

Functions
---------

.. function:: test_normal_ppf_scipy_validation() -> void:

   Validates normal distribution PPF against scipy values

.. function:: test_exponential_ppf_scipy_validation() -> void:

   Validates exponential distribution PPF against scipy values

.. function:: test_uniform_ppf_scipy_validation() -> void:

   Validates uniform distribution PPF against scipy values

.. function:: test_pareto_ppf_scipy_validation() -> void:

   Validates pareto distribution PPF against scipy values

.. function:: test_weibull_ppf_scipy_validation() -> void:

   Validates weibull distribution PPF against scipy values

