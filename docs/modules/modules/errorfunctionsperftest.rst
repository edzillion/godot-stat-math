StatMath.ErrorFunctionsPerfTest
===============================

Performance Test Suite for StatMath.

ErrorFunctions Module
Tests computationally intensive error function calculations and related
mathematical functions like incomplete gamma and beta functions.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.ErrorFunctionsPerfTest.function_name(parameters)

Constants
---------

.. data:: TEST_VALUES

   Value: ``Array[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]``

.. data:: INVERSE_ERROR_FUNCTION_VALUES

   Value: ``Array[float] = [-0.9, -0.5, 0.0, 0.5, 0.9]``

.. data:: INVERSE_COMP_ERROR_FUNCTION_VALUES

   Value: ``Array[float] = [0.1, 0.5, 1.0, 1.5, 1.9]``

Functions
---------

.. function:: get_module_name() -> String:

.. function:: test_error_function_performance() -> void:

.. function:: test_complementary_error_function_performance() -> void:

.. function:: test_error_function_inverse_performance() -> void:

.. function:: test_complementary_error_function_inverse_performance() -> void:

