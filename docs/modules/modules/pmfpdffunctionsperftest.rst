StatMath.PmfPdfFunctionsPerfTest
================================

Performance Test Suite for StatMath.

PmfPdfFunctions Module
Tests probability mass and density function calculations
across discrete and continuous distributions.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PmfPdfFunctionsPerfTest.function_name(parameters)

Constants
---------

.. data:: TEST_VALUES

   Value: ``Array[float] = [0.5, 1.0, 2.0, 5.0]``

Functions
---------

.. function:: get_module_name() -> String:

.. function:: test_normal_pdf_performance() -> void:

.. function:: test_exponential_pdf_performance() -> void:

.. function:: test_uniform_pdf_performance() -> void:

.. function:: test_gamma_pdf_performance() -> void:

.. function:: test_beta_pdf_performance() -> void:

.. function:: test_binomial_pmf_performance() -> void:

.. function:: test_poisson_pmf_performance() -> void:

.. function:: test_chi_squared_pdf_performance() -> void:

.. function:: test_t_pdf_performance() -> void:

.. function:: test_f_pdf_performance() -> void:

.. function:: test_negative_binomial_pmf_performance() -> void:

