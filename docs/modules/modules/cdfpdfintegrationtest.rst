StatMath.CdfPdfIntegrationTest
==============================

This test suite verifies integration and end-to-end statistical workflows:
• CDF and PDF consistency through probability calculations
• CDFs should be monotonically increasing
• End-to-end statistical computation validation
• Numerical stability under various conditions
• Cross-distribution mathematical relationships

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.CdfPdfIntegrationTest.function_name(parameters)

Constants
---------

.. data:: CDF_PDF_INTEGRATION_TEST_DATA

   Value: ``preload("res://addons/godot-stat-math/tables/cdf_pdf_integration_test_data.gd")``

Functions
---------

.. function:: test_scipy_derivative_validation() -> void:

   Tests derivative relationship using scipy-validated data

.. function:: test_scipy_monotonicity_validation() -> void:

   Tests monotonicity using scipy-validated data

.. function:: test_scipy_cross_function_validation() -> void:

   Tests cross-function consistency using scipy-validated data

.. function:: test_scipy_boundary_validation() -> void:

   Tests boundary behavior using scipy-validated data

.. function:: test_normal_cdf_pdf_derivative_relationship() -> void:

   Tests that the numerical derivative of Normal CDF approximates Normal PDF

.. function:: test_exponential_cdf_pdf_derivative_relationship() -> void:

   Tests that the numerical derivative of Exponential CDF approximates Exponential PDF

.. function:: test_uniform_cdf_pdf_derivative_relationship() -> void:

   Tests that the numerical derivative of Uniform CDF approximates Uniform PDF

.. function:: test_beta_cdf_pdf_derivative_relationship() -> void:

   Tests that the numerical derivative of Beta CDF approximates Beta PDF

.. function:: test_weibull_cdf_pdf_derivative_relationship() -> void:

   Tests that the numerical derivative of Weibull CDF approximates Weibull PDF

.. function:: test_gamma_exponential_mathematical_relationship() -> void:

   Tests Gamma distribution probability consistency (Gamma(1,scale) = Exponential(1/scale))

.. function:: test_end_to_end_normal_distribution_workflow() -> void:

   Tests a complete statistical workflow: data generation → analysis → validation

.. function:: test_numerical_stability_extreme_values() -> void:

   Tests numerical stability under extreme parameter values

.. function:: test_integration_parameter_validation() -> void:

   Tests parameter validation for integration functions

