StatMath.DistributionsParameterValidationTest
=============================================

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.DistributionsParameterValidationTest.function_name(parameters)

Functions
---------

.. function:: test_randi_bernoulli_invalid_p_too_low() -> void:

.. function:: test_randi_bernoulli_invalid_p_too_high() -> void:

.. function:: test_randi_binomial_invalid_p_too_low() -> void:

.. function:: test_randi_binomial_invalid_p_too_high() -> void:

.. function:: test_randi_binomial_invalid_n_negative() -> void:

.. function:: test_randi_geometric_invalid_p_zero() -> void:

.. function:: test_randi_geometric_invalid_p_too_low() -> void:

.. function:: test_randi_geometric_invalid_p_too_high() -> void:

.. function:: test_randi_poisson_invalid_lambda_zero() -> void:

.. function:: test_randi_poisson_invalid_lambda_negative() -> void:

.. function:: test_randi_pseudo_invalid_c_param_zero() -> void:

.. function:: test_randi_pseudo_invalid_c_param_negative() -> void:

.. function:: test_randi_pseudo_invalid_c_param_too_high() -> void:

.. function:: test_randi_seige_invalid_w_too_low() -> void:

.. function:: test_randi_seige_invalid_w_too_high() -> void:

.. function:: test_randi_seige_invalid_c0_too_low() -> void:

.. function:: test_randi_seige_invalid_c0_too_high() -> void:

.. function:: test_randi_uniform_invalid_min_greater_than_max() -> void:

.. function:: test_randf_uniform_invalid_a_greater_than_b() -> void:

.. function:: test_randf_exponential_invalid_lambda_zero() -> void:

.. function:: test_randf_exponential_invalid_lambda_negative() -> void:

.. function:: test_randf_erlang_invalid_k_zero() -> void:

.. function:: test_randf_erlang_invalid_k_negative() -> void:

.. function:: test_randf_erlang_invalid_lambda_zero() -> void:

.. function:: test_randf_erlang_invalid_lambda_negative() -> void:

.. function:: test_randf_normal_invalid_sigma_negative() -> void:

.. function:: test_randf_normal_invalid_sigma_zero() -> void:

.. function:: test_randf_cauchy_invalid_scale_zero() -> void:

.. function:: test_randf_cauchy_invalid_scale_negative() -> void:

.. function:: test_randv_histogram_empty_values() -> void:

.. function:: test_randv_histogram_empty_probabilities() -> void:

.. function:: test_randv_histogram_mismatched_sizes() -> void:

.. function:: test_randv_histogram_non_numeric_probability() -> void:

.. function:: test_randv_histogram_negative_probability() -> void:

.. function:: test_randv_histogram_zero_sum_probabilities() -> void:

.. function:: test_randf_gamma_invalid_shape_zero() -> void:

.. function:: test_randf_gamma_invalid_shape_negative() -> void:

.. function:: test_randf_gamma_invalid_scale_zero() -> void:

.. function:: test_randf_gamma_invalid_scale_negative() -> void:

.. function:: test_randf_beta_invalid_alpha_zero() -> void:

.. function:: test_randf_beta_invalid_alpha_negative() -> void:

.. function:: test_randf_beta_invalid_beta_zero() -> void:

.. function:: test_randf_beta_invalid_beta_negative() -> void:

.. function:: test_randf_triangular_invalid_mode_too_low() -> void:

.. function:: test_randf_triangular_invalid_mode_too_high() -> void:

.. function:: test_randf_triangular_invalid_max_less_than_min() -> void:

.. function:: test_randf_triangular_invalid_max_equal_min_with_different_mode() -> void:

.. function:: test_randf_pareto_invalid_scale_zero() -> void:

.. function:: test_randf_pareto_invalid_scale_negative() -> void:

.. function:: test_randf_pareto_invalid_shape_zero() -> void:

.. function:: test_randf_pareto_invalid_shape_negative() -> void:

.. function:: test_randf_weibull_invalid_scale_zero() -> void:

.. function:: test_randf_weibull_invalid_scale_negative() -> void:

.. function:: test_randf_weibull_invalid_shape_zero() -> void:

.. function:: test_randf_weibull_invalid_shape_negative() -> void:

