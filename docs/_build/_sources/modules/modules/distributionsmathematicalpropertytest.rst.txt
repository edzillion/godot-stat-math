StatMath.DistributionsMathematicalPropertyTest
==============================================

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.DistributionsMathematicalPropertyTest.function_name(parameters)

Constants
---------

.. data:: TEST_SEED

   Value: ``int = 777``

Functions
---------

.. function:: test_randi_uniform_range_properties() -> void:

.. function:: test_randi_uniform_deterministic_with_seed() -> void:

.. function:: test_randf_uniform_range_properties() -> void:

.. function:: test_randf_exponential_non_negative_property() -> void:

.. function:: test_randf_erlang_non_negative_property() -> void:

.. function:: test_randf_erlang_exponential_equivalence() -> void:

.. function:: test_randf_gaussian_returns_float() -> void:

.. function:: test_randf_normal_default_parameters() -> void:

.. function:: test_randf_normal_sigma_zero() -> void:

.. function:: test_randf_normal_typical_case() -> void:

.. function:: test_randf_normal_negative_mu() -> void:

.. function:: test_randf_normal_statistical_properties() -> void:

.. function:: test_randf_cauchy_basic() -> void:

.. function:: test_randf_cauchy_with_location() -> void:

.. function:: test_randf_cauchy_with_scale() -> void:

.. function:: test_randf_cauchy_with_location_and_scale() -> void:

.. function:: test_randf_cauchy_negative_location() -> void:

.. function:: test_randf_cauchy_very_small_scale() -> void:

.. function:: test_randf_cauchy_large_scale() -> void:

.. function:: test_randf_cauchy_deterministic_with_seed() -> void:

.. function:: test_randf_cauchy_multiple_calls_different_values() -> void:

.. function:: test_randf_cauchy_statistical_properties() -> void:

.. function:: test_randf_cauchy_damage_variation() -> void:

.. function:: test_randf_cauchy_market_price_fluctuation() -> void:

.. function:: test_randf_cauchy_npc_reaction_time() -> void:

.. function:: test_randf_cauchy_particle_velocity_distribution() -> void:

.. function:: test_randf_cauchy_procedural_terrain_height() -> void:

.. function:: test_randf_cauchy_heavy_tails_property() -> void:

.. function:: test_randv_histogram_basic_case() -> void:

.. function:: test_randv_histogram_probabilities_not_normalized() -> void:

.. function:: test_randv_histogram_single_value() -> void:

.. function:: test_randv_histogram_single_value_non_one_prob() -> void:

.. function:: test_rng_determinism_with_set_seed() -> void:

.. function:: test_randf_gamma_basic() -> void:

.. function:: test_randf_gamma_shape_one() -> void:

.. function:: test_randf_gamma_shape_less_than_one() -> void:

.. function:: test_randf_gamma_shape_greater_than_one() -> void:

.. function:: test_randf_gamma_large_parameters() -> void:

.. function:: test_randf_gamma_deterministic_with_seed() -> void:

.. function:: test_randf_beta_basic() -> void:

.. function:: test_randf_beta_symmetric() -> void:

.. function:: test_randf_beta_skewed_left() -> void:

.. function:: test_randf_beta_skewed_right() -> void:

.. function:: test_randf_beta_uniform() -> void:

.. function:: test_randf_beta_large_parameters() -> void:

.. function:: test_randf_beta_small_parameters() -> void:

.. function:: test_randf_beta_deterministic_with_seed() -> void:

.. function:: test_randf_gamma_statistical_properties() -> void:

.. function:: test_randf_beta_statistical_properties() -> void:

.. function:: test_randf_gamma_damage_variation() -> void:

.. function:: test_randf_beta_quality_scores() -> void:

.. function:: test_combined_gamma_beta_procedural_generation() -> void:

.. function:: test_randf_triangular_basic() -> void:

.. function:: test_randf_triangular_symmetric() -> void:

.. function:: test_randf_triangular_left_skewed() -> void:

.. function:: test_randf_triangular_right_skewed() -> void:

.. function:: test_randf_triangular_mode_at_minimum() -> void:

.. function:: test_randf_triangular_mode_at_maximum() -> void:

.. function:: test_randf_triangular_negative_range() -> void:

.. function:: test_randf_triangular_mixed_sign_range() -> void:

.. function:: test_randf_triangular_small_range() -> void:

.. function:: test_randf_triangular_large_range() -> void:

.. function:: test_randf_triangular_deterministic_with_seed() -> void:

.. function:: test_randf_triangular_multiple_calls_different_values() -> void:

.. function:: test_randf_triangular_degenerate_case_equal_bounds() -> void:

.. function:: test_randf_triangular_nearly_equal_bounds() -> void:

.. function:: test_randf_triangular_statistical_properties() -> void:

.. function:: test_randf_triangular_mode_bias_verification() -> void:

.. function:: test_randf_triangular_ai_decision_confidence() -> void:

.. function:: test_randf_triangular_loot_quality() -> void:

.. function:: test_randf_triangular_multiple_parameters() -> void:

.. function:: test_randf_triangular_npc_stat_generation() -> void:

.. function:: test_randf_triangular_pricing_variation() -> void:

.. function:: test_randf_triangular_procedural_terrain_height() -> void:

.. function:: test_randf_triangular_resource_spawn_rate() -> void:

.. function:: test_randf_triangular_skill_check_difficulty() -> void:

.. function:: test_randf_triangular_weapon_damage() -> void:

.. function:: test_randf_pareto_basic() -> void:

.. function:: test_randf_pareto_scale_parameter() -> void:

.. function:: test_randf_pareto_large_scale() -> void:

.. function:: test_randf_pareto_small_scale() -> void:

.. function:: test_randf_pareto_very_small_shape() -> void:

.. function:: test_randf_pareto_large_shape() -> void:

.. function:: test_randf_pareto_deterministic_with_seed() -> void:

.. function:: test_randf_pareto_multiple_calls_different_values() -> void:

.. function:: test_randf_pareto_statistical_properties() -> void:

.. function:: test_randf_pareto_wealth_distribution() -> void:

.. function:: test_randf_pareto_multiple_applications() -> void:

.. function:: test_randf_pareto_city_population() -> void:

.. function:: test_randf_pareto_concentration_near_minimum() -> void:

.. function:: test_randf_pareto_different_shapes() -> void:

.. function:: test_randf_pareto_heavy_tail_property() -> void:

.. function:: test_randf_pareto_loot_rarity() -> void:

.. function:: test_randf_pareto_market_price_spikes() -> void:

.. function:: test_randf_pareto_network_effect_scaling() -> void:

.. function:: test_randf_pareto_player_skill_gaps() -> void:

.. function:: test_randf_pareto_power_law_scaling() -> void:

.. function:: test_randf_pareto_quest_reward_scaling() -> void:

.. function:: test_randf_pareto_resource_deposits() -> void:

.. function:: test_randf_weibull_basic() -> void:

.. function:: test_randf_weibull_scale_parameter() -> void:

.. function:: test_randf_weibull_exponential_case() -> void:

.. function:: test_randf_weibull_rayleigh_case() -> void:

.. function:: test_randf_weibull_different_shapes() -> void:

.. function:: test_randf_weibull_large_scale() -> void:

.. function:: test_randf_weibull_small_scale() -> void:

.. function:: test_randf_weibull_very_small_shape() -> void:

.. function:: test_randf_weibull_large_shape() -> void:

.. function:: test_randf_weibull_deterministic_with_seed() -> void:

.. function:: test_randf_weibull_multiple_calls_different_values() -> void:

.. function:: test_randf_weibull_statistical_properties() -> void:

.. function:: test_randf_weibull_shape_effect_on_distribution() -> void:

.. function:: test_randf_weibull_exponential_equivalence() -> void:

.. function:: test_randf_weibull_equipment_durability() -> void:

.. function:: test_randf_weibull_wind_speed_simulation() -> void:

.. function:: test_randf_weibull_survival_time_modeling() -> void:

.. function:: test_randf_weibull_component_reliability() -> void:

.. function:: test_randf_weibull_weather_event_duration() -> void:

.. function:: test_randf_weibull_quest_completion_time() -> void:

.. function:: test_randf_weibull_resource_depletion() -> void:

.. function:: test_randf_weibull_player_session_length() -> void:

.. function:: test_randf_weibull_network_latency_spikes() -> void:

.. function:: test_randf_weibull_multiple_reliability_applications() -> void:

