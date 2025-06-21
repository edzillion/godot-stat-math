# Basic Stats Test Migration Checklist

## Original File: `basic_stats_test.gd` (43 tests)
**Target Structure:** 3 specialized files in `addons/godot-stat-math/tests/core/basic_stats/`

---

## SCIPY VALIDATION TESTS (17 tests)
**Target File:** `scipy_validation_test.gd`

- [x] `test_mean_bimodal_data` ✓ Line 12
- [x] `test_mean_integer_like_floats` ✓ Line 18
- [x] `test_mean_single_value` ✓ Line 24
- [x] `test_median_odd_count` ✓ Line 31
- [x] `test_median_even_count` ✓ Line 37
- [x] `test_median_single_value` ✓ Line 43
- [x] `test_variance_bimodal_data` ✓ Line 50
- [x] `test_variance_single_value` ✓ Line 56
- [x] `test_standard_deviation_bimodal_data` ✓ Line 63
- [x] `test_standard_deviation_single_value` ✓ Line 69
- [x] `test_sample_variance_bimodal_data` ✓ Line 76
- [x] `test_sample_variance_very_large_numbers` ✓ Line 82
- [x] `test_sample_standard_deviation_bimodal_data` ✓ Line 89
- [x] `test_median_absolute_deviation_bimodal_data` ✓ Line 96
- [x] `test_range_spread_bimodal_data` ✓ Line 103
- [x] `test_minimum_bimodal_data` ✓ Line 109
- [x] `test_maximum_bimodal_data` ✓ Line 115

---

## MATHEMATICAL PROPERTY TESTS (16 tests)
**Target File:** `mathematical_property_test.gd`

- [x] `test_median_unsorted_data` ✓ Line 12
- [x] `test_median_with_unsorted_array_violation` ✓ Line 21
- [x] `test_median_enhanced_decimal_precision` ✓ Line 43
- [x] `test_median_with_repeated_decimal_values` ✓ Line 53
- [x] `test_right_skewed_data_behavior` ✓ Line 60
- [x] `test_left_skewed_data_behavior` ✓ Line 74
- [x] `test_heavy_tailed_data_robustness` ✓ Line 87
- [x] `test_bimodal_data_characteristics` ✓ Line 108
- [x] `test_power_law_data_handling` ✓ Line 121
- [x] `test_very_large_numbers_stability` ✓ Line 135
- [x] `test_very_small_numbers_precision` ✓ Line 153
- [x] `test_mixed_magnitude_data_robustness` ✓ Line 166
- [x] `test_close_numbers_precision_stability` ✓ Line 181
- [x] `test_identical_values_numerical_stability` ✓ Line 194
- [x] `test_integer_like_float_precision` ✓ Line 210
- [x] `test_variance_standard_deviation_relationship` ✓ Line 225

---

## PARAMETER VALIDATION TESTS (10 tests)
**Target File:** `parameter_validation_test.gd`

- [x] `test_mean_empty_array` ✓ Line 11
- [x] `test_median_empty_array` ✓ Line 20
- [x] `test_variance_empty_array` ✓ Line 29
- [x] `test_standard_deviation_empty_array` ✓ Line 38
- [x] `test_sample_variance_single_value` ✓ Line 47
- [x] `test_sample_standard_deviation_single_value` ✓ Line 54
- [x] `test_range_spread_empty_array` ✓ Line 61
- [x] `test_minimum_empty_array` ✓ Line 70
- [x] `test_maximum_empty_array` ✓ Line 79
- [x] `test_median_absolute_deviation_empty_array` ✓ Line 88

---

## VERIFICATION STATUS
- **Original file tests:** 43
- **Successfully migrated tests:** 43 ✅
- **Additional tests created:** 1 (`test_median_absolute_deviation_basic_properties` in mathematical_property_test.gd)
- **Total tests in reorganized structure:** 44

## MIGRATION SUMMARY
✅ **PHASE 2 COMPLETED SUCCESSFULLY**
- **scipy_validation_test.gd:** 17/17 tests migrated ✅
- **mathematical_property_test.gd:** 16/16 tests migrated + 1 additional test ✅  
- **parameter_validation_test.gd:** 10/10 tests migrated ✅

**ALL TESTS PASSING:** 44/44 ✅
- **0 failures, 0 errors, 0 skipped**
- **Total execution time:** ~1.7 seconds

**All 43 original tests have been successfully migrated to the appropriate specialized files according to the testing organization policy! The `basic_stats` module is now ready for the next phase of reorganization.** 