# Missing Tests Documentation

This document tracks functionality that lacks proper test coverage and needs test implementation.

## Helper Functions Module

### Distribution Helper Functions (Currently Untested)

#### `get_cdf_value()` Function
- [ ] **test_get_cdf_value_with_valid_enum()** - Test with `StatMath.SupportedDistributions` enum
- [ ] **test_get_cdf_value_with_valid_string()** - Test with string distribution names  
- [ ] **test_get_cdf_value_invalid_distribution_type()** - Test error: "Expected StatMath.SupportedDistributions enum or String"
- [ ] **test_get_cdf_value_unimplemented_distribution()** - Test error: "CDF function not implemented for distribution"

#### `get_ppf_value()` Function  
- [ ] **test_get_ppf_value_with_valid_enum()** - Test with `StatMath.SupportedDistributions` enum
- [ ] **test_get_ppf_value_with_valid_string()** - Test with string distribution names
- [ ] **test_get_ppf_value_invalid_distribution_type()** - Test error: "Expected StatMath.SupportedDistributions enum or String"
- [ ] **test_get_ppf_value_unimplemented_distribution()** - Test error: "PPF function not implemented for distribution"

#### `string_to_distribution_enum()` Function
ON HOLD

~~- [ ] **test_string_to_distribution_enum_valid_uppercase()** - Test "NORMAL", "EXPONENTIAL", etc.~~
~~- [ ] **test_string_to_distribution_enum_valid_lowercase()** - Test "normal", "exponential", etc.~~
~~- [ ] **test_string_to_distribution_enum_invalid_string()** - Test error: "Unknown distribution string"~~

### Sampling Validation Functions (Currently Untested)

#### `validate_indices()` Function
- [ ] **test_validate_indices_valid_samples()** - Test valid index arrays
- [ ] **test_validate_indices_negative_index()** - Test error: "Sample index must be non-negative"
- [ ] **test_validate_indices_index_too_large()** - Test error: "Sample index must be less than population size"

#### `validate_unique_indices()` Function
- [ ] **test_validate_unique_indices_valid_unique_samples()** - Test valid unique index arrays
- [ ] **test_validate_unique_indices_duplicate_found()** - Test error: "Sample indices must be unique"
- [ ] **test_validate_unique_indices_size_mismatch()** - Test error: "Number of unique indices must equal sample size"

### Array Preprocessing Functions (Currently Untested)

#### `sanitize_numeric_array()` Function
- [ ] **test_sanitize_numeric_array_mixed_types()** - Test with mixed int/float/string/null inputs
- [ ] **test_sanitize_numeric_array_invalid_strings()** - Test non-numeric strings are filtered out
- [ ] **test_sanitize_numeric_array_infinite_values()** - Test INF/NAN values are filtered out
- [ ] **test_sanitize_numeric_array_empty_input()** - Test empty array input
- [ ] **test_sanitize_numeric_array_sorting()** - Test output is sorted

#### `convert_to_float_array()` Function
- [ ] **test_convert_to_float_array_valid_conversion()** - Test generic Array to Array[float] conversion
- [ ] **test_convert_to_float_array_preserves_order()** - Test order is maintained during conversion

## Implementation Notes

### Error Testing Strategy
All error tests should follow the GDUnit4 pattern:

```gdscript
func test_function_name_error_condition() -> void:
    var test_call: Callable = func():
        StatMath.HelperFunctions.function_name(invalid_params)
    
    # Test that error is logged
    await assert_error(test_call).is_push_error("Expected exact error message")
    
    # Test that sentinel value is returned  
    var result = StatMath.HelperFunctions.function_name(invalid_params)
    assert_that(is_nan(result)).is_true()  # or appropriate sentinel check
```

### Test File Locations
- **Parameter validation tests**: `addons/godot-stat-math/tests/core/helper_functions/parameter_validation_test.gd`
- **Mathematical property tests**: `addons/godot-stat-math/tests/core/helper_functions/mathematical_property_test.gd`  
- **SciPy validation tests**: `addons/godot-stat-math/tests/core/helper_functions/scipy_validation_test.gd`

### Priority
**High Priority**: Distribution helper functions (`get_cdf_value`, `get_ppf_value`, `string_to_distribution_enum`) as these are core utilities used throughout the system.

**Medium Priority**: Sampling validation functions as they support sampling functionality.

**Lower Priority**: Array preprocessing functions as they have simpler logic and fewer edge cases. 