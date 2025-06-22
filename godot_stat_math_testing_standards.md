# Godot Stat Math Testing Standards

## Overview
This document establishes comprehensive testing standards for the Godot Stat Math project, derived from extensive data-driven conversion work and testing best practices. These standards ensure accuracy, maintainability, and consistency across the entire test suite.

## CI/CD Integration and Platform Consistency

### Dedicated AWS Testing Infrastructure
The project uses **dedicated AWS Spot instances** for all test execution, triggered via Lambda functions. This eliminates cross-platform floating-point precision concerns that typically plague statistical computing projects.

#### Development Workflow & Test Execution Points:
1. **PR to `develop`** → Full GDUnit4 test suite on AWS Spot instances
2. **PR to `release`** → Version validation + comprehensive testing  
3. **Manual Test Runs** → Development iterations with committed results

#### Key Benefits:
- **Consistent Hardware**: Identical AWS instances eliminate platform-specific variations
- **Test Result History**: All test reports committed to branches for historical tracking
- **Performance Baselines**: Automated regression detection with snapshot comparisons
- **Isolated Testing**: Fresh, identical environments for every test run

This infrastructure allows us to focus on mathematical accuracy without worrying about hardware-specific floating-point differences that would otherwise require platform-specific tolerance adjustments.

### Performance Testing Integration
Performance tests run alongside standard tests in `run-tests.yaml` and automatically generate baseline snapshots for regression detection.

**Known Regression Handling:**
- Tests with known performance issues can be marked as regression cases
- Marked tests will not execute until new performance baselines are established
- This prevents false failures while allowing continued development

**Baseline Regeneration:**
- Major algorithmic changes may require baseline regeneration across all functions
- Run `run-tests.yaml` multiple times to establish new performance baselines
- The system automatically commits updated baseline data to the repository

This approach maintains performance monitoring while allowing flexibility during development cycles.

## Core Principles

### 1. Data-Driven Testing Philosophy
- **Single Source of Truth**: All expected values must come from scipy-generated data stored in `/addons/godot-stat-math/tables/`
- **Scientific Accuracy**: Every numerical assertion must be traceable to scipy calculations
- **Eliminate Magic Numbers**: No hardcoded expected values in test assertions (except mathematical constants)
- **Scipy Documentation**: All generated table data must include scipy function call documentation

### 2. Alpha Software Development Approach
- **No Backward Compatibility**: Freely modify data structures and APIs for improvement
- **Fix at Source**: Edit `generate_test_data.py` to resolve data structure issues rather than creating workarounds
- **Crash Early Philosophy**: Trust code contracts, fail fast, prioritize rapid development over defensive programming
- **Simplicity First**: Prefer simple solutions over complex ones, smaller additions over larger ones

### 3. Test Organization and Structure
- **Semantic Test Names**: Name tests after what they validate, not implementation details
- **Logical Grouping**: Organize tests into clear sections with descriptive comments
- **Type Safety**: Always use static types and typed assertions (`assert_float`, `assert_int`, etc.)
- **No Redundancy**: Eliminate duplicate tests and merge similar functionality

## Data-Driven Standards

### Required Table Data Structure
```gdscript
# Standard test data pattern with scipy documentation
const VALUES: Dictionary = {
	"function_name": [  # Generated using: scipy.stats.function_name(params)
		{
			"params": [param1, param2, ...],
			"expected": scipy_calculated_result,
			"description": "Human readable test case description"
		}
	]
}
```

### Acceptable Hardcoded Values
**✅ KEEP HARDCODED** - Mathematical constants and well-known limits:
- Zero values: `erf(0.0) = 0.0`, `erfc(0.0) = 1.0`
- Asymptotic limits: `erf(10.0) ≈ 1.0`, `erf(-10.0) ≈ -1.0`
- Mathematical constants: `sqrt(PI)`, factorial identities like `0! = 1`
- Boundary conditions: `uniform_cdf(x) = 0.0` when `x < a`, `= 1.0` when `x > b`
- Identity relationships: `gamma(1.0) = 1.0`

**❌ MUST CONVERT** - Scipy-calculated precision values:
- Complex probability calculations: `0.9750021`, `0.8646647`, `0.59399415`
- Non-obvious mathematical results requiring computation
- Values that appear as "magic numbers" without clear mathematical justification

**Exception: Simple, Illustrative Test Data**

It is acceptable to hardcode simple data directly within a single test function if its primary purpose is to illustrate a specific behavior, data handling characteristic, or edge case, rather than to validate a complex mathematical result against a scientific standard.

**Criteria for "Simple Test Data" Exception:**
- **No computed expected values**: The data is not the result of scientific/mathematical calculations
- **Self-evident from context**: Data structure and purpose is immediately clear (e.g., `[1, 2, 3, 4, 5]` for median testing)
- **Single test usage**: Data is not reused across multiple tests
- **Illustrative purpose**: Primary goal is demonstrating behavior, not validating mathematical accuracy

**Examples of acceptable simple data:**
```gdscript
# ✅ Acceptable - simple array for median behavior
func test_median_with_odd_count() -> void:
	var data: Array[float] = [1.0, 3.0, 2.0, 5.0, 4.0]
	assert_float(StatMath.BasicStats.median(data)).is_equal_approx(3.0, StatMath.FLOAT_TOLERANCE)

# ✅ Acceptable - edge case demonstration  
func test_empty_array_handling() -> void:
	var empty_data: Array[float] = []
	assert_that(is_nan(StatMath.BasicStats.mean(empty_data))).is_true()
```

**Move to tables when:**
- Data is needed for more than one test
- Expected values represent calculated results
- Data represents complex scenarios requiring scipy validation

This approach maintains test readability for simple cases while enforcing data-driven standards for mathematical validation.

### Data Generation Requirements
- **Scipy Call Documentation**: Every function must include exact scipy call used
- **Format**: `"function_name": [  # Generated using: stats.norm.cdf(x, mu, sigma)`
- **Comprehensive Coverage**: Include all test scenarios needed by actual tests
- **Optimized Structure**: Design for test consumption, not generation convenience

## Test Implementation Patterns

### Types of Testing Approaches

#### 1. **Example-Based Testing** (Current Primary Approach)
Tests specific input → output pairs using known correct values:
```gdscript
func test_normal_cdf_specific_case() -> void:
	var result = StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)
	assert_float(result).is_equal_approx(0.97500210, StatMath.FLOAT_TOLERANCE)
```

#### 2. **Data-Driven Testing** (Implemented via Scipy Tables)
Uses precomputed test cases from scipy for validation:
```gdscript
func test_normal_cdf_scipy_validation() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["normal_cdf"]
	var case: Dictionary = test_data[0]
	var result: float = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
	assert_float(result).is_equal_approx(case["expected"], StatMath.FLOAT_TOLERANCE)
```

#### 3. **Property-Based Testing** (Future Enhancement)
Tests mathematical relationships that should always hold using randomly generated inputs:
```gdscript
func test_cdf_range_property() -> void:
	var rng = RandomNumberGenerator.new()
	rng.seed = 12345
	
	for i in range(100):
		var x = rng.randf_range(-10.0, 10.0)
		var mu = rng.randf_range(-3.0, 3.0) 
		var sigma = rng.randf_range(0.1, 2.0)
		
		var result = StatMath.CdfFunctions.normal_cdf(x, mu, sigma)
		assert_float(result).is_between(0.0, 1.0)
```

> **Note**: Our current test suite has excellent mathematical property testing with fixed examples but lacks true property-based testing with random parameter generation. This enhancement will be undertaken at a later date. See: **[https://github.com/edzillion/godot-stat-math/issues/10]** for details.

### Standard Test Function Pattern
```gdscript
func test_function_name_descriptive_case() -> void:
	var test_data: Array = TEST_DATA_TABLE.VALUES["function_name"]
	var case: Dictionary = test_data[index]  # function_name(params) -> expected
	var result: float = StatMath.Module.function_name(case["params"][0], case["params"][1])
	assert_float(result).is_equal_approx(case["expected"], StatMath.APPROPRIATE_TOLERANCE)
```

### Parametrized Test Pattern
```gdscript
func test_function_comprehensive_validation() -> void:
	var test_data: Array = TEST_DATA_TABLE.VALUES["function_name"]
	for case in test_data:
		var result: float = StatMath.Module.function_name(case["params"][0], case["params"][1])
		assert_float(result).is_equal_approx(case["expected"], StatMath.APPROPRIATE_TOLERANCE)
```

### Error Testing Pattern
```gdscript
func test_function_invalid_parameter() -> void:
	var test_call: Callable = func():
		StatMath.Module.function_name(invalid_param)
	
	# Test error logging
	await assert_error(test_call).is_push_error("Expected exact error message")
	
	# Test sentinel return value
	var result = StatMath.Module.function_name(invalid_param)
	assert_that(is_nan(result)).is_true()  # or assert_that(result).is_null()
```

## Anti-Patterns to Avoid

### 🔥 Terrible Abstraction Patterns (ELIMINATED)
**❌ NEVER USE:**
- String-to-enum conversion functions (`_string_to_enum()`)
- Generic helper wrapper functions (`_get_cdf_value()`, `_get_ppf_value()`)
- Variant parameter types for distribution selection
- Fallback error handling that hides problems
- Manual data transformation functions (`_build_*_test_parameters()`)

**✅ USE INSTEAD:**
- Direct StatMath function calls with proper typing
- StatMath.SupportedDistributions enum for type safety
- Crash early with clear error messages
- Direct table data lookups without transformation

### Deprecated Test Patterns
**❌ OLD PATTERN:**
```gdscript
func test_with_hardcoded_values() -> void:
	assert_float(StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)).is_equal_approx(0.975, tolerance)
```

**✅ NEW PATTERN:**
```gdscript
func test_normal_cdf_scipy_validated() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["normal_cdf"]
	var case: Dictionary = test_data[2]  # normal_cdf(1.96, 0.0, 1.0) -> 0.97500210
	var result: float = StatMath.CdfFunctions.normal_cdf(case["params"][0], case["params"][1], case["params"][2])
	assert_float(result).is_equal_approx(case["expected"], StatMath.INVERSE_FUNCTION_TOLERANCE)
```

## Tolerance Constants Usage

### Tolerance Decision Tree
Use this decision tree to eliminate tolerance selection paralysis:

```
├── Testing mathematical identity or boundary condition?
│   └── YES → StatMath.BOUNDARY_TOLERANCE (1.0e-10)
│
├── Testing against scipy validation data?
│   ├── Basic statistical functions (mean, variance, etc.)
│   │   └── StatMath.FLOAT_TOLERANCE (1.0e-7)
│   ├── Approximation algorithms (erf, gamma, etc.)
│   │   └── StatMath.ERF_APPROX_TOLERANCE (1.0e-5)
│   ├── Inverse functions (ppf, quantiles)
│   │   └── StatMath.INVERSE_FUNCTION_TOLERANCE
│   └── Probability density/mass functions
│       └── StatMath.PROBABILITY_TOLERANCE (1.0e-6)
│
├── Testing numerical integration or iterative methods?
│   └── StatMath.NUMERICAL_INTEGRATION_TOLERANCE (5.0e-3)
│
└── High-precision mathematical operations?
    └── StatMath.HIGH_PRECISION_TOLERANCE (1.0e-9)
```

### Standard Tolerances
- `StatMath.FLOAT_TOLERANCE = 1.0e-7` - General floating-point comparisons
- `StatMath.HIGH_PRECISION_TOLERANCE = 1.0e-9` - High-precision operations
- `StatMath.BOUNDARY_TOLERANCE = 1.0e-10` - Boundary conditions

### Specialized Tolerances
- `StatMath.ERF_APPROX_TOLERANCE = 1.0e-5` - Error function approximations
- `StatMath.PROBABILITY_TOLERANCE = 1.0e-6` - Probability values
- `StatMath.NUMERICAL_TOLERANCE = 1.0e-5` - Numerical methods
- `StatMath.NUMERICAL_INTEGRATION_TOLERANCE = 5.0e-3` - PDF integration
- `StatMath.INVERSE_FUNCTION_TOLERANCE` - PPF and inverse functions
- `StatMath.CDF_PPF_CONSISTENCY_TOLERANCE` - Round-trip consistency tests

### Tolerance Selection Guidelines
- Use the decision tree above to eliminate guesswork
- Consider numerical method precision when selecting tolerances
- Document why specific tolerances are chosen for edge cases
- When in doubt, start with the most restrictive tolerance and adjust upward if tests fail

## File Organization Standards

### Test File Structure
```gdscript
# res://addons/godot-stat-math/tests/core/module_test.gd
class_name ModuleTest extends GdUnitTestSuite

const MODULE_TEST_DATA = preload("res://addons/godot-stat-math/tables/module_test_data.gd")

# =============================================================================
# SCIPY VALIDATION TESTS - DATA-DRIVEN
# =============================================================================

## Validates function against scipy values
func test_function_scipy_validation() -> void:
	# Implementation using table data

# =============================================================================
# MATHEMATICAL PROPERTY TESTS
# =============================================================================

## Tests mathematical properties and relationships
func test_function_mathematical_properties() -> void:
	# Implementation testing known mathematical relationships

# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

## Tests parameter validation and error handling
func test_function_parameter_validation() -> void:
	# Implementation testing error conditions
```

### Table Data File Structure
```gdscript
# res://addons/godot-stat-math/tables/module_test_data.gd
class_name ModuleTestData

const VALUES: Dictionary = {
	"function_name": [  # Generated using: scipy.stats.function_name(params)
		{
			"params": [param1, param2],
			"expected": scipy_result,
			"description": "Descriptive test case"
		}
	]
}
```

## Testing Workflow

### 1. Data Generation Process
1. Identify missing or inadequate test data in `generate_test_data.py`
2. Add scipy function calls with proper parameter ranges
3. Include scipy call documentation in comments
4. Generate comprehensive test data covering edge cases
5. Validate generated data through spot checks

### 2. Test Implementation Process
1. Review existing test patterns for consistency
2. Implement data-driven tests using table lookups
3. Eliminate hardcoded expected values (except mathematical constants)
4. Use appropriate tolerance constants
5. Test both success and error conditions

### 3. Quality Assurance Process
1. Run full test suite to ensure no regressions
2. Verify all tests use data-driven patterns
3. Check that scipy documentation is present
4. Confirm elimination of anti-patterns
5. Validate test coverage and edge cases

## Performance Considerations

### Test Execution Efficiency
- Use parallel tool execution for information gathering
- Minimize sequential operations in test discovery
- Cache table data appropriately for repeated access
- Avoid redundant calculations in parametrized tests

### Memory Management
- Use `auto_free()` and `queue_free()` for node cleanup
- Monitor for orphaned nodes in complex tests
- Clean up temporary resources after test completion

## Documentation Standards

### Test Documentation
- Include clear descriptions of what each test validates
- Document mathematical relationships being tested
- Explain tolerance choices for edge cases
- Reference scipy functions used for expected values

### Code Comments
```gdscript
# Mathematical constant - well-known limit
assert_float(StatMath.ErrorFunctions.erf(10.0)).is_equal_approx(1.0, StatMath.FLOAT_TOLERANCE)

# Scipy-validated precision value
var case: Dictionary = test_data[1]  # erf(1.0) -> 0.84270079
```

### Scipy Call Documentation
```gdscript
const VALUES: Dictionary = {
	"normal_cdf": [  # Generated using: stats.norm.cdf(x, loc=mu, scale=sigma)
		{
			"params": [1.96, 0.0, 1.0],
			"expected": 0.97500210485177963,
			"description": "Standard normal 97.5th percentile"
		}
	]
}
```

## Error Handling Standards

### Error Testing Requirements
- Every error condition must have a dedicated test
- Test both error logging AND return values
- Error messages must match exactly between implementation and test
- Use sentinel values (NAN, null) for invalid operations

### Error Message Format
```gdscript
if not (valid_condition):
    push_error("Clear description of what went wrong. Received: %s" % actual_value)
    return NAN
```

## Migration Guidelines

### Converting Existing Tests
1. **Audit Phase**: Identify hardcoded expected values vs. mathematical constants
2. **Data Generation**: Add missing functions to `generate_test_data.py`
3. **Conversion**: Replace hardcoded values with table lookups
4. **Validation**: Ensure all tests pass with new data-driven approach
5. **Cleanup**: Remove deprecated helper functions and anti-patterns

### Maintaining Data-Driven Standards
- Regular audits to ensure no hardcoded values creep back in
- Update table data when mathematical implementations change
- Maintain scipy call documentation for traceability
- Continuously improve data structures based on usage patterns

## Success Metrics

### Conversion Success Indicators
- **100% Data-Driven**: All expected values from scipy tables or mathematical constants
- **Zero Anti-Patterns**: No string-based abstractions or generic helpers
- **High Test Coverage**: Comprehensive validation of all functions
- **Scientific Traceability**: All values traceable to scipy calculations
- **Maintainability**: Clear, readable, and modifiable test code

### Quality Metrics
- Test pass rate > 99% (accounting for known algorithmic issues)
- Zero hardcoded "magic numbers" in assertions
- Complete scipy documentation for all generated data
- Consistent use of appropriate tolerance constants
- Elimination of redundant and duplicate tests

## Test Data Governance Process

### Scipy Version Control
All test data files generated by `generate_test_data.py` include version information for scipy and numpy in their headers:
```gdscript
# Generated with: scipy 1.11.3, numpy 1.24.3
```

This provides traceability for test data generation and ensures reproducibility when debugging edge cases.

### Minimal Governance Philosophy
Our data governance approach is intentionally minimal since we are dealing with **mathematical constants and relationships** that should not change across scipy versions. Mathematical functions like `norm.cdf(1.96, 0.0, 1.0)` represent universal mathematical truths, not implementation-specific behaviors.

**Key Principles:**
- Test data represents mathematical constants, not software behavior
- Scipy version changes should not affect mathematical correctness
- Data regeneration is only needed when adding new functions or test cases
- Version tracking provides debugging context, not compatibility requirements

This approach differs from typical software testing where external API changes require extensive data migration and compatibility testing.

## Conclusion

These standards represent the culmination of extensive data-driven conversion work and establish the foundation for maintaining accuracy and code quality in the Godot Stat Math project. By following these guidelines, future development will benefit from:

- **Accuracy**: All calculations validated against scipy
- **Maintainability**: Clear, traceable, and modifiable test code
- **Consistency**: Uniform patterns across the entire test suite
- **Reliability**: Robust error handling and edge case coverage
- **Performance**: Efficient test execution and resource management

These standards should be treated as living documentation, updated as new patterns emerge and the project evolves. 