# Add Property-Based Testing with Random Parameter Generation

**Label: enhancement**

## Overview
Our current test suite has excellent mathematical property testing but lacks true property-based testing with random parameter generation. We should complement our existing example-based tests with property-based tests that use randomly generated inputs to catch edge cases.

## Current State: Strong Foundation ✅
- Excellent test organization with dedicated `mathematical_property_test.gd` files
- Testing the RIGHT mathematical properties (monotonicity, round-trip consistency, boundary conditions)
- Great coverage of numerical stability and edge cases
- Comprehensive scipy validation with data-driven approach

## Missing: True Property-Based Testing ❌
Our tests currently use **fixed test cases** rather than **generated random inputs**:

**Current Pattern (Example-Based):**
```gdscript
func test_pareto_cdf_monotonicity() -> void:
    # Fixed test points
    var x1: float = 2.5
    var x2: float = 3.0
    var x3: float = 4.0
    # Test specific cases...
```

**Needed Pattern (Property-Based):**
```gdscript
func test_pareto_cdf_monotonicity_property() -> void:
    var rng = RandomNumberGenerator.new()
    rng.seed = 12345  # Deterministic for CI
    
    # Generate many random test cases
    for i in range(100):
        var scale = rng.randf_range(0.1, 10.0)
        var shape = rng.randf_range(0.5, 5.0)
        var x1 = rng.randf_range(scale, scale * 10.0)
        var x2 = x1 + rng.randf_range(0.1, 5.0)
        
        # Property: CDF should be monotonic
        var cdf1 = StatMath.CdfFunctions.pareto_cdf(x1, scale, shape)
        var cdf2 = StatMath.CdfFunctions.pareto_cdf(x2, scale, shape)
        assert_float(cdf1).is_less_equal(cdf2)
```

## Proposed Improvements

### 1. CDF Functions
- Random parameter generation for range validation (CDF ∈ [0,1])
- Random monotonicity testing across parameter spaces
- Random boundary condition validation

### 2. Basic Statistics  
- Generated data property testing (variance ≥ 0)
- Random data transformation properties
- Scale/translation invariance with random data

### 3. PPF Functions
- Enhanced round-trip consistency with random parameters
- Random monotonicity validation
- Random boundary behavior testing

## Implementation Strategy
- **Keep existing tests**: Current scipy validation and example-based tests are valuable
- **Add complementary property-based tests**: New test functions with `_property` suffix
- **Use deterministic seeds**: Ensure CI reproducibility
- **Reasonable test counts**: Balance coverage vs execution time (50-200 iterations per property)

## Success Criteria
- [ ] All CDF functions have property-based range validation
- [ ] All statistical functions have property-based invariant testing  
- [ ] All PPF functions have enhanced round-trip property testing
- [ ] Maintain existing test pass rates
- [ ] No significant CI execution time increase

## Priority
**Medium** - This enhancement will improve test coverage and catch edge cases, but existing test suite already provides strong validation. 