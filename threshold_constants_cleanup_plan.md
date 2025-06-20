# Threshold Constants Cleanup Plan

## Overview
This focused planning document addresses the critical issue of hardcoded tolerance values in test files. We need to distinguish between appropriate hardcoded values (simple edge cases) and inappropriate hardcoded tolerance values that should use StatMath constants.

**Mission:** Achieve 100% consistency in tolerance usage across all test files.

## ✅ What's Perfect As-Is (Type A: Keep Hardcoded Values)
```gdscript
func test_randi_uniform_single_value_range() -> void:
	var min_val: int = 42
	var max_val: int = 42
	var result: int = StatMath.Distributions.randi_uniform(min_val, max_val)
	assert_int(result).is_equal(42)
```
**Keep these hardcoded values - they are test input parameters, not tolerance thresholds.**

## ❌ What Must Be Fixed (Type C: Replace with StatMath Constants)
```gdscript
# BAD - hardcoded tolerance
assert_float(result).is_equal_approx(expected, 1e-9)

# GOOD - proper StatMath constant  
assert_float(result).is_equal_approx(expected, StatMath.HIGH_PRECISION_TOLERANCE)
```

## Available StatMath Tolerance Constants

Based on codebase analysis, these tolerance constants are available in StatMath:

### **Primary Tolerance Constants**
- `StatMath.FLOAT_TOLERANCE` (1e-7) - Standard floating point comparisons
- `StatMath.HIGH_PRECISION_TOLERANCE` (1e-9) - High-precision calculations  
- `StatMath.PROBABILITY_TOLERANCE` (1e-6) - Probability calculations
- `StatMath.BOUNDARY_TOLERANCE` (1e-10) - Extreme boundary conditions
- `StatMath.NUMERICAL_TOLERANCE` (1e-5) - Numerical algorithms

### **Specialized Mathematical Constants**
- `StatMath.ERF_APPROX_TOLERANCE` (1e-5) - Error function approximations
- `StatMath.CDF_PPF_CONSISTENCY_TOLERANCE` (1e-5) - Round-trip CDF↔PPF validation
- `StatMath.DERIVATIVE_TOLERANCE` (1e-3) - CDF-PDF relationship validation
- `StatMath.INVERSE_FUNCTION_TOLERANCE` (2e-6) - PPF calculations
- `StatMath.INTERPOLATION_TOLERANCE` (1e-4) - Percentile interpolation
- `StatMath.ASYMPTOTIC_TOLERANCE` (1e-2) - Large-parameter approximations
- `StatMath.SYMMETRY_TOLERANCE` (1e-4) - Mathematical symmetry tests
- `StatMath.SAMPLING_TOLERANCE` (1e-6) - Statistical sampling validation
- `StatMath.DETERMINISM_TOLERANCE` (1e-7) - Reproducible sequences
- `StatMath.NUMERICAL_INTEGRATION_TOLERANCE` (5e-3) - Integration computations

### **Distribution-Specific Tolerances**
- `StatMath.HYPERGEOMETRIC_TOLERANCE` (0.15) - Hypergeometric distribution
- `StatMath.NEGATIVE_BINOMIAL_TOLERANCE` (0.2) - Negative binomial distribution  
- `StatMath.HIGH_DISTRIBUTION_TOLERANCE` (0.5) - High-variability distributions
- `StatMath.BETA_TOLERANCE` (0.1) - Beta distribution

### **Algorithmic Tolerances**
- `StatMath.STABILITY_TOLERANCE` (1e-6) - Algorithm stability
- `StatMath.INTERFACE_TOLERANCE` (1e-7) - API consistency
- `StatMath.INVERSE_CONSISTENCY_TOLERANCE` (1e-5) - Inverse function accuracy

---

## Phase 1: Fix All Hardcoded Tolerance Values

### **🎯 CRITICAL RULES:**
1. **ALL** hardcoded tolerance values in `is_equal_approx()` calls must use StatMath constants
2. **NO GENERIC TOLERANCES** like `SCIPY_COMPARISON_TOLERANCE` allowed
3. **CONTEXT-APPROPRIATE** tolerances must be used based on mathematical function
4. **REUSE EXISTING** constants - don't duplicate similar tolerance values
5. **MATHEMATICAL CONTEXT** determines which tolerance constant to use

### **Task Checklist by File:**

#### ✅ **Task 1: `stat_math_test.gd` - HIGHEST PRIORITY**
**Found Issues:**
- `1e-20` for FLOAT_EPSILON comparison
- `1e-12` for EPSILON comparison  
- `1e-7` for LANCZOS_G comparison
- `1e-12` for LANCZOS_P comparison
- `1e-7` for error function constants
- `1e-7` for basic stats functions
- `1e-10` for deterministic tests
- `1e-6` for error function values

**Actions Required:**
- [ ] Replace `1e-20` with `StatMath.BOUNDARY_TOLERANCE` (machine epsilon validation)
- [ ] Replace `1e-12` with `StatMath.HIGH_PRECISION_TOLERANCE` (high-precision constants)
- [ ] Replace `1e-7` with `StatMath.FLOAT_TOLERANCE` (standard comparisons)
- [ ] Replace `1e-10` with `StatMath.DETERMINISM_TOLERANCE` (reproducibility tests)
- [ ] Replace `1e-6` with `StatMath.ERF_APPROX_TOLERANCE` (error function tests)

#### ✅ **Task 2: `basic_stats_test.gd`**
**Found Issues:**
- `1e-10` hardcoded in extreme value tests

**Actions Required:**
- [ ] Replace `1e-10` with `StatMath.HIGH_PRECISION_TOLERANCE` (extreme small numbers)

#### ✅ **Task 3: `distributions_test.gd`**  
**Found Issues:**
- `1e-17` and `1e-6` used as test parameters (not tolerances)

**Actions Required:**
- [ ] **VERIFY** these are test parameters, not tolerance values
- [ ] If they are tolerances, replace with appropriate StatMath constants

#### ✅ **Task 4: `cdf_pdf_integration_test.gd`**
**Found Issues:**
- `1e-6`, `1e-10` used as test parameters for extreme conditions

**Actions Required:**
- [ ] **VERIFY** these are test parameters for extreme scenarios, not tolerance values
- [ ] If they are tolerances, replace with appropriate StatMath constants

#### ✅ **Task 5: Scan Remaining Test Files**
**Files to Check:**
- [ ] `cdf_functions_test.gd`
- [ ] `pmf_pdf_functions_test.gd`  
- [ ] `ppf_functions_test.gd`
- [ ] `sampling_gen_test.gd`
- [ ] `error_functions_test.gd`
- [ ] `helper_functions_test.gd`

### **Decision Matrix for Tolerance Selection:**

| **Function Type** | **Recommended Tolerance** | **Use Cases** |
|------------------|---------------------------|---------------|
| Basic Statistics | `FLOAT_TOLERANCE` | mean, median, variance, std dev |
| Probability Calculations | `PROBABILITY_TOLERANCE` | CDF, PMF, PDF values |
| PPF/Quantile Functions | `INVERSE_FUNCTION_TOLERANCE` | Normal PPF, etc. |
| Error Functions | `ERF_APPROX_TOLERANCE` | erf, erfc, gamma |
| CDF↔PPF Round-trip | `CDF_PPF_CONSISTENCY_TOLERANCE` | Inverse validation |
| High Precision Math | `HIGH_PRECISION_TOLERANCE` | Small numbers, constants |
| Boundary Conditions | `BOUNDARY_TOLERANCE` | Extreme values, limits |
| Sampling/Random | `SAMPLING_TOLERANCE` | Distribution sampling |
| Deterministic Tests | `DETERMINISM_TOLERANCE` | Reproducible sequences |
| Interpolation | `INTERPOLATION_TOLERANCE` | Percentile calculations |

### **Process for Each Fix:**

1. **Identify the mathematical context** of the test function
2. **Select appropriate tolerance** from the decision matrix above
3. **Replace hardcoded value** with StatMath constant
4. **Verify test still passes** with new tolerance
5. **Document the reasoning** if tolerance choice is non-obvious

---

## Phase 2: Scipy Comparison Data Migration (FUTURE TASK)

**Status:** 🚧 **DEFERRED - DO NOT IMPLEMENT YET**

This phase will address hardcoded scipy comparison values that should be generated by `generate_test_data.py`. Examples include:

```gdscript
# Type B violations (defer to Phase 2)
func test_some_function() -> void:
    # Hardcoded expected value that should come from scipy
    var expected: float = 0.84270079  # Should be from test data
    var result: float = StatMath.SomeFunction(1.0)
    assert_float(result).is_equal_approx(expected, proper_tolerance)
```

**Phase 2 will systematically replace these with data-driven tests using scipy-generated test data.**

---

## Success Criteria

### **Phase 1 Complete When:**
- [ ] **Zero hardcoded tolerance values** in any `is_equal_approx()` call
- [ ] **All tolerances use StatMath constants** with appropriate mathematical context
- [ ] **100% test suite passes** with new tolerance constants
- [ ] **No duplicate tolerance constants** created

### **Documentation Updates:**
- [ ] Update this plan with completion status for each task
- [ ] Note any tolerance constant additions needed
- [ ] Record any mathematical reasoning for tolerance choices

---

## Notes
- **Test names should describe the test, not implementation**
- **Use StatMath.SupportedDistributions enum, not strings**
- **This is alpha software - no need to document API changes**
- **Focus on mathematical correctness and consistency**
- **When in doubt, choose the more restrictive (smaller) appropriate tolerance**

---

**Next Action:** Begin with `stat_math_test.gd` (highest priority) and work through the checklist systematically. 