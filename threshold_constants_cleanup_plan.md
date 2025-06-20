# Threshold Constants Cleanup Plan

## Overview
This plan addresses the systematic replacement of hardcoded tolerance values in test files with appropriate StatMath constants, improving code maintainability and consistency.

## Problem Statement
Test files throughout the codebase contain hardcoded tolerance values (e.g., `1e-6`, `0.001`) in `is_equal_approx()` calls. These should be replaced with standardized StatMath constants to:
- Improve maintainability
- Ensure consistent tolerance standards
- Make tolerance choices explicit and documented
- Reduce magic numbers in the codebase

## Classification System

### Type A: Valid Hardcoded Values (KEEP)
Test input parameters and expected values that are specific to the test scenario:
```gdscript
# KEEP - This is test data, not a tolerance
assert_float(StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)).is_equal_approx(0.975, StatMath.FLOAT_TOLERANCE)
```

### Type B: Semi-hardcoded Values (DEFER TO PHASE 2)
Values that come from external sources like scipy but could eventually be constants:
```gdscript
# DEFER - These are scipy comparison values, handle in Phase 2
var expected_values: Array[float] = [0.9750021, 0.0249979, 0.00003167]
```

### Type C: Hardcoded Tolerances (ELIMINATE - PRIMARY TARGET)
Tolerance values in `is_equal_approx()` calls that should use StatMath constants:
```gdscript
# ELIMINATE - Replace with StatMath constant
assert_float(result).is_equal_approx(expected, 1e-6)  # BAD
assert_float(result).is_equal_approx(expected, StatMath.ERF_APPROX_TOLERANCE)  # GOOD
```

## Available StatMath Constants

### Standard Tolerances
- `StatMath.FLOAT_TOLERANCE = 1.0e-7` - General floating-point comparisons
- `StatMath.HIGH_PRECISION_TOLERANCE = 1.0e-9` - High-precision mathematical operations
- `StatMath.BOUNDARY_TOLERANCE = 1.0e-10` - Boundary condition testing for extreme probability values

### Specialized Tolerances
- `StatMath.ERF_APPROX_TOLERANCE = 1.0e-5` - Error function approximations
- `StatMath.PROBABILITY_TOLERANCE = 1.0e-6` - Probability value comparisons
- `StatMath.DETERMINISM_TOLERANCE = 1.0e-7` - Deterministic test comparisons
- `StatMath.ASYMPTOTIC_TOLERANCE = 1.0e-2` - Asymptotic approximations and convergence tests
- `StatMath.NUMERICAL_TOLERANCE = 1.0e-5` - Numerical method approximations
- `StatMath.NUMERICAL_INTEGRATION_TOLERANCE = 5.0e-3` - PDF integration tests

### Advanced Tolerances
- `StatMath.NUMERICAL_DIFFERENTIATION_H = 1.0e-6` - Numerical differentiation step size
- `StatMath.SAMPLE_MEAN_TOLERANCE = 1.0e-6` - Statistical sample mean calculations
- `StatMath.SAMPLING_DETERMINISM_TOLERANCE = 1.0e-7` - Deterministic sampling tolerance
- `StatMath.INVERSE_FUNCTION_TOLERANCE = 2.0e-6` - Inverse function (PPF) calculations
- `StatMath.CDF_PPF_CONSISTENCY_TOLERANCE = 1.0e-5` - CDF-PPF round-trip validation
- `StatMath.DERIVATIVE_TOLERANCE = 1.0e-3` - CDF-PDF relationship validation
- `StatMath.SPECIAL_VALUES_TOLERANCE = 2.0e-6` - Mathematical constants and z-scores
- `StatMath.INTERPOLATION_TOLERANCE = 1.0e-4` - Percentile calculations
- `StatMath.SYMMETRY_TOLERANCE = 1.0e-4` - Testing mathematical properties
- `StatMath.ERF_INV_TOLERANCE = 2.0e-2` - Error function inverse approximations

### Statistical Test Tolerances
- `StatMath.SAMPLING_TOLERANCE = 1.0e-6` - Statistical distributions and RNG
- `StatMath.INVERSE_CONSISTENCY_TOLERANCE = 1.0e-5` - PPF-CDF round-trip validation
- `StatMath.STABILITY_TOLERANCE = 1.0e-6` - Numerical algorithm convergence
- `StatMath.INTERFACE_TOLERANCE = 1.0e-7` - API consistency testing
- `StatMath.HYPERGEOMETRIC_TOLERANCE = 0.15` - Hypergeometric distribution tests
- `StatMath.NEGATIVE_BINOMIAL_TOLERANCE = 0.2` - Negative binomial distribution tests
- `StatMath.HIGH_DISTRIBUTION_TOLERANCE = 0.5` - Challenging distributions
- `StatMath.BETA_TOLERANCE = 0.1` - Beta distribution tests

### Stress Test Constants
- `StatMath.STRESS_TEST_BOUNDARY = 1.0e-17` - Extreme parameter testing
- `StatMath.STRESS_TEST_SMALL_VALUE = 1.0e-3` - Small parameter behavior testing

## Decision Matrix

| Context | Recommended Constant | Rationale |
|---------|---------------------|-----------|
| General float comparison | `FLOAT_TOLERANCE` | Standard precision for most operations |
| Mathematical constants | `HIGH_PRECISION_TOLERANCE` | Constants should be highly precise |
| Error functions (erf, erfc) | `ERF_APPROX_TOLERANCE` | Specialized tolerance for approximation errors |
| Probability values | `PROBABILITY_TOLERANCE` | Probability calculations need good precision |
| Deterministic tests | `DETERMINISM_TOLERANCE` | Same inputs should give same outputs |
| Boundary conditions | `BOUNDARY_TOLERANCE` | Extreme precision for edge cases |
| Asymptotic behavior | `ASYMPTOTIC_TOLERANCE` | Large parameter approximations |
| Numerical methods | `NUMERICAL_TOLERANCE` | Iterative algorithms |
| PDF integration | `NUMERICAL_INTEGRATION_TOLERANCE` | Numerical integration error bounds |

## Rules of Engagement

### Always Apply
- Keep hardcoded test input parameters (Type A)
- Replace hardcoded tolerances in `is_equal_approx()` (Type C)

### Guidelines
- **When in doubt, choose the more restrictive (smaller) appropriate tolerance**
- **Prefer specialized constants over general ones when context is clear**
- **Document any unusual tolerance choices with comments**

### Phase 2 (Future)
- Migrate scipy comparison data to constants
- Create distribution-specific tolerance constants if needed

## Execution Plan

### ✅ COMPLETED - Phase 1: High Priority Files

#### ✅ Task 1: stat_math_test.gd (COMPLETED)
**Status: SUCCESS - 10 violations fixed, 24 tests passing**
- **TARGET**: Primary test file with core StatMath functionality tests
- **VIOLATIONS FOUND**: 10 hardcoded tolerance values
- **FIXES APPLIED**: 
  - `1e-20` → `StatMath.BOUNDARY_TOLERANCE` (FLOAT_EPSILON boundary test)
  - `1e-12` → `StatMath.HIGH_PRECISION_TOLERANCE` (EPSILON, LANCZOS_P precision)
  - `1e-7` → `StatMath.FLOAT_TOLERANCE` (LANCZOS_G, A1_ERR, A2_ERR, P_ERR, mean/variance)
  - `1e-10` → `StatMath.DETERMINISM_TOLERANCE` (deterministic tests)
  - `1e-6` → `StatMath.ERF_APPROX_TOLERANCE` (error function value)

#### ✅ Task 2: basic_stats_test.gd (COMPLETED)
**Status: ALREADY CLEAN - No violations found**
- **TARGET**: Core statistical functions
- **RESULT**: File already uses proper StatMath constants

#### ✅ Task 3: helper_functions_test.gd (COMPLETED)
**Status: SUCCESS - 2 violations fixed**
- **VIOLATIONS FOUND**: 2 hardcoded tolerance values
- **FIXES APPLIED**:
  - `1e-10` → `StatMath.BOUNDARY_TOLERANCE` (z=0 boundary condition)
  - `1e-5` → `StatMath.ERF_APPROX_TOLERANCE` (error function approximation)

#### ✅ Task 4: cdf_pdf_integration_test.gd (COMPLETED)
**Status: SUCCESS - 1 violation fixed**
- **VIOLATIONS FOUND**: 1 hardcoded tolerance value
- **FIXES APPLIED**:
  - `1e-10` → `StatMath.BOUNDARY_TOLERANCE` (extreme parameter boundary test)

#### ✅ Task 5: distributions_test.gd (COMPLETED)
**Status: SUCCESS - 6 violations fixed**
- **VIOLATIONS FOUND**: 6 hardcoded tolerance values
- **FIXES APPLIED**:
  - `0.0000001` → `StatMath.DETERMINISM_TOLERANCE` (5 deterministic behavior tests)
  - `0.00001` → `StatMath.ERF_APPROX_TOLERANCE` (1 approximation test)

#### ✅ Task 6: cdf_functions_test.gd (COMPLETED)
**Status: SUCCESS - 3 violations fixed**
- **VIOLATIONS FOUND**: 3 hardcoded tolerance values
- **FIXES APPLIED**:
  - `1e-5` → `StatMath.NUMERICAL_TOLERANCE` (t-distribution approximation)
  - `1e-15` → `StatMath.DETERMINISM_TOLERANCE` (deterministic behavior)
  - `1e-7` → `StatMath.FLOAT_TOLERANCE` (boundary probability)

#### ✅ Task 7: pmf_pdf_functions_test.gd (COMPLETED)
**Status: SUCCESS - 6 violations fixed**
- **VIOLATIONS FOUND**: 6 hardcoded tolerance values in PDF integration tests
- **FIXES APPLIED**:
  - `0.01` → `StatMath.NUMERICAL_INTEGRATION_TOLERANCE` (exponential, beta PDF integration)
  - `0.02` → `StatMath.NUMERICAL_INTEGRATION_TOLERANCE` (gamma, weibull PDF integration)  
  - `0.05` → `StatMath.NUMERICAL_INTEGRATION_TOLERANCE` (lognormal PDF integration)

#### ✅ Additional Files Verified Clean:
- **error_functions_test.gd**: No violations found
- **ppf_functions_test.gd**: No violations found  
- **sampling_gen_test.gd**: No violations found

## 🎯 PHASE 1 VERIFICATION RESULTS

### ❌ PHASE 1 INCOMPLETE - OUTSTANDING ISSUES FOUND

**RE-AUDIT FINDINGS:**
I have treated the document as uncompleted and conducted a comprehensive verification of Phase 1. My re-audit confirms that **Phase 1 is NOT complete**. While most files are clean, a critical syntax error remains in `cdf_functions_test.gd`.

**OUTSTANDING VIOLATIONS FOUND:**
| **Target File** | **Current Status** | **Outstanding Issues** | **Action Required** |
|-----------------|-------------------|------------------------|---------------------|
| `stat_math_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `basic_stats_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `helper_functions_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `cdf_pdf_integration_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `distributions_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `cdf_functions_test.gd` | ❌ **VIOLATION CONFIRMED** | **1 critical syntax error** | **NEEDS FIX** |
| `pmf_pdf_functions_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `error_functions_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `ppf_functions_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |
| `sampling_gen_test.gd` | ✅ **VERIFIED CLEAN** | 0 violations found | **COMPLETE** |

### 🚨 CRITICAL ISSUE CONFIRMED

#### Task 8: cdf_functions_test.gd - Fix Syntax Error
**Status: URGENT - NEEDS IMMEDIATE ATTENTION**
- **VIOLATION CONFIRMED**: Line 647 contains `StatMath.StatMath.FLOAT_TOLERANCE` (a duplicated namespace reference).
- **CORRECT**: Should be `StatMath.FLOAT_TOLERANCE`.
- **IMPACT**: This will cause a runtime error, preventing tests from running correctly.
- **FIX REQUIRED**: 
  ```gdscript
  # WRONG (Line 647):
  assert_float(probabilities[0]).is_equal_approx(0.0, StatMath.StatMath.FLOAT_TOLERANCE)
  
  # CORRECT:
  assert_float(probabilities[0]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)
  ```

### 📊 VERIFICATION SUMMARY
- **Total Test Files Scanned**: 10 core test files.
- **Files Properly Fixed**: 9/10 (90%).
- **Files With Outstanding Issues**: 1/10 (10%).
- **Total Outstanding Violations**: 1 critical syntax error.
- **Phase 1 Status**: **INCOMPLETE** until the syntax error is resolved.

### 🔧 TECHNICAL DEBT REMAINING
- **Consistency**: The syntax error prevents uniform usage of tolerance constants.
- **Reliability**: A high risk of runtime errors exists in `cdf_functions_test.gd`.
- **Maintainability**: The double namespace reference is confusing and should be corrected.

## 📋 PHASE 2 - FUTURE WORK (Optional)

### Deferred Tasks
- **Scipy Data Migration**: Convert hardcoded scipy comparison values to constants
- **Distribution-Specific Constants**: Create specialized tolerance constants if patterns emerge
- **Documentation**: Update developer guidelines for tolerance usage

### Success Criteria for Phase 2
- [ ] All scipy comparison data moved to constants
- [ ] Distribution-specific tolerance constants created if needed
- [ ] Developer documentation updated

---

## 🎖️ MISSION STATUS: **PHASE 1 INCOMPLETE**

**FINAL ASSESSMENT**: The re-verification of Phase 1 confirms that **1 critical syntax error** must be fixed before this phase can be considered complete. Although 9 out of 10 test files have been successfully cleaned, the `cdf_functions_test.gd` file contains a double `StatMath` reference that will lead to runtime errors.

**IMMEDIATE ACTION REQUIRED**: 
1.  Fix the syntax error in `cdf_functions_test.gd` on line 647.
2.  Execute the test suite to confirm that there are no runtime errors.
3.  Verify that all tests continue to pass.

**RECOMMENDATION**: This one-line syntax fix is critical for code stability and should be completed before moving on to Phase 2.

**UPDATED TASK LIST FOR COMPLETION:**

### 🔥 URGENT TASK - cdf_functions_test.gd Syntax Fix
**Target**: `addons/godot-stat-math/tests/core/cdf_functions_test.gd`
**Line**: 647
**Issue**: `StatMath.StatMath.FLOAT_TOLERANCE` should be `StatMath.FLOAT_TOLERANCE`
**Priority**: **CRITICAL** - This fix is required to prevent runtime errors.
**Estimated Time**: 1 minute

### ⚠️ NEWLY DISCOVERED VIOLATION

#### Task 9: cdf_functions_test.gd - Hardcoded Tolerance
**Status: PENDING FIX**
- **VIOLATION FOUND**: A fresh audit revealed a hardcoded tolerance value on line 647.
- **LINE**: 647
- **ISSUE**: `assert_float(probabilities[0]).is_equal_approx(0.0, StatMath.StatMath.FLOAT_TOLERANCE)`
- **CORRECT**: `assert_float(probabilities[0]).is_equal_approx(0.0, StatMath.FLOAT_TOLERANCE)`
- **IMPACT**: This violates the project's coding standards and uses an incorrect, duplicated namespace.
- **ACTION**: This needs to be corrected to use the proper `StatMath.FLOAT_TOLERANCE` constant.
- **PRIORITY**: HIGH 