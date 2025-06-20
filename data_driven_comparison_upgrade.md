# Data Driven Comparison Upgrade Plan - Phase 2

## Overview
This plan addresses the systematic replacement of hardcoded test assertion values with scipy-generated data stored in the `/addons/godot-stat-math/tables/` folder, creating a fully data-driven testing approach.

## Problem Statement
Test files contain hardcoded expected values that should be replaced with scipy-generated data to:
- Eliminate magic numbers in test assertions
- Ensure scientific accuracy through scipy validation
- Improve test maintainability and traceability
- Create a single source of truth for expected values

## Target Classification: Type B Values

### Current Pattern (BAD)
```gdscript
# Hardcoded expected values - these need to be eliminated
assert_float(StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)).is_equal_approx(0.975, StatMath.FLOAT_TOLERANCE)
assert_float(StatMath.ErrorFunctions.erf(2.0)).is_equal_approx(0.9953223, StatMath.ERF_APPROX_TOLERANCE)
```

### Target Pattern (GOOD)
```gdscript
# Data-driven approach using scipy-generated values
func test_error_function_positive() -> void:
	var test_data: Array = ERROR_FUNCTIONS_TEST_DATA.VALUES["erf"]
	var case: Dictionary = test_data[1]  # erf(1.0) -> 0.84270079
	var result: float = StatMath.ErrorFunctions.erf(case["params"][0])
	# Using larger tolerance for error function approximation precision
	assert_float(result).is_equal_approx(case["expected"], StatMath.ERF_APPROX_TOLERANCE)
```

## Available Test Data Files

### Existing Tables (✅ Available)
- `basic_stats_test_data.gd` - Statistical calculations
- `cdf_test_data.gd` - Cumulative distribution functions
- `distributions_test_data.gd` - Distribution parameters and results
- `error_functions_test_data.gd` - Error function values
- `helper_functions_test_data.gd` - Helper function calculations
- `pmf_pdf_test_data.gd` - Probability mass/density functions
- `ppf_test_data.gd` - Percent point functions
- `sampling_test_data.gd` - Sampling and random generation
- `cdf_pdf_integration_test_data.gd` - Integration tests

### Data Structure Pattern
```gdscript
# Standard test data structure
const VALUES: Dictionary = {
	"function_name": [
		{
			"params": [param1, param2, ...],
			"expected": scipy_calculated_result,
			"description": "Human readable test case description"
		}
	]
}
```

## Available StatMath Constants

### Standard Tolerances (Keep Using)
- `StatMath.FLOAT_TOLERANCE = 1.0e-7` - General floating-point comparisons
- `StatMath.HIGH_PRECISION_TOLERANCE = 1.0e-9` - High-precision operations
- `StatMath.BOUNDARY_TOLERANCE = 1.0e-10` - Boundary conditions

### Specialized Tolerances (Keep Using)
- `StatMath.ERF_APPROX_TOLERANCE = 1.0e-5` - Error function approximations
- `StatMath.PROBABILITY_TOLERANCE = 1.0e-6` - Probability values
- `StatMath.NUMERICAL_TOLERANCE = 1.0e-5` - Numerical methods
- `StatMath.NUMERICAL_INTEGRATION_TOLERANCE = 5.0e-3` - PDF integration

## Rules of Engagement

### Alpha Software Principles
- ❌ **NO backward compatibility concerns** - freely modify data structures
- ✅ **Fix data generation at source** - edit `generate_test_data.py` for structure issues
- ✅ **Follow established patterns** - study existing table files and test usage
- ✅ **Eliminate workarounds** - remove manual data transformation functions

### Always Apply
- ✅ Replace hardcoded expected values with direct table data lookups
- ✅ Keep using appropriate StatMath tolerance constants
- ✅ Maintain test readability and documentation
- ✅ Preserve test case descriptions and context

### Guidelines
- **Optimize data structures for test consumption, not data generation convenience**
- **Remove manual transformation functions like `_build_*_test_parameters()`**
- **Design table data to match actual test usage patterns**
- **Generate missing data in `generate_test_data.py` following existing patterns**

### Notes
- **Use StatMath.SupportedDistributions enum where possible**
- **Tests with multiple similar scenarios use gdunit4 parametrized tests**
- **Tests should be named after what they are testing, not implementation**


## Current State Assessment

### ✅ Already Data-Driven
Some test files have already been upgraded to use table data:
- Tests with `_build_*_test_parameters()` functions 
- Tests using `*_TEST_DATA.VALUES` imports
- Example: `cdf_functions_test.gd` has `_build_cdf_test_parameters()`

### 🔧 Data Structure Issues Identified
The `_build_cdf_test_parameters()` function reveals poor data design:
```gdscript
# BAD: Manual data transformation needed
for case in data["normal_cdf"]:
    test_params.append(["normal", case.params[0], [case.params[1], case.params[2]], case.expected])
```

**ROOT CAUSE**: Table data structure doesn't match test usage patterns.
**SOLUTION**: Fix data generation in `generate_test_data.py`, not test workarounds.

## Execution Checklist

### 📋 Task 1: Audit Phase - Assess Current State
- [ ] Identify which tests are already data-driven vs. hardcoded
- [ ] Audit existing table data structures for usability issues
- [ ] Find manual data transformation functions like `_build_*_test_parameters()`
- [ ] Document data structure improvements needed in `generate_test_data.py`
- [ ] Scan remaining tests for hardcoded expected values in assertions

### 📋 Task 2: Data Structure Optimization
**PRIORITY**: Fix table data generation before converting remaining tests.

#### 🔧 Improve `generate_test_data.py`
- [ ] Analyze usage patterns in existing data-driven tests
- [ ] Redesign data structures to eliminate manual transformations
- [ ] Follow established patterns in the codebase
- [ ] Regenerate all table data with improved structures
- [ ] **Note**: No backward compatibility needed (alpha software)

#### 📊 Target Data Structure Improvements:
- [ ] Eliminate need for `_build_*_test_parameters()` helper functions
- [ ] Create test-friendly data shapes that match usage patterns
- [ ] Ensure consistent data organization across all table files
- [ ] Add missing test data for any gaps identified

### 📋 Task 3: Conversion Phase - File by File Upgrades

#### 📁 Core Test Files Status:
- [ ] `stat_math_test.gd` - Core StatMath functionality
- [ ] `basic_stats_test.gd` - Statistical functions  
- [🔄] `cdf_functions_test.gd` - CDF calculations (PARTIALLY UPGRADED - needs data structure fix)
- [ ] `distributions_test.gd` - Distribution functions
- [✅] `error_functions_test.gd` - Error function calculations (CHECK IF COMPLETE)
- [ ] `helper_functions_test.gd` - Helper utilities
- [ ] `pmf_pdf_functions_test.gd` - PMF/PDF functions
- [ ] `ppf_functions_test.gd` - Percent point functions
- [ ] `sampling_gen_test.gd` - Sampling and generation
- [ ] `cdf_pdf_integration_test.gd` - Integration tests

#### 🔍 For Each File:
- [ ] Remove any manual data transformation functions
- [ ] Replace hardcoded assertions with direct table lookups
- [ ] Verify all test cases are covered by optimized table data
- [ ] Ensure proper tolerance constants are used
- [ ] Run tests to verify no regressions
- [ ] Document any special cases or exceptions

### 📋 Task 4: Validation Phase - Quality Assurance
- [ ] Run full test suite to ensure no regressions
- [ ] Verify scipy data accuracy through spot checks
- [ ] Confirm all hardcoded assertion values have been eliminated
- [ ] Review test coverage and completeness
- [ ] Update documentation if needed

### 📋 Task 5: Documentation Phase - Update Records
- [ ] Update test file headers with table dependencies
- [ ] Document any new patterns or conventions
- [ ] Record any limitations or special cases
- [ ] Create migration summary report

## Success Criteria

### ✅ Phase 2 Complete When:
- [ ] All hardcoded expected values replaced with table data
- [ ] All tests continue to pass
- [ ] No scipy-generated magic numbers remain in test files
- [ ] Test data tables are properly utilized
- [ ] Code is more maintainable and scientifically traceable

### 📊 Quality Metrics:
- **Coverage**: 100% of eligible hardcoded values converted
- **Accuracy**: All scipy data properly mapped to test cases
- **Maintainability**: Clear data-driven test patterns established
- **Traceability**: All expected values traceable to scipy calculations

## Example Conversion Pattern

### Before (Hardcoded)
```gdscript
func test_normal_cdf_standard_values() -> void:
	# These hardcoded values need to be replaced
	assert_float(StatMath.CdfFunctions.normal_cdf(0.0, 0.0, 1.0)).is_equal_approx(0.5, StatMath.FLOAT_TOLERANCE)
	assert_float(StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)).is_equal_approx(0.975, StatMath.FLOAT_TOLERANCE)
```

### After (Data-Driven)
```gdscript
func test_normal_cdf_standard_values() -> void:
	var test_data: Array = CDF_TEST_DATA.VALUES["normal_cdf"]
	
	# Test case: normal_cdf(0.0, 0.0, 1.0) -> 0.5
	var case_median: Dictionary = test_data[0]
	var result_median: float = StatMath.CdfFunctions.normal_cdf(
		case_median["params"][0], 
		case_median["params"][1], 
		case_median["params"][2]
	)
	assert_float(result_median).is_equal_approx(case_median["expected"], StatMath.FLOAT_TOLERANCE)
	
	# Test case: normal_cdf(1.96, 0.0, 1.0) -> 0.975
	var case_975: Dictionary = test_data[1]
	var result_975: float = StatMath.CdfFunctions.normal_cdf(
		case_975["params"][0],
		case_975["params"][1], 
		case_975["params"][2]
	)
	assert_float(result_975).is_equal_approx(case_975["expected"], StatMath.FLOAT_TOLERANCE)
```

---

## 🎯 MISSION STATUS: **READY TO EXECUTE**

**OBJECTIVE**: Replace all hardcoded test assertion values with scipy-generated data from tables folder.

**IMPACT**: This upgrade will eliminate the final category of magic numbers, creating a fully data-driven, scientifically accurate test suite.

**PREREQUISITES**: 
- ✅ Phase 1 magic numbers cleanup completed
- ✅ Table data files already exist and contain scipy-generated values
- ✅ StatMath tolerance constants properly established

**NEXT STEPS**: Begin with Task 1 audit to assess current data-driven state and identify data structure optimization opportunities in `generate_test_data.py`. 


## UNCOVERED ISSUES: **WILL BE ADDED TO AS WE PROGRESS**

### ✅ ISSUE #1: CDF Data Structure Mismatch (RESOLVED)
**Location**: `cdf_functions_test.gd` line 8-34
**Problem**: The `_build_cdf_test_parameters()` function manually reshapes data because table structure doesn't match test usage
**Data Issue**: Table stores `{ "params": [1.96, 0.0, 1.0], "expected": 0.975 }` but test needs `["normal", 1.96, [0.0, 1.0], 0.975]`
**✅ RESOLUTION**: Fixed `generate_test_data.py` to output test-ready structure: `["normal", 1.96, [0.0, 1.0], 0.97500210]`
**✅ IMPACT**: Eliminated `_build_cdf_test_parameters()` function, all 152 CDF tests passing with direct table usage

### 🎯 AUDIT PHASE COMPLETE - BATTLE ASSESSMENT:

#### ✅ ALREADY DATA-DRIVEN (Keep as reference examples):
- **error_functions_test.gd** - ✅ COMPLETE - Perfect table data usage
- **cdf_functions_test.gd** - 🔄 PARTIAL - Has data structure issue requiring fix

#### ❌ HARDCODED MAGIC NUMBER INFESTATIONS:
- **stat_math_test.gd** - 🔥 MASSIVE (constants testing has hardcoded values)
- ✅ **basic_stats_test.gd** - ✅ **CONVERTED** (60/60 tests passed with scipy data!)
- **pmf_pdf_functions_test.gd** - 🔥 MAJOR (calculated hardcoded probability values)
- **ppf_functions_test.gd** - ⚠️ LIGHT (special value hardcoded assertions)
- **helper_functions_test.gd** - ⚠️ MODERATE (utility function hardcoded values)
- **cdf_pdf_integration_test.gd** - ⚠️ LIGHT (boundary value hardcoded assertions)

#### 📊 INFECTION STATISTICS:
- **Total Files Contaminated**: 6 out of 9 core test files (67% infection rate!)
- **Hardcoded Assertion Count**: 150+ detected instances
- **Priority Files for Conversion**: `basic_stats_test.gd`, `pmf_pdf_functions_test.gd`, `stat_math_test.gd`

### ✅ ISSUE #2: Basic Stats Hardcoded Values (RESOLVED)
**Location**: `basic_stats_test.gd` lines 17-311
**Problem**: Extensive hardcoded expected values in test assertions
**Examples**: `assert_float(result).is_equal_approx(3.0, StatMath.FLOAT_TOLERANCE)`
**✅ RESOLUTION**: Converted all hardcoded values to use `BASIC_STATS_TEST_DATA.VALUES` 
**✅ IMPACT**: All 60 tests passing, eliminated 50+ hardcoded magic numbers

### 🚨 NEXT TARGETS:
1. **pmf_pdf_functions_test.gd** - 🔥 MAJOR contamination priority
2. **stat_math_test.gd** - 🔥 MASSIVE contamination  
3. **helper_functions_test.gd** - ⚠️ MODERATE contamination