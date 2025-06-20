# Test Improvement Plan

## Overview
This document outlines the observations, suggestions, and outstanding tasks for improving the test suites in the Godot Stat Math project. The goal is to ensure comprehensive test coverage, consistent testing patterns, and reliable validation against established statistical libraries.

**Status: Phase 2 COMPLETED (728 tests passing). Ready for Phase 3.**

## General Observations & Patterns

### Consistency Improvements
1. Floating-point comparison tolerance:
   - Move `FLOAT_TOLERANCE` constant (1e-7) to `StatMath` class with other constants
   - Can be used both in tests and core functionality where appropriate
   - Remove individual test file tolerance definitions once moved

**NOTE: IMPORTANT. NO MAGIC NUMBERS - generate real values to test against using generate_test_data.py**


2. Test Data Generation: ✓
   - New pattern established using `generate_test_data.py` with scipy/numpy validation ✓
   - Stores pre-calculated test values in `/tables` directory ✓
   - Provides reliable reference values for complex mathematical functions ✓

### Best Practices
1. Test Organization: ✓
   - Group tests by function/distribution ✓
   - Include basic functionality, edge cases, and special mathematical relationships ✓
   - Add property-based tests where applicable ✓

2. Test Coverage Categories: ✓
   - Basic functionality ✓
   - Edge cases (zero, negative values, boundaries) ✓
   - Parameter validation ✓
   - Special mathematical relationships ✓
   - Numerical stability ✓
   - Distribution-specific properties ✓

## Test Suite Specific Analysis

### PMF/PDF Functions Test Suite

#### Current State
- Basic test coverage for all distributions ✓
- Some special case relationships tested ✓
- Parameter validation implemented ✓
- Some property-based tests present ✓

#### Improvements Needed
1. Data-Driven Testing: ✓
   - Implement scipy-validated test data for:
     - Student's t-distribution PDF ✓
     - Beta PDF ✓
     - Chi-squared PDF ✓
     - Lognormal PDF (already implemented) ✓
     - Weibull PDF (already implemented) ✓
     - F-distribution PDF (already implemented) ✓

2. Property Testing: ✓ **COMPLETED**
   - Restore non-negativity tests for Lognormal and Weibull PDFs ✓
   - Add integration tests (total probability = 1 for valid ranges) ✓
   - Add monotonicity tests where applicable ✓

3. Special Cases: ✓
   - Restore Beta PDF boundary behavior tests ✓
   - Add tests for relationships between distributions ✓
   - Add limit behavior tests ✓

### Outstanding Tasks

#### PMF/PDF Functions - **PHASE 2 COMPLETED** ✅
- [✓] Generate scipy-validated test data for all continuous distributions
- [✓] Add integration tests to verify PDF properties (7 comprehensive integration tests added)
- [✓] Restore and enhance property-based tests **COMPLETED**
- [✓] Add comprehensive boundary tests for all distributions
- [✓] Implement numerical stability tests for extreme values
- [✓] Add tests for special mathematical relationships between distributions
- [✓] Restore Weibull distribution tests for:
  - Monotonicity ✓
  - Special case relationships (Exponential and Rayleigh) ✓
  - Boundary conditions ✓
  - Deterministic behavior ✓
  - Parameter validation ✓
- [✓] Add tests for unsorted data in median calculations
- [✓] Add comprehensive tests for percentile function
- [✓] Enhance test coverage for decimal data (high-precision tests added)

#### Basic Stats - **PHASE 2 COMPLETED** ✅
- [✓] Enhanced decimal precision tests for mean, variance, standard deviation, percentiles
- [✓] Comprehensive unsorted data behavior tests for median calculations  
- [✓] High-precision decimal data validation (up to 1e-9 tolerance)
- [✓] Repeated decimal value testing
- [✓] Percentile interpolation with high-precision data

#### CDF Functions - **PHASE 3 TASKS**
- [ ] Generate scipy-validated test data for all CDFs
- [ ] Add monotonicity tests
- [ ] Test relationship with PDFs (derivative relationship)
- [ ] Add boundary value tests (limits at ±∞)
- [ ] Test special values (median, quartiles)
- [ ] Add known-value tests validated against scipy.stats
- [ ] Enhance parameter validation tests for probability bounds

#### PPF Functions - **PHASE 3 TASKS**
- [ ] Generate scipy-validated test data for all PPFs
- [ ] Test inverse relationship with CDFs
- [ ] Add boundary tests (0 and 1 probabilities)
- [ ] Test standard probability points (median, quartiles)

#### Basic Stats - **PHASE 3 TASKS**
- [ ] Add tests for large datasets (stress testing)
- [ ] Test with non-normal distributions
- [ ] Add stress tests for numerical stability
- [ ] Test with integer vs float inputs
- [ ] Add edge case tests for single-element arrays

## Implementation Strategy

### Phase 1: Test Data Generation ✓ **COMPLETED**
1. Extend `generate_test_data.py` to cover all distributions ✓
2. Add validation data for special mathematical relationships ✓
3. Include edge cases and boundary values ✓

### Phase 2: Test Enhancement ✅ **COMPLETED** 
1. Implement data-driven tests using generated test data ✓
2. Restore and enhance property-based tests ✓ **COMPLETED**
3. Add comprehensive boundary and special case tests ✓
4. **Added:** PDF integration tests (7 distributions) ✓
5. **Added:** Enhanced decimal precision testing ✓
6. **Added:** Comprehensive unsorted data behavior tests ✓

### Phase 3: Integration Testing **READY TO START**
1. Add tests for relationships between different functions (CDF ↔ PDF relationships)
2. Implement end-to-end statistical computation tests
3. Add performance benchmarks for critical operations
4. **Add:** Cross-function mathematical relationship validation
5. **Add:** Large dataset stress testing
6. **Add:** Numerical stability under extreme conditions

## Phase 2 Summary: **728 Tests Passing** 

### Key Achievements:
- **PDF Integration Testing:** 7 comprehensive tests ensuring PDFs integrate to 1.0
- **Weibull Distribution:** Complete test coverage (monotonicity, special cases, boundary conditions)
- **Decimal Precision:** High-precision testing (1e-9 tolerance) for statistical functions
- **Unsorted Data Behavior:** "Crash early" philosophy validation for median calculations
- **Enhanced Percentile Testing:** Interpolation behavior with high-precision data
- **Mathematical Relationships:** Exponential↔Weibull, Rayleigh↔Weibull, Normal↔Lognormal

### Test Count Breakdown:
- **PMF/PDF Functions:** 162 tests
- **Basic Stats:** Enhanced with precision and edge case tests
- **Overall:** 728 tests passing across all test suites

## Notes
- All new tests follow the "crash early" philosophy ✓
- Focus on mathematical correctness and numerical stability ✓
- Maintain balance between test coverage and execution time ✓
- Document any assumptions or limitations in test cases ✓
- This is alpha software, we do not need to document changes
- Tests with multiple similar scenarios use gdunit4 parametrized tests ✓